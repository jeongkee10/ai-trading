"""
AI Trading System - 5모델 스태킹 메타 러너 앙상블
XGBoost + LightGBM + LSTM + GRU + Transformer
→ Logistic Regression 메타 러너 → 최종 예측
"""

import logging
import numpy as np
import pandas as pd
import pickle
from datetime import date
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import MODEL_CONFIG, TRADING_RULES, get_all_stocks
from data.preprocessor import FeatureBuilder, TechScoreCalculator
from models.xgboost_model import XGBStockModel
from models.lightgbm_model import LGBStockModel
from models.lstm_model import LSTMStockModel
from models.gru_model import GRUStockModel
from models.transformer_model import TransformerStockModel

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")
os.makedirs(MODEL_DIR, exist_ok=True)


class EnsemblePredictor:
    """
    5모델 스태킹 앙상블 예측기
    1단계: 5개 모델이 각각 확률 예측
    2단계: 메타 러너(LogisticRegression)가 최종 확률 산출
    3단계: 모델 합의도 + 기술점수 반영 → 종합 점수
    """

    LABEL_MAP = {0: "DOWN", 1: "HOLD", 2: "UP"}
    MODEL_NAMES = ["xgb", "lgb", "lstm", "gru", "transformer"]

    def __init__(self):
        self.fb = FeatureBuilder()
        self.tc = TechScoreCalculator()
        self.models = {}      # {horizon: {"xgb": model, "lgb": model, ...}}
        self.meta_models = {} # {horizon: LogisticRegression}
        self.cfg = MODEL_CONFIG

    def train_all(self, combined_df: pd.DataFrame) -> dict:
        """전체 종목 합산 DataFrame으로 5개 모델 + 메타 러너 학습"""
        results = {}
        for horizon in self.cfg["prediction_horizons"]:
            logger.info(f"=== Horizon {horizon}d 학습 시작 ===")

            df_with_label = self.fb.create_labels(combined_df, horizon)
            self.models[horizon] = {}

            # ── 1. 개별 모델 학습 ───────────────────
            # XGBoost
            xgb = XGBStockModel(horizon=horizon)
            results[f"xgb_h{horizon}"] = xgb.train(df_with_label)
            self.models[horizon]["xgb"] = xgb

            # LightGBM
            lgb = LGBStockModel(horizon=horizon)
            results[f"lgb_h{horizon}"] = lgb.train(df_with_label)
            self.models[horizon]["lgb"] = lgb

            # LSTM
            lstm = LSTMStockModel(horizon=horizon)
            results[f"lstm_h{horizon}"] = lstm.train(df_with_label)
            self.models[horizon]["lstm"] = lstm

            # GRU
            gru = GRUStockModel(horizon=horizon)
            results[f"gru_h{horizon}"] = gru.train(df_with_label)
            self.models[horizon]["gru"] = gru

            # Transformer
            tf_model = TransformerStockModel(horizon=horizon)
            results[f"transformer_h{horizon}"] = tf_model.train(df_with_label)
            self.models[horizon]["transformer"] = tf_model

            # ── 2. 메타 러너 학습 ───────────────────
            meta_result = self._train_meta_learner(df_with_label, horizon)
            results[f"meta_h{horizon}"] = meta_result

            # 성과 요약 로그
            accs = {}
            for name in self.MODEL_NAMES:
                key = f"{name}_h{horizon}"
                r = results.get(key, {})
                accs[name] = r.get("cv_accuracy_mean", r.get("val_accuracy", 0))
            logger.info(f"  H{horizon} 개별: {', '.join(f'{k}={v:.3f}' for k,v in accs.items())}")
            if meta_result:
                logger.info(f"  H{horizon} 메타러너: {meta_result.get('meta_cv_accuracy', 'N/A'):.4f}")

        return results

    def _train_meta_learner(self, df: pd.DataFrame, horizon: int) -> dict:
        """
        메타 러너 학습 (경량): 샘플링으로 메모리 절약
        5개 모델의 예측 확률을 입력 → Logistic Regression
        """
        label_col = f"label_{horizon}d"
        avail_cols = [c for c in self.fb.get_feature_columns() if c in df.columns and df[c].isna().mean() <= 0.5]
        df_clean = df[avail_cols + ["trade_date", label_col]].dropna()

        if len(df_clean) < 200:
            logger.warning(f"  메타 러너 학습 데이터 부족: {len(df_clean)}건")
            return {}

        # 뒤 30%에서 최대 2000건만 샘플링 (메모리 절약)
        split_idx = int(len(df_clean) * 0.7)
        meta_df = df_clean.iloc[split_idx:].reset_index(drop=True)
        max_samples = 2000
        lookback = self.cfg["lookback_days"]

        # 균일 간격 샘플링
        indices = list(range(lookback, len(meta_df)))
        if len(indices) > max_samples:
            step = len(indices) // max_samples
            indices = indices[::step][:max_samples]

        meta_features = []
        meta_labels = []
        models = self.models.get(horizon, {})

        for i in indices:
            row_features = []
            for name in self.MODEL_NAMES:
                model = models.get(name)
                if model is None:
                    row_features.extend([1/3, 1/3, 1/3])
                    continue
                try:
                    if name in ["xgb", "lgb"]:
                        proba = model.predict_proba(meta_df[avail_cols].iloc[i:i+1])[0]
                    else:
                        proba = model.predict_proba(meta_df.iloc[max(0,i-lookback):i+1])[0]
                    row_features.extend(proba.tolist())
                except Exception:
                    row_features.extend([1/3, 1/3, 1/3])

            if len(row_features) == 15:
                meta_features.append(row_features)
                meta_labels.append(int(meta_df[label_col].iloc[i]))

        if len(meta_features) < 50:
            logger.warning(f"  메타 러너 피처 부족: {len(meta_features)}건")
            return {}

        X_meta = np.array(meta_features)
        y_meta = np.array(meta_labels)
        logger.info(f"  메타 러너 학습 데이터: {len(X_meta)}건")

        # Logistic Regression 메타 러너 학습
        meta_model = LogisticRegression(
            max_iter=1000, solver="lbfgs", C=1.0
        )

        # 교차 검증
        tscv = TimeSeriesSplit(n_splits=3)
        meta_scores = []
        for train_idx, val_idx in tscv.split(X_meta):
            meta_model.fit(X_meta[train_idx], y_meta[train_idx])
            y_pred = meta_model.predict(X_meta[val_idx])
            meta_scores.append(accuracy_score(y_meta[val_idx], y_pred))

        # 전체 학습
        meta_model.fit(X_meta, y_meta)
        self.meta_models[horizon] = meta_model
        self._save_meta_model(horizon)

        meta_cv_acc = float(np.mean(meta_scores))
        logger.info(f"  Meta Learner H{horizon} | CV Acc: {meta_cv_acc:.4f}")

        return {
            "meta_cv_accuracy": meta_cv_acc,
            "meta_samples": len(X_meta),
            "meta_coefs": meta_model.coef_.tolist() if hasattr(meta_model, 'coef_') else None,
        }

    def predict_single(self, ticker: str, price_df: pd.DataFrame,
                        ti_df: pd.DataFrame = None,
                        pred_date: date = None) -> list:
        """단일 종목 5모델 앙상블 예측"""
        if pred_date is None:
            pred_date = date.today()

        feat_df = self.fb.build_features(price_df, ti_df)
        if feat_df.empty or len(feat_df) < 25:
            return []

        latest = feat_df.iloc[-1]
        tech_score = self.tc.calculate_score(latest)

        predictions = []
        for horizon in self.cfg["prediction_horizons"]:
            # ── 1단계: 5개 모델 개별 예측 ───────────
            model_probas = {}
            for name in self.MODEL_NAMES:
                try:
                    model = self._get_or_load_model(name, horizon)
                    if name in ["xgb", "lgb"]:
                        proba = model.predict_proba(feat_df.tail(1))[0]
                    else:
                        proba = model.predict_proba(feat_df)[0]
                    model_probas[name] = proba
                except Exception as e:
                    model_probas[name] = np.array([1/3, 1/3, 1/3])

            # ── 2단계: 앙상블 최종 예측 ──────────
            # 활성 모델만 필터 (uniform [1/3,1/3,1/3]이 아닌 모델)
            active_models = {
                n: p for n, p in model_probas.items()
                if not np.allclose(p, [1/3, 1/3, 1/3], atol=0.01)
            }

            if len(active_models) >= 2:
                # 활성 모델만으로 가중 평균
                weights = self._calc_fallback_weights(horizon)
                total_w = sum(weights.get(n, 0.2) for n in active_models)
                ensemble_proba = sum(
                    (weights.get(n, 0.2) / total_w) * p
                    for n, p in active_models.items()
                )
            elif len(active_models) == 1:
                ensemble_proba = list(active_models.values())[0]
            else:
                # 모든 모델 실패 시 메타 러너 fallback
                meta_input = np.concatenate([model_probas[n] for n in self.MODEL_NAMES]).reshape(1, -1)
                meta_model = self._get_or_load_meta(horizon)
                if meta_model is not None:
                    ensemble_proba = meta_model.predict_proba(meta_input)[0]
                else:
                    ensemble_proba = np.array([1/3, 1/3, 1/3])

            pred_label_idx = int(np.argmax(ensemble_proba))
            pred_label = self.LABEL_MAP[pred_label_idx]

            # ── 3단계: 모델 합의도 계산 ─────────────
            individual_labels = [
                self.LABEL_MAP[int(np.argmax(model_probas[n]))]
                for n in self.MODEL_NAMES
            ]
            agreement_count = max(
                individual_labels.count("UP"),
                individual_labels.count("DOWN"),
                individual_labels.count("HOLD"),
            )
            agreement_ratio = agreement_count / len(self.MODEL_NAMES)

            # 개별 모델 시그널 (대시보드 표시용: 상위 2개)
            xgb_label = self.LABEL_MAP[int(np.argmax(model_probas["xgb"]))]
            lstm_label = self.LABEL_MAP[int(np.argmax(model_probas["lstm"]))]

            # ── 추세 컨펌 점수 ──────────────────────
            trend_confirm = self.tc.calculate_trend_confirmation(latest)

            up_prob = float(ensemble_proba[2])
            down_prob = float(ensemble_proba[0])
            hold_prob = float(ensemble_proba[1])

            # ── 5개 모델 개별 라벨 ────────────────
            lgb_label = self.LABEL_MAP[int(np.argmax(model_probas["lgb"]))]
            gru_label = self.LABEL_MAP[int(np.argmax(model_probas["gru"]))]
            tf_label = self.LABEL_MAP[int(np.argmax(model_probas["transformer"]))]

            # ── 모델 합의 기반 종합 점수 (Disagreement Penalty) ──
            # 1) 합의도 계산
            up_votes = individual_labels.count("UP")
            down_votes = individual_labels.count("DOWN")
            hold_votes = individual_labels.count("HOLD")
            n_models = len(individual_labels)

            # 2) 불일치 페널티 (업계 Practice: Agreement Filter + Penalty)
            max_votes = max(up_votes, down_votes, hold_votes)
            if max_votes >= 5:
                confidence_multiplier = 1.0    # 만장일치 → 100% 신뢰
            elif max_votes >= 4:
                confidence_multiplier = 0.85   # 4:1 → 85% 신뢰
            elif max_votes >= 3:
                confidence_multiplier = 0.60   # 3:2 → 60% 신뢰
            else:
                confidence_multiplier = 0.30   # 2:2:1 → 30% 신뢰 (거의 무의미)

            # 3) 개별 모델 확신도 (각 모델이 얼마나 확신하는지)
            avg_confidence = np.mean([
                float(np.max(model_probas[n])) for n in self.MODEL_NAMES
            ])

            # 4) 방향성 점수: -1(강한 매도) ~ 0(중립) ~ +1(강한 매수)
            ai_direction = up_prob - down_prob

            # 5) 기술/추세 점수 방향 변환
            tech_direction = tech_score - 0.5
            trend_direction = trend_confirm - 0.5

            # 6) 종합 점수 산출 (방향 반영 + 불일치 페널티)
            raw_score = (
                ai_direction * 0.50 +       # AI 예측 방향
                tech_direction * 0.25 +     # 기술적 분석 방향
                trend_direction * 0.15 +    # 추세 확인 방향
                (avg_confidence - 0.5) * 0.10  # 모델 확신도
            )

            # 불일치 페널티 적용: 모델끼리 엇갈리면 점수를 0.5(중립) 쪽으로 끌어당김
            raw_score = raw_score * confidence_multiplier

            # 0~1 정규화 (0=강한 매도, 0.5=중립, 1=강한 매수)
            total_score = np.clip((raw_score + 1) / 2, 0, 1)

            # 7) 시그널 강도 결정
            if max_votes <= 2:
                # 합의 부족 → 강제 HOLD로 재분류
                signal_strength = "WEAK"
            elif max_votes >= 4 and avg_confidence > 0.6:
                signal_strength = "STRONG"
            else:
                signal_strength = "MODERATE"

            predictions.append({
                "ticker":        ticker,
                "pred_date":     pred_date,
                "horizon":       horizon,
                "pred_up_prob":  round(up_prob, 4),
                "pred_hold_prob":round(hold_prob, 4),
                "pred_down_prob":round(down_prob, 4),
                "pred_label":    pred_label,
                "xgb_signal":    xgb_label,
                "lstm_signal":   lstm_label,
                "lgb_signal":    lgb_label,
                "gru_signal":    gru_label,
                "transformer_signal": tf_label,
                "tech_score":    round(tech_score, 4),
                "supply_score":  round(agreement_ratio, 4),
                "total_score":   round(float(total_score), 4),
                "agreement_ratio": round(agreement_ratio, 4),
                "confidence_mult": round(confidence_multiplier, 4),
                "model_version": "v2.3",
            })

        return predictions

    def explain_prediction(self, ticker: str, price_df: pd.DataFrame,
                           ti_df: pd.DataFrame = None, horizon: int = 1) -> dict:
        """
        Explainable AI: 예측 근거를 사람이 읽을 수 있는 형태로 생성
        Returns: {"summary": str, "factors": list, "model_details": dict, ...}
        """
        feat_df = self.fb.build_features(price_df, ti_df)
        if feat_df.empty or len(feat_df) < 25:
            return {"summary": "데이터 부족", "factors": []}

        latest = feat_df.iloc[-1]
        factors = []

        # ── 1. AI 모델 분석 ────────────────────────
        model_votes = {}
        for name in self.MODEL_NAMES:
            try:
                model = self._get_or_load_model(name, horizon)
                if name in ["xgb", "lgb"]:
                    proba = model.predict_proba(feat_df.tail(1))[0]
                else:
                    proba = model.predict_proba(feat_df)[0]
                label = self.LABEL_MAP[int(np.argmax(proba))]
                model_votes[name] = {"label": label, "confidence": float(np.max(proba))}
            except Exception:
                model_votes[name] = {"label": "N/A", "confidence": 0}

        up_votes = sum(1 for v in model_votes.values() if v["label"] == "UP")
        down_votes = sum(1 for v in model_votes.values() if v["label"] == "DOWN")
        factors.append({
            "category": "AI 모델",
            "signal": "BUY" if up_votes >= 3 else ("SELL" if down_votes >= 3 else "NEUTRAL"),
            "detail": f"5개 모델 중 {up_votes}개 상승, {down_votes}개 하락 예측",
            "models": model_votes,
        })

        # ── 2. 이동평균 분석 ───────────────────────
        close = latest.get("close", np.nan)
        sma5  = latest.get("sma5", np.nan)
        sma20 = latest.get("sma20", np.nan)
        sma60 = latest.get("sma60", np.nan)

        if all(pd.notna([close, sma5, sma20, sma60])):
            if close > sma5 > sma20 > sma60:
                ma_signal, ma_detail = "BUY", "정배열 (종가 > 5일 > 20일 > 60일) — 강한 상승 추세"
            elif close > sma20 > sma60:
                ma_signal, ma_detail = "BUY", f"중기 상승 추세 (종가 > 20일선 > 60일선)"
            elif close < sma5 < sma20 < sma60:
                ma_signal, ma_detail = "SELL", "역배열 (종가 < 5일 < 20일 < 60일) — 강한 하락 추세"
            elif close < sma20:
                ma_signal, ma_detail = "SELL", f"종가가 20일 이동평균 아래 — 약세"
            else:
                ma_signal, ma_detail = "NEUTRAL", "이동평균 혼조"

            golden = latest.get("golden_cross", 0)
            death = latest.get("death_cross", 0)
            if golden == 1:
                ma_detail += " + 골든크로스 발생!"
                ma_signal = "BUY"
            elif death == 1:
                ma_detail += " + 데드크로스 발생!"
                ma_signal = "SELL"

            factors.append({
                "category": "이동평균",
                "signal": ma_signal,
                "detail": ma_detail,
                "values": {"종가": f"{close:,.0f}", "5일선": f"{sma5:,.0f}",
                           "20일선": f"{sma20:,.0f}", "60일선": f"{sma60:,.0f}"},
            })

        # ── 3. RSI 분석 ───────────────────────────
        rsi = latest.get("rsi14", np.nan)
        if pd.notna(rsi):
            if rsi < 30:
                rsi_signal, rsi_detail = "BUY", f"RSI {rsi:.1f} — 과매도 구간 (반등 기대)"
            elif rsi < 40:
                rsi_signal, rsi_detail = "BUY", f"RSI {rsi:.1f} — 매수 유리 구간"
            elif rsi > 70:
                rsi_signal, rsi_detail = "SELL", f"RSI {rsi:.1f} — 과매수 구간 (조정 주의)"
            elif rsi > 60:
                rsi_signal, rsi_detail = "NEUTRAL", f"RSI {rsi:.1f} — 상승세 유지 중"
            else:
                rsi_signal, rsi_detail = "NEUTRAL", f"RSI {rsi:.1f} — 중립 구간"
            factors.append({"category": "RSI", "signal": rsi_signal, "detail": rsi_detail})

        # ── 4. MACD 분석 ──────────────────────────
        macd = latest.get("macd", np.nan)
        macd_sig = latest.get("macd_signal", np.nan)
        macd_hist = latest.get("macd_hist", np.nan)
        macd_cross = latest.get("macd_cross_up", 0)
        if all(pd.notna([macd, macd_sig])):
            if macd_cross == 1:
                mc_signal, mc_detail = "BUY", "MACD 골든크로스 발생 — 상승 전환 신호"
            elif macd > macd_sig and macd > 0:
                mc_signal, mc_detail = "BUY", "MACD 양수 + 시그널선 위 — 상승 추세 지속"
            elif macd > macd_sig:
                mc_signal, mc_detail = "BUY", "MACD가 시그널선 위 — 전환 초기"
            elif macd < macd_sig and macd < 0:
                mc_signal, mc_detail = "SELL", "MACD 음수 + 시그널선 아래 — 하락 지속"
            else:
                mc_signal, mc_detail = "SELL", "MACD가 시그널선 아래 — 약세"
            factors.append({"category": "MACD", "signal": mc_signal, "detail": mc_detail})

        # ── 5. 볼린저밴드 분석 ────────────────────
        bb_pct = latest.get("bb_pct", np.nan)
        bb_width = latest.get("bb_width", np.nan)
        if pd.notna(bb_pct):
            if bb_pct < 0.1:
                bb_signal, bb_detail = "BUY", f"볼린저밴드 하단 돌파 (위치: {bb_pct:.1%}) — 매수 기회"
            elif bb_pct < 0.3:
                bb_signal, bb_detail = "BUY", f"볼린저밴드 하단 근처 (위치: {bb_pct:.1%})"
            elif bb_pct > 0.9:
                bb_signal, bb_detail = "SELL", f"볼린저밴드 상단 돌파 (위치: {bb_pct:.1%}) — 과열"
            elif bb_pct > 0.7:
                bb_signal, bb_detail = "NEUTRAL", f"볼린저밴드 상단 근처 (위치: {bb_pct:.1%})"
            else:
                bb_signal, bb_detail = "NEUTRAL", f"볼린저밴드 중간 (위치: {bb_pct:.1%})"
            if pd.notna(bb_width) and bb_width < 0.05:
                bb_detail += " + 밴드 수축 중 (큰 움직임 임박)"
            factors.append({"category": "볼린저밴드", "signal": bb_signal, "detail": bb_detail})

        # ── 6. 거래량 분석 ────────────────────────
        vol_ratio = latest.get("volume_ratio", np.nan)
        ret_1d = latest.get("ret_1d", np.nan)
        if pd.notna(vol_ratio):
            if vol_ratio > 2.0 and pd.notna(ret_1d) and ret_1d > 0:
                vl_signal, vl_detail = "BUY", f"거래량 {vol_ratio:.1f}배 급증 + 양봉 — 강한 매수세"
            elif vol_ratio > 2.0 and pd.notna(ret_1d) and ret_1d < 0:
                vl_signal, vl_detail = "SELL", f"거래량 {vol_ratio:.1f}배 급증 + 음봉 — 매도 압력"
            elif vol_ratio > 1.5:
                vl_signal, vl_detail = "NEUTRAL", f"거래량 평소 대비 {vol_ratio:.1f}배 — 관심 증가"
            else:
                vl_signal, vl_detail = "NEUTRAL", f"거래량 평소 수준 ({vol_ratio:.1f}배)"
            factors.append({"category": "거래량", "signal": vl_signal, "detail": vl_detail})

        # ── 종합 판단 ─────────────────────────────
        buy_count = sum(1 for f in factors if f["signal"] == "BUY")
        sell_count = sum(1 for f in factors if f["signal"] == "SELL")
        total_factors = len(factors)

        if buy_count >= 4:
            summary = f"강한 매수 신호 ({buy_count}/{total_factors}개 지표 매수)"
        elif buy_count >= 3:
            summary = f"매수 우세 ({buy_count}/{total_factors}개 지표 매수)"
        elif sell_count >= 4:
            summary = f"강한 매도 신호 ({sell_count}/{total_factors}개 지표 매도)"
        elif sell_count >= 3:
            summary = f"매도 우세 ({sell_count}/{total_factors}개 지표 매도)"
        else:
            summary = f"혼조세 (매수 {buy_count} vs 매도 {sell_count})"

        return {
            "summary": summary,
            "factors": factors,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "neutral_count": total_factors - buy_count - sell_count,
        }

    def predict_all_stocks(self, price_dict: dict,
                            ti_dict: dict = None) -> pd.DataFrame:
        all_preds = []
        for stock in get_all_stocks():
            ticker = stock["ticker"]
            if ticker not in price_dict:
                continue
            price_df = price_dict[ticker]
            ti_df = ti_dict.get(ticker) if ti_dict else None
            preds = self.predict_single(ticker, price_df, ti_df)
            all_preds.extend(preds)
        return pd.DataFrame(all_preds)

    def _calc_fallback_weights(self, horizon: int) -> dict:
        """메타 러너 없을 때 적응형 가중치 (4주 롤링 성과 기반)"""
        try:
            from models.adaptive_weights import load_weights
            return load_weights()
        except Exception:
            return {"xgb": 0.25, "lgb": 0.25, "lstm": 0.15, "gru": 0.15, "transformer": 0.20}

    def _get_or_load_model(self, name: str, horizon: int):
        if horizon not in self.models:
            self.models[horizon] = {}
        if name not in self.models[horizon]:
            model_classes = {
                "xgb": XGBStockModel,
                "lgb": LGBStockModel,
                "lstm": LSTMStockModel,
                "gru": GRUStockModel,
                "transformer": TransformerStockModel,
            }
            m = model_classes[name](horizon=horizon)
            m._load()
            self.models[horizon][name] = m
        return self.models[horizon][name]

    def _get_or_load_meta(self, horizon: int):
        if horizon not in self.meta_models:
            meta_path = os.path.join(MODEL_DIR, f"meta_lr_h{horizon}.pkl")
            if os.path.exists(meta_path):
                with open(meta_path, "rb") as f:
                    self.meta_models[horizon] = pickle.load(f)
            else:
                return None
        return self.meta_models[horizon]

    def _save_meta_model(self, horizon: int):
        meta_path = os.path.join(MODEL_DIR, f"meta_lr_h{horizon}.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump(self.meta_models[horizon], f)

    def validate_past_predictions(self, db, horizon: int = 1, lookback_days: int = 30) -> dict:
        """Self-consistency 검증: 과거 예측 vs 실제 결과 비교"""
        try:
            with db.engine.connect() as conn:
                from sqlalchemy import text
                pred_df = pd.read_sql(text("""
                    SELECT p.ticker, p.pred_date, p.horizon, p.pred_label, p.pred_up_prob
                    FROM predictions p
                    WHERE p.horizon = :horizon
                      AND p.pred_date >= CURRENT_DATE - :lookback
                      AND p.pred_date < CURRENT_DATE - :horizon
                    ORDER BY p.pred_date
                """), conn, params={"horizon": horizon, "lookback": lookback_days})

            if pred_df.empty or len(pred_df) < 10:
                logger.info(f"  Self-check: 검증할 과거 예측 부족 ({len(pred_df)}건)")
                return {"accuracy": None, "need_retrain": False, "total_checked": 0}

            correct, up_correct, up_total, total = 0, 0, 0, 0
            for _, row in pred_df.iterrows():
                price_df = db.get_stock_prices(row["ticker"], days=lookback_days + horizon + 5)
                if price_df.empty:
                    continue

                price_df["trade_date"] = pd.to_datetime(price_df["trade_date"]).dt.date
                pred_date_val = row["pred_date"].date() if hasattr(row["pred_date"], 'date') else row["pred_date"]
                mask = price_df["trade_date"] == pred_date_val
                if not mask.any():
                    continue
                idx = price_df[mask].index[0]
                future_idx = idx + horizon
                if future_idx >= len(price_df):
                    continue

                ret = float(price_df.loc[future_idx, "close"]) / float(price_df.loc[idx, "close"]) - 1
                actual = "UP" if ret > 0.02 else ("DOWN" if ret < -0.02 else "HOLD")

                if row["pred_label"] == actual:
                    correct += 1
                if row["pred_label"] == "UP":
                    up_total += 1
                    if actual == "UP":
                        up_correct += 1
                total += 1

            if total == 0:
                return {"accuracy": None, "need_retrain": False, "total_checked": 0}

            accuracy = correct / total
            precision_up = up_correct / up_total if up_total > 0 else 0
            need_retrain = accuracy < 0.40

            if need_retrain:
                logger.warning(f"  Self-check H{horizon}: {accuracy:.1%} - 재학습 필요!")
            else:
                logger.info(f"  Self-check H{horizon}: {accuracy:.1%}, UP정밀도 {precision_up:.1%} ({total}건)")

            return {"accuracy": round(accuracy, 4), "precision_up": round(precision_up, 4),
                    "total_checked": total, "need_retrain": need_retrain}
        except Exception as e:
            logger.warning(f"  Self-check 실패: {e}")
            return {"accuracy": None, "need_retrain": False, "total_checked": 0}


class SignalGenerator:
    """예측 결과 → 매수/매도 시그널 생성"""

    def __init__(self):
        self.cfg = MODEL_CONFIG
        self.rules = TRADING_RULES

    def generate_buy_signals(self, pred_df: pd.DataFrame,
                              horizon: int = 1) -> pd.DataFrame:
        df = pred_df[pred_df["horizon"] == horizon].copy()
        buy = df[
            (df["pred_up_prob"] >= self.cfg["signal_threshold_up"]) &
            (df["pred_label"] == "UP")
        ].sort_values("total_score", ascending=False)
        return buy.head(10)

    def generate_sell_signals(self, pred_df: pd.DataFrame,
                               horizon: int = 1) -> pd.DataFrame:
        df = pred_df[pred_df["horizon"] == horizon].copy()
        sell = df[
            (df["pred_down_prob"] >= self.cfg["signal_threshold_down"]) &
            (df["pred_label"] == "DOWN")
        ].sort_values("pred_down_prob", ascending=False)
        return sell.head(10)

    def get_today_report(self, pred_df: pd.DataFrame) -> dict:
        return {
            "buy_top5_h1":  self.generate_buy_signals(pred_df, horizon=1).head(5),
            "buy_top5_h5":  self.generate_buy_signals(pred_df, horizon=5).head(5),
            "buy_top5_h20": self.generate_buy_signals(pred_df, horizon=20).head(5),
            "sell_top5":    self.generate_sell_signals(pred_df, horizon=1).head(5),
        }
