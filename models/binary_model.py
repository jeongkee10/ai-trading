"""
AI Trading System - 2분류 (UP/DOWN) 모델
모델 A (07:30): HOLD 없이 방향만 예측
"""

import logging
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.preprocessor import FeatureBuilder
from config.settings import MODEL_CONFIG

logger = logging.getLogger(__name__)
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")
os.makedirs(MODEL_DIR, exist_ok=True)


class BinaryEnsemblePredictor:
    """
    2분류 앙상블: XGBoost + LightGBM → 가중평균 → UP/DOWN
    모델 A (07:30) / 모델 B (17:00) 공통
    뉴스 센티멘트 피처 포함
    """

    LABEL_MAP = {0: "DOWN", 1: "UP"}

    # 뉴스 센티멘트 피처 (모델에 추가되는 변수)
    NEWS_FEATURES = ["news_sentiment", "news_count", "news_positive_ratio", "news_negative_ratio", "news_momentum"]

    def __init__(self):
        self.fb = FeatureBuilder()
        self.models = {}  # {horizon: {"xgb": model, "lgb": model}}
        self.cfg = MODEL_CONFIG

    def train_all(self, combined_df: pd.DataFrame, news_df: pd.DataFrame = None) -> dict:
        """2분류 모델 학습 (UP/DOWN only) — 뉴스 센티멘트 피처 포함"""
        # 뉴스 피처 머지 (있으면)
        if news_df is not None and not news_df.empty:
            for col in self.NEWS_FEATURES:
                if col not in combined_df.columns:
                    combined_df[col] = 0.0
            if "ticker" in news_df.columns:
                news_map = news_df.set_index("ticker").to_dict("index")
                for col in self.NEWS_FEATURES:
                    if col in news_df.columns:
                        combined_df[col] = combined_df["ticker"].map(
                            lambda t: news_map.get(t, {}).get(col, 0.0)
                        )
        else:
            # 뉴스 데이터 없으면 0으로 채움 (학습 시에는 0이지만 피처 칼럼은 존재)
            for col in self.NEWS_FEATURES:
                if col not in combined_df.columns:
                    combined_df[col] = 0.0

        results = {}
        for horizon in self.cfg["prediction_horizons"]:
            logger.info(f"=== Binary H{horizon} 학습 시작 ===")

            df_labeled = self.fb.create_labels_binary(combined_df, horizon)
            self.models[horizon] = {}

            # XGBoost
            xgb_result = self._train_single("xgb", df_labeled, horizon)
            results[f"xgb_h{horizon}"] = xgb_result

            # LightGBM
            lgb_result = self._train_single("lgb", df_labeled, horizon)
            results[f"lgb_h{horizon}"] = lgb_result

            logger.info(f"  Binary H{horizon}: XGB={xgb_result.get('cv_accuracy', 0):.4f} "
                        f"LGB={lgb_result.get('cv_accuracy', 0):.4f}")

        return results

    def _train_single(self, model_type: str, df: pd.DataFrame, horizon: int) -> dict:
        label_col = f"label_{horizon}d"
        if label_col not in df.columns:
            return {}

        all_feature_cols = self.fb.get_feature_columns() + self.NEWS_FEATURES
        avail_cols = [c for c in all_feature_cols
                      if c in df.columns and df[c].isna().mean() <= 0.5]
        df_clean = df[avail_cols + [label_col]].dropna()

        if len(df_clean) < 100:
            logger.warning(f"  Binary {model_type} H{horizon} 학습 데이터 부족: {len(df_clean)}")
            return {}

        X = df_clean[avail_cols].values
        y = df_clean[label_col].astype(int).values

        # F1 최적화: class_weight='balanced' + scale_pos_weight
        n_down = int((y == 0).sum())
        n_up = int((y == 1).sum())
        scale = n_down / n_up if n_up > 0 else 1.0

        if model_type == "xgb":
            model = XGBClassifier(
                n_estimators=400, max_depth=5, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.7,
                scale_pos_weight=scale,  # 클래스 불균형 보정
                min_child_weight=5, gamma=0.1,
                reg_alpha=0.1, reg_lambda=1.5,
                use_label_encoder=False, eval_metric="logloss", random_state=42
            )
        else:
            model = LGBMClassifier(
                n_estimators=400, max_depth=5, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.7,
                scale_pos_weight=scale,  # 클래스 불균형 보정
                min_child_weight=5, reg_alpha=0.1, reg_lambda=1.5,
                random_state=42, verbose=-1
            )

        # TimeSeriesSplit CV with F1 tracking
        from sklearn.metrics import f1_score as sk_f1_score, precision_score, recall_score
        tscv = TimeSeriesSplit(n_splits=5)
        acc_scores = []
        f1_scores = []
        for train_idx, val_idx in tscv.split(X):
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[val_idx])
            acc_scores.append(accuracy_score(y[val_idx], y_pred))
            f1_scores.append(sk_f1_score(y[val_idx], y_pred, average="macro"))

        # 전체 학습
        model.fit(X, y)
        self.models[horizon][model_type] = model

        # Threshold tuning: 최적 threshold 탐색 (F1 최대화)
        y_proba = model.predict_proba(X)[:, 1]
        best_threshold = 0.5
        best_f1 = 0
        for thr in np.arange(0.35, 0.65, 0.01):
            y_thr = (y_proba >= thr).astype(int)
            f1 = sk_f1_score(y, y_thr, average="macro")
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thr

        # 저장 (threshold 포함)
        path = os.path.join(MODEL_DIR, f"binary_{model_type}_h{horizon}.pkl")
        with open(path, "wb") as f:
            pickle.dump({"model": model, "trained_cols": avail_cols, "threshold": best_threshold}, f)

        cv_acc = float(np.mean(acc_scores))
        cv_f1 = float(np.mean(f1_scores))
        logger.info(f"  Binary {model_type} H{horizon} | Acc: {cv_acc:.4f} | F1: {cv_f1:.4f} | Threshold: {best_threshold:.2f} ({len(df_clean)} samples)")
        return {"cv_accuracy": cv_acc, "cv_f1": cv_f1, "threshold": best_threshold, "samples": len(df_clean)}

    def predict_single(self, ticker: str, price_df: pd.DataFrame, news_features: dict = None) -> list:
        """단��� 종목 2분류 예측"""
        feat_df = self.fb.build_features(price_df)
        if feat_df.empty or len(feat_df) < 25:
            return []

        # 뉴스 피처 추가
        if news_features:
            for col in self.NEWS_FEATURES:
                feat_df[col] = news_features.get(col, 0.0)
        else:
            for col in self.NEWS_FEATURES:
                if col not in feat_df.columns:
                    feat_df[col] = 0.0

        predictions = []
        for horizon in self.cfg["prediction_horizons"]:
            xgb_proba = self._predict_model("xgb", feat_df, horizon)
            lgb_proba = self._predict_model("lgb", feat_df, horizon)

            # 가중평균 (50:50)
            ensemble_proba = 0.5 * xgb_proba + 0.5 * lgb_proba
            up_prob = float(ensemble_proba[1]) if len(ensemble_proba) > 1 else 0.5
            down_prob = float(ensemble_proba[0]) if len(ensemble_proba) > 0 else 0.5

            # F1 최적화된 threshold 적용
            threshold = self._get_threshold(horizon)
            pred_label = "UP" if up_prob > threshold else "DOWN"
            xgb_label = "UP" if xgb_proba[1] > 0.5 else "DOWN"
            lgb_label = "UP" if lgb_proba[1] > 0.5 else "DOWN"

            predictions.append({
                "ticker": ticker,
                "pred_date": pd.Timestamp.now().date().isoformat(),
                "horizon": horizon,
                "pred_label": pred_label,
                "pred_up_prob": round(up_prob, 4),
                "pred_down_prob": round(down_prob, 4),
                "xgb_signal": xgb_label,
                "lgb_signal": lgb_label,
                "confidence": round(abs(up_prob - 0.5) * 2, 4),  # 0~1 확신도
                "model_type": "binary_A",
            })

        return predictions

    def _predict_model(self, model_type: str, feat_df: pd.DataFrame, horizon: int) -> np.ndarray:
        model_data = self._load_model(model_type, horizon)
        if model_data is None:
            return np.array([0.5, 0.5])

        model = model_data["model"]
        cols = [c for c in model_data["trained_cols"] if c in feat_df.columns]
        X = feat_df[cols].tail(1).fillna(0).values

        try:
            return model.predict_proba(X)[0]
        except Exception:
            return np.array([0.5, 0.5])

    def _load_model(self, model_type: str, horizon: int):
        # 메모리 캐시
        if horizon in self.models and model_type in self.models[horizon]:
            model = self.models[horizon][model_type]
            path = os.path.join(MODEL_DIR, f"binary_{model_type}_h{horizon}.pkl")
            with open(path, "rb") as f:
                data = pickle.load(f)
            return {"model": model, "trained_cols": data.get("trained_cols", [])}

        path = os.path.join(MODEL_DIR, f"binary_{model_type}_h{horizon}.pkl")
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            data = pickle.load(f)
        if horizon not in self.models:
            self.models[horizon] = {}
        self.models[horizon][model_type] = data["model"]
        return data

    def _get_threshold(self, horizon: int) -> float:
        """F1 최적화된 threshold 로드"""
        path = os.path.join(MODEL_DIR, f"binary_xgb_h{horizon}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
                return data.get("threshold", 0.5)
        return 0.5

    def is_trained(self) -> bool:
        return os.path.exists(os.path.join(MODEL_DIR, "binary_xgb_h1.pkl"))
