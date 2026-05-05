"""
AI Trading System - 피처 엔지니어링
기술적 지표 계산 + 모델 학습용 피처/레이블 생성
"""

import logging
import numpy as np
import pandas as pd
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import MODEL_CONFIG

logger = logging.getLogger(__name__)


class TechnicalIndicatorCalculator:
    """기술적 지표 계산기"""

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        OHLCV → 기술지표 DataFrame 반환
        입력: trade_date, open, high, low, close, volume
        """
        df = df.copy().sort_values("trade_date").reset_index(drop=True)
        if len(df) < 30:
            return pd.DataFrame()

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]

        result = pd.DataFrame()
        result["trade_date"] = df["trade_date"]
        if "ticker" in df.columns:
            result["ticker"] = df["ticker"]

        # ── 이동평균 ───────────────────────────────────
        result["sma5"]   = SMAIndicator(close, window=5).sma_indicator()
        result["sma20"]  = SMAIndicator(close, window=20).sma_indicator()
        result["sma60"]  = SMAIndicator(close, window=60).sma_indicator()
        result["sma120"] = SMAIndicator(close, window=120).sma_indicator()
        result["ema5"]   = EMAIndicator(close, window=5).ema_indicator()
        result["ema20"]  = EMAIndicator(close, window=20).ema_indicator()

        # ── MACD ───────────────────────────────────────
        macd_ind = MACD(close, window_slow=26, window_fast=12, window_sign=9)
        result["macd"]        = macd_ind.macd()
        result["macd_signal"] = macd_ind.macd_signal()
        result["macd_hist"]   = macd_ind.macd_diff()

        # ── RSI ────────────────────────────────────────
        result["rsi14"] = RSIIndicator(close, window=14).rsi()

        # ── 볼린저밴드 ─────────────────────────────────
        bb = BollingerBands(close, window=20, window_dev=2)
        result["bb_upper"]  = bb.bollinger_hband()
        result["bb_middle"] = bb.bollinger_mavg()
        result["bb_lower"]  = bb.bollinger_lband()
        result["bb_pct"]    = bb.bollinger_pband()
        result["bb_width"]  = bb.bollinger_wband()

        # ── 스토캐스틱 ─────────────────────────────────
        stoch = StochasticOscillator(high, low, close, window=14, smooth_window=3)
        result["stoch_k"] = stoch.stoch()
        result["stoch_d"] = stoch.stoch_signal()

        # ── OBV ────────────────────────────────────────
        result["obv"] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()

        # ── 거래량 비율 ────────────────────────────────
        vol_ma20 = volume.rolling(20).mean()
        result["volume_ratio"] = volume / vol_ma20.replace(0, np.nan)

        # ── ATR ────────────────────────────────────────
        result["atr14"] = AverageTrueRange(high, low, close, window=14).average_true_range()

        return result.dropna(subset=["sma20", "rsi14"]).reset_index(drop=True)


class FeatureBuilder:
    """모델 학습용 피처 + 레이블 생성"""

    def __init__(self):
        self.ti_calc = TechnicalIndicatorCalculator()
        self.cfg = MODEL_CONFIG

    def build_features(self, price_df: pd.DataFrame,
                       ti_df: pd.DataFrame = None,
                       macro_df: pd.DataFrame = None,
                       us_lag_days: int = 0) -> pd.DataFrame:
        """
        가격 + 기술지표 + 거시경제 → 피처 DataFrame 생성
        us_lag_days: 미국 데이터 lag 일수 (0=당일새벽마감, 1=전일마감)
          - 모델 A (07:30): us_lag_days=0 (당일 새벽에 미국 마감)
          - 모델 B (17:00): us_lag_days=1 (전일 미국 마감)
        """
        df = price_df.copy().sort_values("trade_date").reset_index(drop=True)

        if ti_df is None:
            ti_df = self.ti_calc.calculate(df)

        # merge 기술지표
        feat_cols = ["trade_date","sma5","sma20","sma60","sma120","ema5","ema20",
                     "macd","macd_signal","macd_hist","rsi14",
                     "bb_upper","bb_middle","bb_lower","bb_pct","bb_width",
                     "stoch_k","stoch_d","obv","volume_ratio","atr14"]
        feat_cols = [c for c in feat_cols if c in ti_df.columns]
        df = df.merge(ti_df[feat_cols], on="trade_date", how="left")

        close = df["close"]

        # ── 파생 피처 ─────────────────────────────────
        # 이동평균 대비 가격 위치
        df["price_vs_sma5"]   = (close / df["sma5"]  - 1).replace([np.inf,-np.inf], np.nan)
        df["price_vs_sma20"]  = (close / df["sma20"] - 1).replace([np.inf,-np.inf], np.nan)
        df["price_vs_sma60"]  = (close / df["sma60"] - 1).replace([np.inf,-np.inf], np.nan)
        df["price_vs_sma120"] = (close / df["sma120"]- 1).replace([np.inf,-np.inf], np.nan)

        # 골든/데드 크로스 신호
        df["golden_cross"]  = ((df["sma5"] > df["sma20"]) &
                               (df["sma5"].shift(1) <= df["sma20"].shift(1))).astype(int)
        df["death_cross"]   = ((df["sma5"] < df["sma20"]) &
                               (df["sma5"].shift(1) >= df["sma20"].shift(1))).astype(int)

        # 일간 수익률
        df["ret_1d"]  = close.pct_change(1)
        df["ret_5d"]  = close.pct_change(5)
        df["ret_20d"] = close.pct_change(20)

        # 변동성
        df["volatility_5d"]  = df["ret_1d"].rolling(5).std()
        df["volatility_20d"] = df["ret_1d"].rolling(20).std()

        # 거래량 트렌드
        df["vol_trend"] = (df["volume"] / df["volume"].rolling(5).mean()).replace(
            [np.inf,-np.inf], np.nan)

        # MACD 크로스오버
        df["macd_cross_up"] = ((df["macd"] > df["macd_signal"]) &
                                (df["macd"].shift(1) <= df["macd_signal"].shift(1))).astype(int)

        # RSI 과매도/과매수 존
        df["rsi_oversold"]  = (df["rsi14"] < 30).astype(int)
        df["rsi_overbought"]= (df["rsi14"] > 70).astype(int)

        # ── 거시경제 피처 (DB에서 로드) ──────────────
        if macro_df is None:
            try:
                from database.db_manager import DBManager
                db = DBManager()
                macro_df = pd.read_sql(
                    "SELECT data_date as trade_date, usd_krw, us_10y, vix, kospi, kosdaq, sp500, nasdaq, sox, dji, wti, gold, dxy, btc FROM macro_data",
                    db.engine
                )
            except Exception:
                macro_df = pd.DataFrame()

        if not macro_df.empty and "trade_date" in macro_df.columns:
            macro_df = macro_df.copy()
            macro_df["trade_date"] = pd.to_datetime(macro_df["trade_date"])
            df["trade_date"] = pd.to_datetime(df["trade_date"])

            macro_df = macro_df.sort_values("trade_date").drop_duplicates("trade_date")

            # ── 핵심: 미국 시장은 한국 대비 시차(lag) 존재 ──
            # 미국 T일 마감 → 한국 T+1일 개장에 영향
            # 따라서 미국 지표는 1일 shift하여 병합

            us_cols = ["sp500", "nasdaq", "sox", "vix", "us_10y", "usd_krw", "dji", "wti", "gold", "dxy", "btc"]
            kr_cols = ["kospi", "kosdaq"]

            # 1) 미국 지표 변화율 (동일 날짜 기준으로 계산 후 shift)
            for col in us_cols + kr_cols:
                if col in macro_df.columns:
                    macro_df[f"{col}_chg1d"] = macro_df[col].pct_change(1, fill_method=None)
                    macro_df[f"{col}_chg5d"] = macro_df[col].pct_change(5, fill_method=None)

            # 2) 미국 시차 피처: T일 미국 → T+1일 한국에 매칭
            #    "어제 미국 장이 이랬다" → "오늘 한국 장 예측"
            us_lag_df = macro_df[["trade_date"]].copy()
            for col in us_cols:
                if col in macro_df.columns:
                    # 전일 미국 종가
                    us_lag_df[f"us_prev_{col}"] = macro_df[col].values
                    # 전일 미국 변화율
                    if f"{col}_chg1d" in macro_df.columns:
                        us_lag_df[f"us_prev_{col}_chg"] = macro_df[f"{col}_chg1d"].values
                    # 전일 미국 5일 변화율
                    if f"{col}_chg5d" in macro_df.columns:
                        us_lag_df[f"us_prev_{col}_chg5d"] = macro_df[f"{col}_chg5d"].values

            # shift: 미국 T일 데이터를 한국 T+1일로 이동
            us_lag_df["trade_date"] = us_lag_df["trade_date"] + pd.Timedelta(days=1)
            # 주말 보정: 금요일 미국 → 월요일 한국
            # (merge시 left join이므로 토/일은 자동 무시됨)

            # 3) 미국 시장 간 상호작용 피처
            if "sp500" in macro_df.columns and "nasdaq" in macro_df.columns:
                # NASDAQ-S&P500 괴리 (기술주 과열/저평가)
                macro_df["nasdaq_sp500_ratio"] = macro_df["nasdaq"] / macro_df["sp500"].replace(0, pd.NA)
                us_lag_df["us_prev_nasdaq_sp500_ratio"] = macro_df["nasdaq_sp500_ratio"].values

            if "sox" in macro_df.columns and "nasdaq" in macro_df.columns:
                # SOX-NASDAQ 괴리 (반도체 상대강도)
                macro_df["sox_nasdaq_ratio"] = macro_df["sox"] / macro_df["nasdaq"].replace(0, pd.NA)
                us_lag_df["us_prev_sox_nasdaq_ratio"] = macro_df["sox_nasdaq_ratio"].values

            if "vix" in macro_df.columns:
                # VIX 급등 (공포 신호: 20 이상이면 경계, 30 이상이면 패닉)
                macro_df["vix_high"] = (macro_df["vix"] > 25).astype(int)
                macro_df["vix_panic"] = (macro_df["vix"] > 30).astype(int)
                us_lag_df["us_prev_vix_high"] = macro_df["vix_high"].values
                us_lag_df["us_prev_vix_panic"] = macro_df["vix_panic"].values

            # 4) 한국 지표 (동일 날짜, 시차 없음)
            kr_merge = macro_df[["trade_date"]].copy()
            for col in kr_cols:
                if col in macro_df.columns:
                    if f"{col}_chg1d" in macro_df.columns:
                        kr_merge[f"{col}_chg1d"] = macro_df[f"{col}_chg1d"].values
                    if f"{col}_chg5d" in macro_df.columns:
                        kr_merge[f"{col}_chg5d"] = macro_df[f"{col}_chg5d"].values

            # 5) 병합: 한국 지표는 동일 날짜, 미국 지표는 1일 lag
            df = df.merge(kr_merge, on="trade_date", how="left")
            df = df.merge(us_lag_df, on="trade_date", how="left")

        # ── 수급 피처 (DB에서 로드) ────────────────
        ticker_val = None
        if "ticker" in df.columns:
            ticker_val = df["ticker"].iloc[0] if len(df) > 0 else None

        if ticker_val:
            try:
                from database.db_manager import DBManager
                db = DBManager()
                sd_df = pd.read_sql(
                    f"SELECT trade_date, foreign_net, institution_net FROM supply_demand WHERE ticker = '{ticker_val}' ORDER BY trade_date",
                    db.engine
                )
                if not sd_df.empty:
                    sd_df["trade_date"] = pd.to_datetime(sd_df["trade_date"])
                    df["trade_date"] = pd.to_datetime(df["trade_date"])
                    # 5일 누적 수급
                    sd_df = sd_df.sort_values("trade_date").drop_duplicates("trade_date")
                    sd_df["foreign_net_5d"] = sd_df["foreign_net"].rolling(5).sum()
                    sd_df["institution_net_5d"] = sd_df["institution_net"].rolling(5).sum()
                    df = df.merge(
                        sd_df[["trade_date", "foreign_net", "institution_net",
                               "foreign_net_5d", "institution_net_5d"]],
                        on="trade_date", how="left"
                    )
            except Exception:
                pass

        return df

    def create_labels(self, df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
        """
        horizon일 후 수익률 기반 레이블 생성 (ATR 동적 임계값)
        UP(2), HOLD(1), DOWN(0)
        - 종목별 변동성(ATR) 대비 의미 있는 움직임을 기준으로 분류
        - 변동성 큰 종목: 임계값 높음 / 변동성 작은 종목: 임계값 낮음
        """
        df = df.copy()
        future_ret = df["close"].pct_change(horizon).shift(-horizon)
        df[f"future_ret_{horizon}d"] = future_ret

        # ATR 기반 동적 임계값 (ATR 14일 / 종가 = 변동성 비율)
        if "atr14" in df.columns and "close" in df.columns:
            atr_pct = (df["atr14"] / df["close"]).rolling(20).mean()
            # 임계값 = ATR% × 스케일팩터 (horizon에 따라 조정)
            scale = np.sqrt(horizon)  # 기간이 길수록 임계값 확대
            threshold_up   = (atr_pct * scale * 0.5).clip(lower=0.005, upper=0.10)
            threshold_down = (-atr_pct * scale * 0.5).clip(lower=-0.10, upper=-0.005)
        else:
            # ATR 없으면 고정 임계값 (fallback)
            threshold_up   = self.cfg["label_up_pct"]
            threshold_down = self.cfg["label_down_pct"]

        # 동적 레이블 할당
        labels = pd.Series(1, index=df.index, dtype=float)  # 기본 HOLD
        labels[future_ret > threshold_up] = 2   # UP
        labels[future_ret < threshold_down] = 0  # DOWN
        labels[future_ret.isna()] = np.nan

        df[f"label_{horizon}d"] = labels
        return df

    def create_labels_binary(self, df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
        """
        2분류 레이블: UP(1) / DOWN(0) — HOLD 없음
        기준: future_return > 0% → UP, ≤ 0% → DOWN
        """
        df = df.copy()
        future_ret = df["close"].pct_change(horizon).shift(-horizon)
        df[f"future_ret_{horizon}d"] = future_ret

        labels = pd.Series(np.nan, index=df.index, dtype=float)
        labels[future_ret > 0] = 1   # UP
        labels[future_ret <= 0] = 0  # DOWN
        labels[future_ret.isna()] = np.nan

        df[f"label_{horizon}d"] = labels
        return df

    def get_feature_columns(self) -> list:
        return [
            # 기술지표
            "sma5","sma20","sma60","sma120","ema5","ema20",
            "macd","macd_signal","macd_hist",
            "rsi14","bb_pct","bb_width","stoch_k","stoch_d",
            "obv","volume_ratio","atr14",
            # 파생 피처
            "price_vs_sma5","price_vs_sma20","price_vs_sma60","price_vs_sma120",
            "golden_cross","death_cross",
            "ret_1d","ret_5d","ret_20d",
            "volatility_5d","volatility_20d","vol_trend",
            "macd_cross_up","rsi_oversold","rsi_overbought",
            # 거시경제 피처 — 미국 지수
            "us_prev_sp500_chg","us_prev_sp500_chg5d",
            "us_prev_nasdaq_chg","us_prev_nasdaq_chg5d",
            "us_prev_sox_chg","us_prev_sox_chg5d",
            "us_prev_dji_chg","us_prev_dji_chg5d",
            "us_prev_vix_chg","us_prev_vix_chg5d",
            "us_prev_vix_high","us_prev_vix_panic",
            "us_prev_nasdaq_sp500_ratio","us_prev_sox_nasdaq_ratio",
            # 거시경제 피처 — 원자재/통화
            "us_prev_wti_chg","us_prev_wti_chg5d",
            "us_prev_gold_chg","us_prev_gold_chg5d",
            "us_prev_dxy_chg","us_prev_dxy_chg5d",
            "us_prev_btc_chg","us_prev_btc_chg5d",
            "us_prev_usd_krw_chg","us_prev_usd_krw_chg5d",
            "us_prev_us_10y_chg","us_prev_us_10y_chg5d",
            # 거시경제 피처 — 한국 지수
            "kospi_chg1d","kospi_chg5d",
            "kosdaq_chg1d","kosdaq_chg5d",
            # 수급 피처
            "foreign_net","institution_net",
            "foreign_net_5d","institution_net_5d",
        ]

    def get_available_features(self, df: pd.DataFrame, max_nan_ratio: float = 0.5) -> list:
        """데이터에서 실제 사용 가능한 피처만 반환 (NaN 비율 초과 칼럼 제외)"""
        all_cols = self.get_feature_columns()
        available = []
        for col in all_cols:
            if col in df.columns and df[col].isna().mean() <= max_nan_ratio:
                available.append(col)
        return available

    def build_lstm_sequences(self, df: pd.DataFrame,
                              feature_cols: list,
                              lookback: int = 20) -> tuple:
        """
        LSTM 입력용 3D 시퀀스 생성
        Returns: (X, dates) — X shape: (samples, lookback, features)
        """
        df = df.dropna(subset=feature_cols).reset_index(drop=True)
        X, dates = [], []
        for i in range(lookback, len(df)):
            seq = df[feature_cols].iloc[i-lookback:i].values
            X.append(seq)
            dates.append(df["trade_date"].iloc[i])
        return np.array(X, dtype=np.float32), dates

    def normalize_features(self, train_df: pd.DataFrame,
                            test_df: pd.DataFrame,
                            feature_cols: list) -> tuple:
        """훈련셋 통계로 정규화 (표준화)"""
        mean = train_df[feature_cols].mean()
        std  = train_df[feature_cols].std().replace(0, 1)

        train_norm = train_df.copy()
        test_norm  = test_df.copy()
        train_norm[feature_cols] = (train_df[feature_cols] - mean) / std
        test_norm[feature_cols]  = (test_df[feature_cols]  - mean) / std
        return train_norm, test_norm, mean, std


class TechScoreCalculator:
    """
    기술적 종합 점수 계산 (0~1)
    3개 카테고리로 분리: 기술 시그널 / 추세 컨펌 / 리스크
    """

    def calculate_score(self, row: pd.Series) -> float:
        """기술적 시그널 종합 점수 (0~1, 높을수록 매수 유리)"""
        scores = {}

        # ── 1. 모멘텀 시그널 (가중치 30%) ──────────
        momentum_scores = []

        # RSI 점수
        rsi = row.get("rsi14", 50)
        if pd.notna(rsi):
            if rsi < 30:    momentum_scores.append(0.9)   # 과매도 → 반등 기대
            elif rsi < 40:  momentum_scores.append(0.7)
            elif rsi < 60:  momentum_scores.append(0.5)
            elif rsi < 70:  momentum_scores.append(0.3)
            else:           momentum_scores.append(0.1)   # 과매수 → 조정 기대

        # 스토캐스틱
        stoch_k = row.get("stoch_k", np.nan)
        stoch_d = row.get("stoch_d", np.nan)
        if all(pd.notna([stoch_k, stoch_d])):
            if stoch_k < 20 and stoch_k > stoch_d:
                momentum_scores.append(0.9)   # 과매도에서 골든크로스
            elif stoch_k > 80 and stoch_k < stoch_d:
                momentum_scores.append(0.1)   # 과매수에서 데드크로스
            elif stoch_k > stoch_d:
                momentum_scores.append(0.6)
            else:
                momentum_scores.append(0.4)

        # MACD 히스토그램 방향
        macd_hist = row.get("macd_hist", np.nan)
        if pd.notna(macd_hist):
            if macd_hist > 0:
                momentum_scores.append(0.7)
            else:
                momentum_scores.append(0.3)

        scores["momentum"] = float(np.mean(momentum_scores)) if momentum_scores else 0.5

        # ── 2. 추세 시그널 (가중치 35%) ────────────
        trend_scores = []

        # 이동평균 배열
        sma5   = row.get("sma5", np.nan)
        sma20  = row.get("sma20", np.nan)
        sma60  = row.get("sma60", np.nan)
        sma120 = row.get("sma120", np.nan)
        close  = row.get("close", np.nan)

        if all(pd.notna([sma5, sma20, sma60, close])):
            if close > sma5 > sma20 > sma60:
                trend_scores.append(0.95)  # 완전 정배열
            elif close > sma5 > sma20:
                trend_scores.append(0.8)
            elif close > sma20:
                trend_scores.append(0.6)
            elif close > sma60:
                trend_scores.append(0.4)
            elif close < sma5 < sma20 < sma60:
                trend_scores.append(0.05)  # 완전 역배열
            else:
                trend_scores.append(0.3)

        # MACD 크로스 + 방향
        macd     = row.get("macd", np.nan)
        macd_sig = row.get("macd_signal", np.nan)
        if all(pd.notna([macd, macd_sig])):
            if macd > macd_sig and macd > 0:
                trend_scores.append(0.85)  # 상승 추세 + 강세
            elif macd > macd_sig:
                trend_scores.append(0.65)  # 전환 초기
            elif macd < macd_sig and macd < 0:
                trend_scores.append(0.15)  # 하락 추세 + 약세
            else:
                trend_scores.append(0.35)

        # 골든/데드 크로스 (당일 발생 시 강한 신호)
        golden = row.get("golden_cross", 0)
        death  = row.get("death_cross", 0)
        if golden == 1:
            trend_scores.append(0.9)
        elif death == 1:
            trend_scores.append(0.1)

        # 가격 vs 이동평균 거리 (평균 회귀 반영)
        price_vs_sma20 = row.get("price_vs_sma20", np.nan)
        if pd.notna(price_vs_sma20):
            if price_vs_sma20 > 0.10:
                trend_scores.append(0.3)   # 20일선 대비 10% 이상 → 과열
            elif price_vs_sma20 > 0.03:
                trend_scores.append(0.6)
            elif price_vs_sma20 > -0.03:
                trend_scores.append(0.5)
            elif price_vs_sma20 > -0.10:
                trend_scores.append(0.6)   # 적당한 이격 → 반등 기대
            else:
                trend_scores.append(0.7)   # 급락 후 반등 기대

        scores["trend"] = float(np.mean(trend_scores)) if trend_scores else 0.5

        # ── 3. 변동성/거래량 시그널 (가중치 20%) ───
        vol_scores = []

        # 볼린저밴드 위치
        bb_pct = row.get("bb_pct", np.nan)
        if pd.notna(bb_pct):
            if bb_pct < 0.1:
                vol_scores.append(0.85)    # 하단 돌파 → 매수
            elif bb_pct < 0.3:
                vol_scores.append(0.7)
            elif bb_pct > 0.9:
                vol_scores.append(0.15)    # 상단 돌파 → 매도
            elif bb_pct > 0.7:
                vol_scores.append(0.3)
            else:
                vol_scores.append(0.5)

        # 볼린저밴드 폭 (변동성)
        bb_width = row.get("bb_width", np.nan)
        if pd.notna(bb_width):
            if bb_width < 0.05:
                vol_scores.append(0.7)     # 밴드 수축 → 큰 움직임 임박
            elif bb_width > 0.15:
                vol_scores.append(0.4)     # 밴드 확장 → 이미 변동 중
            else:
                vol_scores.append(0.5)

        # 거래량 비율
        vol_ratio = row.get("volume_ratio", np.nan)
        if pd.notna(vol_ratio):
            ret_1d = row.get("ret_1d", 0)
            if pd.notna(ret_1d) and vol_ratio > 1.5 and ret_1d > 0:
                vol_scores.append(0.85)    # 거래량 급증 + 양봉 = 강한 매수세
            elif pd.notna(ret_1d) and vol_ratio > 1.5 and ret_1d < 0:
                vol_scores.append(0.2)     # 거래량 급증 + 음봉 = 매도 압력
            elif vol_ratio > 1.2:
                vol_scores.append(0.6)
            else:
                vol_scores.append(0.5)

        scores["volatility"] = float(np.mean(vol_scores)) if vol_scores else 0.5

        # ── 4. 수익률 모멘텀 (가중치 15%) ──────────
        ret_scores = []

        ret_5d = row.get("ret_5d", np.nan)
        ret_20d = row.get("ret_20d", np.nan)
        if pd.notna(ret_5d):
            if ret_5d > 0.05:
                ret_scores.append(0.7)
            elif ret_5d > 0:
                ret_scores.append(0.6)
            elif ret_5d > -0.05:
                ret_scores.append(0.4)
            else:
                ret_scores.append(0.5)     # 급락 후 반등 기대

        if pd.notna(ret_20d):
            if ret_20d > 0.10:
                ret_scores.append(0.6)
            elif ret_20d > 0:
                ret_scores.append(0.55)
            elif ret_20d > -0.10:
                ret_scores.append(0.45)
            else:
                ret_scores.append(0.55)    # 20일 급락 → 반등 기대

        scores["return_momentum"] = float(np.mean(ret_scores)) if ret_scores else 0.5

        # ── 가중 합산 ─────────────────────────────
        final = (
            scores.get("momentum", 0.5) * 0.30 +
            scores.get("trend", 0.5) * 0.35 +
            scores.get("volatility", 0.5) * 0.20 +
            scores.get("return_momentum", 0.5) * 0.15
        )
        return round(float(final), 4)

    def calculate_trend_confirmation(self, row: pd.Series) -> float:
        """
        추세 컨펌 점수 (0~1): 여러 시간 프레임의 추세가 일치하는지
        단기/중기/장기가 모두 같은 방향이면 높은 점수
        """
        signals = []

        # 단기 추세 (5일선 vs 20일선)
        sma5 = row.get("sma5", np.nan)
        sma20 = row.get("sma20", np.nan)
        if all(pd.notna([sma5, sma20])):
            signals.append(1 if sma5 > sma20 else 0)

        # 중기 추세 (20일선 vs 60일선)
        sma60 = row.get("sma60", np.nan)
        if all(pd.notna([sma20, sma60])):
            signals.append(1 if sma20 > sma60 else 0)

        # MACD 방향
        macd = row.get("macd", np.nan)
        macd_sig = row.get("macd_signal", np.nan)
        if all(pd.notna([macd, macd_sig])):
            signals.append(1 if macd > macd_sig else 0)

        # RSI 중립 이상
        rsi = row.get("rsi14", np.nan)
        if pd.notna(rsi):
            signals.append(1 if rsi > 50 else 0)

        # 종가 위치
        close = row.get("close", np.nan)
        if all(pd.notna([close, sma20])):
            signals.append(1 if close > sma20 else 0)

        if not signals:
            return 0.5
        return round(sum(signals) / len(signals), 4)
