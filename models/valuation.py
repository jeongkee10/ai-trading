"""
AI Trading System - 적정가격 산출 모듈
PER/PBR 밸류에이션 + ATR 기반 가격 밴드 예측
"""

import logging
import numpy as np
import pandas as pd
from datetime import date

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


class ValuationCalculator:
    """기업별 적정가격 산출 — DB 기반 동적 업종 평균"""

    # 폴백용 (DB 조회 실패 시)
    _FALLBACK_PER = {
        "L1_메모리반도체": 22, "L2_반도체장비": 42, "L3_반도체소재부품": 37,
        "L4_AI인프라": 40, "L5_AI플랫폼소프트웨어": 36,
        "L6_AI응용로봇의료": 24, "L7_시스템반도체팹리스": 50,
    }
    _FALLBACK_PBR = {
        "L1_메모리반도체": 4.1, "L2_반도체장비": 5.5, "L3_반도체소재부품": 3.0,
        "L4_AI인프라": 5.7, "L5_AI플랫폼소프트웨어": 1.8,
        "L6_AI응용로봇의료": 2.8, "L7_시스템반도체팹리스": 5.9,
    }

    _sector_cache = None

    @property
    def SECTOR_AVG_PER(self):
        self._load_sector_avg()
        return {k: v["per"] for k, v in (self._sector_cache or {}).items()} or self._FALLBACK_PER

    @property
    def SECTOR_AVG_PBR(self):
        self._load_sector_avg()
        return {k: v["pbr"] for k, v in (self._sector_cache or {}).items()} or self._FALLBACK_PBR

    def _load_sector_avg(self):
        """DB에서 레이어별 PER/PBR 중앙값 동적 산출"""
        if self._sector_cache is not None:
            return
        try:
            from database.db_manager import DBManager
            from sqlalchemy import text
            db = DBManager()
            with db.engine.connect() as conn:
                df = pd.read_sql(text("""
                    SELECT f.ticker, f.per, f.pbr, s.layer
                    FROM financial_data f
                    JOIN (SELECT ticker, layer,
                          ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY trade_date DESC) as rn
                          FROM stock_prices) s
                    ON f.ticker = s.ticker AND s.rn = 1
                    WHERE f.per IS NOT NULL AND f.per > 0 AND f.per < 500
                      AND f.pbr IS NOT NULL AND f.pbr > 0
                """), conn)
            if df.empty:
                self._sector_cache = {}
                return
            grouped = df.groupby("layer").agg(per=("per","median"), pbr=("pbr","median"))
            self._sector_cache = grouped.to_dict("index")
            logger.info(f"  업종 PER/PBR 동적 산출: {len(self._sector_cache)}개 레이어")
        except Exception as e:
            logger.warning(f"  업종 평균 DB 조회 실패: {e}")
            self._sector_cache = {}

    def calculate_fair_value(self, ticker: str, current_price: float,
                              financial: dict, layer: str) -> dict:
        """
        적정가격 산출 (PER + PBR 블렌딩)
        Returns: {fair_value, per_value, pbr_value, upside, grade, ...}
        """
        result = {
            "ticker": ticker,
            "current_price": current_price,
            "fair_value": None,
            "per_value": None,
            "pbr_value": None,
            "upside_pct": None,
            "grade": "N/A",
            "valuation_status": "데이터 부족",
        }

        per = financial.get("per")
        pbr = financial.get("pbr")
        eps = financial.get("eps")

        sector_per = self.SECTOR_AVG_PER.get(layer, 15)
        sector_pbr = self.SECTOR_AVG_PBR.get(layer, 2.0)

        values = []
        weights = []

        # PER 기반 적정가격: EPS × 업종 평균 PER
        if eps and per and per > 0 and eps > 0:
            per_fair = eps * sector_per
            result["per_value"] = round(per_fair)
            values.append(per_fair)
            weights.append(0.6)

        # PBR 기반 적정가격: 현재가 / PBR × 업종 평균 PBR
        if pbr and pbr > 0:
            bps = current_price / pbr
            pbr_fair = bps * sector_pbr
            result["pbr_value"] = round(pbr_fair)
            values.append(pbr_fair)
            weights.append(0.4)

        if values:
            total_w = sum(weights)
            fair_value = sum(v * w for v, w in zip(values, weights)) / total_w
            result["fair_value"] = round(fair_value)
            upside = (fair_value / current_price - 1) * 100
            result["upside_pct"] = round(upside, 1)

            # 등급 부여
            if upside > 30:
                result["grade"] = "강력매수"
                result["valuation_status"] = "심각한 저평가"
            elif upside > 15:
                result["grade"] = "매수"
                result["valuation_status"] = "저평가"
            elif upside > -5:
                result["grade"] = "적정"
                result["valuation_status"] = "적정가격 근처"
            elif upside > -20:
                result["grade"] = "관망"
                result["valuation_status"] = "고평가"
            else:
                result["grade"] = "매도"
                result["valuation_status"] = "심각한 고평가"
        else:
            result["valuation_status"] = "재무 데이터 부족"

        return result

    def calculate_price_band(self, price_df: pd.DataFrame,
                              horizons: list = None) -> dict:
        """
        ATR 기반 가격 밴드 예측 (상한/하한)
        """
        if horizons is None:
            horizons = [1, 5, 20]

        if price_df.empty or len(price_df) < 20:
            return {}

        df = price_df.sort_values("trade_date").reset_index(drop=True)
        current = float(df["close"].iloc[-1])

        # ATR 14일 계산
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr14 = float(tr.rolling(14).mean().iloc[-1])

        # 최근 20일 수익률 추세
        ret_20d = float(close.pct_change(20).iloc[-1]) if len(df) >= 20 else 0

        bands = {}
        for h in horizons:
            # 변동폭 = ATR × √거래일 × 보정계수
            spread = atr14 * np.sqrt(h) * 1.2
            # 추세 반영: 상승 추세면 상한 확대
            trend_bias = ret_20d * 0.3 * current

            upper = round(current + spread + max(trend_bias, 0))
            lower = round(current - spread + min(trend_bias, 0))
            mid = round((upper + lower) / 2)

            bands[h] = {
                "upper": upper,
                "lower": lower,
                "mid": mid,
                "current": round(current),
                "spread_pct": round(spread / current * 100, 1),
            }

        return bands

    def get_all_valuations(self, db) -> pd.DataFrame:
        """전체 종목 적정가격 일괄 산출"""
        from config.settings import get_all_stocks
        from sqlalchemy import text

        all_stocks = get_all_stocks()
        results = []

        for stock in all_stocks:
            ticker = stock["ticker"]
            layer = stock["layer"]

            # 현재가 조회
            price_df = db.get_stock_prices(ticker, days=30)
            if price_df.empty:
                continue
            current_price = float(price_df.sort_values("trade_date").iloc[-1]["close"])

            # 재무 데이터 조회
            try:
                with db.engine.connect() as conn:
                    fdf = pd.read_sql(text(
                        "SELECT per, pbr, eps, roe, op_margin FROM financial_data WHERE ticker = :t ORDER BY period DESC LIMIT 1"
                    ), conn, params={"t": ticker})
                financial = fdf.iloc[0].to_dict() if not fdf.empty else {}
            except Exception:
                financial = {}

            # 적정가격 산출
            val = self.calculate_fair_value(ticker, current_price, financial, layer)
            val["name"] = stock["name"]
            val["layer"] = layer

            # 가격 밴드
            bands = self.calculate_price_band(price_df)
            if bands.get(1):
                val["band_1d_upper"] = bands[1]["upper"]
                val["band_1d_lower"] = bands[1]["lower"]
            if bands.get(5):
                val["band_5d_upper"] = bands[5]["upper"]
                val["band_5d_lower"] = bands[5]["lower"]
            if bands.get(20):
                val["band_20d_upper"] = bands[20]["upper"]
                val["band_20d_lower"] = bands[20]["lower"]

            results.append(val)

        return pd.DataFrame(results)
