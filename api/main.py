"""
AI Trading System - FastAPI Backend
Next.js 프론트엔드를 위한 REST API
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from datetime import date, timedelta
from typing import Optional
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_manager import DBManager
from config.settings import STOCK_UNIVERSE, get_all_stocks, get_layer_color, TRADING_RULES

app = FastAPI(title="J.Insight AI Trading API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    return DBManager()


# ── 시장 개요 ─────────────────────────────────────────
@app.get("/api/macro")
def get_macro():
    db = get_db()
    macro = db.get_macro_latest()
    return macro


# ── 예측 데이터 ───────────────────────────────────────
@app.get("/api/predictions")
def get_predictions(
    pred_date: Optional[date] = None,
    horizon: int = 1,
    layer: Optional[str] = None,
):
    db = get_db()
    if pred_date is None:
        pred_date = date.today()
    df = db.get_latest_predictions(pred_date)
    if df.empty:
        return {"predictions": [], "summary": {"total": 0, "up": 0, "hold": 0, "down": 0, "pred_date": str(pred_date)}}

    # model_type 필터 (중복 방지)
    if "model_type" in df.columns:
        df = df[df["model_type"].isin(["3class", None, ""])]
    df = df[df["horizon"] == horizon]
    if layer:
        df = df[df["layer"] == layer]

    records = df.to_dict(orient="records")
    summary = {
        "total": len(df),
        "up": len(df[df["pred_label"] == "UP"]),
        "hold": len(df[df["pred_label"] == "HOLD"]),
        "down": len(df[df["pred_label"] == "DOWN"]),
        "pred_date": str(pred_date),
    }
    return {"predictions": records, "summary": summary}


# ── 매수/매도 시그널 TOP N ────────────────────────────
@app.get("/api/signals")
def get_signals(horizon: int = 1, top_n: int = 5):
    db = get_db()
    df = db.get_latest_predictions(date.today())
    if df.empty:
        return {"buy": [], "sell": []}

    h_df = df[df["horizon"] == horizon]

    buy = h_df[h_df["pred_label"] == "UP"].sort_values("total_score", ascending=False).head(top_n)
    sell = h_df[h_df["pred_label"] == "DOWN"].sort_values("pred_down_prob", ascending=False).head(top_n)

    return {
        "buy": buy.to_dict(orient="records"),
        "sell": sell.to_dict(orient="records"),
    }


# ── 종목 주가 ────────────────────────────────────────
@app.get("/api/prices/{ticker}")
def get_prices(ticker: str, days: int = 120, end_date: Optional[date] = None):
    db = get_db()
    from sqlalchemy import text
    if end_date:
        with db.engine.connect() as conn:
            df = pd.read_sql(text("""
                SELECT trade_date, open, high, low, close, volume, adj_close
                FROM stock_prices WHERE ticker = :ticker AND trade_date <= :end_date
                ORDER BY trade_date DESC LIMIT :days
            """), conn, params={"ticker": ticker, "end_date": end_date, "days": days})
    else:
        df = db.get_stock_prices(ticker, days=days)
    if df.empty:
        return {"prices": []}
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values("trade_date")
    return {"prices": df.to_dict(orient="records")}


# ── 종목 목록 ────────────────────────────────────────
@app.get("/api/stocks")
def get_stocks():
    stocks = get_all_stocks()
    layers = [{"key": k, "desc": v["description"], "count": len(v["stocks"])}
              for k, v in STOCK_UNIVERSE.items()]
    return {"stocks": stocks, "layers": layers, "colors": get_layer_color()}


# ── 모델 A/B (2분류 UP/DOWN) 예측 — DB에서 읽기 ──────
@app.get("/api/predictions-binary")
def get_binary_predictions(horizon: int = 1, model: str = "model_A"):
    """모델 A/B 2분류 예측 (DB에서 읽기 — 배치로 미리 저장됨)"""
    db = get_db()
    from sqlalchemy import text
    with db.engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT p.ticker, p.pred_date, p.horizon, p.pred_up_prob, p.pred_down_prob,
                   p.pred_label, p.xgb_signal, p.lgb_signal, p.total_score,
                   p.model_type,
                   COALESCE(p.name, sp.name) as name,
                   COALESCE(p.layer, sp.layer) as layer
            FROM predictions p
            LEFT JOIN LATERAL (
                SELECT name, layer FROM stock_prices
                WHERE ticker = p.ticker ORDER BY trade_date DESC LIMIT 1
            ) sp ON true
            WHERE p.model_type = :model
              AND p.horizon = :horizon
              AND p.pred_date = (SELECT MAX(pred_date) FROM predictions WHERE model_type = :model)
            ORDER BY p.pred_up_prob DESC
        """), conn, params={"horizon": horizon, "model": model})

    if df.empty:
        return {"predictions": [], "summary": {"total": 0, "up": 0, "down": 0}}

    up_count = int((df["pred_label"] == "UP").sum())
    return {
        "predictions": df.to_dict(orient="records"),
        "summary": {
            "total": len(df),
            "up": up_count,
            "down": len(df) - up_count,
            "model_type": model,
            "pred_date": str(df["pred_date"].iloc[0]),
        }
    }


# ── 뉴스 센티멘트 ───────────────────────────────────
@app.get("/api/news/{ticker}")
def get_news_sentiment(ticker: str):
    from data.news_collector import NewsCollector
    nc = NewsCollector()
    return nc.get_sentiment_features(ticker)


# ── 기술적 지표 ──────────────────────────────────────
@app.get("/api/indicators/{ticker}")
def get_indicators(ticker: str, days: int = 120, end_date: Optional[date] = None):
    import numpy as np
    db = get_db()
    from sqlalchemy import text
    date_filter = "AND trade_date <= :end_date" if end_date else ""
    params = {"ticker": ticker, "days": days}
    if end_date:
        params["end_date"] = end_date
    with db.engine.connect() as conn:
        df = pd.read_sql(text(f"""
            SELECT trade_date, sma5, sma20, sma60, sma120, ema5, ema20,
                   macd, macd_signal, macd_hist,
                   rsi14, bb_upper, bb_middle, bb_lower, bb_pct, bb_width,
                   stoch_k, stoch_d, obv, volume_ratio, atr14
            FROM technical_indicators
            WHERE ticker = :ticker {date_filter}
            ORDER BY trade_date DESC LIMIT :days
        """), conn, params=params)
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values("trade_date")
    df = df.replace({np.nan: None, np.inf: None, -np.inf: None})
    return {"indicators": df.to_dict(orient="records")}


# ── 재무 데이터 ──────────────────────────────────────
@app.get("/api/financials/{ticker}")
def get_financials(ticker: str):
    db = get_db()
    from sqlalchemy import text
    import numpy as np
    with db.engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT period, revenue, operating_income, net_income, eps,
                   roe, per, pbr, debt_ratio, op_margin, eps_yoy
            FROM financial_data
            WHERE ticker = :ticker
            ORDER BY period DESC LIMIT 8
        """), conn, params={"ticker": ticker})
    df = df.replace({np.nan: None, np.inf: None, -np.inf: None})
    return {"financials": df.to_dict(orient="records")}


# ── 수급 데이터 ──────────────────────────────────────
@app.get("/api/supply/{ticker}")
def get_supply(ticker: str, days: int = 60):
    db = get_db()
    from sqlalchemy import text
    with db.engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT trade_date, foreign_net, institution_net, individual_net,
                   foreign_net_5d, institution_net_5d
            FROM supply_demand
            WHERE ticker = :ticker
            ORDER BY trade_date DESC LIMIT :days
        """), conn, params={"ticker": ticker, "days": days})
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values("trade_date")
    return {"supply": df.to_dict(orient="records")}


# ── 백테스트 적중률 (AI 모델 예측 실행) ──────────────
@app.get("/api/backtest")
def get_backtest_accuracy(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    ticker: Optional[str] = None,
    horizon: int = 1,
):
    """과거 날짜 기준으로 AI 모델 예측을 실행하고 실제 결과와 비교"""
    import warnings
    warnings.filterwarnings("ignore")

    if start_date is None:
        start_date = date.today() - timedelta(days=7)
    if end_date is None:
        end_date = date.today() - timedelta(days=1)

    db = get_db()
    from data.preprocessor import FeatureBuilder
    from models.ensemble import EnsemblePredictor
    from sqlalchemy import text

    predictor = EnsemblePredictor()

    # 기간 내 거래일 조회
    with db.engine.connect() as conn:
        trade_dates_df = pd.read_sql(text("""
            SELECT DISTINCT trade_date FROM stock_prices
            WHERE ticker = '005930.KS'
              AND trade_date >= :start AND trade_date <= :end
            ORDER BY trade_date
        """), conn, params={"start": start_date, "end": end_date})

    if trade_dates_df.empty:
        return {"results": [], "summary": {}}

    dates_list = trade_dates_df["trade_date"].tolist()

    # 종목 결정
    if ticker:
        tickers = [ticker]
    else:
        tickers = [s["ticker"] for s in get_all_stocks()]

    results = []
    for sim_date in dates_list:
        for tk in tickers:
            try:
                price_df = db.get_stock_prices(tk, days=250)
                if price_df.empty or len(price_df) < 50:
                    continue

                price_df["trade_date"] = pd.to_datetime(price_df["trade_date"]).dt.date
                past_df = price_df[price_df["trade_date"] <= sim_date]
                # horizon에 따라 N일 후 종가
                future_df = price_df[price_df["trade_date"] > sim_date].head(horizon)

                if past_df.empty or future_df.empty or len(past_df) < 30:
                    continue
                if len(future_df) < horizon:
                    continue

                preds = predictor.predict_single(tk, past_df)
                if not preds:
                    continue

                # horizon별 예측 선택
                p = next((x for x in preds if x.get("horizon") == horizon), preds[0])

                base_close = float(past_df["close"].iloc[-1])
                actual_close = float(future_df["close"].iloc[-1])
                return_pct = round((actual_close - base_close) / base_close * 100, 2)

                actual_label = "UP" if return_pct > 0 else "DOWN"

                results.append({
                    "sim_date": str(sim_date),
                    "ticker": tk,
                    "name": p.get("name", tk),
                    "pred_label": p["pred_label"],
                    "pred_up_prob": round(p["pred_up_prob"], 4),
                    "pred_down_prob": round(p["pred_down_prob"], 4),
                    "actual_label": actual_label,
                    "base_close": base_close,
                    "actual_close": actual_close,
                    "return_pct": return_pct,
                    "hit": p["pred_label"] == actual_label,
                    "horizon": horizon,
                })
            except Exception:
                continue

    # 요약 계산
    total = len(results)
    hits = sum(1 for r in results if r["hit"])
    up_preds = [r for r in results if r["pred_label"] == "UP"]
    down_preds = [r for r in results if r["pred_label"] == "DOWN"]
    up_hits = sum(1 for r in up_preds if r["hit"])
    down_hits = sum(1 for r in down_preds if r["hit"])
    avg_return = sum(r["return_pct"] for r in results) / total if total > 0 else 0
    up_avg_return = sum(r["return_pct"] for r in up_preds) / len(up_preds) if up_preds else 0

    return {
        "results": results,
        "summary": {
            "total": total,
            "accuracy": round(hits / total, 4) if total > 0 else 0,
            "up_count": len(up_preds),
            "up_hits": up_hits,
            "down_count": len(down_preds),
            "down_hits": down_hits,
            "avg_return": round(avg_return, 2),
            "up_avg_return": round(up_avg_return, 2),
            "start": str(start_date),
            "end": str(end_date),
            "horizon": horizon,
        }
    }


# ── 시스템 상태 ──────────────────────────────────────
@app.get("/api/status")
def get_status():
    db = get_db()
    from sqlalchemy import text
    with db.engine.connect() as conn:
        stats = {}
        for table in ["stock_prices", "predictions", "macro_data", "technical_indicators"]:
            r = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).fetchone()
            stats[table] = r[0]
        r = conn.execute(text("SELECT MAX(trade_date) FROM stock_prices")).fetchone()
        stats["latest_price_date"] = str(r[0]) if r[0] else None
        r = conn.execute(text("SELECT MAX(pred_date) FROM predictions")).fetchone()
        stats["latest_pred_date"] = str(r[0]) if r[0] else None
    return stats


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
