"""
AI Trading System - 매일 추론 배치
모델 A (07:30): 전일종가 + 당일새벽 미국마감 + 밤뉴스 → 오늘 UP/DOWN
모델 B (17:00): 당일종가 + 전일 미국마감 + 오후뉴스 → 내일 UP/DOWN
"""

import time
import logging
import os
import sys
import pandas as pd
from datetime import date

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import get_all_stocks, MODEL_CONFIG
from data.collector import StockCollector, MacroCollector
from data.news_collector import NewsCollector
from data.preprocessor import FeatureBuilder
from models.advanced_model import AdvancedEnsemblePredictor
from database.db_manager import DBManager

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(f"logs/inference_{date.today()}.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def run_inference(model_type: str = "model_A"):
    """
    매일 추론 실행
    model_type: "model_A" (07:30) or "model_B" (17:00)
    """
    start_time = time.time()
    logger.info("=" * 60)
    logger.info(f"[{model_type.upper()}] 추론 시작: {date.today()}")
    logger.info("=" * 60)

    db = DBManager()
    nc = NewsCollector()

    # 모델 로드
    predictor = AdvancedEnsemblePredictor()
    loaded = predictor.load(horizon=1)
    if not loaded:
        logger.error("모델 미학습 상태! weekly_train을 먼저 실행하세요.")
        return

    # 뉴스 시간대 결정
    news_window = "morning" if model_type == "model_A" else "afternoon"
    logger.info(f"  뉴스 시간대: {news_window}")

    # STEP 1: 데이터 수집 (모델 B는 당일 종가 업데이트)
    if model_type == "model_B":
        logger.info("--- 당일 종가 수집 ---")
        collector = StockCollector()
        today_str = date.today().strftime("%Y-%m-%d")
        for stock in get_all_stocks():
            try:
                df = collector.fetch_prices(stock["ticker"], today_str, today_str)
                if not df.empty:
                    df["ticker"] = stock["ticker"]
                    df["name"] = stock["name"]
                    df["layer"] = stock["layer"]
                    db.upsert_stock_prices(df)
            except Exception:
                pass
            time.sleep(0.1)

    # STEP 2: 전 종목 추론
    logger.info("--- 추론 실행 ---")
    all_preds = []
    for stock in get_all_stocks():
        ticker = stock["ticker"]
        try:
            price_df = db.get_stock_prices(ticker, days=200)
            if price_df.empty or len(price_df) < 30:
                continue

            # 뉴스 센티멘트
            news = nc.get_sentiment_features(ticker, time_window=news_window)
            news_feat = {
                "news_sentiment": news.get("sentiment_score", 0),
                "news_count": news.get("news_count", 0),
                "news_positive_ratio": news.get("positive_ratio", 0),
                "news_negative_ratio": news.get("negative_ratio", 0),
                "news_momentum": news.get("news_momentum", 0),
            }

            # 각 horizon 예측
            for horizon in MODEL_CONFIG["prediction_horizons"]:
                if not predictor.is_trained(horizon):
                    predictor.load(horizon)
                result = predictor.predict_single(ticker, price_df, news_features=news_feat, horizon=horizon)
                if result:
                    result["name"] = stock["name"]
                    result["layer"] = stock["layer"]
                    result["model_type"] = model_type
                    result["pred_date"] = date.today().isoformat()
                    all_preds.append(result)
        except Exception as e:
            logger.debug(f"  {ticker} 실패: {e}")
        time.sleep(0.3)  # 뉴스 API 부하 방지

    # STEP 3: DB 저장
    if all_preds:
        pred_df = pd.DataFrame(all_preds)
        pred_df["model_version"] = f"advanced_{model_type}"
        n = db.upsert_predictions(pred_df)
        up = (pred_df["pred_label"] == "UP").sum()
        logger.info(f"  저장: {n}건 (UP={up} DOWN={len(pred_df)-up})")
    else:
        logger.warning("  예측 결과 없음!")

    elapsed = time.time() - start_time
    logger.info(f"[{model_type.upper()}] 완료 | {elapsed:.0f}초")


def run_monitoring():
    """전일 예측 vs 실제 비교 → 적중률 기록"""
    logger.info("--- 모니터링: 전일 적중률 확인 ---")
    db = DBManager()
    from sqlalchemy import text

    with db.engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT p.pred_date, p.pred_label, p.model_type,
                   sp_base.close as base_close, sp_next.close as next_close
            FROM predictions p
            LEFT JOIN LATERAL (
                SELECT close FROM stock_prices
                WHERE ticker = p.ticker AND trade_date < p.pred_date
                ORDER BY trade_date DESC LIMIT 1
            ) sp_base ON true
            LEFT JOIN LATERAL (
                SELECT close FROM stock_prices
                WHERE ticker = p.ticker AND trade_date >= p.pred_date
                ORDER BY trade_date ASC LIMIT 1
            ) sp_next ON true
            WHERE p.pred_date = (CURRENT_DATE - INTERVAL '1 day')::date
              AND p.horizon = 1
              AND sp_base.close IS NOT NULL AND sp_next.close IS NOT NULL
        """), conn)

    if df.empty:
        logger.info("  전일 예측 데이터 없음")
        return

    df["actual_label"] = df.apply(
        lambda r: "UP" if r["next_close"] > r["base_close"] else "DOWN", axis=1)
    df["hit"] = df["pred_label"] == df["actual_label"]

    for mt in df["model_type"].unique():
        sub = df[df["model_type"] == mt]
        acc = sub["hit"].mean()
        logger.info(f"  {mt}: 적중률 {acc:.1%} ({sub['hit'].sum()}/{len(sub)})")

        # 3일 연속 40% 이하면 경고
        if acc < 0.4:
            logger.warning(f"  ⚠️ {mt} 적중률 저하! 재학습 필요 가능성")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["A", "B", "monitor"], default="A")
    args = parser.parse_args()

    if args.model == "monitor":
        run_monitoring()
    else:
        run_inference(f"model_{args.model}")
