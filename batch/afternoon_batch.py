"""
AI Trading System - 오후 배치 (17:00 KST)
모델 B: 당일 종가 + 뉴스(15:30~17:00) → 내일 UP/DOWN 예측
"""

import time
import logging
import os
import sys
import pandas as pd
from datetime import date

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import get_all_stocks
from data.collector import StockCollector
from data.news_collector import NewsCollector
from models.binary_model import BinaryEnsemblePredictor
from database.db_manager import DBManager

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(f"logs/model_B_{date.today()}.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def run_model_b_batch():
    """모델 B (17:00): 당일종가 + 오후뉴스(15:30~17:00) → 내일 예측"""
    start_time = time.time()
    logger.info("=" * 60)
    logger.info(f"[MODEL B] 오후 배치 시작: {date.today()} 17:00")
    logger.info("=" * 60)

    db = DBManager()
    nc = NewsCollector()
    predictor = BinaryEnsemblePredictor()

    # STEP 1: 당일 종가 업데이트
    logger.info("--- STEP 1: 당일 종가 수집 ---")
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

    # STEP 2: 오후 뉴스 수집 (15:30~17:00)
    logger.info("--- STEP 2: 오후 뉴스 수집 (15:30~17:00) ---")
    news_map = {}
    for stock in get_all_stocks():
        try:
            news = nc.get_sentiment_features(stock["ticker"], time_window="afternoon")
            news_map[stock["ticker"]] = {
                "news_sentiment": news.get("sentiment_score", 0),
                "news_count": news.get("news_count", 0),
                "news_positive_ratio": news.get("positive_ratio", 0),
                "news_negative_ratio": news.get("negative_ratio", 0),
                "news_momentum": news.get("news_momentum", 0),
            }
        except Exception:
            news_map[stock["ticker"]] = {}
        time.sleep(0.3)

    # STEP 3: 예측 실행
    logger.info("--- STEP 3: 모델 B 예측 ---")
    all_preds = []
    for stock in get_all_stocks():
        ticker = stock["ticker"]
        price_df = db.get_stock_prices(ticker, days=200)
        if price_df.empty:
            continue
        try:
            preds = predictor.predict_single(ticker, price_df, news_features=news_map.get(ticker, {}))
            for p in preds:
                p["name"] = stock["name"]
                p["layer"] = stock["layer"]
                p["model_type"] = "model_B"
            all_preds.extend(preds)
        except Exception:
            pass

    # STEP 4: DB 저장
    if all_preds:
        pred_df = pd.DataFrame(all_preds)
        pred_df["model_version"] = "binary_B_v1"
        n = db.upsert_predictions(pred_df)
        up = (pred_df["pred_label"] == "UP").sum()
        logger.info(f"  저장: {n}건 (UP={up} DOWN={len(pred_df)-up})")

    elapsed = time.time() - start_time
    logger.info(f"[MODEL B] 완료 | {elapsed:.1f}초")


def run_model_a_batch():
    """모델 A (07:30): 전일종가 + 밤뉴스(전일15:30~07:30) → 오늘 예측"""
    start_time = time.time()
    logger.info("=" * 60)
    logger.info(f"[MODEL A] 오전 배치 시작: {date.today()} 07:30")
    logger.info("=" * 60)

    db = DBManager()
    nc = NewsCollector()
    predictor = BinaryEnsemblePredictor()

    # 아침 뉴스 수집 (전일 15:30 ~ 당일 07:30)
    logger.info("--- 아침 뉴스 수집 (전일15:30~07:30) ---")
    news_map = {}
    for stock in get_all_stocks():
        try:
            news = nc.get_sentiment_features(stock["ticker"], time_window="morning")
            news_map[stock["ticker"]] = {
                "news_sentiment": news.get("sentiment_score", 0),
                "news_count": news.get("news_count", 0),
                "news_positive_ratio": news.get("positive_ratio", 0),
                "news_negative_ratio": news.get("negative_ratio", 0),
                "news_momentum": news.get("news_momentum", 0),
            }
        except Exception:
            news_map[stock["ticker"]] = {}
        time.sleep(0.3)

    # 예측 실행
    logger.info("--- 모델 A 예측 ---")
    all_preds = []
    for stock in get_all_stocks():
        ticker = stock["ticker"]
        price_df = db.get_stock_prices(ticker, days=200)
        if price_df.empty:
            continue
        try:
            preds = predictor.predict_single(ticker, price_df, news_features=news_map.get(ticker, {}))
            for p in preds:
                p["name"] = stock["name"]
                p["layer"] = stock["layer"]
                p["model_type"] = "model_A"
            all_preds.extend(preds)
        except Exception:
            pass

    # DB 저장
    if all_preds:
        pred_df = pd.DataFrame(all_preds)
        pred_df["model_version"] = "binary_A_v1"
        n = db.upsert_predictions(pred_df)
        up = (pred_df["pred_label"] == "UP").sum()
        logger.info(f"  저장: {n}건 (UP={up} DOWN={len(pred_df)-up})")

    elapsed = time.time() - start_time
    logger.info(f"[MODEL A] 완료 | {elapsed:.1f}초")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["A", "B"], default="B")
    args = parser.parse_args()
    if args.model == "A":
        run_model_a_batch()
    else:
        run_model_b_batch()
