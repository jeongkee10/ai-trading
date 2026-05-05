"""
AI Trading System - 주간 학습 (매주 일요일 02:00)
5모델 Optuna 최적화 + PyTorch 딥러닝 학습
"""

import time
import logging
import os
import sys
import pandas as pd
from datetime import date

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import get_all_stocks, MODEL_CONFIG
from data.preprocessor import FeatureBuilder
from data.news_collector import NewsCollector
from models.advanced_model import AdvancedEnsemblePredictor
from database.db_manager import DBManager

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(f"logs/weekly_train_{date.today()}.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def run_weekly_train(optuna_trials: int = 30):
    """주간 모델 학습 (일요일 실행)"""
    start_time = time.time()
    logger.info("=" * 60)
    logger.info(f"[WEEKLY TRAIN] 주간 모델 학습 시작: {date.today()}")
    logger.info("=" * 60)

    db = DBManager()
    fb = FeatureBuilder()

    # STEP 1: 전체 종목 피처 빌드
    logger.info("--- STEP 1: 피처 빌드 ---")
    all_dfs = []
    for stock in get_all_stocks():
        price_df = db.get_stock_prices(stock["ticker"], days=800)
        if price_df.empty or len(price_df) < 100:
            continue
        feat_df = fb.build_features(price_df)
        if not feat_df.empty:
            feat_df["ticker"] = stock["ticker"]
            all_dfs.append(feat_df)

    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"  데이터: {len(all_dfs)}종목, {len(combined):,}행")

    # STEP 2: 뉴스 센티멘트 (학습용은 0으로 채움 — 과거 뉴스 없음)
    # 실제 추론 시에만 실시간 뉴스 사용

    # STEP 3: 각 horizon별 학습
    predictor = AdvancedEnsemblePredictor()
    all_results = {}

    for horizon in MODEL_CONFIG["prediction_horizons"]:
        logger.info(f"\n{'='*40}")
        logger.info(f"  Horizon T+{horizon} 학습 시작")
        results = predictor.train_all(combined, horizon=horizon, optuna_trials=optuna_trials)
        all_results[f"H{horizon}"] = results

    elapsed = time.time() - start_time
    logger.info(f"\n[WEEKLY TRAIN] 전체 완료 | 소요: {elapsed:.0f}초 ({elapsed/60:.1f}분)")

    # 성능 리포트 저장
    report_path = os.path.join("logs", f"model_report_{date.today()}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Weekly Model Training Report - {date.today()}\n")
        f.write("=" * 50 + "\n")
        for h, results in all_results.items():
            f.write(f"\n{h}:\n")
            for model_name, metrics in results.items():
                if metrics:
                    f.write(f"  {model_name}: F1={metrics.get('f1', 0):.4f}\n")
    logger.info(f"  리포트 저장: {report_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30, help="Optuna trials per model")
    args = parser.parse_args()
    run_weekly_train(optuna_trials=args.trials)
