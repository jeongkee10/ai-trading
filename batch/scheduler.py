"""
AI Trading System - 스케줄러
일요일: 모델 학습 (02:00)
월~금 07:30: 모델 A 추론 (오전)
일~목 17:00: 모델 B 추론 (오후 - 다음날 예측)
매일 07:00: 데이터 수집 + 모니터링
"""

import logging
import os
import sys
from datetime import date

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(f"logs/scheduler_{date.today()}.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def job_data_collection():
    """매일 07:00 - 데이터 수집"""
    try:
        from batch.daily_batch import run_daily_batch
        run_daily_batch()
    except Exception as e:
        logger.error(f"데이터 수집 실패: {e}")


def job_model_a():
    """월~금 07:30 - 데이터 수집 + 모델 A 추론"""
    try:
        from batch.daily_batch import _collect_prices, _collect_macro, _calc_indicators
        from batch.daily_inference import run_inference
        from database.db_manager import DBManager
        db = DBManager()
        logger.info("[MODEL A] 오전 배치 시작")
        _collect_prices(db)
        _collect_macro(db)
        _calc_indicators(db)
        run_inference("model_A")
        logger.info("[MODEL A] 오전 배치 완료")
    except Exception as e:
        logger.error(f"모델 A 실패: {e}")


def job_model_b():
    """일~목 17:00 - 당일종가 수집 + 모델 B 추론"""
    try:
        from batch.daily_batch import _collect_prices, _calc_indicators
        from batch.daily_inference import run_inference
        from database.db_manager import DBManager
        db = DBManager()
        logger.info("[MODEL B] 오후 배치 시작")
        _collect_prices(db)
        _calc_indicators(db)
        run_inference("model_B")
        logger.info("[MODEL B] 오후 배치 완료")
    except Exception as e:
        logger.error(f"모델 B 실패: {e}")


def job_weekly_train():
    """일요일 02:00 - 주간 모델 학습"""
    from batch.weekly_train import run_weekly_train
    run_weekly_train(optuna_trials=30)


def job_monitoring():
    """매일 08:00 - 전일 적중률 모니터링"""
    from batch.daily_inference import run_monitoring
    run_monitoring()


def start_scheduler():
    scheduler = BlockingScheduler(timezone="Asia/Seoul")

    # 매일 07:00 - 데이터 수집
    scheduler.add_job(job_data_collection,
                      CronTrigger(hour=7, minute=0, day_of_week="mon-fri", timezone="Asia/Seoul"),
                      id="data_collection", replace_existing=True)

    # 월~금 07:30 - 모델 A 추론
    scheduler.add_job(job_model_a,
                      CronTrigger(hour=7, minute=30, day_of_week="mon-fri", timezone="Asia/Seoul"),
                      id="model_a_inference", replace_existing=True)

    # 일~목 17:00 - 모델 B 추론 (다음 거래일 예측)
    scheduler.add_job(job_model_b,
                      CronTrigger(hour=17, minute=0, day_of_week="sun-thu", timezone="Asia/Seoul"),
                      id="model_b_inference", replace_existing=True)

    # 일요일 02:00 - 주간 모델 학습
    scheduler.add_job(job_weekly_train,
                      CronTrigger(hour=2, minute=0, day_of_week="sun", timezone="Asia/Seoul"),
                      id="weekly_train", replace_existing=True)

    # 매일 08:00 - 모니터링
    scheduler.add_job(job_monitoring,
                      CronTrigger(hour=8, minute=0, day_of_week="mon-fri", timezone="Asia/Seoul"),
                      id="monitoring", replace_existing=True)

    logger.info("[SCHEDULER] 스케줄 등록 완료:")
    logger.info("  매일 07:00 (월-금) - 데이터 수집")
    logger.info("  매일 07:30 (월-금) - 모델 A 추론")
    logger.info("  매일 17:00 (일-목) - 모델 B 추론")
    logger.info("  일요일 02:00       - 주간 모델 학습")
    logger.info("  매일 08:00 (월-금) - 적중률 모니터링")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("스케줄러 종료")


if __name__ == "__main__":
    start_scheduler()
