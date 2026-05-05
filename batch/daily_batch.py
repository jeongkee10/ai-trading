"""
AI Trading System - 일배치 실행기
매일 오전 7:00 자동 실행 (APScheduler)
수동 실행: python batch/daily_batch.py --run-now
"""

import time
import logging
import argparse
import os
import sys
import pandas as pd
from datetime import datetime, date, timedelta

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import BATCH_CONFIG, get_all_stocks, STOCK_UNIVERSE
from data.collector import StockCollector, MacroCollector, DARTCollector, NaverFinanceCollector
from data.preprocessor import TechnicalIndicatorCalculator, FeatureBuilder
from models.ensemble import EnsemblePredictor, SignalGenerator
from models.xgboost_model import XGBStockModel
from database.db_manager import DBManager

# 로깅 설정
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(f"logs/batch_{date.today()}.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def run_daily_batch():
    """메인 배치 실행 함수"""
    start_time = time.time()
    today = date.today()
    logger.info("=" * 60)
    logger.info(f"[START] AI Trading 일배치 시작: {today}")
    logger.info("=" * 60)

    db = DBManager()

    # ── STEP 1: 주가 데이터 수집 ────────────────────────
    _run_step(db, "STEP1_주가수집", _collect_prices, db)

    # ── STEP 2: 거시 데이터 수집 ────────────────────────
    _run_step(db, "STEP2_거시수집", _collect_macro, db)

    # ── STEP 3: 기술적 지표 계산 ────────────────────────
    _run_step(db, "STEP3_기술지표", _calc_indicators, db)

    # ── STEP 4: 재무 데이터 수집 ────────────────────────
    _run_step(db, "STEP4_재무수집", _collect_financials, db)

    # ── STEP 4.5: 수급 데이터 수집 ────────────────────
    _run_step(db, "STEP4.5_수급수집", _collect_supply_demand, db)

    # ── STEP 4.7: Self-consistency 검증 ─────────────────
    _run_step(db, "STEP4.7_자기검증", _self_consistency_check, db)

    # ── STEP 4.8: 주간 적응형 가중치 업데이트 (월요일) ───
    if date.today().weekday() == 0:
        _run_step(db, "STEP4.8_가중치조정", _adaptive_weight_update, db)

    # ── STEP 5: 모델 학습 (최초 or 주 1회 재학습 or 검증 실패) ───
    _run_step(db, "STEP5_모델학습", _train_models, db)

    # ── STEP 6: 예측 실행 ───────────────────────────────
    _run_step(db, "STEP6_예측실행", _run_predictions, db)

    # ── STEP 7: 주간 롤링 백테스트 (월요일) ──────────────
    if date.today().weekday() == 0:
        _run_step(db, "STEP7_롤링백테스트", _rolling_backtest, db)

    elapsed = time.time() - start_time
    logger.info(f"[DONE] 배치 완료 | 소요시간: {elapsed:.1f}초")
    db.log_batch("BATCH_COMPLETE", "SUCCESS",
                 f"총 소요시간: {elapsed:.1f}초", elapsed)


def _run_step(db: DBManager, step_name: str, func, *args):
    """단계 실행 + 로깅"""
    t = time.time()
    logger.info(f"--- {step_name} 시작 ---")
    try:
        func(*args)
        elapsed = time.time() - t
        db.log_batch(step_name, "SUCCESS", "", elapsed)
        logger.info(f"--- {step_name} 완료 ({elapsed:.1f}s) ---")
    except Exception as e:
        elapsed = time.time() - t
        db.log_batch(step_name, "FAILED", str(e), elapsed)
        logger.error(f"--- {step_name} 실패: {e} ---")


def _collect_prices(db: DBManager):
    """주가 증분 수집 (DB 마지막 날짜 이후분만)"""
    collector = StockCollector()
    today_str = date.today().strftime("%Y-%m-%d")

    total_rows = 0
    for layer_key, layer_data in STOCK_UNIVERSE.items():
        for stock in layer_data["stocks"]:
            ticker = stock["ticker"]
            name   = stock["name"]
            try:
                last_date = db.get_last_trade_date(ticker)
                if last_date:
                    # 마지막 수집일 다음날부터 수집
                    start_str = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
                    if start_str > today_str:
                        continue  # 이미 최신
                else:
                    # DB에 데이터 없으면 3년치 수집
                    start_str = (date.today() - timedelta(days=365*3)).strftime("%Y-%m-%d")

                df = collector.fetch_prices(ticker, start_str, today_str)
                if not df.empty:
                    df["ticker"] = ticker
                    df["name"]   = name
                    df["layer"]  = layer_key
                    n = db.upsert_stock_prices(df)
                    total_rows += n
                    logger.info(f"  {name}: {start_str}~{today_str} -> {n}건 수집")
            except Exception as e:
                logger.warning(f"  {name} 주가 수집 실패: {e}")
            time.sleep(0.2)

    logger.info(f"  주가 총 {total_rows}건 저장 완료")


def _collect_macro(db: DBManager):
    """거시경제 증분 수집 (DB 마지막 날짜 이후분만)"""
    collector = MacroCollector()
    last_date = db.get_last_macro_date()

    if last_date:
        start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        end = date.today().strftime("%Y-%m-%d")
        if start > end:
            logger.info("  거시데이터 이미 최신")
            return
        df = collector.fetch_macro_range(start, end)
    else:
        df = collector.fetch_macro(years=3)

    if not df.empty:
        n = db.upsert_macro_data(df)
        logger.info(f"  거시데이터 {n}건 저장 완료")
    else:
        logger.info("  거시데이터 신규 없음")


def _calc_indicators(db: DBManager):
    """기술적 지표 계산 및 저장"""
    calc = TechnicalIndicatorCalculator()
    total = 0
    for stock in get_all_stocks():
        ticker = stock["ticker"]
        try:
            price_df = db.get_stock_prices(ticker, days=200)
            if price_df.empty:
                continue
            ti_df = calc.calculate(price_df)
            if not ti_df.empty:
                ti_df["ticker"] = ticker
                n = db.upsert_technical_indicators(ti_df)
                total += n
        except Exception as e:
            logger.warning(f"  {ticker} 기술지표 계산 실패: {e}")
    logger.info(f"  기술지표 {total}건 저장 완료")


def _collect_financials(db: DBManager):
    """재무 데이터 수집 (네이버 금융 + yfinance 보완)"""
    naver = NaverFinanceCollector()
    success = 0
    for stock in get_all_stocks():
        ticker = stock["ticker"]
        try:
            metrics = naver.get_financial_metrics(ticker)
            if metrics and len(metrics) > 2:
                with db.engine.connect() as conn:
                    from sqlalchemy import text
                    conn.execute(text("""
                        INSERT INTO financial_data
                        (ticker, period, per, pbr, roe, op_margin, revenue, net_income, eps)
                        VALUES (:ticker, :period, :per, :pbr, :roe, :op_margin,
                                :revenue, :net_income, :eps)
                        ON CONFLICT (ticker, period) DO UPDATE SET
                            per=EXCLUDED.per, pbr=EXCLUDED.pbr, roe=EXCLUDED.roe,
                            op_margin=EXCLUDED.op_margin, revenue=EXCLUDED.revenue,
                            net_income=EXCLUDED.net_income, eps=EXCLUDED.eps
                    """), {
                        "ticker": metrics.get("ticker", ticker),
                        "period": metrics.get("period", ""),
                        "per": metrics.get("per", None),
                        "pbr": metrics.get("pbr", None),
                        "roe": metrics.get("roe", None),
                        "op_margin": metrics.get("op_margin", None),
                        "revenue": metrics.get("revenue", None),
                        "net_income": metrics.get("net_income", None),
                        "eps": metrics.get("eps", None),
                    })
                    conn.commit()
                success += 1
        except Exception as e:
            logger.warning(f"  {ticker} 재무 저장 실패: {e}")
        time.sleep(0.5)
    logger.info(f"  재무데이터 {success}건 수집 완료")


def _collect_supply_demand(db: DBManager):
    """수급 데이터 수집 (네이버 금융 크롤링)"""
    naver = NaverFinanceCollector()
    try:
        sd_df = naver.fetch_all_supply_demand(pages=1)
        if not sd_df.empty:
            n = db.upsert_supply_demand(sd_df)
            logger.info(f"  수급 데이터 {n}건 저장 완료")
        else:
            logger.info("  수급 데이터 없음 (휴장일)")
    except Exception as e:
        logger.warning(f"  수급 데이터 수집 실패: {e}")


_force_retrain = False  # Self-consistency 검증 결과에 따른 강제 재학습 플래그

def _self_consistency_check(db: DBManager):
    """Self-consistency 검증: 과거 예측 정확도 확인 → 재학습 트리거"""
    global _force_retrain
    predictor = EnsemblePredictor()
    for horizon in [1, 5, 20]:
        result = predictor.validate_past_predictions(db, horizon=horizon, lookback_days=30)
        if result.get("need_retrain"):
            _force_retrain = True
            logger.warning(f"  H{horizon} 정확도 저하 감지 → 강제 재학습 예정")
    if not _force_retrain:
        logger.info("  Self-consistency 검증 통과")


def _train_models(db: DBManager):
    """
    모델 학습 (최초 실행 or 월요일 재학습 or 검증 실패 시 강제)
    """
    from models.xgboost_model import XGBStockModel
    from models.lstm_model import LSTMStockModel

    global _force_retrain

    # 이미 학습된 모델이 있고 오늘이 월요일이 아니고 검증도 통과했으면 스킵
    xgb_h1 = XGBStockModel(horizon=1)
    if xgb_h1.is_trained() and date.today().weekday() != 0 and not _force_retrain:
        logger.info("  기존 모델 사용 (월요일에 재학습)")
        return

    if _force_retrain:
        logger.info("  Self-consistency 검증 실패 → 강제 재학습 시작")
        _force_retrain = False

    logger.info("  전체 종목 데이터 로드 중...")
    fb = FeatureBuilder()
    all_dfs = []

    for stock in get_all_stocks():
        ticker = stock["ticker"]
        price_df = db.get_stock_prices(ticker, days=800)
        if price_df.empty or len(price_df) < 100:
            continue
        feat_df = fb.build_features(price_df)
        if feat_df.empty:
            continue
        feat_df["ticker"] = ticker
        all_dfs.append(feat_df)

    if not all_dfs:
        logger.error("  학습 데이터 없음")
        return

    import pandas as pd
    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"  학습 데이터: {len(combined)}행")

    predictor = EnsemblePredictor()
    results = predictor.train_all(combined)
    logger.info(f"  모델 학습 완료: {results}")


def _run_predictions(db: DBManager):
    """전체 종목 예측 실행 및 저장"""
    predictor = EnsemblePredictor()
    all_preds = []

    for stock in get_all_stocks():
        ticker = stock["ticker"]
        price_df = db.get_stock_prices(ticker, days=200)
        if price_df.empty:
            continue
        try:
            preds = predictor.predict_single(ticker, price_df)
            all_preds.extend(preds)
        except Exception as e:
            logger.warning(f"  {ticker} 예측 실패: {e}")

    if all_preds:
        import pandas as pd
        pred_df = pd.DataFrame(all_preds)
        n = db.upsert_predictions(pred_df)
        logger.info(f"  예측 {n}건 저장 완료")


def _adaptive_weight_update(db: DBManager):
    """4주 롤링 성과 기반 앙상블 가중치 조정 (매주 월요일)"""
    from models.adaptive_weights import run_weekly_weight_update
    for horizon in [1, 5, 20]:
        result = run_weekly_weight_update(db, horizon=horizon)
        if result.get("updated"):
            logger.info(f"  H{horizon} 가중치 업데이트 완료: {result['new_weights']}")
        else:
            logger.info(f"  H{horizon} 가중치 변경 없음")


def _rolling_backtest(db: DBManager):
    """주간 롤링 백테스트 (최근 4주 예측 성과 평가)"""
    from backtest.backtester import WalkForwardBacktester

    bt = WalkForwardBacktester()
    results = []

    for stock in get_all_stocks():
        ticker = stock["ticker"]
        price_df = db.get_stock_prices(ticker, days=60)
        if price_df.empty or len(price_df) < 25:
            continue
        try:
            result = bt.run(ticker, price_df, pd.DataFrame())
            if result:
                db.save_backtest_result(result)
                results.append(result)
        except Exception as e:
            logger.warning(f"  {stock['name']} 백테스트 실패: {e}")

    if results:
        avg_return = sum(r.get("total_return", 0) for r in results) / len(results)
        avg_winrate = sum(r.get("win_rate", 0) for r in results) / len(results)
        logger.info(f"  롤링 백테스트 완료: {len(results)}종목 | "
                     f"평균수익 {avg_return:.2%} | 평균승률 {avg_winrate:.1%}")
    else:
        logger.info("  롤링 백테스트: 실행 가능 종목 없음")


# ── 초기 설정 (최초 1회 실행) ────────────────────────────
def run_initial_setup():
    """
    최초 실행 시:
    1. DB 테이블 생성
    2. 3년치 과거 데이터 수집
    3. 기술지표 계산
    4. 모델 학습
    5. 백테스트 실행
    """
    logger.info("=" * 60)
    logger.info("[INIT] 초기 설정 시작 (최초 1회 실행)")
    logger.info("=" * 60)

    db = DBManager()

    # 1. 테이블 생성
    logger.info("--- 테이블 초기화 ---")
    db.init_tables()

    # 2. 3년치 과거 주가 수집
    logger.info("--- 과거 주가 수집 (3년) ---")
    collector = StockCollector()
    all_price_df = collector.fetch_all_prices(years=3)
    if not all_price_df.empty:
        n = db.upsert_stock_prices(all_price_df)
        logger.info(f"  주가 {n}건 저장 완료")

    # 3. 거시 데이터 수집
    logger.info("--- 거시 데이터 수집 ---")
    macro = MacroCollector()
    macro_df = macro.fetch_macro(years=3)
    if not macro_df.empty:
        n = db.upsert_macro_data(macro_df)
        logger.info(f"  거시데이터 {n}건 저장 완료")

    # 4. 기술지표 계산
    logger.info("--- 기술지표 계산 ---")
    _calc_indicators(db)

    # 5. 모델 학습
    logger.info("--- 모델 학습 ---")
    _collect_financials(db)

    # 강제 학습 실행
    from models.xgboost_model import XGBStockModel
    import pandas as pd
    fb = FeatureBuilder()
    all_dfs = []
    for stock in get_all_stocks():
        ticker = stock["ticker"]
        price_df = db.get_stock_prices(ticker, days=800)
        if price_df.empty:
            continue
        feat_df = fb.build_features(price_df)
        if not feat_df.empty:
            feat_df["ticker"] = ticker
            all_dfs.append(feat_df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        predictor = EnsemblePredictor()
        predictor.train_all(combined)

    # 6. 초기 예측 실행
    logger.info("--- 초기 예측 실행 ---")
    _run_predictions(db)

    logger.info("[DONE] 초기 설정 완료!")


# ── 스케줄러 실행 ────────────────────────────────────────
def start_scheduler():
    """APScheduler로 매일 오전 7시 배치 실행"""
    scheduler = BlockingScheduler(timezone="Asia/Seoul")
    scheduler.add_job(
        run_daily_batch,
        trigger=CronTrigger(
            hour=BATCH_CONFIG["hour"],
            minute=BATCH_CONFIG["minute"],
            timezone="Asia/Seoul"
        ),
        id="daily_batch",
        name="AI Trading Daily Batch",
        replace_existing=True,
    )
    logger.info(f"[SCHEDULER] 스케줄러 시작: 매일 {BATCH_CONFIG['hour']:02d}:{BATCH_CONFIG['minute']:02d} KST")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("스케줄러 종료")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Trading Daily Batch")
    parser.add_argument("--run-now",  action="store_true", help="즉시 배치 실행")
    parser.add_argument("--init",     action="store_true", help="초기 설정 (최초 1회)")
    parser.add_argument("--schedule", action="store_true", help="스케줄러 시작")
    args = parser.parse_args()

    if args.init:
        run_initial_setup()
    elif args.run_now:
        run_daily_batch()
    elif args.schedule:
        start_scheduler()
    else:
        # 기본: 스케줄러 시작
        start_scheduler()
