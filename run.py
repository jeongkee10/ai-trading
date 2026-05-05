"""
AI Trading System - 빠른 실행 스크립트
사용법:
  python run.py app        # Streamlit UI 실행
  python run.py init       # 초기 설정 (최초 1회)
  python run.py batch      # 배치 즉시 실행
  python run.py schedule   # 스케줄러 시작 (매일 7시 자동)
  python run.py backtest   # 백테스트 실행
"""

import sys
import os
import subprocess
from datetime import date

# Windows 콘솔 인코딩 설정
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _needs_batch() -> bool:
    """오늘 배치가 실행됐는지 확인. 미실행이면 True."""
    try:
        from database.db_manager import DBManager
        db = DBManager()
        preds = db.get_latest_predictions(pred_date=date.today())
        return preds.empty
    except Exception:
        return True


def _auto_batch():
    """오늘 배치가 없으면 자동 실행"""
    if not _needs_batch():
        print(f"[OK] 오늘({date.today()}) 배치 이미 완료 — 바로 앱 시작")
        return
    print(f"[AUTO] 오늘({date.today()}) 배치 미반영 → 자동 실행 중...")
    try:
        from batch.daily_batch import run_daily_batch
        run_daily_batch()
        print("[OK] 자동 배치 완료")
    except Exception as e:
        print(f"[WARN] 자동 배치 실패 ({e}) — 기존 데이터로 앱 시작")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "app"

    if cmd == "app":
        _auto_batch()
        print("[APP] Streamlit UI 시작...")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "app/streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--theme.base", "dark",
        ])

    elif cmd == "dash":
        _auto_batch()
        print("[DASH] Plotly Dash UI 시작...")
        from app.dash_app import run_dash
        run_dash(debug=True, port=8050)

    elif cmd == "init":
        print("[INIT] 초기 설정 시작...")
        from batch.daily_batch import run_initial_setup
        run_initial_setup()

    elif cmd == "train":
        print("[TRAIN] 주간 모델 학습 (5모델 Optuna + PyTorch)...")
        from batch.weekly_train import run_weekly_train
        run_weekly_train()

    elif cmd == "infer-a" or cmd == "morning":
        print("[MODEL A] 오전 추론 (07:30)...")
        from batch.daily_inference import run_inference
        run_inference("model_A")

    elif cmd == "infer-b" or cmd == "afternoon":
        print("[MODEL B] 오후 추론 (17:00)...")
        from batch.daily_inference import run_inference
        run_inference("model_B")

    elif cmd == "monitor":
        print("[MONITOR] 전일 적중률 확인...")
        from batch.daily_inference import run_monitoring
        run_monitoring()

    elif cmd == "batch-pm":
        print("[MODEL B] 오후 배치 실행 (레거시)...")
        from batch.afternoon_batch import run_afternoon_batch
        run_afternoon_batch()

    elif cmd == "batch":
        print("[BATCH] 오전 배치 즉시 실행 (07:30 모델)...")
        from batch.daily_batch import run_daily_batch
        run_daily_batch()

    elif cmd == "schedule":
        print("[SCHEDULE] 전체 스케줄러 시작 (07:30 모델A + 17:00 모델B + 일요일 학습)...")
        from batch.scheduler import start_scheduler
        start_scheduler()

    elif cmd == "backtest":
        print("[BACKTEST] 백테스트 실행...")
        from database.db_manager import DBManager
        from backtest.backtester import WalkForwardBacktester
        from config.settings import get_all_stocks
        import pandas as pd

        db = DBManager()
        bt = WalkForwardBacktester()
        results = []

        for stock in get_all_stocks():
            ticker = stock["ticker"]
            price_df = db.get_stock_prices(ticker, days=800)
            if not price_df.empty:
                result = bt.run(ticker, price_df, pd.DataFrame())
                if result:
                    db.save_backtest_result(result)
                    results.append(result)
                    print(f"  {stock['name']}: return {result.get('total_return',0):.2%}, "
                          f"win_rate {result.get('win_rate',0):.2%}")

        print(f"\n[DONE] 백테스트 완료: {len(results)}개 종목")

    elif cmd == "createdb":
        print("[DB] 테이블 생성...")
        from database.db_manager import DBManager
        db = DBManager()
        db.init_tables()
        print("[DONE] 테이블 생성 완료")

    else:
        print(f"알 수 없는 명령어: {cmd}")
        print(__doc__)

if __name__ == "__main__":
    main()
