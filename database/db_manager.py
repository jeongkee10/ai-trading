"""
AI Trading System - PostgreSQL DB 관리자
테이블 생성, CRUD, 연결 관리
"""

import logging
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from sqlalchemy import create_engine, text
from datetime import datetime, date

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DB_CONFIG, DB_URL

logger = logging.getLogger(__name__)


class DBManager:
    def __init__(self):
        self.db_url = DB_URL
        self.engine = None
        self._connect()

    def _connect(self):
        try:
            self.engine = create_engine(self.db_url, pool_pre_ping=True)
            logger.info("PostgreSQL 연결 성공")
        except Exception as e:
            logger.error(f"PostgreSQL 연결 실패: {e}")
            raise

    def init_tables(self):
        """전체 테이블 초기화 (최초 1회 실행)"""
        ddl_statements = [
            # 1. 주가 일별 데이터
            """
            CREATE TABLE IF NOT EXISTS stock_prices (
                id          BIGSERIAL PRIMARY KEY,
                ticker      VARCHAR(20) NOT NULL,
                name        VARCHAR(50),
                trade_date  DATE NOT NULL,
                open        NUMERIC(15,2),
                high        NUMERIC(15,2),
                low         NUMERIC(15,2),
                close       NUMERIC(15,2),
                volume      BIGINT,
                adj_close   NUMERIC(15,2),
                layer       VARCHAR(50),
                created_at  TIMESTAMP DEFAULT NOW(),
                UNIQUE(ticker, trade_date)
            );
            CREATE INDEX IF NOT EXISTS idx_sp_ticker_date ON stock_prices(ticker, trade_date DESC);
            """,

            # 2. 기술적 지표
            """
            CREATE TABLE IF NOT EXISTS technical_indicators (
                id              BIGSERIAL PRIMARY KEY,
                ticker          VARCHAR(20) NOT NULL,
                trade_date      DATE NOT NULL,
                sma5            NUMERIC(15,4),
                sma20           NUMERIC(15,4),
                sma60           NUMERIC(15,4),
                sma120          NUMERIC(15,4),
                ema5            NUMERIC(15,4),
                ema20           NUMERIC(15,4),
                macd            NUMERIC(15,4),
                macd_signal     NUMERIC(15,4),
                macd_hist       NUMERIC(15,4),
                rsi14           NUMERIC(8,4),
                bb_upper        NUMERIC(15,4),
                bb_middle       NUMERIC(15,4),
                bb_lower        NUMERIC(15,4),
                bb_pct          NUMERIC(8,4),
                bb_width        NUMERIC(8,4),
                stoch_k         NUMERIC(8,4),
                stoch_d         NUMERIC(8,4),
                obv             NUMERIC(20,2),
                volume_ratio    NUMERIC(8,4),
                atr14           NUMERIC(15,4),
                created_at      TIMESTAMP DEFAULT NOW(),
                UNIQUE(ticker, trade_date)
            );
            CREATE INDEX IF NOT EXISTS idx_ti_ticker_date ON technical_indicators(ticker, trade_date DESC);
            """,

            # 3. 재무 데이터 (분기)
            """
            CREATE TABLE IF NOT EXISTS financial_data (
                id              BIGSERIAL PRIMARY KEY,
                ticker          VARCHAR(20) NOT NULL,
                corp_code       VARCHAR(20),
                period          VARCHAR(10),      -- 예: 2025Q3
                revenue         NUMERIC(20,2),
                operating_income NUMERIC(20,2),
                net_income      NUMERIC(20,2),
                eps             NUMERIC(10,2),
                roe             NUMERIC(8,4),
                per             NUMERIC(8,2),
                pbr             NUMERIC(8,2),
                debt_ratio      NUMERIC(8,2),
                op_margin       NUMERIC(8,4),
                eps_yoy         NUMERIC(8,4),
                updated_at      TIMESTAMP DEFAULT NOW(),
                UNIQUE(ticker, period)
            );
            """,

            # 4. 수급 데이터 (외국인·기관)
            """
            CREATE TABLE IF NOT EXISTS supply_demand (
                id                  BIGSERIAL PRIMARY KEY,
                ticker              VARCHAR(20) NOT NULL,
                trade_date          DATE NOT NULL,
                foreign_buy         BIGINT DEFAULT 0,
                foreign_sell        BIGINT DEFAULT 0,
                foreign_net         BIGINT DEFAULT 0,
                institution_buy     BIGINT DEFAULT 0,
                institution_sell    BIGINT DEFAULT 0,
                institution_net     BIGINT DEFAULT 0,
                individual_net      BIGINT DEFAULT 0,
                short_sell_vol      BIGINT DEFAULT 0,
                foreign_net_5d      BIGINT DEFAULT 0,
                institution_net_5d  BIGINT DEFAULT 0,
                created_at          TIMESTAMP DEFAULT NOW(),
                UNIQUE(ticker, trade_date)
            );
            CREATE INDEX IF NOT EXISTS idx_sd_ticker_date ON supply_demand(ticker, trade_date DESC);
            """,

            # 5. 거시경제 데이터
            """
            CREATE TABLE IF NOT EXISTS macro_data (
                id          BIGSERIAL PRIMARY KEY,
                data_date   DATE NOT NULL,
                usd_krw     NUMERIC(10,2),
                us_10y      NUMERIC(8,4),
                vix         NUMERIC(8,4),
                kospi       NUMERIC(10,2),
                kosdaq      NUMERIC(10,2),
                sp500       NUMERIC(10,2),
                nasdaq      NUMERIC(10,2),
                sox         NUMERIC(10,2),
                created_at  TIMESTAMP DEFAULT NOW(),
                UNIQUE(data_date)
            );
            CREATE INDEX IF NOT EXISTS idx_macro_date ON macro_data(data_date DESC);
            """,

            # 6. 예측 결과
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id              BIGSERIAL PRIMARY KEY,
                ticker          VARCHAR(20) NOT NULL,
                name            VARCHAR(50),
                layer           VARCHAR(50),
                pred_date       DATE NOT NULL,
                horizon         INT NOT NULL,       -- 1, 5, 20 (거래일)
                pred_up_prob    NUMERIC(6,4),
                pred_hold_prob  NUMERIC(6,4),
                pred_down_prob  NUMERIC(6,4),
                pred_label      VARCHAR(10),        -- UP/HOLD/DOWN
                xgb_signal      VARCHAR(10),
                lgb_signal      VARCHAR(10),
                lstm_signal     VARCHAR(10),
                gru_signal      VARCHAR(10),
                transformer_signal VARCHAR(10),
                agreement_ratio NUMERIC(6,4),
                confidence_mult NUMERIC(6,4),
                tech_score      NUMERIC(6,4),
                supply_score    NUMERIC(6,4),
                total_score     NUMERIC(6,4),
                model_version   VARCHAR(20),
                created_at      TIMESTAMP DEFAULT NOW(),
                UNIQUE(ticker, pred_date, horizon)
            );
            CREATE INDEX IF NOT EXISTS idx_pred_date ON predictions(pred_date DESC);
            """,

            # 7. 백테스트 결과
            """
            CREATE TABLE IF NOT EXISTS backtest_results (
                id                  BIGSERIAL PRIMARY KEY,
                run_date            DATE NOT NULL,
                ticker              VARCHAR(20) NOT NULL,
                horizon             INT NOT NULL,
                start_date          DATE,
                end_date            DATE,
                total_trades        INT,
                win_trades          INT,
                loss_trades         INT,
                win_rate            NUMERIC(6,4),
                total_return        NUMERIC(10,4),
                annualized_return   NUMERIC(10,4),
                max_drawdown        NUMERIC(10,4),
                sharpe_ratio        NUMERIC(8,4),
                benchmark_return    NUMERIC(10,4),
                excess_return       NUMERIC(10,4),
                model_version       VARCHAR(20),
                created_at          TIMESTAMP DEFAULT NOW()
            );
            """,

            # 8. 배치 실행 로그
            """
            CREATE TABLE IF NOT EXISTS batch_log (
                id          BIGSERIAL PRIMARY KEY,
                run_date    DATE NOT NULL,
                run_time    TIMESTAMP NOT NULL,
                step        VARCHAR(50),
                status      VARCHAR(20),    -- SUCCESS / FAILED / PARTIAL
                message     TEXT,
                duration_sec NUMERIC(10,2),
                created_at  TIMESTAMP DEFAULT NOW()
            );
            """,
        ]

        with self.engine.connect() as conn:
            for ddl in ddl_statements:
                try:
                    conn.execute(text(ddl))
                    conn.commit()
                except Exception as e:
                    logger.error(f"테이블 생성 오류: {e}")
                    conn.rollback()

        logger.info("[DONE] 전체 테이블 초기화 완료")

    # =============================================
    # 주가 데이터 CRUD
    # =============================================
    def upsert_stock_prices(self, df: pd.DataFrame) -> int:
        """주가 데이터 upsert"""
        if df.empty:
            return 0
        required = ["ticker", "name", "trade_date", "open", "high", "low",
                    "close", "volume", "adj_close", "layer"]
        df = df[required].dropna(subset=["ticker", "trade_date", "close"])

        sql = """
            INSERT INTO stock_prices (ticker, name, trade_date, open, high, low, close, volume, adj_close, layer)
            VALUES %s
            ON CONFLICT (ticker, trade_date) DO UPDATE SET
                open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low,
                close=EXCLUDED.close, volume=EXCLUDED.volume,
                adj_close=EXCLUDED.adj_close
        """
        rows = [tuple(r) for r in df.itertuples(index=False)]
        try:
            with psycopg2.connect(**DB_CONFIG) as conn:
                with conn.cursor() as cur:
                    execute_values(cur, sql, rows)
                conn.commit()
            return len(rows)
        except Exception as e:
            logger.error(f"주가 upsert 오류: {e}")
            return 0

    def upsert_technical_indicators(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        cols = ["ticker","trade_date","sma5","sma20","sma60","sma120","ema5","ema20",
                "macd","macd_signal","macd_hist","rsi14","bb_upper","bb_middle",
                "bb_lower","bb_pct","bb_width","stoch_k","stoch_d","obv",
                "volume_ratio","atr14"]
        df = df[[c for c in cols if c in df.columns]].dropna(subset=["ticker","trade_date"])

        col_str = ",".join(df.columns)
        update_str = ",".join([f"{c}=EXCLUDED.{c}" for c in df.columns
                               if c not in ["ticker","trade_date"]])
        sql = f"""
            INSERT INTO technical_indicators ({col_str})
            VALUES %s
            ON CONFLICT (ticker, trade_date) DO UPDATE SET {update_str}
        """
        rows = [tuple(r) for r in df.itertuples(index=False)]
        try:
            with psycopg2.connect(**DB_CONFIG) as conn:
                with conn.cursor() as cur:
                    execute_values(cur, sql, rows)
                conn.commit()
            return len(rows)
        except Exception as e:
            logger.error(f"기술지표 upsert 오류: {e}")
            return 0

    def upsert_macro_data(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        # 컬럼 정규화 (DataFrame에 있는 것만 사용)
        all_cols = ["data_date", "usd_krw", "us_10y", "vix", "kospi", "kosdaq",
                     "sp500", "nasdaq", "sox"]
        avail_cols = [c for c in all_cols if c in df.columns]
        df_sub = df[avail_cols].copy()
        for c in all_cols:
            if c not in df_sub.columns:
                df_sub[c] = None
        df_sub = df_sub[all_cols]  # 순서 보장

        sql = """
            INSERT INTO macro_data (data_date, usd_krw, us_10y, vix, kospi, kosdaq,
                                     sp500, nasdaq, sox)
            VALUES %s
            ON CONFLICT (data_date) DO UPDATE SET
                usd_krw=COALESCE(EXCLUDED.usd_krw, macro_data.usd_krw),
                us_10y=COALESCE(EXCLUDED.us_10y, macro_data.us_10y),
                vix=COALESCE(EXCLUDED.vix, macro_data.vix),
                kospi=COALESCE(EXCLUDED.kospi, macro_data.kospi),
                kosdaq=COALESCE(EXCLUDED.kosdaq, macro_data.kosdaq),
                sp500=COALESCE(EXCLUDED.sp500, macro_data.sp500),
                nasdaq=COALESCE(EXCLUDED.nasdaq, macro_data.nasdaq),
                sox=COALESCE(EXCLUDED.sox, macro_data.sox)
        """
        rows = [tuple(r) for r in df_sub.itertuples(index=False)]
        try:
            with psycopg2.connect(**DB_CONFIG) as conn:
                with conn.cursor() as cur:
                    execute_values(cur, sql, rows)
                conn.commit()
            return len(rows)
        except Exception as e:
            logger.error(f"거시데이터 upsert 오류: {e}")
            return 0

    def upsert_predictions(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        cols = ["ticker","pred_date","horizon","pred_up_prob","pred_hold_prob",
                "pred_down_prob","pred_label","xgb_signal","lstm_signal",
                "lgb_signal","gru_signal","transformer_signal",
                "tech_score","supply_score","total_score",
                "agreement_ratio","confidence_mult","model_version","model_type"]
        for c in cols:
            if c not in df.columns:
                df[c] = None
        if df["model_type"].isna().all():
            df["model_type"] = "3class"
        df_sub = df[cols]
        sql = """
            INSERT INTO predictions (ticker,pred_date,horizon,pred_up_prob,pred_hold_prob,
                pred_down_prob,pred_label,xgb_signal,lstm_signal,
                lgb_signal,gru_signal,transformer_signal,
                tech_score,supply_score,total_score,
                agreement_ratio,confidence_mult,model_version,model_type)
            VALUES %s
            ON CONFLICT (ticker, pred_date, horizon, model_type) DO UPDATE SET
                pred_up_prob=EXCLUDED.pred_up_prob,
                pred_hold_prob=EXCLUDED.pred_hold_prob,
                pred_down_prob=EXCLUDED.pred_down_prob,
                pred_label=EXCLUDED.pred_label,
                xgb_signal=EXCLUDED.xgb_signal,
                lstm_signal=EXCLUDED.lstm_signal,
                lgb_signal=EXCLUDED.lgb_signal,
                gru_signal=EXCLUDED.gru_signal,
                transformer_signal=EXCLUDED.transformer_signal,
                tech_score=EXCLUDED.tech_score,
                total_score=EXCLUDED.total_score,
                agreement_ratio=EXCLUDED.agreement_ratio,
                confidence_mult=EXCLUDED.confidence_mult,
                model_version=EXCLUDED.model_version
        """
        rows = [tuple(r) for r in df_sub.itertuples(index=False)]
        try:
            with psycopg2.connect(**DB_CONFIG) as conn:
                with conn.cursor() as cur:
                    execute_values(cur, sql, rows)
                conn.commit()
            return len(rows)
        except Exception as e:
            logger.error(f"예측 upsert 오류: {e}")
            return 0

    def save_backtest_result(self, result: dict):
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO backtest_results
                    (run_date,ticker,horizon,start_date,end_date,total_trades,
                     win_trades,loss_trades,win_rate,total_return,annualized_return,
                     max_drawdown,sharpe_ratio,benchmark_return,excess_return,model_version)
                    VALUES
                    (:run_date,:ticker,:horizon,:start_date,:end_date,:total_trades,
                     :win_trades,:loss_trades,:win_rate,:total_return,:annualized_return,
                     :max_drawdown,:sharpe_ratio,:benchmark_return,:excess_return,:model_version)
                """), result)
                conn.commit()
        except Exception as e:
            logger.error(f"백테스트 저장 오류: {e}")

    def log_batch(self, step: str, status: str, message: str = "", duration: float = 0):
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO batch_log (run_date, run_time, step, status, message, duration_sec)
                    VALUES (:run_date, :run_time, :step, :status, :message, :duration_sec)
                """), {
                    "run_date": date.today(),
                    "run_time": datetime.now(),
                    "step": step,
                    "status": status,
                    "message": message,
                    "duration_sec": duration,
                })
                conn.commit()
        except Exception as e:
            logger.error(f"배치 로그 저장 오류: {e}")

    # =============================================
    # 증분 수집용 조회 함수
    # =============================================
    def get_last_trade_date(self, ticker: str) -> date:
        """종목별 마지막 수집일 조회"""
        sql = "SELECT MAX(trade_date) as last_date FROM stock_prices WHERE ticker = :ticker"
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), {"ticker": ticker}).fetchone()
        if result and result[0]:
            return result[0]
        return None

    def get_last_macro_date(self) -> date:
        """거시데이터 마지막 수집일 조회"""
        sql = "SELECT MAX(data_date) as last_date FROM macro_data"
        with self.engine.connect() as conn:
            result = conn.execute(text(sql)).fetchone()
        if result and result[0]:
            return result[0]
        return None

    # =============================================
    # 조회 함수
    # =============================================
    def get_stock_prices(self, ticker: str, days: int = 500) -> pd.DataFrame:
        sql = """
            SELECT trade_date, open, high, low, close, volume, adj_close
            FROM stock_prices
            WHERE ticker = :ticker
            ORDER BY trade_date DESC
            LIMIT :days
        """
        with self.engine.connect() as conn:
            df = pd.read_sql(text(sql), conn, params={"ticker": ticker, "days": days})
        return df.sort_values("trade_date").reset_index(drop=True)

    def get_latest_predictions(self, pred_date: date = None) -> pd.DataFrame:
        if pred_date is None:
            pred_date = date.today()
        sql = """
            SELECT p.id, p.ticker, p.pred_date, p.horizon,
                   p.pred_up_prob, p.pred_hold_prob, p.pred_down_prob,
                   p.pred_label, p.xgb_signal, p.lstm_signal,
                   p.lgb_signal, p.gru_signal, p.transformer_signal,
                   p.agreement_ratio, p.confidence_mult,
                   p.tech_score, p.supply_score, p.total_score,
                   p.model_version, p.created_at,
                   COALESCE(p.name, sp.name) as name,
                   COALESCE(p.layer, sp.layer) as layer
            FROM predictions p
            LEFT JOIN (
                SELECT DISTINCT ON (ticker) ticker, name, layer
                FROM stock_prices ORDER BY ticker, trade_date DESC
            ) sp ON p.ticker = sp.ticker
            WHERE p.pred_date = :pred_date
            ORDER BY p.total_score DESC
        """
        with self.engine.connect() as conn:
            return pd.read_sql(text(sql), conn, params={"pred_date": pred_date})

    def get_macro_latest(self) -> dict:
        """각 지표별로 가장 최근 유효값을 가져옴 (NaN 건너뜀)"""
        sql = """
            SELECT
                (SELECT usd_krw FROM macro_data WHERE usd_krw IS NOT NULL AND usd_krw != 'NaN' ORDER BY data_date DESC LIMIT 1) as usd_krw,
                (SELECT us_10y FROM macro_data WHERE us_10y IS NOT NULL AND us_10y != 'NaN' ORDER BY data_date DESC LIMIT 1) as us_10y,
                (SELECT vix FROM macro_data WHERE vix IS NOT NULL AND vix != 'NaN' ORDER BY data_date DESC LIMIT 1) as vix,
                (SELECT kospi FROM macro_data WHERE kospi IS NOT NULL AND kospi != 'NaN' ORDER BY data_date DESC LIMIT 1) as kospi,
                (SELECT kosdaq FROM macro_data WHERE kosdaq IS NOT NULL AND kosdaq != 'NaN' ORDER BY data_date DESC LIMIT 1) as kosdaq,
                (SELECT data_date FROM macro_data ORDER BY data_date DESC LIMIT 1) as data_date
        """
        with self.engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)
        return df.iloc[0].to_dict() if not df.empty else {}

    def get_batch_logs(self, days: int = 7) -> pd.DataFrame:
        sql = """
            SELECT * FROM batch_log
            WHERE run_date >= CURRENT_DATE - :days
            ORDER BY created_at DESC
            LIMIT 100
        """
        with self.engine.connect() as conn:
            return pd.read_sql(text(sql), conn, params={"days": days})

    def get_backtest_summary(self) -> pd.DataFrame:
        sql = """
            SELECT ticker, horizon, win_rate, total_return, sharpe_ratio,
                   max_drawdown, benchmark_return, excess_return, run_date
            FROM backtest_results
            WHERE run_date = (SELECT MAX(run_date) FROM backtest_results)
            ORDER BY total_return DESC
        """
        with self.engine.connect() as conn:
            return pd.read_sql(text(sql), conn)

    def get_prediction_history(self, ticker: str, horizon: int = 1, days: int = 30) -> pd.DataFrame:
        sql = """
            SELECT pred_date, pred_up_prob, pred_down_prob, pred_label, total_score
            FROM predictions
            WHERE ticker = :ticker AND horizon = :horizon
              AND pred_date >= CURRENT_DATE - :days
            ORDER BY pred_date
        """
        with self.engine.connect() as conn:
            return pd.read_sql(text(sql), conn,
                               params={"ticker": ticker, "horizon": horizon, "days": days})

    def upsert_supply_demand(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        cols = ["ticker","trade_date","foreign_buy","foreign_sell","foreign_net",
                "institution_buy","institution_sell","institution_net",
                "individual_net","short_sell_vol","foreign_net_5d","institution_net_5d"]
        df = df[[c for c in cols if c in df.columns]]
        sql = """
            INSERT INTO supply_demand (ticker,trade_date,foreign_buy,foreign_sell,foreign_net,
                institution_buy,institution_sell,institution_net,individual_net,
                short_sell_vol,foreign_net_5d,institution_net_5d)
            VALUES %s
            ON CONFLICT (ticker, trade_date) DO UPDATE SET
                foreign_net=EXCLUDED.foreign_net,
                institution_net=EXCLUDED.institution_net,
                individual_net=EXCLUDED.individual_net
        """
        rows = [tuple(r) for r in df.itertuples(index=False)]
        try:
            with psycopg2.connect(**DB_CONFIG) as conn:
                with conn.cursor() as cur:
                    execute_values(cur, sql, rows)
                conn.commit()
            return len(rows)
        except Exception as e:
            logger.error(f"수급 upsert 오류: {e}")
            return 0
