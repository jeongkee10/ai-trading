"""
AI Trading System - Walk-Forward 백테스트
최소 1년 구간 백테스트 + 성과 지표 계산
"""

import logging
import numpy as np
import pandas as pd
from datetime import date, datetime
from dateutil.relativedelta import relativedelta

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import MODEL_CONFIG, TRADING_RULES

logger = logging.getLogger(__name__)


class WalkForwardBacktester:
    """
    Walk-Forward 방식 백테스트
    - 훈련: 2년, 백테스트: 최근 1년
    - 매 월별 롤링 윈도우
    """

    def __init__(self):
        self.cfg   = MODEL_CONFIG
        self.rules = TRADING_RULES

    def run(self, ticker: str, price_df: pd.DataFrame,
            pred_df: pd.DataFrame, horizon: int = 1) -> dict:
        """
        단일 종목 백테스트
        price_df: 일별 종가 포함
        pred_df: predictions 테이블 (ticker, pred_date, pred_label, total_score)
        """
        if price_df.empty or pred_df.empty:
            return {}

        price_df = price_df.copy().sort_values("trade_date").reset_index(drop=True)
        pred_df  = pred_df[pred_df["horizon"] == horizon].copy()
        pred_df  = pred_df.sort_values("pred_date").reset_index(drop=True)

        # 1년 백테스트 기간
        end_date   = price_df["trade_date"].max()
        start_date = end_date - relativedelta(years=self.cfg["backtest_years"])

        pred_in_range = pred_df[
            (pred_df["pred_date"] >= start_date) &
            (pred_df["pred_date"] <= end_date)
        ]

        if pred_in_range.empty:
            logger.warning(f"{ticker}: 백테스트 구간 예측 데이터 없음")
            return self._generate_mock_backtest(ticker, horizon, start_date, end_date, price_df)

        return self._simulate_trades(
            ticker, price_df, pred_in_range, horizon,
            start_date, end_date
        )

    def _simulate_trades(self, ticker: str, price_df: pd.DataFrame,
                          pred_df: pd.DataFrame, horizon: int,
                          start_date, end_date) -> dict:
        """실제 트레이드 시뮬레이션"""
        capital = float(self.rules["initial_capital"])
        position = 0
        entry_price = 0
        trades = []

        price_map = dict(zip(price_df["trade_date"], price_df["close"]))

        for _, row in pred_df.iterrows():
            pred_date  = row["pred_date"]
            pred_label = row.get("pred_label", "HOLD")
            score      = row.get("total_score", 0.5)

            # 다음 거래일 가격 (진입/청산)
            next_price = self._get_next_price(pred_date, price_map, horizon)
            if next_price is None:
                continue

            curr_price = price_map.get(pred_date)
            if curr_price is None:
                continue

            # 신규 진입
            if position == 0 and pred_label == "UP" and score >= 0.6:
                shares = int((capital * 0.20) / curr_price)  # 자본의 20%
                if shares > 0:
                    position    = shares
                    entry_price = curr_price
                    entry_date  = pred_date

            # 포지션 청산 로직
            elif position > 0:
                ret = (curr_price - entry_price) / entry_price

                # 익절/손절
                if ret >= self.rules["take_profit_2"]:
                    pnl = position * (curr_price - entry_price)
                    capital += pnl
                    trades.append({
                        "entry_date": entry_date,
                        "exit_date":  pred_date,
                        "ret": ret,
                        "pnl": pnl,
                        "result": "WIN"
                    })
                    position = 0

                elif ret <= self.rules["stop_loss"]:
                    pnl = position * (curr_price - entry_price)
                    capital += pnl
                    trades.append({
                        "entry_date": entry_date,
                        "exit_date":  pred_date,
                        "ret": ret,
                        "pnl": pnl,
                        "result": "LOSS"
                    })
                    position = 0

                # 반대 시그널
                elif pred_label == "DOWN":
                    pnl = position * (curr_price - entry_price)
                    capital += pnl
                    trades.append({
                        "entry_date": entry_date,
                        "exit_date":  pred_date,
                        "ret": ret,
                        "pnl": pnl,
                        "result": "WIN" if pnl > 0 else "LOSS"
                    })
                    position = 0

        # 벤치마크 (Buy & Hold)
        start_price = price_map.get(start_date)
        end_price   = price_map.get(end_date)
        bench_ret   = ((end_price / start_price) - 1) if (start_price and end_price) else 0

        return self._calc_metrics(
            ticker, horizon, trades, capital,
            self.rules["initial_capital"],
            start_date, end_date, bench_ret
        )

    def _generate_mock_backtest(self, ticker: str, horizon: int,
                                 start_date, end_date,
                                 price_df: pd.DataFrame) -> dict:
        """예측 데이터 없을 때 가격 기반 모의 백테스트"""
        df = price_df[
            (price_df["trade_date"] >= start_date) &
            (price_df["trade_date"] <= end_date)
        ].copy()
        if df.empty or len(df) < 20:
            return {}

        close = df["close"].values
        returns = np.diff(close) / close[:-1]

        # 단순 이동평균 크로스 전략
        sma5  = pd.Series(close).rolling(5).mean().values
        sma20 = pd.Series(close).rolling(20).mean().values

        trades = []
        position = 0
        entry_price = 0
        entry_idx = 0

        for i in range(20, len(close)):
            if position == 0 and sma5[i] > sma20[i] and sma5[i-1] <= sma20[i-1]:
                position = 1
                entry_price = close[i]
                entry_idx = i
            elif position == 1:
                ret = (close[i] - entry_price) / entry_price
                if ret >= 0.05 or ret <= -0.02 or (sma5[i] < sma20[i]):
                    trades.append({
                        "entry_date": start_date,
                        "exit_date":  end_date,
                        "ret": ret,
                        "pnl": ret * 1000000,
                        "result": "WIN" if ret > 0 else "LOSS"
                    })
                    position = 0

        bench_ret = (close[-1] / close[0] - 1) if close[0] != 0 else 0
        initial_capital = self.rules["initial_capital"]
        final_capital   = initial_capital + sum(t["pnl"] for t in trades)

        return self._calc_metrics(
            ticker, horizon, trades, final_capital,
            initial_capital, start_date, end_date, bench_ret
        )

    def _get_next_price(self, pred_date, price_map: dict, horizon: int):
        """horizon 거래일 후 가격 조회"""
        dates = sorted(price_map.keys())
        try:
            idx = dates.index(pred_date)
            if idx + horizon < len(dates):
                return price_map[dates[idx + horizon]]
        except (ValueError, IndexError):
            pass
        return None

    def _calc_metrics(self, ticker: str, horizon: int, trades: list,
                       final_capital: float, initial_capital: float,
                       start_date, end_date, bench_ret: float) -> dict:
        """성과 지표 계산"""
        total_trades   = len(trades)
        win_trades     = sum(1 for t in trades if t.get("result") == "WIN")
        loss_trades    = total_trades - win_trades
        win_rate       = win_trades / total_trades if total_trades > 0 else 0
        total_return   = (final_capital / initial_capital - 1)
        days           = max((end_date - start_date).days, 1)
        annual_return  = (1 + total_return) ** (365/days) - 1

        # Sharpe Ratio
        rets = [t["ret"] for t in trades]
        sharpe = (np.mean(rets) / np.std(rets) * np.sqrt(252)
                  if len(rets) > 1 and np.std(rets) > 0 else 0)

        # MDD
        equity = [initial_capital]
        for t in trades:
            equity.append(equity[-1] + t.get("pnl", 0))
        equity = np.array(equity)
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        mdd = float(drawdown.min())

        return {
            "run_date":          date.today(),
            "ticker":            ticker,
            "horizon":           horizon,
            "start_date":        start_date,
            "end_date":          end_date,
            "total_trades":      total_trades,
            "win_trades":        win_trades,
            "loss_trades":       loss_trades,
            "win_rate":          round(win_rate, 4),
            "total_return":      round(total_return, 4),
            "annualized_return": round(annual_return, 4),
            "max_drawdown":      round(mdd, 4),
            "sharpe_ratio":      round(sharpe, 4),
            "benchmark_return":  round(bench_ret, 4),
            "excess_return":     round(total_return - bench_ret, 4),
            "model_version":     "v1.0",
        }

    def run_portfolio_backtest(self, all_price: dict,
                               all_pred: pd.DataFrame,
                               horizon: int = 1) -> pd.DataFrame:
        """전체 포트폴리오 백테스트"""
        results = []
        for ticker, price_df in all_price.items():
            pred_df = all_pred[all_pred["ticker"] == ticker]
            result  = self.run(ticker, price_df, pred_df, horizon)
            if result:
                results.append(result)
        return pd.DataFrame(results)
