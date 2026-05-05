"""
AI Trading System - 데이터 수집기
yfinance(주가/거시), DART(재무), ECOS(한국은행), KRX(수급)
"""

import time
import logging
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta

import sys, os
import shutil
import certifi

# SSL 인증서 경로 설정 (한글 경로 문제 우회 - ASCII 경로로 복사)
_cert_dst = os.path.join(os.environ.get("LOCALAPPDATA", "C:/Users/jeong/AppData/Local"), "cacert.pem")
if not os.path.exists(_cert_dst):
    shutil.copy2(certifi.where(), _cert_dst)
os.environ["SSL_CERT_FILE"] = _cert_dst
os.environ["REQUESTS_CA_BUNDLE"] = _cert_dst
os.environ["CURL_CA_BUNDLE"] = _cert_dst

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    STOCK_UNIVERSE, BENCHMARKS, DART_API_KEY, ECOS_API_KEY,
    get_all_stocks, BATCH_CONFIG
)

logger = logging.getLogger(__name__)


class StockCollector:
    """주가 데이터 수집 (네이버 금융 우선, yfinance 폴백)"""

    NAVER_HEADERS = {"User-Agent": "Mozilla/5.0 (AITrading/3.0)"}

    def __init__(self):
        self.all_stocks = get_all_stocks()

    def fetch_prices(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """단일 종목 주가 수집 — 네이버 금융 API 우선"""
        # 종목코드 추출 (005930.KS → 005930)
        code = ticker.replace(".KS", "").replace(".KQ", "")

        # 1차: 네이버 금융
        df = self._fetch_naver(code, start, end)
        if not df.empty:
            return df

        # 2차: yfinance 폴백
        return self._fetch_yfinance(ticker, start, end)

    def _fetch_naver(self, code: str, start: str, end: str) -> pd.DataFrame:
        """네이버 금융 차트 API (API키 불필요)"""
        try:
            # 일봉 데이터 (최대 500건)
            url = f"https://fchart.stock.naver.com/siseJson.nhn"
            start_dt = start.replace("-", "")
            end_dt = end.replace("-", "")

            # 필요 일수 계산
            from datetime import datetime as dt
            d1 = dt.strptime(start, "%Y-%m-%d")
            d2 = dt.strptime(end, "%Y-%m-%d")
            count = max(10, (d2 - d1).days + 5)

            params = {
                "symbol": code, "requestType": 1,
                "startTime": start_dt, "endTime": end_dt,
                "timeframe": "day", "count": count,
            }
            r = requests.get(url, params=params, headers=self.NAVER_HEADERS, timeout=10)
            text = r.text.strip()

            # JSON-like 응답 파싱
            import json
            text = text.replace("'", '"')
            rows = json.loads(text)

            if not rows or len(rows) < 2:
                return pd.DataFrame()

            # 첫 행은 헤더: ["날짜", "시가", "고가", "저가", "종가", "거래량"]
            data = []
            for row in rows[1:]:
                if len(row) < 6:
                    continue
                date_str = str(row[0]).strip().replace('"', '')
                if len(date_str) < 8:
                    continue
                try:
                    trade_date = pd.to_datetime(date_str).date()
                    data.append({
                        "trade_date": trade_date,
                        "open": float(row[1]),
                        "high": float(row[2]),
                        "low": float(row[3]),
                        "close": float(row[4]),
                        "volume": int(row[5]),
                        "adj_close": float(row[4]),
                    })
                except (ValueError, TypeError):
                    continue

            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)
            # start~end 범위 필터
            s = pd.to_datetime(start).date()
            e = pd.to_datetime(end).date()
            df = df[(df.trade_date >= s) & (df.trade_date <= e)]
            return df

        except Exception as e:
            logger.debug(f"  Naver 수집 실패 ({code}): {e}")
            return pd.DataFrame()

    def _fetch_yfinance(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """yfinance 폴백"""
        try:
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=True, progress=False)
            if df.empty:
                return pd.DataFrame()

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume"
            })
            df["adj_close"] = df["close"]
            df["trade_date"] = pd.to_datetime(df.index).date
            return df[["trade_date", "open", "high", "low", "close", "volume", "adj_close"]]
        except Exception as e:
            logger.warning(f"{ticker} yfinance 폴백 실패: {e}")
            return pd.DataFrame()

    def fetch_all_prices(self, years: int = 3) -> pd.DataFrame:
        """전체 58개 종목 주가 수집"""
        end = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - relativedelta(years=years)).strftime("%Y-%m-%d")

        all_records = []
        for layer_key, layer_data in STOCK_UNIVERSE.items():
            for stock in layer_data["stocks"]:
                ticker = stock["ticker"]
                name   = stock["name"]
                logger.info(f"  수집 중: {name} ({ticker})")
                df = self.fetch_prices(ticker, start, end)
                if not df.empty:
                    df["ticker"] = ticker
                    df["name"]   = name
                    df["layer"]  = layer_key
                    all_records.append(df)
                time.sleep(0.3)  # API 부하 방지

        if not all_records:
            return pd.DataFrame()
        return pd.concat(all_records, ignore_index=True)

    def fetch_single_stock_prices(self, ticker: str, name: str,
                                   layer: str, years: int = 3) -> pd.DataFrame:
        end = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - relativedelta(years=years)).strftime("%Y-%m-%d")
        df = self.fetch_prices(ticker, start, end)
        if not df.empty:
            df["ticker"] = ticker
            df["name"]   = name
            df["layer"]  = layer
        return df


class MacroCollector:
    """거시경제 지표 수집 (네이버 금융 우선, yfinance 폴백)"""

    NAVER_HEADERS = {"User-Agent": "Mozilla/5.0 (AITrading/3.0)"}

    def fetch_macro(self, years: int = 3) -> pd.DataFrame:
        end = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - relativedelta(years=years)).strftime("%Y-%m-%d")

        # 네이버 금융으로 최신 매크로 수집
        df = self._fetch_naver_macro(start, end)
        if not df.empty:
            return df

        # yfinance 폴백
        return self._fetch_yfinance_macro(start, end)

    def _fetch_naver_macro(self, start: str, end: str) -> pd.DataFrame:
        """네이버(국내) + Google Finance(해외) 매크로 수집"""
        import re
        today = date.today()
        row = {"data_date": today}

        # ── 국내 지수 (네이버 금융) ──────────────────
        for code, col in [("KOSPI", "kospi"), ("KOSDAQ", "kosdaq")]:
            try:
                r = requests.get(f"https://m.stock.naver.com/api/index/{code}/basic",
                                 headers=self.NAVER_HEADERS, timeout=8)
                if r.status_code == 200:
                    d = r.json()
                    row[col] = float(str(d.get("closePrice", "0")).replace(",", ""))
            except Exception:
                pass

        # ── 해외 지수 + 환율 (yfinance) ────────
        yf_map = {
            "^GSPC": "sp500",
            "^IXIC": "nasdaq",
            "^SOX": "sox",
            "KRW=X": "usd_krw",
        }
        for yf_sym, col in yf_map.items():
            try:
                hist = yf.Ticker(yf_sym).history(period="5d")
                if not hist.empty:
                    row[col] = float(hist["Close"].iloc[-1])
            except Exception:
                pass

        # ── VIX + US 10Y (FRED API) ──────────────────
        fred_key = os.getenv("FRED_API_KEY", "")
        if fred_key:
            for series_id, col in [("VIXCLS", "vix"), ("DGS10", "us_10y")]:
                try:
                    r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                                     params={"series_id": series_id, "api_key": fred_key,
                                             "file_type": "json", "sort_order": "desc",
                                             "limit": 1}, timeout=8)
                    if r.status_code == 200:
                        obs = r.json().get("observations", [])
                        if obs and obs[0].get("value") != ".":
                            row[col] = float(obs[0]["value"])
                            logger.info(f"  [FRED] {col}: {row[col]}")
                except Exception:
                    pass

        # ── Finnhub: 미국 시장 심리 (선택) ──────────
        finnhub_key = os.getenv("FINNHUB_KEY", "")
        if finnhub_key:
            try:
                r = requests.get("https://finnhub.io/api/v1/news",
                                 params={"category": "general", "token": finnhub_key},
                                 timeout=8)
                if r.status_code == 200:
                    news = r.json()
                    row["finnhub_news_count"] = len(news)
                    logger.info(f"  [Finnhub] Market news: {len(news)} articles")
            except Exception:
                pass

        # 기본값 보장
        for col in ["vix", "us_10y", "kospi", "kosdaq", "usd_krw", "sp500", "nasdaq", "sox"]:
            row.setdefault(col, np.nan)

        # 최소 1개 지수 있으면 유효
        if pd.notna(row.get("kospi")) or pd.notna(row.get("nasdaq")):
            df = pd.DataFrame([row])
            logger.info(f"  [Macro] KOSPI: {row.get('kospi')}, NASDAQ: {row.get('nasdaq')}, "
                         f"SOX: {row.get('sox')}, S&P500: {row.get('sp500')}, "
                         f"USD/KRW: {row.get('usd_krw')}")
            return df

        return pd.DataFrame()

    def _fetch_yfinance_macro(self, start: str, end: str) -> pd.DataFrame:
        """yfinance 폴백"""
        records = {}
        tickers_map = {"KRW=X": "usd_krw", "^TNX": "us_10y",
                       "^VIX": "vix", "^KS11": "kospi", "^KQ11": "kosdaq"}
        for yf_ticker, col_name in tickers_map.items():
            try:
                df = yf.download(yf_ticker, start=start, end=end,
                                 auto_adjust=True, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if not df.empty:
                    for idx, row in df.iterrows():
                        d = str(idx.date())
                        records.setdefault(d, {})[col_name] = float(row.get("Close", np.nan))
            except Exception:
                pass

        if not records:
            return pd.DataFrame()

        rows = [{"data_date": d, **vals} for d, vals in records.items()]
        df = pd.DataFrame(rows)
        df["data_date"] = pd.to_datetime(df["data_date"]).dt.date
        for col in ["usd_krw", "us_10y", "vix", "kospi", "kosdaq"]:
            if col not in df.columns:
                df[col] = np.nan
        return df.dropna(subset=["data_date"])

    def fetch_macro_range(self, start: str, end: str) -> pd.DataFrame:
        """지정 기간 거시경제 수집 (증분용) — 네이버 우선"""
        df = self._fetch_naver_macro(start, end)
        if not df.empty:
            return df
        return self._fetch_yfinance_macro(start, end)


class NaverFinanceCollector:
    """네이버 금융 크롤링 — PER/PBR/EPS + 수급 데이터"""

    HEADERS = {"User-Agent": "Mozilla/5.0"}

    def get_financial_metrics(self, ticker: str) -> dict:
        """
        네이버 금융에서 PER/PBR/EPS/배당수익률 크롤링
        + yfinance info에서 ROE/영업이익률/매출 등 보완
        """
        import re
        from bs4 import BeautifulSoup

        stock_code = ticker.replace(".KS", "").replace(".KQ", "")
        result = {
            "ticker": ticker,
            "period": datetime.today().strftime("%YQ") + str((datetime.today().month-1)//3+1),
        }

        # 1) 네이버 금융 — PER, PBR, EPS, 배당
        try:
            url = f"https://finance.naver.com/item/main.naver?code={stock_code}"
            resp = requests.get(url, headers=self.HEADERS, timeout=10)
            soup = BeautifulSoup(resp.text, "lxml")

            tag_map = {"_per": "per", "_pbr": "pbr", "_eps": "eps", "_dvr": "dividend_yield"}
            for tag_id, key in tag_map.items():
                tag = soup.find("em", id=tag_id)
                if tag:
                    val_str = tag.text.strip().replace(",", "")
                    try:
                        result[key] = float(val_str)
                    except ValueError:
                        result[key] = np.nan
        except Exception as e:
            logger.warning(f"네이버 재무 크롤링 실패 ({ticker}): {e}")

        # 2) 추가 지표 — yfinance 없이 기본값 설정 (네이버에서 못 가져오는 항목)
        for key in ["roe", "op_margin", "revenue", "net_income",
                     "debt_ratio", "forward_per", "beta", "current_ratio", "free_cashflow"]:
            result.setdefault(key, np.nan)

        return result

    def fetch_supply_demand(self, ticker: str, pages: int = 1) -> pd.DataFrame:
        """
        네이버 금융에서 외국인/기관 매매 데이터 크롤링
        Returns: DataFrame [trade_date, foreign_net, institution_net, foreign_hold_pct]
        """
        import re
        from bs4 import BeautifulSoup

        stock_code = ticker.replace(".KS", "").replace(".KQ", "")
        records = []

        for page in range(1, pages + 1):
            try:
                url = f"https://finance.naver.com/item/frgn.naver?code={stock_code}&page={page}"
                resp = requests.get(url, headers=self.HEADERS, timeout=10)
                soup = BeautifulSoup(resp.text, "html.parser")

                for tr in soup.select("tr"):
                    tds = tr.select("td span.tah")
                    if len(tds) < 8:
                        continue
                    vals = [td.text.strip() for td in tds]
                    if not re.match(r"\d{4}\.\d{2}\.\d{2}", vals[0]):
                        continue

                    try:
                        trade_date = datetime.strptime(vals[0], "%Y.%m.%d").date()
                        close = int(vals[1].replace(",", ""))
                        foreign_net = int(vals[5].replace(",", "").replace("+", ""))
                        inst_net = int(vals[6].replace(",", "").replace("+", ""))
                        foreign_hold = int(vals[7].replace(",", ""))

                        records.append({
                            "ticker": ticker,
                            "trade_date": trade_date,
                            "close": close,
                            "foreign_net": foreign_net,
                            "institution_net": inst_net,
                            "foreign_hold": foreign_hold,
                        })
                    except (ValueError, IndexError):
                        continue
                time.sleep(0.3)
            except Exception as e:
                logger.warning(f"네이버 수급 크롤링 실패 ({ticker}, p{page}): {e}")

        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records)

    def fetch_all_supply_demand(self, pages: int = 1) -> pd.DataFrame:
        """전체 종목 수급 수집 (네이버 금융)"""
        all_stocks = get_all_stocks()
        all_dfs = []
        for stock in all_stocks:
            ticker = stock["ticker"]
            df = self.fetch_supply_demand(ticker, pages=pages)
            if not df.empty:
                all_dfs.append(df)
            time.sleep(0.3)

        if not all_dfs:
            return pd.DataFrame()

        result = pd.concat(all_dfs, ignore_index=True)
        # supply_demand 테이블 형식에 맞추기
        for col in ["foreign_buy", "foreign_sell", "institution_buy",
                     "institution_sell", "individual_net", "short_sell_vol",
                     "foreign_net_5d", "institution_net_5d"]:
            if col not in result.columns:
                result[col] = 0
        return result


class DARTCollector:
    """DART Open API 재무 데이터 수집 (API 키 필요)"""

    BASE_URL = "https://opendart.fss.or.kr/api"

    def __init__(self):
        self.api_key = DART_API_KEY

    def fetch_financial_summary(self, corp_code: str, year: int,
                                 reprt_code: str = "11011") -> dict:
        if not self.api_key:
            return {}
        url = f"{self.BASE_URL}/fnlttSinglAcntAll.json"
        params = {
            "crtfc_key": self.api_key,
            "corp_code": corp_code,
            "bsns_year": str(year),
            "reprt_code": reprt_code,
            "fs_div": "CFS",
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            data = resp.json()
            if data.get("status") == "000":
                return data.get("list", [])
            return {}
        except Exception as e:
            logger.warning(f"DART 재무 수집 실패: {e}")
            return {}

    def get_financial_metrics(self, ticker: str) -> dict:
        """NaverFinanceCollector로 위임"""
        return NaverFinanceCollector().get_financial_metrics(ticker)


class SupplyDemandCollector:
    """KRX 수급 데이터 수집 (requests 기반)"""

    KRX_URL = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"

    def fetch_supply_demand(self, stock_code: str, trade_date: str) -> dict:
        """
        KRX 투자자별 매매 현황 수집
        trade_date: YYYYMMDD 형식
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Referer": "http://data.krx.co.kr/",
            }
            params = {
                "bld": "dbms/MDC/STAT/standard/MDCSTAT02301",
                "isuCd": stock_code,
                "strtDd": trade_date,
                "endDd": trade_date,
                "share": "1",
                "money": "1",
                "csvxls_isNo": "false",
            }
            resp = requests.post(self.KRX_URL, data=params,
                                  headers=headers, timeout=15)
            data = resp.json()
            if not data.get("OutBlock_1"):
                return {}

            rows = data["OutBlock_1"]
            result = {}
            for row in rows:
                inv_type = row.get("INVST_TP_NM", "")
                if "외국인" in inv_type:
                    result["foreign_buy"]  = int(str(row.get("BUY_TRDVAL","0")).replace(",","") or 0)
                    result["foreign_sell"] = int(str(row.get("SLL_TRDVAL","0")).replace(",","") or 0)
                    result["foreign_net"]  = int(str(row.get("NETBUY_TRDVAL","0")).replace(",","") or 0)
                elif "기관" in inv_type:
                    result["institution_buy"]  = int(str(row.get("BUY_TRDVAL","0")).replace(",","") or 0)
                    result["institution_sell"] = int(str(row.get("SLL_TRDVAL","0")).replace(",","") or 0)
                    result["institution_net"]  = int(str(row.get("NETBUY_TRDVAL","0")).replace(",","") or 0)
                elif "개인" in inv_type:
                    result["individual_net"] = int(str(row.get("NETBUY_TRDVAL","0")).replace(",","") or 0)
            return result
        except Exception as e:
            logger.warning(f"KRX 수급 수집 실패 ({stock_code}): {e}")
            return {}

    def fetch_all_supply_demand(self, trade_date: str = None) -> pd.DataFrame:
        """전체 종목 수급 수집"""
        if trade_date is None:
            trade_date = datetime.today().strftime("%Y%m%d")

        all_stocks = get_all_stocks()
        records = []

        for stock in all_stocks:
            code = stock["code"]
            ticker = stock["ticker"]
            sd = self.fetch_supply_demand(code, trade_date)
            if sd:
                sd["ticker"] = ticker
                sd["trade_date"] = datetime.strptime(trade_date, "%Y%m%d").date()
                records.append(sd)
            time.sleep(0.5)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        # 5일 누적 수급은 별도 계산 (DB에서)
        for col in ["foreign_buy","foreign_sell","foreign_net",
                    "institution_buy","institution_sell","institution_net",
                    "individual_net","short_sell_vol","foreign_net_5d","institution_net_5d"]:
            if col not in df.columns:
                df[col] = 0
        return df
