"""
AI Trading System - 네이버 금융 뉴스 수집 + 센티멘트 분석
API 키 불필요 (네이버 모바일 금융 API)
"""

import logging
import time
import requests
import pandas as pd
import numpy as np
from datetime import date

logger = logging.getLogger(__name__)

POSITIVE_KEYWORDS = [
    "상향", "목표가", "호실적", "사상최대", "수주", "흑자", "전환", "상승",
    "매수", "비중확대", "신고가", "성장", "확대", "돌파", "급등", "호재",
    "수혜", "계약", "투자", "증가", "개선", "회복", "강세", "추천",
    "아웃퍼폼", "오버웨이트", "탑픽", "최선호", "기대", "긍정",
]

NEGATIVE_KEYWORDS = [
    "하향", "적자", "손실", "하락", "급락", "매도", "리스크", "우려",
    "감소", "축소", "부진", "악화", "위기", "소송", "제재", "벌금",
    "언더퍼폼", "비중축소", "과열", "거품", "경고", "조정", "약세",
    "실적부진", "하회", "감익", "적자전환", "공매도", "불확실",
]


class NewsCollector:
    """네이버 금융 뉴스 크롤링 + 센티멘트 분석"""

    HEADERS = {"User-Agent": "Mozilla/5.0 (AITrading/3.0)"}

    def fetch_stock_news(self, code: str, page_size: int = 20) -> list:
        """종목 관련 뉴스 제목 수집 (네이버 모바일 금융 API)"""
        articles = []
        try:
            url = f"https://m.stock.naver.com/api/news/stock/{code}?pageSize={page_size}"
            r = requests.get(url, headers=self.HEADERS, timeout=8)
            if r.status_code != 200 or not r.text:
                return []
            data = r.json()
            if isinstance(data, list):
                for page_data in data:
                    for item in page_data.get("items", []):
                        articles.append({
                            "title": item.get("title", ""),
                            "date": item.get("datetime", ""),
                        })
        except Exception as e:
            logger.debug(f"News fetch failed {code}: {e}")
        return articles

    def analyze_sentiment(self, articles: list) -> dict:
        """뉴스 제목 기반 센티멘트 점수 계산"""
        if not articles:
            return {
                "news_count": 0, "positive_count": 0, "negative_count": 0,
                "positive_ratio": 0.0, "negative_ratio": 0.0,
                "sentiment_score": 0.0, "news_momentum": 0.0,
            }

        pos_count = 0
        neg_count = 0

        for article in articles:
            title = article.get("title", "")
            pos_hit = sum(1 for kw in POSITIVE_KEYWORDS if kw in title)
            neg_hit = sum(1 for kw in NEGATIVE_KEYWORDS if kw in title)
            if pos_hit > neg_hit:
                pos_count += 1
            elif neg_hit > pos_hit:
                neg_count += 1

        total = len(articles)
        pos_ratio = pos_count / total if total > 0 else 0
        neg_ratio = neg_count / total if total > 0 else 0
        sentiment_score = pos_ratio - neg_ratio

        return {
            "news_count": total,
            "positive_count": pos_count,
            "negative_count": neg_count,
            "positive_ratio": round(pos_ratio, 4),
            "negative_ratio": round(neg_ratio, 4),
            "sentiment_score": round(sentiment_score, 4),
            "news_momentum": round(total / 20.0, 4),
        }

    def get_sentiment_features(self, ticker: str, time_window: str = "all") -> dict:
        """
        종목 센티멘트 피처 추출
        time_window:
          "all" - 전체 최근 뉴스
          "morning" - 전일 15:30 ~ 당일 07:30 (모델 A용)
          "afternoon" - 당일 15:30 ~ 17:00 (모델 B용)
        """
        code = ticker.replace(".KS", "").replace(".KQ", "")
        articles = self.fetch_stock_news(code)

        # 시간대 필터링
        if time_window != "all" and articles:
            from datetime import datetime
            now = datetime.now()
            filtered = []
            for a in articles:
                try:
                    # datetime format: "202605051510"
                    dt_str = a.get("date", "")
                    if len(dt_str) >= 12:
                        dt = datetime.strptime(dt_str[:12], "%Y%m%d%H%M")
                        if time_window == "morning":
                            # 전일 15:30 ~ 당일 07:30
                            yesterday_1530 = now.replace(hour=15, minute=30, second=0) - pd.Timedelta(days=1)
                            today_0730 = now.replace(hour=7, minute=30, second=0)
                            if yesterday_1530 <= dt <= today_0730:
                                filtered.append(a)
                        elif time_window == "afternoon":
                            # 당일 15:30 ~ 17:00
                            today_1530 = now.replace(hour=15, minute=30, second=0)
                            today_1700 = now.replace(hour=17, minute=0, second=0)
                            if today_1530 <= dt <= today_1700:
                                filtered.append(a)
                except Exception:
                    pass
            if filtered:
                articles = filtered

        features = self.analyze_sentiment(articles)
        features["ticker"] = ticker
        features["time_window"] = time_window
        return features

    def get_all_sentiments(self, tickers: list) -> pd.DataFrame:
        """전체 종목 센티멘트 수집"""
        results = []
        for ticker in tickers:
            try:
                features = self.get_sentiment_features(ticker)
                features["collected_date"] = date.today().isoformat()
                results.append(features)
            except Exception as e:
                logger.warning(f"  {ticker} sentiment failed: {e}")
            time.sleep(0.3)
        return pd.DataFrame(results) if results else pd.DataFrame()
