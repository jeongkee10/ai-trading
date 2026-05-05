# 📈 J.Insight AI Trading System

코스피·코스닥 AI 가치사슬 58개 종목 데이 트레이딩 시뮬레이션 시스템

## 📁 프로젝트 구조

```
ai_trading/
├── config/
│   └── settings.py          # 58개 종목 유니버스 + 시스템 설정
├── data/
│   ├── collector.py         # yfinance, DART, ECOS, KRX 수집
│   └── preprocessor.py      # 기술지표 + 피처 엔지니어링
├── models/
│   ├── xgboost_model.py     # XGBoost 분류 모델
│   ├── lstm_model.py        # LSTM 시계열 모델
│   ├── ensemble.py          # 앙상블 + 시그널 생성
│   └── saved/               # 학습된 모델 저장 (자동 생성)
├── backtest/
│   └── backtester.py        # Walk-Forward 백테스트
├── database/
│   └── db_manager.py        # PostgreSQL CRUD
├── batch/
│   └── daily_batch.py       # APScheduler 일배치
├── app/
│   └── streamlit_app.py     # Streamlit UI
├── logs/                    # 배치 로그 (자동 생성)
├── run.py                   # 통합 실행 스크립트
├── requirements.txt
└── .env.example
```

## ⚙️ 설치 순서

### 1. 사전 준비
```bash
# Python 3.10+ 필요
# PostgreSQL 설치 및 실행 (포트 5432 기본값)
# VS Code에서 프로젝트 폴더 열기
```

### 2. 가상환경 설정
```bash
cd ai_trading
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. 패키지 설치
```bash
pip install -r requirements.txt
```

### 4. 환경변수 설정
```bash
cp .env.example .env
# .env 파일 편집:
# DB_PASSWORD=your_postgres_password
# DART_API_KEY=your_dart_key     # https://opendart.fss.or.kr (무료)
# ECOS_API_KEY=your_ecos_key     # https://ecos.bok.or.kr (무료)
```

### 5. PostgreSQL DB 생성
```sql
-- psql에서 실행
CREATE DATABASE ai_trading;
```

### 6. 초기 설정 (최초 1회 — 약 20~40분 소요)
```bash
python run.py init
```
이 명령어가 수행하는 작업:
- DB 테이블 전체 생성
- 3년치 과거 주가 수집 (58개 종목)
- 거시경제 데이터 수집
- 기술적 지표 계산
- XGBoost + LSTM 모델 학습
- 초기 예측 실행

### 7. Streamlit UI 실행
```bash
python run.py app
# 브라우저에서 http://localhost:8501 접속
```

## 🔄 일상 사용법

```bash
# UI 실행
python run.py app

# 배치 수동 실행 (UI에서도 가능)
python run.py batch

# 스케줄러 시작 (매일 오전 7시 자동 배치)
python run.py schedule

# 백테스트 재실행
python run.py backtest
```

## 📊 7개 가치사슬 레이어 (58개 종목)

| 레이어 | 설명 | 종목 수 |
|--------|------|---------|
| L1 메모리반도체 | 삼성전자, SK하이닉스 | 2 |
| L2 반도체장비 | 한미반도체, 원익IPS, 주성엔지니어링 외 | 12 |
| L3 반도체소재부품 | 리노공업, HPSP, 솔브레인, 삼성전기 외 | 12 |
| L4 AI인프라 | 효성중공업, HD현대일렉트릭, LS ELECTRIC 외 | 12 |
| L5 AI플랫폼 | 네이버, 카카오, 솔트룩스, 마음AI 외 | 10 |
| L6 AI응용로봇 | 두산로보틱스, 레인보우로보틱스, 루닛 외 | 8 |
| L7 시스템반도체 | 가온칩스, 오픈엣지테크놀로지 | 2 |

## 🤖 예측 모델

- **XGBoost**: 31개 기술적 피처로 UP/HOLD/DOWN 3분류
- **LSTM**: 20일 시계열 시퀀스로 방향 예측
- **앙상블**: XGB(60%) + LSTM(40%) 가중 평균
- **예측 기간**: T+1(내일), T+5(1주일), T+20(1개월)
- **백테스트**: Walk-Forward 1년 구간

## 📈 주요 지표

- **이동평균**: SMA 5/20/60/120, EMA 5/20
- **MACD** (12/26/9)
- **RSI** (14)
- **볼린저밴드** (20, 2σ)
- **스토캐스틱** (14, 3)
- **OBV**, **거래량비율**, **ATR**

## ⚠️ 주의사항

- 이 시스템은 **투자 참고용**입니다. 실제 투자 판단은 본인 책임입니다.
- 모든 API는 **무료** (yfinance, DART, ECOS, KRX)
- TensorFlow 미설치 시 LSTM 비활성화, XGBoost만 동작
- 초기 설정은 네트워크 속도에 따라 20~60분 소요될 수 있습니다
