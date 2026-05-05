"""
AI Trading System - 설정 파일
가치사슬별 58개 종목 유니버스 + 시스템 설정
"""

import os, sys
# 공유 설정 우선 로드
try:
    sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from shared_config import load_shared_env
    load_shared_env()
except ImportError:
    pass
from dotenv import load_dotenv
load_dotenv()  # 앱별 .env 오버라이드

# =============================================
# DATABASE 설정
# =============================================
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME", "ai_trading"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

DB_URL = os.getenv("DATABASE_URL",
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

# =============================================
# API 키
# =============================================
DART_API_KEY = os.getenv("DART_API_KEY", "")
ECOS_API_KEY = os.getenv("ECOS_API_KEY", "")

# =============================================
# 가치사슬별 종목 유니버스 (58개)
# yfinance 코드: KOSPI → {코드}.KS, KOSDAQ → {코드}.KQ
# =============================================

STOCK_UNIVERSE = {
    # ═══════════════════════════════════════════════
    # AI 반도체 (Upstream → Midstream → Downstream)
    # ═══════════════════════════════════════════════
    "L1_반도체소재": {
        "description": "웨이퍼·포토레지스트·케미컬·블랭크마스크",
        "weight": 0.08,
        "stocks": [
            {"name": "솔브레인",      "code": "357780", "market": "KQ", "ticker": "357780.KQ"},
            {"name": "동진쎄미켐",    "code": "005290", "market": "KQ", "ticker": "005290.KQ"},
            {"name": "에스앤에스텍",  "code": "101490", "market": "KQ", "ticker": "101490.KQ"},
            {"name": "원익머트리얼즈","code": "104830", "market": "KQ", "ticker": "104830.KQ"},
            {"name": "레이크머티리얼즈","code": "281740", "market": "KQ", "ticker": "281740.KQ"},
            {"name": "덕산네오룩스",  "code": "213420", "market": "KQ", "ticker": "213420.KQ"},
        ]
    },
    "L2_반도체장비": {
        "description": "식각·증착·검사·레이저·테스트 장비",
        "weight": 0.12,
        "stocks": [
            {"name": "한미반도체",    "code": "042700", "market": "KQ", "ticker": "042700.KQ"},
            {"name": "원익IPS",       "code": "240810", "market": "KQ", "ticker": "240810.KQ"},
            {"name": "주성엔지니어링","code": "036930", "market": "KQ", "ticker": "036930.KQ"},
            {"name": "이오테크닉스",  "code": "039030", "market": "KQ", "ticker": "039030.KQ"},
            {"name": "HPSP",          "code": "403870", "market": "KQ", "ticker": "403870.KQ"},
            {"name": "유진테크",      "code": "084370", "market": "KQ", "ticker": "084370.KQ"},
            {"name": "피에스케이홀딩스","code":"031980","market": "KQ", "ticker": "031980.KQ"},
            {"name": "테크윙",        "code": "089030", "market": "KQ", "ticker": "089030.KQ"},
            {"name": "파크시스템스",  "code": "140860", "market": "KQ", "ticker": "140860.KQ"},
            {"name": "오로스테크놀로지","code": "322310", "market": "KQ", "ticker": "322310.KQ"},
            {"name": "에이피티씨",    "code": "200160", "market": "KQ", "ticker": "200160.KQ"},
            {"name": "피에스케이",    "code": "319660", "market": "KQ", "ticker": "319660.KQ"},
            {"name": "티에스이",      "code": "131290", "market": "KQ", "ticker": "131290.KQ"},
            {"name": "티이엠씨씨엔에스","code": "241790", "market": "KQ", "ticker": "241790.KQ"},
            {"name": "프로텍",        "code": "053610", "market": "KQ", "ticker": "053610.KQ"},
            {"name": "케이씨텍",      "code": "281820", "market": "KQ", "ticker": "281820.KQ"},
            {"name": "디이엔티",      "code": "079810", "market": "KQ", "ticker": "079810.KQ"},
            {"name": "에스티아이",    "code": "039440", "market": "KQ", "ticker": "039440.KQ"},
        ]
    },
    "L3_반도체설계제조": {
        "description": "IDM·파운드리·팹리스·IP설계",
        "weight": 0.12,
        "stocks": [
            {"name": "삼성전자",      "code": "005930", "market": "KS", "ticker": "005930.KS"},
            {"name": "SK하이닉스",    "code": "000660", "market": "KS", "ticker": "000660.KS"},
            {"name": "DB하이텍",      "code": "000990", "market": "KS", "ticker": "000990.KS"},
            {"name": "오픈엣지테크놀로지","code": "394280", "market": "KQ", "ticker": "394280.KQ"},
            {"name": "칩스앤미디어",  "code": "094360", "market": "KQ", "ticker": "094360.KQ"},
            {"name": "LX세미콘",      "code": "108320", "market": "KS", "ticker": "108320.KS"},
            {"name": "가온칩스",      "code": "399720", "market": "KQ", "ticker": "399720.KQ"},
            {"name": "에이디테크놀로지","code": "200710", "market": "KQ", "ticker": "200710.KQ"},
        ]
    },
    "L4_반도체후공정기판": {
        "description": "OSAT·패키징·PCB·기판·부품",
        "weight": 0.10,
        "stocks": [
            {"name": "하나마이크론",  "code": "067310", "market": "KQ", "ticker": "067310.KQ"},
            {"name": "SFA반도체",     "code": "036540", "market": "KQ", "ticker": "036540.KQ"},
            {"name": "네패스",        "code": "033640", "market": "KQ", "ticker": "033640.KQ"},
            {"name": "엘비세미콘",    "code": "061970", "market": "KQ", "ticker": "061970.KQ"},
            {"name": "이수페타시스",  "code": "007660", "market": "KQ", "ticker": "007660.KQ"},
            {"name": "대덕전자",      "code": "353200", "market": "KS", "ticker": "353200.KS"},
            {"name": "삼성전기",      "code": "009150", "market": "KS", "ticker": "009150.KS"},
            {"name": "해성디에스",    "code": "195870", "market": "KQ", "ticker": "195870.KQ"},
            {"name": "심텍",          "code": "222800", "market": "KQ", "ticker": "222800.KQ"},
            {"name": "티엘비",        "code": "356860", "market": "KQ", "ticker": "356860.KQ"},
            {"name": "리노공업",      "code": "058470", "market": "KQ", "ticker": "058470.KQ"},
            {"name": "ISC",           "code": "095340", "market": "KQ", "ticker": "095340.KQ"},
            {"name": "원익QnC",       "code": "074600", "market": "KQ", "ticker": "074600.KQ"},
            {"name": "하나머티리얼즈","code": "166090", "market": "KQ", "ticker": "166090.KQ"},
            {"name": "두산테스나",    "code": "131970", "market": "KQ", "ticker": "131970.KQ"},
            {"name": "티씨케이",      "code": "064760", "market": "KQ", "ticker": "064760.KQ"},
            {"name": "RFHIC",         "code": "218410", "market": "KQ", "ticker": "218410.KQ"},
        ]
    },
    # ═══════════════════════════════════════════════
    # AI 소프트웨어 (Upstream → Midstream → Downstream)
    # ═══════════════════════════════════════════════
    "L5_AI인프라전력": {
        "description": "IDC·데이터센터·전력기기·네트워크",
        "weight": 0.12,
        "stocks": [
            {"name": "효성중공업",    "code": "298040", "market": "KS", "ticker": "298040.KS"},
            {"name": "HD현대일렉트릭","code": "267260", "market": "KS", "ticker": "267260.KS"},
            {"name": "LS ELECTRIC",   "code": "010120", "market": "KS", "ticker": "010120.KS"},
            {"name": "일진전기",      "code": "103590", "market": "KS", "ticker": "103590.KS"},
            {"name": "산일전기",      "code": "062040", "market": "KS", "ticker": "062040.KS"},
            {"name": "KT",            "code": "030200", "market": "KS", "ticker": "030200.KS"},
            {"name": "SK텔레콤",      "code": "017670", "market": "KS", "ticker": "017670.KS"},
            {"name": "LG유플러스",    "code": "032640", "market": "KS", "ticker": "032640.KS"},
            {"name": "지엔씨에너지",  "code": "119850", "market": "KQ", "ticker": "119850.KQ"},
            {"name": "모아데이타",    "code": "288980", "market": "KQ", "ticker": "288980.KQ"},
            {"name": "링네트",        "code": "042500", "market": "KQ", "ticker": "042500.KQ"},
            {"name": "코오롱인더",    "code": "120110", "market": "KS", "ticker": "120110.KS"},
        ]
    },
    "L6_AI플랫폼LLM": {
        "description": "LLM·AI플랫폼·클라우드SI·음성AI",
        "weight": 0.10,
        "stocks": [
            {"name": "네이버",        "code": "035420", "market": "KS", "ticker": "035420.KS"},
            {"name": "카카오",        "code": "035720", "market": "KS", "ticker": "035720.KS"},
            {"name": "삼성에스디에스","code": "018260", "market": "KS", "ticker": "018260.KS"},
            {"name": "LG CNS",        "code": "064400", "market": "KS", "ticker": "064400.KS"},
            {"name": "솔트룩스",      "code": "304100", "market": "KQ", "ticker": "304100.KQ"},
            {"name": "마음AI",        "code": "377480", "market": "KQ", "ticker": "377480.KQ"},
            {"name": "폴라리스AI",    "code": "039980", "market": "KQ", "ticker": "039980.KQ"},
            {"name": "셀바스AI",      "code": "108860", "market": "KQ", "ticker": "108860.KQ"},
            {"name": "크래프톤",      "code": "259960", "market": "KS", "ticker": "259960.KS"},
            {"name": "코난테크놀로지","code": "402030", "market": "KQ", "ticker": "402030.KQ"},
            {"name": "씨이랩",        "code": "189330", "market": "KQ", "ticker": "189330.KQ"},
            {"name": "폴라리스오피스","code": "041020", "market": "KQ", "ticker": "041020.KQ"},
            {"name": "플리토",        "code": "300080", "market": "KQ", "ticker": "300080.KQ"},
        ]
    },
    "L7_AI응용서비스": {
        "description": "의료AI·보안AI·금융AI·교육AI·콘텐츠AI",
        "weight": 0.08,
        "stocks": [
            {"name": "루닛",          "code": "328130", "market": "KQ", "ticker": "328130.KQ"},
            {"name": "뷰노",          "code": "338220", "market": "KQ", "ticker": "338220.KQ"},
            {"name": "딥노이드",      "code": "315640", "market": "KQ", "ticker": "315640.KQ"},
            {"name": "제이엘케이",    "code": "322510", "market": "KQ", "ticker": "322510.KQ"},
            {"name": "이글루코퍼레이션","code": "067920", "market": "KQ", "ticker": "067920.KQ"},
            {"name": "안랩",          "code": "053800", "market": "KQ", "ticker": "053800.KQ"},
            {"name": "쿠콘",          "code": "294570", "market": "KQ", "ticker": "294570.KQ"},
            {"name": "메가스터디교육","code": "215200", "market": "KQ", "ticker": "215200.KQ"},
            {"name": "와이즈넛",      "code": "460870", "market": "KQ", "ticker": "460870.KQ"},
            {"name": "클래시스",      "code": "214150", "market": "KQ", "ticker": "214150.KQ"},
            {"name": "카카오페이",    "code": "377300", "market": "KS", "ticker": "377300.KS"},
        ]
    },
    # ═══════════════════════════════════════════════
    # Physical AI (센서·구동 → 로봇 → 자율주행·국방)
    # ═══════════════════════════════════════════════
    "L8_피지컬AI센서구동": {
        "description": "라이다·IMU·서보모터·감속기·액추에이터",
        "weight": 0.08,
        "stocks": [
            {"name": "에스오에스랩",  "code": "464080", "market": "KQ", "ticker": "464080.KQ"},
            {"name": "나무가",        "code": "190510", "market": "KQ", "ticker": "190510.KQ"},
            {"name": "아이쓰리시스템","code": "214430", "market": "KQ", "ticker": "214430.KQ"},
            {"name": "삼현",          "code": "437730", "market": "KQ", "ticker": "437730.KQ"},
            {"name": "에스피지",      "code": "058610", "market": "KQ", "ticker": "058610.KQ"},
            {"name": "에스비비테크",  "code": "389500", "market": "KQ", "ticker": "389500.KQ"},
            {"name": "로보티즈",      "code": "108490", "market": "KQ", "ticker": "108490.KQ"},
            {"name": "삼익THK",       "code": "004380", "market": "KQ", "ticker": "004380.KQ"},
            {"name": "하이젠알앤엠",  "code": "288490", "market": "KQ", "ticker": "288490.KQ"},
        ]
    },
    "L9_로봇제조": {
        "description": "협동로봇·산업로봇·서비스로봇·물류로봇",
        "weight": 0.10,
        "stocks": [
            {"name": "두산로보틱스",  "code": "454910", "market": "KS", "ticker": "454910.KS"},
            {"name": "레인보우로보틱스","code":"277810","market": "KQ", "ticker": "277810.KQ"},
            {"name": "뉴로메카",      "code": "348340", "market": "KQ", "ticker": "348340.KQ"},
            {"name": "로보스타",      "code": "090360", "market": "KQ", "ticker": "090360.KQ"},
            {"name": "유일로보틱스",  "code": "388720", "market": "KQ", "ticker": "388720.KQ"},
            {"name": "싸이맥스",      "code": "160980", "market": "KQ", "ticker": "160980.KQ"},
            {"name": "티로보틱스",    "code": "117730", "market": "KQ", "ticker": "117730.KQ"},
            {"name": "유진로봇",      "code": "056080", "market": "KQ", "ticker": "056080.KQ"},
            {"name": "큐렉소",        "code": "060280", "market": "KQ", "ticker": "060280.KQ"},
            {"name": "에브리봇",      "code": "270660", "market": "KQ", "ticker": "270660.KQ"},
            {"name": "휴림로봇",      "code": "090710", "market": "KQ", "ticker": "090710.KQ"},
        ]
    },
    "L10_자율주행국방": {
        "description": "자율주행·모빌리티·드론·국방무인체계",
        "weight": 0.10,
        "stocks": [
            {"name": "현대차",        "code": "005380", "market": "KS", "ticker": "005380.KS"},
            {"name": "현대모비스",    "code": "012330", "market": "KS", "ticker": "012330.KS"},
            {"name": "HD현대",        "code": "267250", "market": "KS", "ticker": "267250.KS"},
            {"name": "쏘카",          "code": "403550", "market": "KS", "ticker": "403550.KS"},
            {"name": "한화에어로스페이스","code": "012450", "market": "KS", "ticker": "012450.KS"},
            {"name": "한국항공우주",  "code": "047810", "market": "KS", "ticker": "047810.KS"},
            {"name": "인텔리안테크",  "code": "189300", "market": "KQ", "ticker": "189300.KQ"},
            {"name": "베셀",          "code": "177350", "market": "KQ", "ticker": "177350.KQ"},
        ]
    },
}

# 벤치마크 인덱스
BENCHMARKS = {
    "KOSPI":  "^KS11",
    "KOSDAQ": "^KQ11",
    "VIX":    "^VIX",
    "US10Y":  "^TNX",
    "USD_KRW":"KRW=X",
}

# 편의 함수: 전체 종목 리스트
def get_all_stocks():
    stocks = []
    for layer_key, layer_data in STOCK_UNIVERSE.items():
        for s in layer_data["stocks"]:
            stock = dict(s)
            stock["layer"] = layer_key
            stock["layer_desc"] = layer_data["description"]
            stocks.append(stock)
    return stocks

def get_all_tickers():
    return [s["ticker"] for s in get_all_stocks()]

def get_ticker_map():
    """ticker → name 매핑"""
    return {s["ticker"]: s["name"] for s in get_all_stocks()}

def get_layer_color():
    return {
        # 현재 레이어
        "L1_반도체소재":        "#1f77b4",
        "L2_반도체장비":        "#ff7f0e",
        "L3_반도체설계제조":    "#2ca02c",
        "L4_반도체후공정기판":  "#d62728",
        "L5_AI인프라전력":      "#9467bd",
        "L6_AI플랫폼LLM":      "#8c564b",
        "L7_AI응용서비스":      "#e377c2",
        "L8_피지컬AI센서구동":  "#7f7f7f",
        "L9_로봇제조":          "#bcbd22",
        "L10_자율주행국방":     "#17becf",
        # 이전 레이어 (호환용)
        "L1_메모리반도체":      "#1f77b4",
        "L3_반도체소재부품":    "#2ca02c",
        "L4_AI인프라":          "#9467bd",
        "L5_AI플랫폼소프트웨어":"#8c564b",
        "L6_AI응용로봇의료":    "#e377c2",
        "L7_시스템반도체팹리스":"#7f7f7f",
    }

# =============================================
# 모델 설정
# =============================================
MODEL_CONFIG = {
    "lookback_days": 20,          # LSTM 시퀀스 길이
    "train_years": 2,             # 훈련 데이터 기간
    "backtest_years": 1,          # 백테스트 기간
    "prediction_horizons": [1, 5, 20],  # 예측 기간 (거래일)
    "signal_threshold_up": 0.60,  # 상승 확률 임계값
    "signal_threshold_down": 0.55,# 하락 확률 임계값
    "label_up_pct": 0.02,         # 상승 레이블 기준 (+2%)
    "label_down_pct": -0.02,      # 하락 레이블 기준 (-2%)
}

# =============================================
# 배치 설정
# =============================================
BATCH_CONFIG = {
    "hour": int(os.getenv("BATCH_HOUR", 7)),
    "minute": int(os.getenv("BATCH_MINUTE", 0)),
    "history_years": 3,           # 과거 데이터 수집 기간
}

# =============================================
# 매매 규칙
# =============================================
TRADING_RULES = {
    "take_profit_1": 0.03,   # 1차 익절 (+3%)
    "take_profit_2": 0.05,   # 2차 전량 익절 (+5%)
    "stop_loss": -0.02,      # 손절 (-2%)
    "max_position_pct": 0.20, # 단일 종목 최대 비중 20%
    "initial_capital": 10_000_000,  # 시뮬레이션 초기 자본 (1천만원)
}

# 로그 설정
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = "logs"
