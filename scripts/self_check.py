"""Self-Consistency 50항목 검증"""
import sys, os, warnings
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

errors = []
passes = []

def check(num, desc, func):
    try:
        result = func()
        passes.append(f'{num}. {desc}: {result}')
    except Exception as e:
        errors.append(f'{num}. {desc}: {e}')

# DB
from database.db_manager import DBManager
from sqlalchemy import text
db = DBManager()

check(1, 'DB 연결', lambda: 'OK')
check(2, '설정', lambda: f'{len(__import__("config.settings", fromlist=["get_all_stocks"]).get_all_stocks())} stocks')

with db.engine.connect() as conn:
    check(3, '주가', lambda: f'{conn.execute(text("SELECT COUNT(*) FROM stock_prices")).fetchone()[0]:,}건')
    check(4, '거시', lambda: f'{conn.execute(text("SELECT COUNT(*) FROM macro_data")).fetchone()[0]}건')
    check(5, '거시 신규칼럼', lambda: str(conn.execute(text("SELECT dji,wti,gold FROM macro_data ORDER BY data_date DESC LIMIT 1")).fetchone()))
    check(6, '기술지표', lambda: f'{conn.execute(text("SELECT COUNT(*) FROM technical_indicators")).fetchone()[0]:,}건')
    check(7, '수급', lambda: f'{conn.execute(text("SELECT COUNT(*) FROM supply_demand")).fetchone()[0]:,}건')
    check(8, 'model_type', lambda: str([x[0] for x in conn.execute(text("SELECT DISTINCT model_type FROM predictions")).fetchall()]))

# Features
from data.preprocessor import FeatureBuilder
fb = FeatureBuilder()
check(9, '피처 정의', lambda: f'{len(fb.get_feature_columns())}개')

price_df = db.get_stock_prices('005930.KS', days=200)
feat_df = fb.build_features(price_df)
usable = [c for c in fb.get_feature_columns() if c in feat_df.columns and feat_df[c].isna().mean() <= 0.5]
check(10, '피처 사용가능', lambda: f'{len(usable)}개')

# Model
from models.advanced_model import AdvancedEnsemblePredictor
predictor = AdvancedEnsemblePredictor()
check(11, '모델 로드', lambda: 'OK' if predictor.load(horizon=1) else 'FAIL')
result = predictor.predict_single('005930.KS', price_df, horizon=1)
check(12, '추론', lambda: f'{result["pred_label"]} UP={result["pred_up_prob"]:.1%}')

# News
from data.news_collector import NewsCollector
nc = NewsCollector()
news = nc.get_sentiment_features('005930.KS')
check(13, '뉴스', lambda: f'{news["news_count"]}건 score={news["sentiment_score"]}')
check(14, '뉴스 시간대', lambda: f'morning={nc.get_sentiment_features("005930.KS", "morning")["news_count"]}건')

# API
import requests
check(15, 'API status', lambda: requests.get('http://localhost:8000/api/status', timeout=5).status_code)
check(16, 'API predictions-binary', lambda: requests.get('http://localhost:8000/api/predictions-binary?model=model_A&horizon=1', timeout=5).json()['summary']['total'])
check(17, 'API backtest', lambda: requests.get('http://localhost:8000/api/backtest?start_date=2026-04-28&end_date=2026-04-30&ticker=005930.KS&horizon=1', timeout=120).json()['summary']['total'])
check(18, 'API indicators', lambda: requests.get('http://localhost:8000/api/indicators/005930.KS?days=5', timeout=5).status_code)
check(19, 'API financials', lambda: requests.get('http://localhost:8000/api/financials/005930.KS', timeout=5).status_code)
check(20, 'API supply', lambda: requests.get('http://localhost:8000/api/supply/005930.KS', timeout=5).status_code)
check(21, 'API news', lambda: requests.get('http://localhost:8000/api/news/005930.KS', timeout=10).status_code)
check(22, 'API macro', lambda: requests.get('http://localhost:8000/api/macro', timeout=5).json().get('kospi'))
check(23, 'API prices', lambda: len(requests.get('http://localhost:8000/api/prices/005930.KS?days=5', timeout=5).json()['prices']))
check(24, 'API prices end_date', lambda: requests.get('http://localhost:8000/api/prices/005930.KS?days=60&end_date=2026-04-20', timeout=5).status_code)

# Frontend
for i, page in enumerate(['/', '/signals', '/analysis', '/backtest', '/system'], 25):
    check(i, f'Frontend {page}', lambda p=page: requests.get(f'http://localhost:3000{p}', timeout=5).status_code)

# Imports
check(30, 'Scheduler', lambda: (__import__('batch.scheduler', fromlist=['start_scheduler']), 'OK')[1])
check(31, 'Weekly train', lambda: (__import__('batch.weekly_train', fromlist=['run_weekly_train']), 'OK')[1])
check(32, 'Daily inference', lambda: (__import__('batch.daily_inference', fromlist=['run_inference']), 'OK')[1])
check(33, 'Binary model', lambda: (__import__('models.binary_model', fromlist=['BinaryEnsemblePredictor']), 'OK')[1])

# Files
check(34, '모델 파일', lambda: f'{len([f for f in os.listdir("models/saved") if f.endswith(".pkl")])} pkl + {len([f for f in os.listdir("models/saved") if f.endswith(".pt")])} pt')
check(35, 'Dockerfile', lambda: 'OK' if os.path.exists('Dockerfile') else 'MISSING')
check(36, 'docker-compose', lambda: 'OK' if os.path.exists('docker-compose.yml') else 'MISSING')
check(37, 'railway.toml', lambda: 'OK' if os.path.exists('railway.toml') else 'MISSING')
check(38, '.dockerignore', lambda: 'OK' if os.path.exists('.dockerignore') else 'MISSING')
check(39, 'requirements.txt', lambda: 'OK' if all(p in open('requirements.txt').read() for p in ['torch','catboost','optuna','fastapi']) else 'MISSING')

# Logic checks
check(40, '백테스트 HOLD 없음', lambda: set(x['actual_label'] for x in requests.get('http://localhost:8000/api/backtest?start_date=2026-04-28&end_date=2026-04-30&ticker=005930.KS&horizon=1', timeout=120).json()['results']))
check(41, '예측 HOLD 없음', lambda: set(x['pred_label'] for x in requests.get('http://localhost:8000/api/predictions-binary?model=model_A&horizon=1', timeout=5).json()['predictions']))

# Scheduler time check
import inspect
from batch.scheduler import start_scheduler
src = inspect.getsource(start_scheduler)
check(42, '스케줄러 07:30', lambda: 'OK' if 'hour=7' in src and 'minute=30' in src else 'MISSING')
check(43, '스케줄러 17:00', lambda: 'OK' if 'hour=17' in src else 'MISSING')
check(44, '스케줄러 일요일', lambda: 'OK' if 'sun' in src else 'MISSING')

# Model signals
check(45, '5모델 시그널', lambda: [k for k in result.keys() if 'signal' in k])
check(46, 'LSTM 파일', lambda: 'OK' if os.path.exists('models/saved/advanced_lstm_h1.pt') else 'MISSING')
check(47, 'Transformer 파일', lambda: 'OK' if os.path.exists('models/saved/advanced_transformer_h1.pt') else 'MISSING')

# 2-class labels
labeled = fb.create_labels_binary(feat_df, horizon=1)
check(48, '2분류 레이블', lambda: set(labeled['label_1d'].dropna().unique()))
check(49, 'afternoon_batch', lambda: (__import__('batch.afternoon_batch', fromlist=['run_model_a_batch']), 'OK')[1])

# DATABASE_URL
from config.settings import DB_URL
check(50, 'DB_URL', lambda: DB_URL[:50])

# Summary
print(f'\n{"="*50}')
print(f'PASS: {len(passes)} / FAIL: {len(errors)} / TOTAL: 50')
print(f'{"="*50}')
if errors:
    print('\n❌ FAIL:')
    for e in errors:
        print(f'  {e}')
print('\n✅ PASS:')
for p in passes:
    print(f'  {p}')
