"""
models/adaptive_weights.py — 4주 롤링 성과 기반 적응형 앙상블 가중치
매주 월요일: 과거 4주 예측 vs 실현 비교 → 모델별 정확도 → 가중치 재조정
"""
import os, json, logging
import numpy as np
import pandas as pd
from datetime import date, timedelta

logger = logging.getLogger(__name__)

WEIGHT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved", "adaptive_weights.json")

MODEL_NAMES = ["xgb", "lgb", "lstm", "gru", "transformer"]

# 최소 가중치 (특정 모델이 0이 되는 것 방지)
MIN_WEIGHT = 0.05
# 기본 가중치 (데이터 부족 시)
DEFAULT_WEIGHTS = {"xgb": 0.25, "lgb": 0.25, "lstm": 0.15, "gru": 0.15, "transformer": 0.20}


def evaluate_model_accuracy(db, horizon: int = 1, lookback_days: int = 20) -> dict:
    """
    과거 4주(20 거래일) 동안 각 모델의 개별 정확도 측정

    Returns: {
        "xgb": {"accuracy": 0.42, "up_precision": 0.50, "total": 58},
        "lgb": {"accuracy": 0.38, ...}, ...
    }
    """
    try:
        from sqlalchemy import text
        with db.engine.connect() as conn:
            pred_df = pd.read_sql(text("""
                SELECT ticker, pred_date, horizon, pred_label,
                       xgb_signal, lstm_signal
                FROM predictions
                WHERE horizon = :horizon
                  AND pred_date >= CURRENT_DATE - :lookback
                  AND pred_date < CURRENT_DATE - :horizon
                ORDER BY pred_date
            """), conn, params={"horizon": horizon, "lookback": lookback_days})

        if pred_df.empty or len(pred_df) < 20:
            logger.info(f"  Rolling eval: 데이터 부족 ({len(pred_df)}건) → 기본 가중치 유지")
            return {}

        # 실현 수익률 계산
        results = {name: {"correct": 0, "total": 0, "up_correct": 0, "up_total": 0}
                   for name in MODEL_NAMES}

        for _, row in pred_df.iterrows():
            price_df = db.get_stock_prices(row["ticker"], days=lookback_days + horizon + 10)
            if price_df.empty:
                continue

            price_df["trade_date"] = pd.to_datetime(price_df["trade_date"]).dt.date
            pred_date_val = row["pred_date"].date() if hasattr(row["pred_date"], "date") else row["pred_date"]

            mask = price_df["trade_date"] == pred_date_val
            if not mask.any():
                continue
            idx = price_df[mask].index[0]
            future_idx = idx + horizon
            if future_idx >= len(price_df):
                continue

            ret = float(price_df.loc[future_idx, "close"]) / float(price_df.loc[idx, "close"]) - 1
            actual = "UP" if ret > 0.02 else ("DOWN" if ret < -0.02 else "HOLD")

            # 앙상블 라벨 (pred_label)은 전체 합의 → 개별 모델 시그널 활용
            # DB에 xgb_signal, lstm_signal만 저장되어 있음
            # 나머지는 pred_label로 대리 평가
            model_labels = {
                "xgb": row.get("xgb_signal", row["pred_label"]),
                "lgb": row["pred_label"],      # 개별 저장 안 됨 → 앙상블 대리
                "lstm": row.get("lstm_signal", row["pred_label"]),
                "gru": row["pred_label"],       # 개별 저장 안 됨
                "transformer": row["pred_label"],
            }

            for name in MODEL_NAMES:
                ml = model_labels[name]
                results[name]["total"] += 1
                if ml == actual:
                    results[name]["correct"] += 1
                if ml == "UP":
                    results[name]["up_total"] += 1
                    if actual == "UP":
                        results[name]["up_correct"] += 1

        # 정확도 계산
        eval_results = {}
        for name in MODEL_NAMES:
            r = results[name]
            if r["total"] > 0:
                eval_results[name] = {
                    "accuracy": r["correct"] / r["total"],
                    "up_precision": r["up_correct"] / r["up_total"] if r["up_total"] > 0 else 0,
                    "total": r["total"],
                }
            else:
                eval_results[name] = {"accuracy": 0.33, "up_precision": 0.33, "total": 0}

        return eval_results

    except Exception as e:
        logger.warning(f"  Rolling eval 실패: {e}")
        return {}


def compute_adaptive_weights(eval_results: dict) -> dict:
    """
    모델별 정확도 → softmax 정규화 → 적응형 가중치

    알고리즘:
      1. 정확도 + UP정밀도를 7:3으로 결합 → combined_score
      2. softmax(combined_score / temperature) → 가중치
      3. MIN_WEIGHT 이하 방지 → 재정규화
    """
    if not eval_results:
        return DEFAULT_WEIGHTS.copy()

    scores = {}
    for name in MODEL_NAMES:
        r = eval_results.get(name, {})
        acc = r.get("accuracy", 0.33)
        up_prec = r.get("up_precision", 0.33)
        # 정확도 70% + UP 정밀도 30% (매수 신호가 더 중요)
        scores[name] = acc * 0.7 + up_prec * 0.3

    # Softmax with temperature (온도 낮을수록 차이 극대화)
    temperature = 0.15
    score_arr = np.array([scores[n] for n in MODEL_NAMES])
    exp_scores = np.exp((score_arr - score_arr.max()) / temperature)
    softmax_weights = exp_scores / exp_scores.sum()

    # MIN_WEIGHT 보장
    weights = {}
    for i, name in enumerate(MODEL_NAMES):
        weights[name] = max(MIN_WEIGHT, float(softmax_weights[i]))

    # 재정규화 (합=1)
    total = sum(weights.values())
    weights = {k: round(v / total, 4) for k, v in weights.items()}

    return weights


def run_weekly_weight_update(db, horizon: int = 1) -> dict:
    """
    주간 가중치 업데이트 메인 함수
    매주 월요일 배치에서 호출

    Returns: {
        "previous_weights": {...},
        "eval_results": {...},
        "new_weights": {...},
        "updated": True/False,
    }
    """
    logger.info("=" * 50)
    logger.info("  [ADAPTIVE] 4주 롤링 성과 기반 가중치 조정")
    logger.info("=" * 50)

    # 이전 가중치 로드
    prev_weights = load_weights()
    logger.info(f"  현재 가중치: {prev_weights}")

    # 4주(20 거래일) 성과 평가
    eval_results = evaluate_model_accuracy(db, horizon=horizon, lookback_days=20)

    if not eval_results:
        logger.info("  평가 데이터 부족 → 가중치 변경 없음")
        return {"previous_weights": prev_weights, "eval_results": {},
                "new_weights": prev_weights, "updated": False}

    # 성과 로그
    for name, r in eval_results.items():
        logger.info(f"    {name:12s}: acc={r['accuracy']:.3f}, "
                     f"up_prec={r['up_precision']:.3f}, n={r['total']}")

    # 새 가중치 계산
    new_weights = compute_adaptive_weights(eval_results)
    logger.info(f"  새 가중치: {new_weights}")

    # 변화량 로그
    for name in MODEL_NAMES:
        delta = new_weights[name] - prev_weights.get(name, DEFAULT_WEIGHTS[name])
        if abs(delta) > 0.01:
            arrow = "▲" if delta > 0 else "▼"
            logger.info(f"    {name}: {arrow} {delta:+.4f}")

    # 저장
    save_weights(new_weights, eval_results)
    logger.info("  [OK] 가중치 업데이트 완료")

    return {"previous_weights": prev_weights, "eval_results": eval_results,
            "new_weights": new_weights, "updated": True}


def load_weights() -> dict:
    """저장된 가중치 로드 (없으면 기본값)"""
    if os.path.exists(WEIGHT_FILE):
        try:
            with open(WEIGHT_FILE, "r") as f:
                data = json.load(f)
            return data.get("weights", DEFAULT_WEIGHTS.copy())
        except Exception:
            pass
    return DEFAULT_WEIGHTS.copy()


def save_weights(weights: dict, eval_results: dict = None):
    """가중치 + 평가 이력 저장"""
    data = {
        "weights": weights,
        "updated_at": str(date.today()),
        "eval_results": {k: {kk: round(vv, 4) for kk, vv in v.items()}
                         for k, v in (eval_results or {}).items()},
    }

    # 이력 보존 (최근 12주)
    history = []
    if os.path.exists(WEIGHT_FILE):
        try:
            with open(WEIGHT_FILE, "r") as f:
                old = json.load(f)
            history = old.get("history", [])
        except Exception:
            pass

    history.append({"date": str(date.today()), "weights": weights})
    history = history[-12:]  # 최근 12주만 유지
    data["history"] = history

    os.makedirs(os.path.dirname(WEIGHT_FILE), exist_ok=True)
    with open(WEIGHT_FILE, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
