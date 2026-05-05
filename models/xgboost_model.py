"""
AI Trading System - XGBoost 분류 모델
"""

import logging
import numpy as np
import pandas as pd
import pickle
import os
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import MODEL_CONFIG
from data.preprocessor import FeatureBuilder

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")
os.makedirs(MODEL_DIR, exist_ok=True)


class XGBStockModel:
    """종목별 XGBoost 3분류 모델 (UP/HOLD/DOWN)"""

    def __init__(self, horizon: int = 1):
        self.horizon = horizon
        self.feature_cols = FeatureBuilder().get_feature_columns()
        self.trained_cols = self.feature_cols
        self.model = None
        self.model_path = os.path.join(MODEL_DIR, f"xgb_h{horizon}.pkl")

    def _build_model(self, params: dict = None):
        default_params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "eval_metric": "mlogloss",
            "random_state": 42,
            "n_jobs": -1,
        }
        if params:
            default_params.update(params)
        return XGBClassifier(**default_params)

    def train(self, df: pd.DataFrame) -> dict:
        """
        df: 피처 + 레이블 포함 DataFrame
        Returns: 학습 성과 지표
        """
        label_col = f"label_{self.horizon}d"
        if label_col not in df.columns:
            logger.error(f"레이블 컬럼 {label_col} 없음")
            return {}

        avail_cols = [c for c in self.feature_cols if c in df.columns and df[c].isna().mean() <= 0.5]
        self.trained_cols = avail_cols
        df_clean = df[avail_cols + [label_col]].dropna()
        if len(df_clean) < 100:
            logger.warning(f"학습 데이터 부족: {len(df_clean)}건")
            return {}

        X = df_clean[avail_cols].values
        y = df_clean[label_col].astype(int).values

        # 클래스 불균형 처리 - sample_weight 계산
        from collections import Counter
        class_counts = Counter(y)
        total = len(y)
        n_classes = len(class_counts)
        sample_weights = np.array([
            total / (n_classes * class_counts[label]) for label in y
        ])

        # Optuna 하이퍼파라미터 탐색 (월요일 or 최초 학습 시)
        best_params = self._optuna_search(X, y, sample_weights)

        self.model = self._build_model(best_params)
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            sw_tr = sample_weights[train_idx]
            self.model.fit(
                X_tr, y_tr,
                sample_weight=sw_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            y_pred = self.model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            scores.append(acc)
            logger.debug(f"  Fold {fold+1} accuracy: {acc:.4f}")

        # 전체 데이터 최종 학습 (가중치 포함)
        self.model.fit(X, y, sample_weight=sample_weights, verbose=False)
        self._save()

        # Self-consistency 검증: 학습 데이터 재예측으로 과적합/미학습 확인
        y_train_pred = self.model.predict(X)
        train_acc = accuracy_score(y, y_train_pred)
        cv_acc = float(np.mean(scores))
        overfit_gap = train_acc - cv_acc

        if overfit_gap > 0.15:
            logger.warning(f"XGB H{self.horizon} 과적합 의심 | Train:{train_acc:.4f} CV:{cv_acc:.4f} Gap:{overfit_gap:.4f}")
        if cv_acc < 0.35:
            logger.warning(f"XGB H{self.horizon} 미학습 의심 | CV Acc:{cv_acc:.4f}")

        metrics = {
            "horizon": self.horizon,
            "train_samples": len(df_clean),
            "cv_accuracy_mean": cv_acc,
            "cv_accuracy_std": float(np.std(scores)),
            "train_accuracy": train_acc,
            "overfit_gap": overfit_gap,
            "class_distribution": dict(class_counts),
            "feature_cols": avail_cols,
        }
        logger.info(f"XGB H{self.horizon} 학습 완료 | CV Acc: {cv_acc:.4f} | Train Acc: {train_acc:.4f} | Gap: {overfit_gap:.4f}")
        return metrics

    def _optuna_search(self, X: np.ndarray, y: np.ndarray,
                       sample_weights: np.ndarray, n_trials: int = 15) -> dict:
        """Optuna로 최적 하이퍼파라미터 탐색"""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("Optuna 미설치 - 기본 파라미터 사용")
            return {}

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 0.5),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
            }
            model = self._build_model(params)
            tscv = TimeSeriesSplit(n_splits=3)
            fold_scores = []
            for train_idx, val_idx in tscv.split(X):
                model.fit(X[train_idx], y[train_idx],
                          sample_weight=sample_weights[train_idx],
                          eval_set=[(X[val_idx], y[val_idx])],
                          verbose=False)
                y_pred = model.predict(X[val_idx])
                fold_scores.append(accuracy_score(y[val_idx], y_pred))
            return np.mean(fold_scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best = study.best_params
        logger.info(f"  Optuna 최적 파라미터 (H{self.horizon}): acc={study.best_value:.4f} params={best}")
        return best

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns shape: (n, 3) — [DOWN_prob, HOLD_prob, UP_prob]
        """
        if self.model is None:
            self._load()
        cols = [c for c in self.trained_cols if c in df.columns]
        X = df[cols].fillna(0).values
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            self._load()
        avail_cols = [c for c in self.feature_cols]
        scores = self.model.feature_importances_
        return pd.DataFrame({
            "feature": avail_cols[:len(scores)],
            "importance": scores
        }).sort_values("importance", ascending=False)

    def _save(self):
        with open(self.model_path, "wb") as f:
            pickle.dump({"model": self.model, "trained_cols": self.trained_cols}, f)

    def _load(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, dict) and "model" in data:
                    self.model = data["model"]
                    self.trained_cols = data.get("trained_cols", self.feature_cols)
                else:
                    self.model = data
                    self.trained_cols = self.feature_cols
        else:
            raise FileNotFoundError(f"모델 파일 없음: {self.model_path}")

    def is_trained(self) -> bool:
        return os.path.exists(self.model_path)
