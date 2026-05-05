"""
AI Trading System - LightGBM 분류 모델
XGBoost 대비 빠르고 대규모 피처에 강함
"""

import logging
import numpy as np
import pandas as pd
import pickle
import os
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from collections import Counter

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import MODEL_CONFIG
from data.preprocessor import FeatureBuilder

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")
os.makedirs(MODEL_DIR, exist_ok=True)


class LGBStockModel:
    """LightGBM 3분류 모델 (UP/HOLD/DOWN)"""

    def __init__(self, horizon: int = 1):
        self.horizon = horizon
        self.feature_cols = FeatureBuilder().get_feature_columns()
        self.trained_cols = self.feature_cols
        self.model = None
        self.model_path = os.path.join(MODEL_DIR, f"lgb_h{horizon}.pkl")

    def _build_model(self, params: dict = None):
        default_params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "num_leaves": 31,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        if params:
            default_params.update(params)
        return LGBMClassifier(**default_params)

    def train(self, df: pd.DataFrame) -> dict:
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

        # 클래스 불균형 처리
        class_counts = Counter(y)
        total = len(y)
        n_classes = len(class_counts)
        sample_weights = np.array([
            total / (n_classes * class_counts[label]) for label in y
        ])

        # Optuna 탐색
        best_params = self._optuna_search(X, y, sample_weights)

        self.model = self._build_model(best_params)
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            self.model.fit(
                X[train_idx], y[train_idx],
                sample_weight=sample_weights[train_idx],
                eval_set=[(X[val_idx], y[val_idx])],
            )
            y_pred = self.model.predict(X[val_idx])
            scores.append(accuracy_score(y[val_idx], y_pred))

        self.model.fit(X, y, sample_weight=sample_weights)
        self._save()

        y_train_pred = self.model.predict(X)
        train_acc = accuracy_score(y, y_train_pred)
        cv_acc = float(np.mean(scores))
        overfit_gap = train_acc - cv_acc

        if overfit_gap > 0.15:
            logger.warning(f"LGB H{self.horizon} 과적합 의심 | Train:{train_acc:.4f} CV:{cv_acc:.4f}")

        metrics = {
            "horizon": self.horizon,
            "train_samples": len(df_clean),
            "cv_accuracy_mean": cv_acc,
            "train_accuracy": train_acc,
            "overfit_gap": overfit_gap,
            "class_distribution": dict(class_counts),
        }
        logger.info(f"LGB H{self.horizon} 학습 완료 | CV Acc: {cv_acc:.4f} | Train Acc: {train_acc:.4f} | Gap: {overfit_gap:.4f}")
        return metrics

    def _optuna_search(self, X, y, sample_weights, n_trials=15):
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            return {}

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
            }
            model = self._build_model(params)
            tscv = TimeSeriesSplit(n_splits=3)
            fold_scores = []
            for train_idx, val_idx in tscv.split(X):
                model.fit(X[train_idx], y[train_idx],
                          sample_weight=sample_weights[train_idx],
                          eval_set=[(X[val_idx], y[val_idx])])
                fold_scores.append(accuracy_score(y[val_idx], model.predict(X[val_idx])))
            return np.mean(fold_scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        logger.info(f"  Optuna LGB 최적 (H{self.horizon}): acc={study.best_value:.4f}")
        return study.best_params

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            self._load()
        cols = [c for c in self.trained_cols if c in df.columns]
        X = df[cols].fillna(0).values
        return self.model.predict_proba(X)

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
