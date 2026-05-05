"""
AI Trading System - GRU 시계열 모델
LSTM보다 경량, 파라미터 적어 과적합 위험 낮음
"""

import logging
import numpy as np
import pandas as pd
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import MODEL_CONFIG
from data.preprocessor import FeatureBuilder

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")
os.makedirs(MODEL_DIR, exist_ok=True)


class GRUStockModel:
    """GRU 기반 주가 방향 예측 모델 (LSTM 경량 대안)"""

    def __init__(self, horizon: int = 1):
        self.horizon = horizon
        self.lookback = MODEL_CONFIG["lookback_days"]
        self.feature_cols = FeatureBuilder().get_feature_columns()
        self.model = None
        self.mean = None
        self.std = None
        self.model_path = os.path.join(MODEL_DIR, f"gru_h{horizon}.keras")
        self.scaler_path = os.path.join(MODEL_DIR, f"gru_scaler_h{horizon}.pkl")

    def _build_model(self, n_features: int, n_classes: int = 3):
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import (GRU, Dense, Dropout,
                                                  BatchNormalization, Input)
            from tensorflow.keras.optimizers import Adam

            model = Sequential([
                Input(shape=(self.lookback, n_features)),
                GRU(96, return_sequences=True),
                BatchNormalization(),
                Dropout(0.3),
                GRU(48, return_sequences=False),
                BatchNormalization(),
                Dropout(0.3),
                Dense(24, activation="relu"),
                Dense(n_classes, activation="softmax"),
            ])
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            return model
        except ImportError:
            logger.warning("TensorFlow 미설치 - GRU 모델 비활성화")
            return None

    def train(self, df: pd.DataFrame) -> dict:
        label_col = f"label_{self.horizon}d"
        avail_cols = [c for c in self.feature_cols if c in df.columns]
        df_clean = df[avail_cols + ["trade_date", label_col]].dropna()

        if len(df_clean) < self.lookback + 50:
            logger.warning(f"GRU 학습 데이터 부족: {len(df_clean)}건")
            return {}

        # 정규화
        self.mean = df_clean[avail_cols].mean()
        self.std = df_clean[avail_cols].std().replace(0, 1)
        df_norm = df_clean.copy()
        df_norm[avail_cols] = (df_clean[avail_cols] - self.mean) / self.std

        # 시퀀스 생성
        X, y = [], []
        vals = df_norm[avail_cols].values
        labels = df_norm[label_col].astype(int).values
        for i in range(self.lookback, len(vals)):
            X.append(vals[i - self.lookback:i])
            y.append(labels[i])
        X, y = np.array(X, dtype=np.float32), np.array(y)

        if len(X) < 50:
            return {}

        # 클래스 불균형 처리
        from collections import Counter
        class_counts = Counter(y)
        total = len(y)
        n_classes = len(class_counts)
        class_weight = {
            cls: total / (n_classes * count) for cls, count in class_counts.items()
        }

        model = self._build_model(n_features=len(avail_cols))
        if model is None:
            return {}

        try:
            from tensorflow.keras.callbacks import EarlyStopping

            split = int(len(X) * 0.85)
            X_tr, X_val = X[:split], X[split:]
            y_tr, y_val = y[:split], y[split:]

            es = EarlyStopping(monitor="val_accuracy", patience=10,
                               restore_best_weights=True)
            history = model.fit(
                X_tr, y_tr,
                validation_data=(X_val, y_val),
                epochs=40,
                batch_size=32,
                callbacks=[es],
                class_weight=class_weight,
                verbose=0,
            )
            self.model = model
            self._save()

            val_acc = float(np.max(history.history.get("val_accuracy", [0])))
            train_acc = float(np.max(history.history.get("accuracy", [0])))
            overfit_gap = train_acc - val_acc

            if overfit_gap > 0.15:
                logger.warning(f"GRU H{self.horizon} 과적합 의심 | Train:{train_acc:.4f} Val:{val_acc:.4f}")

            logger.info(f"GRU H{self.horizon} 학습 완료 | Val Acc: {val_acc:.4f} | Train Acc: {train_acc:.4f} | Gap: {overfit_gap:.4f}")
            return {
                "horizon": self.horizon,
                "val_accuracy": val_acc,
                "train_accuracy": train_acc,
                "overfit_gap": overfit_gap,
                "class_distribution": dict(class_counts),
            }
        except Exception as e:
            logger.error(f"GRU 학습 오류: {e}")
            return {}

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            self._load()
        if self.model is None:
            return np.full((1, 3), 1 / 3)

        avail_cols = [c for c in self.feature_cols if c in df.columns]
        if len(df) < self.lookback:
            return np.full((1, 3), 1 / 3)

        df_norm = df.copy()
        if self.mean is not None:
            df_norm[avail_cols] = (df[avail_cols] - self.mean[avail_cols]) / self.std[avail_cols]

        vals = df_norm[avail_cols].fillna(0).values
        X = np.array([vals[-self.lookback:]], dtype=np.float32)
        try:
            return self.model.predict(X, verbose=0)
        except Exception:
            return np.full((1, 3), 1 / 3)

    def _save(self):
        import pickle
        try:
            self.model.save(self.model_path)
            with open(self.scaler_path, "wb") as f:
                pickle.dump({"mean": self.mean, "std": self.std}, f)
        except Exception as e:
            logger.error(f"GRU 저장 오류: {e}")

    def _load(self):
        import pickle
        try:
            import tensorflow as tf
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, "rb") as f:
                    scaler = pickle.load(f)
                self.mean = scaler["mean"]
                self.std = scaler["std"]
        except Exception as e:
            logger.warning(f"GRU 로드 실패: {e}")
            self.model = None

    def is_trained(self) -> bool:
        return os.path.exists(self.model_path)
