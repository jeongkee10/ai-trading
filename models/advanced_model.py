"""
AI Trading System - 5모델 앙상블 (Optuna 최적화 + PyTorch 딥러닝)
매주 일요일 학습 → 평일 추론
"""

import logging
import os
import pickle
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import optuna

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.preprocessor import FeatureBuilder
from config.settings import MODEL_CONFIG

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")
os.makedirs(MODEL_DIR, exist_ok=True)

NEWS_FEATURES = ["news_sentiment", "news_count", "news_positive_ratio", "news_negative_ratio", "news_momentum"]


# ═══════════════════════════════════════════════════════
# PyTorch LSTM Model
# ═══════════════════════════════════════════════════════
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ═══════════════════════════════════════════════════════
# PyTorch Transformer Model
# ═══════════════════════════════════════════════════════
class TransformerNet(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128,
                                                    dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        return self.fc(x[:, -1, :])


# ═══════════════════════════════════════════════════════
# Advanced Ensemble Predictor
# ═══════════════════════════════════════════════════════
class AdvancedEnsemblePredictor:
    """
    5모델 스태킹 앙상블 + Optuna 최적화
    XGBoost + LightGBM + CatBoost + LSTM + Transformer
    """

    LABEL_MAP = {0: "DOWN", 1: "UP"}
    MODEL_NAMES = ["xgb", "lgb", "catboost", "lstm", "transformer"]
    SEQ_LEN = 20  # 딥러닝 시퀀스 길이

    def __init__(self):
        self.fb = FeatureBuilder()
        self.models = {}
        self.meta_model = None
        self.trained_cols = []
        self.cfg = MODEL_CONFIG

    # ─── 전체 학습 파이프라인 ────────────────────────
    def train_all(self, combined_df: pd.DataFrame, horizon: int = 1,
                  optuna_trials: int = 50) -> dict:
        """주간 학습 (일요일 실행)"""
        start_time = time.time()
        logger.info(f"=== Advanced Ensemble H{horizon} 학습 시작 ===")

        # 뉴스 피처 칼럼 보장
        for col in NEWS_FEATURES:
            if col not in combined_df.columns:
                combined_df[col] = 0.0

        # 레이블 생성
        df = self.fb.create_labels_binary(combined_df, horizon)

        # 피처 선택 (NaN 50% 이하)
        all_feat = self.fb.get_feature_columns() + NEWS_FEATURES
        self.trained_cols = [c for c in all_feat if c in df.columns and df[c].isna().mean() <= 0.5]
        label_col = f"label_{horizon}d"
        df_clean = df[self.trained_cols + [label_col]].dropna()

        if len(df_clean) < 500:
            logger.error(f"학습 데이터 부족: {len(df_clean)}")
            return {}

        X = df_clean[self.trained_cols].values.astype(np.float32)
        y = df_clean[label_col].astype(int).values

        # 최근 데이터 가중 (최근 30%에 3배 가중치)
        self._sample_weights = np.ones(len(X))
        recent_start = int(len(X) * 0.7)
        self._sample_weights[recent_start:] = 3.0

        logger.info(f"  데이터: {len(X)} samples, {len(self.trained_cols)} features")
        logger.info(f"  클래스: UP={sum(y==1)} DOWN={sum(y==0)}")
        logger.info(f"  최근 가중: 상위 30%에 3배 가중치")

        results = {}

        # 1. XGBoost (Optuna)
        logger.info("  [1/5] XGBoost + Optuna...")
        results["xgb"] = self._train_xgb(X, y, optuna_trials)

        # 2. LightGBM (Optuna)
        logger.info("  [2/5] LightGBM + Optuna...")
        results["lgb"] = self._train_lgb(X, y, optuna_trials)

        # 3. CatBoost (Optuna)
        logger.info("  [3/5] CatBoost + Optuna...")
        results["catboost"] = self._train_catboost(X, y, optuna_trials)

        # 4. LSTM (PyTorch)
        logger.info("  [4/5] LSTM...")
        results["lstm"] = self._train_lstm(df_clean, horizon)

        # 5. Transformer (PyTorch)
        logger.info("  [5/5] Transformer...")
        results["transformer"] = self._train_transformer(df_clean, horizon)

        # 6. 메타러너 스태킹
        logger.info("  [Meta] Stacking meta-learner...")
        results["meta"] = self._train_meta(X, y, horizon)

        elapsed = time.time() - start_time
        logger.info(f"=== 학습 완료 ({elapsed:.0f}초) ===")
        for k, v in results.items():
            if v:
                logger.info(f"  {k}: F1={v.get('f1', 0):.4f} Acc={v.get('accuracy', 0):.4f}")

        # 저장
        self._save_all(horizon)
        return results

    # ─── XGBoost + Optuna ────────────────────────────
    def _train_xgb(self, X, y, n_trials):
        sw = self._sample_weights

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 0.5),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
                "scale_pos_weight": sum(y == 0) / max(sum(y == 1), 1),
            }
            model = XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss", random_state=42)
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            for tr, val in tscv.split(X):
                model.fit(X[tr], y[tr], sample_weight=sw[tr])
                scores.append(f1_score(y[val], model.predict(X[val]), average="macro"))
            return np.mean(scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_params["scale_pos_weight"] = sum(y == 0) / max(sum(y == 1), 1)
        model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric="logloss", random_state=42)
        model.fit(X, y, sample_weight=sw)
        self.models["xgb"] = model

        return {"f1": study.best_value, "accuracy": study.best_value, "params": best_params}

    # ─── LightGBM + Optuna ───────────────────────────
    def _train_lgb(self, X, y, n_trials):
        sw = self._sample_weights

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
                "scale_pos_weight": sum(y == 0) / max(sum(y == 1), 1),
            }
            model = LGBMClassifier(**params, random_state=42, verbose=-1)
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            for tr, val in tscv.split(X):
                model.fit(X[tr], y[tr], sample_weight=sw[tr])
                scores.append(f1_score(y[val], model.predict(X[val]), average="macro"))
            return np.mean(scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_params["scale_pos_weight"] = sum(y == 0) / max(sum(y == 1), 1)
        model = LGBMClassifier(**best_params, random_state=42, verbose=-1)
        model.fit(X, y, sample_weight=sw)
        self.models["lgb"] = model

        return {"f1": study.best_value, "accuracy": study.best_value, "params": best_params}

    # ─── CatBoost + Optuna ───────────────────────────
    def _train_catboost(self, X, y, n_trials):
        sw = self._sample_weights

        def objective(trial):
            params = {
                "iterations": trial.suggest_int("iterations", 200, 600, step=50),
                "depth": trial.suggest_int("depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "border_count": trial.suggest_int("border_count", 32, 255),
                "auto_class_weights": "Balanced",
            }
            model = CatBoostClassifier(**params, random_seed=42, verbose=0)
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            for tr, val in tscv.split(X):
                model.fit(X[tr], y[tr], sample_weight=sw[tr])
                scores.append(f1_score(y[val], model.predict(X[val]), average="macro"))
            return np.mean(scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_params["auto_class_weights"] = "Balanced"
        model = CatBoostClassifier(**best_params, random_seed=42, verbose=0)
        model.fit(X, y, sample_weight=sw)
        self.models["catboost"] = model

        return {"f1": study.best_value, "accuracy": study.best_value, "params": best_params}

    # ─── LSTM (PyTorch) ──────────────────────────────
    def _train_lstm(self, df_clean, horizon):
        X_seq, y_seq = self._build_sequences(df_clean, horizon)
        if X_seq is None:
            return {"f1": 0, "accuracy": 0}

        input_size = X_seq.shape[2]
        model = LSTMNet(input_size, hidden_size=128, num_layers=3, dropout=0.3)
        return self._train_pytorch(model, X_seq, y_seq, "lstm", epochs=100, lr=0.0005)

    # ─── Transformer (PyTorch) ───────────────────────
    def _train_transformer(self, df_clean, horizon):
        X_seq, y_seq = self._build_sequences(df_clean, horizon)
        if X_seq is None:
            return {"f1": 0, "accuracy": 0}

        input_size = X_seq.shape[2]
        model = TransformerNet(input_size, d_model=128, nhead=4, num_layers=3, dropout=0.3)
        return self._train_pytorch(model, X_seq, y_seq, "transformer", epochs=80, lr=0.0005)

    def _build_sequences(self, df_clean, horizon):
        label_col = f"label_{horizon}d"
        X_vals = df_clean[self.trained_cols].values.astype(np.float32)
        y_vals = df_clean[label_col].values.astype(np.int64)

        sequences, labels = [], []
        for i in range(self.SEQ_LEN, len(X_vals)):
            sequences.append(X_vals[i - self.SEQ_LEN:i])
            labels.append(y_vals[i])

        if len(sequences) < 200:
            return None, None

        return np.array(sequences), np.array(labels)

    def _train_pytorch(self, model, X_seq, y_seq, name, epochs=100, lr=0.0005):
        # Train/val split (시계열: 뒤 20% validation)
        split = int(len(X_seq) * 0.8)
        X_train, X_val = torch.FloatTensor(X_seq[:split]), torch.FloatTensor(X_seq[split:])
        y_train, y_val = torch.LongTensor(y_seq[:split]), torch.LongTensor(y_seq[split:])

        # 최근 데이터 가중 (최근 50%에 2배 가중치)
        n_train = len(X_train)
        sample_weights = torch.ones(n_train)
        sample_weights[n_train // 2:] = 2.0  # 최근 절반 2배 가중

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss(reduction='none')

        best_f1 = 0
        best_state = None
        patience = 15
        no_improve = 0

        model.train()
        batch_size = 512
        for epoch in range(epochs):
            indices = torch.randperm(n_train)
            epoch_loss = 0
            for i in range(0, n_train, batch_size):
                batch_idx = indices[i:i + batch_size]
                output = model(X_train[batch_idx])
                loss = criterion(output, y_train[batch_idx])
                # 가중 손실
                weighted_loss = (loss * sample_weights[batch_idx]).mean()
                optimizer.zero_grad()
                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += weighted_loss.item()
            scheduler.step()

            # Early stopping (매 10 epoch)
            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_out = model(X_val)
                    val_pred = val_out.argmax(dim=1).numpy()
                    val_f1 = f1_score(y_val.numpy(), val_pred, average="macro")
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience // 10:
                    break
                model.train()

        # Best model 복원
        if best_state:
            model.load_state_dict(best_state)

        # 최종 평가
        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_pred = val_out.argmax(dim=1).numpy()
            f1 = f1_score(y_val.numpy(), val_pred, average="macro")
            acc = accuracy_score(y_val.numpy(), val_pred)

        self.models[name] = model
        logger.info(f"  {name}: F1={f1:.4f} Acc={acc:.4f} (best_f1={best_f1:.4f})")
        return {"f1": f1, "accuracy": acc}

    # ─── 메타러너 (스태킹) ───────────────────────────
    def _train_meta(self, X, y, horizon):
        """3개 tree모델 배치 predict → LogisticRegression 메타러너"""
        # 배치 predict (벡터화 — 빠름)
        X_recent = X[-5000:]
        y_recent = y[-5000:]

        meta_cols = []
        for name in ["xgb", "lgb", "catboost"]:
            model = self.models.get(name)
            if model:
                proba = model.predict_proba(X_recent)  # shape: (N, 2) — 배치
                meta_cols.append(proba)
            else:
                meta_cols.append(np.full((len(X_recent), 2), 0.5))

        X_meta = np.hstack(meta_cols)  # shape: (N, 6)
        y_meta = y_recent

        meta = LogisticRegression(max_iter=1000, C=1.0)
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        for tr, val in tscv.split(X_meta):
            meta.fit(X_meta[tr], y_meta[tr])
            scores.append(f1_score(y_meta[val], meta.predict(X_meta[val]), average="macro"))

        meta.fit(X_meta, y_meta)
        self.meta_model = meta
        f1 = float(np.mean(scores))
        logger.info(f"  Meta: F1={f1:.4f}")
        return {"f1": f1, "accuracy": f1}

    # ─── 추론 ────────────────────────────────────────
    def predict_single(self, ticker: str, price_df: pd.DataFrame,
                       news_features: dict = None, horizon: int = 1) -> dict:
        """단일 종목 예측"""
        feat_df = self.fb.build_features(price_df)
        if feat_df.empty or len(feat_df) < self.SEQ_LEN + 5:
            return {}

        for col in NEWS_FEATURES:
            feat_df[col] = (news_features or {}).get(col, 0.0)

        cols = [c for c in self.trained_cols if c in feat_df.columns]
        if not cols:
            return {}

        # Tree 모델 예측
        X_latest = feat_df[cols].tail(1).fillna(0).values.astype(np.float32)
        tree_probas = {}
        for name in ["xgb", "lgb", "catboost"]:
            model = self.models.get(name)
            if model:
                tree_probas[name] = model.predict_proba(X_latest)[0]
            else:
                tree_probas[name] = np.array([0.5, 0.5])

        # 딥러닝 예측
        dl_probas = {}
        X_seq = feat_df[cols].tail(self.SEQ_LEN).fillna(0).values.astype(np.float32)
        if len(X_seq) == self.SEQ_LEN:
            X_tensor = torch.FloatTensor(X_seq).unsqueeze(0)
            for name in ["lstm", "transformer"]:
                model = self.models.get(name)
                if model:
                    model.eval()
                    with torch.no_grad():
                        out = model(X_tensor)
                        proba = torch.softmax(out, dim=1)[0].numpy()
                        dl_probas[name] = proba

        # 앙상블 (메타러너 or 단순평균)
        if self.meta_model:
            meta_input = np.concatenate([tree_probas[n] for n in ["xgb", "lgb", "catboost"]]).reshape(1, -1)
            ensemble_proba = self.meta_model.predict_proba(meta_input)[0]
        else:
            all_probas = list(tree_probas.values()) + list(dl_probas.values())
            ensemble_proba = np.mean(all_probas, axis=0)

        up_prob = float(ensemble_proba[1])
        pred_label = "UP" if up_prob > 0.5 else "DOWN"

        return {
            "ticker": ticker,
            "pred_label": pred_label,
            "pred_up_prob": round(up_prob, 4),
            "pred_down_prob": round(1 - up_prob, 4),
            "xgb_signal": "UP" if tree_probas["xgb"][1] > 0.5 else "DOWN",
            "lgb_signal": "UP" if tree_probas["lgb"][1] > 0.5 else "DOWN",
            "catboost_signal": "UP" if tree_probas["catboost"][1] > 0.5 else "DOWN",
            "lstm_signal": "UP" if dl_probas.get("lstm", [0.5, 0.5])[1] > 0.5 else "DOWN",
            "transformer_signal": "UP" if dl_probas.get("transformer", [0.5, 0.5])[1] > 0.5 else "DOWN",
            "confidence": round(abs(up_prob - 0.5) * 2, 4),
            "horizon": horizon,
        }

    # ─── 저장/로드 ───────────────────────────────────
    def _save_all(self, horizon):
        path = os.path.join(MODEL_DIR, f"advanced_h{horizon}.pkl")
        save_data = {
            "trained_cols": self.trained_cols,
            "meta_model": self.meta_model,
        }
        # Tree 모델
        for name in ["xgb", "lgb", "catboost"]:
            save_data[name] = self.models.get(name)
        with open(path, "wb") as f:
            pickle.dump(save_data, f)

        # PyTorch 모델
        for name in ["lstm", "transformer"]:
            model = self.models.get(name)
            if model:
                torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"advanced_{name}_h{horizon}.pt"))

        logger.info(f"  모델 저장 완료: {path}")

    def load(self, horizon: int = 1):
        path = os.path.join(MODEL_DIR, f"advanced_h{horizon}.pkl")
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.trained_cols = data["trained_cols"]
        self.meta_model = data["meta_model"]
        for name in ["xgb", "lgb", "catboost"]:
            self.models[name] = data.get(name)
        # PyTorch
        for name in ["lstm", "transformer"]:
            pt_path = os.path.join(MODEL_DIR, f"advanced_{name}_h{horizon}.pt")
            if os.path.exists(pt_path) and self.trained_cols:
                n_features = len(self.trained_cols)
                if name == "lstm":
                    model = LSTMNet(n_features, hidden_size=128, num_layers=3, dropout=0.3)
                else:
                    model = TransformerNet(n_features, d_model=128, nhead=4, num_layers=3, dropout=0.3)
                model.load_state_dict(torch.load(pt_path, weights_only=True))
                model.eval()
                self.models[name] = model
        return True

    def is_trained(self, horizon: int = 1) -> bool:
        return os.path.exists(os.path.join(MODEL_DIR, f"advanced_h{horizon}.pkl"))
