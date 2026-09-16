"""Train LGB + PyTorch on 15:19 session bars → next 09:00 labels."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from kr_stock.config import FEE_RATE, MODEL_DIR
from kr_stock.inference import DAILY_FEATURE_COLS, compute_kline_features
from kr_stock.session_bars import load_session_1520
from research.kline_ml_dl_pipeline import DeepOvernightNet

SEED = 42
TRAIN_START = "2025-02-01"  # 20-day feature warmup after 2025-01-02
TRAIN_END = "2025-08-12"
TEST_START = "2025-08-13"
TEST_END = "2026-08-12"

LGB_1520 = MODEL_DIR / "lgb_kline_1520_model.joblib"
SCALER_1520 = MODEL_DIR / "kline_scaler_1520.joblib"
TORCH_1520 = MODEL_DIR / "pytorch_kline_1520_model.pt"


def _session_frame() -> pd.DataFrame:
    df = load_session_1520().to_pandas()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return compute_kline_features(df)


def train_1520_models() -> Dict[str, float]:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    feat = _session_frame()
    train = feat[
        (feat["date"] >= TRAIN_START)
        & (feat["date"] <= TRAIN_END)
        & feat["next_open"].notna()
        & (feat["close"] > 0)
        & (feat["next_open"] > 0)
    ].dropna(subset=DAILY_FEATURE_COLS + ["is_win"])
    test = feat[
        (feat["date"] >= TEST_START)
        & (feat["date"] <= TEST_END)
        & feat["next_open"].notna()
        & (feat["close"] > 0)
    ].dropna(subset=DAILY_FEATURE_COLS + ["is_win"])

    X_train = train[DAILY_FEATURE_COLS].values
    y_train = train["is_win"].values.astype(int)
    print(
        f"[train] {len(train):,} rows {train['date'].min()}..{train['date'].max()} "
        f"pos={y_train.mean():.2%}"
    )
    print(
        f"[test ] {len(test):,} rows {test['date'].min() if len(test) else '-'}.."
        f"{test['date'].max() if len(test) else '-'} "
        f"pos={test['is_win'].mean():.2%}" if len(test) else "[test] empty"
    )

    lgb_train = lgb.Dataset(X_train, label=y_train)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 4,
        "feature_fraction": 0.8,
        "verbose": -1,
        "seed": SEED,
    }
    gbm = lgb.train(params, lgb_train, num_boost_round=100)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(gbm, LGB_1520)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, SCALER_1520)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=256, shuffle=True)
    model = DeepOvernightNet(input_dim=len(DAILY_FEATURE_COLS)).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for epoch in range(1, 21):
        total = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(bx)
        if epoch % 5 == 0 or epoch == 1:
            print(f"    epoch {epoch:02d}/20 loss={total / len(train):.4f} device={device}")
    torch.save(model.state_dict(), TORCH_1520)

    def _auc(df: pd.DataFrame) -> float:
        if df.empty:
            return float("nan")
        p = gbm.predict(df[DAILY_FEATURE_COLS].values)
        y = df["is_win"].values
        if y.min() == y.max():
            return float("nan")
        return float(roc_auc_score(y, p))

    stats = {
        "train_rows": float(len(train)),
        "test_rows": float(len(test)),
        "train_pos": float(y_train.mean()),
        "test_pos": float(test["is_win"].mean()) if len(test) else 0.0,
        "train_auc": _auc(train),
        "test_auc": _auc(test),
        "fee_rate": FEE_RATE,
    }
    print(
        f"[auc] train={stats['train_auc']:.3f} test={stats['test_auc']:.3f} "
        f"saved {LGB_1520.name} / {TORCH_1520.name}"
    )
    return stats
