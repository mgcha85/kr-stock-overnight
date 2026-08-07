#!/usr/bin/env python3
"""
Kline Technical Feature Extraction & ML/DL Training Pipeline
------------------------------------------------------------
Calculates technical indicators on KRX candles and trains:
1. LightGBM Classifier
2. PyTorch Deep MLP Network

Input: May 2026 candles (In-Sample training)
Target: is_win = (next_open / close - 1 - 0.0023 > 0)
"""

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PARQUET = ROOT_DIR / "data" / "kr_kline_processed.parquet"
MODEL_DIR = ROOT_DIR / "research" / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)

FEE_RATE = 0.0023

class DeepOvernightNet(nn.Module):
    """PyTorch Deep Neural Network for Overnight Win Probability Prediction"""
    def __init__(self, input_dim: int):
        super(DeepOvernightNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def compute_kline_features(df_candles: pd.DataFrame) -> pd.DataFrame:
    """Computes technical indicator features from candle OHLCV data."""
    df = df_candles.copy()
    
    # 1. Price Action Features
    high_low_range = df['high'] - df['low'] + 1e-5
    df['high_close_ratio'] = (df['close'] - df['low']) / high_low_range
    df['body_ratio'] = abs(df['close'] - df['open']) / high_low_range
    df['upper_shadow_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / high_low_range
    
    # 2. Returns
    df['ret_1d'] = df.groupby('ticker')['close'].pct_change(1).fillna(0)
    df['ret_3d'] = df.groupby('ticker')['close'].pct_change(3).fillna(0)
    df['ret_5d'] = df.groupby('ticker')['close'].pct_change(5).fillna(0)
    
    # 3. Volume & Turnover Features
    df['vol_ma5'] = df.groupby('ticker')['turnover'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['vol_ratio_5d'] = df['turnover'] / (df['vol_ma5'] + 1e-5)
    
    # 4. Bollinger Bands (20, 2.0)
    ma20 = df.groupby('ticker')['close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
    std20 = df.groupby('ticker')['close'].transform(lambda x: x.rolling(20, min_periods=1).std()).fillna(1.0)
    df['bb_pct_b'] = (df['close'] - (ma20 - 2 * std20)) / (4 * std20 + 1e-5)
    df['bb_width'] = (4 * std20) / (ma20 + 1e-5)
    
    # 5. RSI (14)
    delta = df.groupby('ticker')['close'].diff().fillna(0)
    gain = (delta.where(delta > 0, 0)).groupby(df['ticker']).transform(lambda x: x.rolling(14, min_periods=1).mean())
    loss = (-delta.where(delta < 0, 0)).groupby(df['ticker']).transform(lambda x: x.rolling(14, min_periods=1).mean())
    rs = gain / (loss + 1e-5)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # Target: is_win
    df['overnight_pnl'] = (df['next_open'] - df['close']) / df['close'] - FEE_RATE
    df['is_win'] = (df['overnight_pnl'] > 0).astype(int)

    return df


def train_ml_dl_models():
    print("=========================================================================")
    print("      TRAINING LIGHTGBM (ML) & PYTORCH (DL) ON MAY 2026 CANDLES         ")
    print("=========================================================================\n")

    # Load May 2026 Candles
    lazy_df = pl.scan_parquet(str(DATA_PARQUET))
    df_raw = (
        lazy_df
        .filter((pl.col("date") >= "2026-05-01") & (pl.col("date") <= "2026-05-31"))
        .collect()
        .to_pandas()
    )
    
    print(f"[Step 1] Loaded {len(df_raw):,} raw candle records for May 2026.")
    
    # Compute Features
    df_feat = compute_kline_features(df_raw)
    feature_cols = [
        'high_close_ratio', 'body_ratio', 'upper_shadow_ratio',
        'ret_1d', 'ret_3d', 'ret_5d', 'vol_ratio_5d',
        'bb_pct_b', 'bb_width', 'rsi_14'
    ]
    
    df_clean = df_feat.dropna(subset=feature_cols + ['is_win']).copy()
    X_train = df_clean[feature_cols].values
    y_train = df_clean['is_win'].values
    
    print(f"[Step 2] Prepared training matrix: {X_train.shape} | Positive ratio: {y_train.mean():.2%}")

    # 1. Train LightGBM Model
    print("\n[Step 3] Training LightGBM Classifier...")
    lgb_train = lgb.Dataset(X_train, label=y_train)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.05,
        'num_leaves': 15,
        'max_depth': 4,
        'feature_fraction': 0.8,
        'verbose': -1,
        'seed': 42
    }
    gbm = lgb.train(params, lgb_train, num_boost_round=100)
    joblib.dump(gbm, MODEL_DIR / "lgb_kline_model.joblib")
    print(" -> LightGBM model saved successfully.")

    # 2. Train PyTorch Deep MLP Model
    print("\n[Step 4] Training PyTorch Deep MLP Model...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, MODEL_DIR / "kline_scaler.joblib")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" -> Training PyTorch model on device: {device}")

    X_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = DeepOvernightNet(input_dim=len(feature_cols)).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    model.train()
    for epoch in range(1, 21):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)
        
        if epoch % 5 == 0 or epoch == 1:
            avg_loss = total_loss / len(dataset)
            print(f"    Epoch {epoch:02d}/20 | Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), MODEL_DIR / "pytorch_kline_model.pt")
    print(" -> PyTorch model saved successfully.")
    print(f"\nAll models trained and saved to {MODEL_DIR}")

if __name__ == "__main__":
    train_ml_dl_models()
