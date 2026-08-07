#!/usr/bin/env python3
"""
Intraday Multi-Timeframe (MTF) ML/DL Feature Training & Benchmark Pipeline
-----------------------------------------------------------------------------
Extracts 15m/1h intraday candle features before 15:30 market close and merges
with Daily candle indicators into a 15-dim MTF feature vector.

Trains:
1. LightGBM (15-dim MTF Features)
2. PyTorch Deep MLP (15-dim MTF Features)

Evaluates June 2026 Walk-Forward performance against Daily-Only baseline.
"""

import sqlite3
import pandas as pd
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib
from typing import Dict, Any, List

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PARQUET = ROOT_DIR / "data" / "kr_kline_processed.parquet"
INTRA_15M_DIR = Path("/mnt/data/finance/candles/KO/interval=15m")
JUDAL_DB_PATH = Path("/mnt/data/projects/marketMosaic/backend/data/judal.db")
SECTOR_DB = Path("/mnt/data/finance/candles/KO/sector_info.db")
MODEL_DIR = ROOT_DIR / "research" / "models"

FEE_RATE = 0.0023

DAILY_FEATURE_COLS = [
    'high_close_ratio', 'body_ratio', 'upper_shadow_ratio',
    'ret_1d', 'ret_3d', 'ret_5d', 'vol_ratio_5d',
    'bb_pct_b', 'bb_width', 'rsi_14'
]

MTF_INTRA_COLS = [
    'intra_ret_15m', 'intra_ret_1h', 'intra_vol_ratio_30m',
    'intra_high_pullback', 'intra_rsi_15m'
]

COMBINED_FEATURE_COLS = DAILY_FEATURE_COLS + MTF_INTRA_COLS


class DeepMTFOvernightNet(nn.Module):
    """PyTorch Deep Neural Network for 15-dim MTF Features"""
    def __init__(self, input_dim: int = 15):
        super().__init__()
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


def compute_daily_features(df_candles: pd.DataFrame) -> pd.DataFrame:
    df = df_candles.copy()
    high_low_range = df['high'] - df['low'] + 1e-5
    df['high_close_ratio'] = (df['close'] - df['low']) / high_low_range
    df['body_ratio'] = abs(df['close'] - df['open']) / high_low_range
    df['upper_shadow_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / high_low_range
    
    df['ret_1d'] = df.groupby('ticker')['close'].pct_change(1).fillna(0)
    df['ret_3d'] = df.groupby('ticker')['close'].pct_change(3).fillna(0)
    df['ret_5d'] = df.groupby('ticker')['close'].pct_change(5).fillna(0)
    
    df['vol_ma5'] = df.groupby('ticker')['turnover'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['vol_ratio_5d'] = df['turnover'] / (df['vol_ma5'] + 1e-5)
    
    ma20 = df.groupby('ticker')['close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
    std20 = df.groupby('ticker')['close'].transform(lambda x: x.rolling(20, min_periods=1).std()).fillna(1.0)
    df['bb_pct_b'] = (df['close'] - (ma20 - 2 * std20)) / (4 * std20 + 1e-5)
    df['bb_width'] = (4 * std20) / (ma20 + 1e-5)
    
    delta = df.groupby('ticker')['close'].diff().fillna(0)
    gain = (delta.where(delta > 0, 0)).groupby(df['ticker']).transform(lambda x: x.rolling(14, min_periods=1).mean())
    loss = (-delta.where(delta < 0, 0)).groupby(df['ticker']).transform(lambda x: x.rolling(14, min_periods=1).mean())
    rs = gain / (loss + 1e-5)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    if 'next_open' in df.columns:
        df['overnight_pnl'] = (df['next_open'] - df['close']) / df['close'] - FEE_RATE
        df['is_win'] = (df['overnight_pnl'] > 0).astype(int)

    return df


def load_intraday_15m_features(year: int, month: int) -> pd.DataFrame:
    """Loads 15m intraday candles and computes MTF features per stock-day."""
    path_pattern = str(INTRA_15M_DIR / f"year={year}" / f"month={month}" / "*.parquet")
    print(f" -> Scanning 15m intraday candles: {path_pattern}")
    
    lazy_15m = pl.scan_parquet(path_pattern)
    df_15m = (
        lazy_15m
        .select(["code", "datetime", "open", "high", "low", "close", "volume"])
        .collect()
        .to_pandas()
    )

    if df_15m.empty:
        return pd.DataFrame()

    df_15m['date'] = df_15m['datetime'].dt.strftime('%Y-%m-%d')
    df_15m['time'] = df_15m['datetime'].dt.strftime('%H:%M')
    df_15m['code'] = df_15m['code'].apply(lambda x: str(x).split('.')[0].zfill(6))
    
    # Sort by code and datetime
    df_15m = df_15m.sort_values(by=['code', 'datetime']).reset_index(drop=True)

    # 15m RSI per stock
    delta = df_15m.groupby('code')['close'].diff().fillna(0)
    gain = (delta.where(delta > 0, 0)).groupby(df_15m['code']).transform(lambda x: x.rolling(14, min_periods=1).mean())
    loss = (-delta.where(delta < 0, 0)).groupby(df_15m['code']).transform(lambda x: x.rolling(14, min_periods=1).mean())
    rs = gain / (loss + 1e-5)
    df_15m['rsi_15m'] = 100 - (100 / (1 + rs))

    records = []
    for (code, date), group in df_15m.groupby(['code', 'date']):
        group = group.sort_values('time')
        if len(group) < 4:
            continue
        
        last_candle = group.iloc[-1]
        last_1h = group.tail(4)
        last_30m = group.tail(2)
        rest_day = group.iloc[:-2]

        intra_ret_15m = (last_candle['close'] - last_candle['open']) / (last_candle['open'] + 1e-5)
        intra_ret_1h = (last_1h.iloc[-1]['close'] - last_1h.iloc[0]['open']) / (last_1h.iloc[0]['open'] + 1e-5)
        
        avg_vol_morning = rest_day['volume'].mean() + 1e-5
        intra_vol_ratio_30m = last_30m['volume'].mean() / avg_vol_morning
        
        day_high = group['high'].max()
        day_close = last_candle['close']
        intra_high_pullback = (day_close - day_high) / (day_high + 1e-5)
        
        intra_rsi_15m = last_candle['rsi_15m']

        records.append({
            'code': code,
            'date': date,
            'intra_ret_15m': float(intra_ret_15m),
            'intra_ret_1h': float(intra_ret_1h),
            'intra_vol_ratio_30m': float(intra_vol_ratio_30m),
            'intra_high_pullback': float(intra_high_pullback),
            'intra_rsi_15m': float(intra_rsi_15m)
        })

    return pd.DataFrame(records)


def train_mtf_models():
    print("=========================================================================")
    print("   TRAINING INTRADAY MULTI-TIMEFRAME (MTF) LIGHTGBM & PYTORCH MODELS    ")
    print("=========================================================================\n")

    # 1. Load May Daily Candles
    lazy_df = pl.scan_parquet(str(DATA_PARQUET))
    df_daily_raw = (
        lazy_df
        .filter((pl.col("date") >= "2026-05-01") & (pl.col("date") <= "2026-05-31"))
        .collect()
        .to_pandas()
    )
    df_daily_feat = compute_daily_features(df_daily_raw)
    df_daily_feat['code'] = df_daily_feat['ticker'].apply(lambda x: str(x).split('.')[0].zfill(6))

    # 2. Load May 15m Intraday Features
    df_intra_may = load_intraday_15m_features(year=2026, month=5)
    print(f" -> Computed Intraday MTF features for {len(df_intra_may):,} stock-days in May 2026.")

    # 3. Merge Daily + Intraday MTF Features
    merged_train = pd.merge(df_daily_feat, df_intra_may, on=['code', 'date'], how='inner')
    merged_train = merged_train.dropna(subset=COMBINED_FEATURE_COLS + ['is_win']).copy()
    
    X_train = merged_train[COMBINED_FEATURE_COLS].values
    y_train = merged_train['is_win'].values
    print(f" -> Combined Training Matrix: {X_train.shape} | Win Ratio: {y_train.mean():.2%}")

    # 4. Train LightGBM MTF Model
    print("\n[Step 1] Training LightGBM MTF Classifier...")
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
    gbm_mtf = lgb.train(params, lgb_train, num_boost_round=100)
    joblib.dump(gbm_mtf, MODEL_DIR / "lgb_mtf_model.joblib")
    print(" -> LightGBM MTF model saved.")

    # 5. Train PyTorch Deep MLP MTF Model
    print("\n[Step 2] Training PyTorch Deep MLP MTF Model...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, MODEL_DIR / "mtf_scaler.joblib")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=256, shuffle=True)
    mlp_mtf = DeepMTFOvernightNet(input_dim=15).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(mlp_mtf.parameters(), lr=1e-3, weight_decay=1e-4)

    mlp_mtf.train()
    for epoch in range(1, 21):
        total_loss = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = mlp_mtf(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)
        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:02d}/20 | Loss: {total_loss / len(loader.dataset):.4f}")

    torch.save(mlp_mtf.state_dict(), MODEL_DIR / "pytorch_mtf_model.pt")
    print(" -> PyTorch MTF Model saved.")


def evaluate_june_mtf_benchmark():
    print("\n=========================================================================")
    print("   JUNE 2026 WALK-FORWARD BENCHMARK: DAILY-ONLY VS MTF INTRADAY ML/DL   ")
    print("=========================================================================\n")

    # Load June Daily Data
    lazy_df = pl.scan_parquet(str(DATA_PARQUET))
    df_june_daily = (
        lazy_df
        .filter((pl.col("date") >= "2026-05-15") & (pl.col("date") <= "2026-06-30"))
        .collect()
        .to_pandas()
    )
    df_daily_feat = compute_daily_features(df_june_daily)
    df_june_daily_feat = df_daily_feat[(df_daily_feat['date'] >= "2026-06-01") & (df_daily_feat['date'] <= "2026-06-30")].copy()
    df_june_daily_feat['code'] = df_june_daily_feat['ticker'].apply(lambda x: str(x).split('.')[0].zfill(6))

    # Load June 15m Intraday Features
    df_intra_june = load_intraday_15m_features(year=2026, month=6)

    # Judal Themes
    conn_judal = sqlite3.connect(str(JUDAL_DB_PATH))
    df_hist = pd.read_sql_query("""
        SELECT crawl_date as date, code, name, change_rate as stock_change, neglect_index_52w, expected_return
        FROM stock_history
        WHERE crawl_date LIKE '2026-06%'
    """, conn_judal)
    df_ts = pd.read_sql_query("SELECT theme_idx, stock_code as code FROM theme_stocks", conn_judal)
    df_themes = pd.read_sql_query("SELECT theme_idx, name as theme_name FROM themes", conn_judal)
    conn_judal.close()

    df_joined = pd.merge(df_hist, df_ts, on='code', how='inner')
    df_joined = pd.merge(df_joined, df_themes, on='theme_idx', how='inner')
    theme_group = df_joined.groupby(['date', 'theme_idx', 'theme_name'])['stock_change'].agg(
        theme_avg_change='mean', theme_max_change='max', theme_stock_cnt='count'
    ).reset_index()
    df_stock_theme = pd.merge(df_joined, theme_group, on=['date', 'theme_idx', 'theme_name'], how='left')

    # Merge Daily + Judal
    merged_daily = pd.merge(df_stock_theme, df_june_daily_feat, on=['date', 'code'], how='inner')

    # Load Daily Models
    gbm_daily = joblib.load(MODEL_DIR / "lgb_kline_model.joblib")
    scaler_daily = joblib.load(MODEL_DIR / "kline_scaler.joblib")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    from research.run_tcn_and_mtf_experiments import DeepOvernightNet
    mlp_daily = DeepOvernightNet(input_dim=10).to(device)
    mlp_daily.load_state_dict(torch.load(MODEL_DIR / "pytorch_kline_model.pt", map_location=device))
    mlp_daily.eval()

    # Track A: Daily Only Inference
    X_daily = merged_daily[DAILY_FEATURE_COLS].fillna(0).values
    merged_daily['p_lgb'] = gbm_daily.predict(X_daily)
    X_daily_scaled = scaler_daily.transform(X_daily)
    with torch.no_grad():
        p_mlp_daily = mlp_daily(torch.tensor(X_daily_scaled, dtype=torch.float32).to(device)).cpu().numpy().flatten()
    merged_daily['p_torch'] = p_mlp_daily

    merged_daily['is_leader'] = merged_daily['stock_change'] >= merged_daily['theme_max_change'] * 0.85
    merged_daily['judal_score'] = (
        (merged_daily['is_leader'].astype(int) * 35.0) +
        (np.clip(merged_daily['theme_avg_change'], -5, 12) * 2.5) +
        (np.clip(merged_daily['stock_change'], 2, 14) * 3.0) +
        (merged_daily['high_close_ratio'] * 30.0) -
        (np.maximum(0, merged_daily['stock_change'] - 15) * 4.0)
    )
    merged_daily['hybrid_score'] = merged_daily['judal_score'] + merged_daily['p_lgb'] * 40.0 + merged_daily['p_torch'] * 40.0

    # Track B: Combined Daily + Intraday MTF Inference
    merged_mtf = pd.merge(merged_daily, df_intra_june, on=['code', 'date'], how='inner')
    
    gbm_mtf = joblib.load(MODEL_DIR / "lgb_mtf_model.joblib")
    scaler_mtf = joblib.load(MODEL_DIR / "mtf_scaler.joblib")
    mlp_mtf = DeepMTFOvernightNet(input_dim=15).to(device)
    mlp_mtf.load_state_dict(torch.load(MODEL_DIR / "pytorch_mtf_model.pt", map_location=device))
    mlp_mtf.eval()

    X_mtf = merged_mtf[COMBINED_FEATURE_COLS].fillna(0).values
    merged_mtf['p_lgb_mtf'] = gbm_mtf.predict(X_mtf)
    X_mtf_scaled = scaler_mtf.transform(X_mtf)
    with torch.no_grad():
        p_mlp_mtf = mlp_mtf(torch.tensor(X_mtf_scaled, dtype=torch.float32).to(device)).cpu().numpy().flatten()
    merged_mtf['p_torch_mtf'] = p_mlp_mtf
    merged_mtf['hybrid_score_mtf'] = merged_mtf['judal_score'] + merged_mtf['p_lgb_mtf'] * 40.0 + merged_mtf['p_torch_mtf'] * 40.0

    # Run Simulations
    from research.run_tcn_and_mtf_experiments import run_simulation

    merged_daily['stock_name'] = merged_daily['name']
    merged_daily['p_tcn'] = 0.0
    merged_daily['aux_bonus'] = 0.0
    res_daily = run_simulation(merged_daily, top_k=3, min_hybrid_score=0.0, model_mode="ensemble_full")
    
    # Custom simulation call for MTF
    merged_mtf_eval = merged_mtf.copy()
    merged_mtf_eval['stock_name'] = merged_mtf_eval['name']
    merged_mtf_eval['p_lgb'] = merged_mtf_eval['p_lgb_mtf']
    merged_mtf_eval['p_torch'] = merged_mtf_eval['p_torch_mtf']
    merged_mtf_eval['p_tcn'] = 0.0
    merged_mtf_eval['aux_bonus'] = 0.0
    res_mtf = run_simulation(merged_mtf_eval, top_k=3, min_hybrid_score=0.0, model_mode="ensemble_full")

    print("\n" + "="*80)
    print("      WALK-FORWARD PERFORMANCE COMPARISON (JUNE 2026)      ")
    print("="*80)
    print(f"Track A: Daily Features Only ML/DL (10-dim)")
    print(f"  -> Total Return  : {res_daily['total_return']*100:+.2f}%")
    print(f"  -> Win Rate       : {res_daily['win_rate']*100:.2f}%")
    print(f"  -> Profit Factor : {res_daily['profit_factor']:.2f}")
    print(f"  -> Sharpe Ratio   : {res_daily['sharpe']:.2f}")
    print(f"  -> Max Drawdown  : {res_daily['mdd']*100:.2f}%\n")

    print(f"Track B: Daily + Intraday MTF Features ML/DL (15-dim)")
    print(f"  -> Total Return  : {res_mtf['total_return']*100:+.2f}%")
    print(f"  -> Win Rate       : {res_mtf['win_rate']*100:.2f}%")
    print(f"  -> Profit Factor : {res_mtf['profit_factor']:.2f}")
    print(f"  -> Sharpe Ratio   : {res_mtf['sharpe']:.2f}")
    print(f"  -> Max Drawdown  : {res_mtf['mdd']*100:.2f}%\n")

    print("="*80)
    print("      TOP-10 HYBRID SCORE THRESHOLD SWEEP (MTF ML/DL MODEL)      ")
    print("="*80)
    scores_grid = [0.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    thresh_records = []
    for sc in scores_grid:
        r = run_simulation(merged_mtf_eval, top_k=10, min_hybrid_score=sc, model_mode="ensemble_full")
        thresh_records.append({
            'min_score': sc,
            'total_return': r['total_return'],
            'win_rate': r['win_rate'],
            'profit_factor': r['profit_factor'],
            'sharpe': r['sharpe'],
            'mdd': r['mdd'],
            'n_trades': r['n_trades']
        })
    df_thresh = pd.DataFrame(thresh_records)
    print(df_thresh.to_string(index=False))


if __name__ == "__main__":
    train_mtf_models()
    evaluate_june_mtf_benchmark()
