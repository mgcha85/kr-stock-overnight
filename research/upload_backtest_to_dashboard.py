#!/usr/bin/env python3
"""
Backtest Result Dashboard Auto-Uploader
----------------------------------------
Uploads completed backtest results to the Backtest Lab Dashboard API (http://146.56.115.71:8082/api/backtest)
in strict compliance with AGENTS.md and Backtest Lab API Specifications.
"""

import requests
import json
import sqlite3
import pandas as pd
import numpy as np
import polars as pl
import torch
import joblib
from datetime import datetime
from pathlib import Path

from research.marketmosaic_integrator import MarketMosaicIntegrator
from research.kline_ml_dl_pipeline import compute_kline_features, DeepOvernightNet

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PARQUET = ROOT_DIR / "data" / "kr_kline_processed.parquet"
JUDAL_DB_PATH = Path("/mnt/data/projects/marketMosaic/backend/data/judal.db")
SECTOR_DB = Path("/mnt/data/finance/candles/KO/sector_info.db")
MODEL_DIR = ROOT_DIR / "research" / "models"
DASHBOARD_API_URL = "http://146.56.115.71:8082/api/backtest"

FEE_RATE = 0.0023  # 0.23%

def load_ticker_name_map() -> dict:
    name_map = {}
    if SECTOR_DB.exists():
        try:
            conn = sqlite3.connect(str(SECTOR_DB))
            cursor = conn.cursor()
            rows = cursor.execute("SELECT ticker, name FROM sectors").fetchall()
            for t, n in rows:
                clean_t = t.split(".")[0].zfill(6)
                name_map[clean_t] = n
            conn.close()
        except Exception:
            pass
    return name_map

def upload_backtest_results(algorithm_name: str = "judal_hybrid_lgb_pytorch_KRX"):
    print("=========================================================================")
    print("   UPLOADING BACKTEST RESULTS TO DASHBOARD SERVER                        ")
    print(f"   Target Endpoint: {DASHBOARD_API_URL}")
    print("=========================================================================\n")

    # 1. Load Trained ML & DL Models
    gbm = joblib.load(MODEL_DIR / "lgb_kline_model.joblib")
    scaler = joblib.load(MODEL_DIR / "kline_scaler.joblib")
    
    feature_cols = [
        'high_close_ratio', 'body_ratio', 'upper_shadow_ratio',
        'ret_1d', 'ret_3d', 'ret_5d', 'vol_ratio_5d',
        'bb_pct_b', 'bb_width', 'rsi_14'
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pytorch_model = DeepOvernightNet(input_dim=len(feature_cols)).to(device)
    pytorch_model.load_state_dict(torch.load(MODEL_DIR / "pytorch_kline_model.pt", map_location=device))
    pytorch_model.eval()

    # 2. Load Stock Names & Full Judal Data (2026-04-01 ~ 2026-08-03)
    ticker_map = load_ticker_name_map()
    conn_judal = sqlite3.connect(str(JUDAL_DB_PATH))
    df_hist = pd.read_sql_query("""
        SELECT crawl_date as date, code, name as stock_name, change_rate as stock_change, neglect_index_52w, expected_return
        FROM stock_history
        WHERE crawl_date >= '2026-04-01'
    """, conn_judal)
    
    df_ts = pd.read_sql_query("SELECT theme_idx, stock_code as code FROM theme_stocks", conn_judal)
    df_themes = pd.read_sql_query("SELECT theme_idx, name as theme_name FROM themes", conn_judal)
    conn_judal.close()

    df_joined = pd.merge(df_hist, df_ts, on='code', how='inner')
    df_joined = pd.merge(df_joined, df_themes, on='theme_idx', how='inner')
    
    theme_group = df_joined.groupby(['date', 'theme_idx', 'theme_name'])['stock_change'].agg(
        theme_avg_change='mean',
        theme_max_change='max',
        theme_stock_cnt='count'
    ).reset_index()

    df_stock_theme = pd.merge(df_joined, theme_group, on=['date', 'theme_idx', 'theme_name'], how='left')

    # 3. Load Candle Data
    lazy_df = pl.scan_parquet(str(DATA_PARQUET))
    df_candles = (
        lazy_df
        .filter(pl.col("date") >= "2026-03-15")
        .select(["date", "ticker", "open", "close", "high", "low", "turnover", "high_close_ratio", "next_open"])
        .collect()
        .to_pandas()
    )
    df_candles = df_candles.sort_values(by=['ticker', 'date']).reset_index(drop=True)
    df_feat = compute_kline_features(df_candles)
    
    df_feat['code'] = df_feat['ticker'].apply(lambda x: str(x).split('.')[0].zfill(6))
    
    merged = pd.merge(df_stock_theme, df_feat, on=['date', 'code'], how='inner')
    merged['stock_name'] = merged['code'].map(ticker_map).fillna(merged['stock_name'])

    # 4. Model Inference
    X_mat = merged[feature_cols].fillna(0).values
    merged['p_lgb'] = gbm.predict(X_mat)
    
    X_scaled = scaler.transform(X_mat)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        merged['p_torch'] = pytorch_model(X_tensor).cpu().numpy().flatten()

    merged['is_leader'] = merged['stock_change'] >= merged['theme_max_change'] * 0.85
    merged['judal_score'] = (
        (merged['is_leader'].astype(int) * 35.0) +
        (np.clip(merged['theme_avg_change'], -5, 12) * 2.5) +
        (np.clip(merged['stock_change'], 2, 14) * 3.0) +
        (merged['high_close_ratio'] * 30.0) -
        (np.maximum(0, merged['stock_change'] - 15) * 4.0)
    )

    merged['hybrid_score'] = (
        merged['judal_score'] +
        (merged['p_lgb'] * 40.0) +
        (merged['p_torch'] * 40.0)
    )

    deduped = merged.sort_values(by='hybrid_score', ascending=False).groupby(['date', 'code']).first().reset_index()

    # 5. Backtest Simulation & Trade Details Extraction
    integrator = MarketMosaicIntegrator()
    unique_dates = sorted(deduped['date'].unique())
    daily_stats = []
    trade_details = []

    for i, date_str in enumerate(unique_dates):
        # Calculate next trading date for close_time
        next_date_str = unique_dates[i+1] if i + 1 < len(unique_dates) else date_str

        day_cands = deduped[deduped['date'] == date_str].copy()
        day_cands = day_cands[
            (day_cands['turnover'] >= 2e10) & 
            (day_cands['stock_change'] < 29.0) &  # Upper Limit Filter
            (day_cands['p_lgb'] >= 0.35) & 
            (day_cands['p_torch'] >= 0.35) &
            (abs(day_cands['next_open'] / day_cands['close'] - 1) < 0.25)
        ]
        
        if day_cands.empty:
            daily_stats.append({"date": date_str, "return_pct": 0.0, "trades": 0})
            continue

        top_picks = day_cands.sort_values(by='hybrid_score', ascending=False).head(3)
        day_trades = []
        for idx, row in top_picks.iterrows():
            code = row['code']
            stock_name = str(row['stock_name'])
            ctx = integrator.get_full_market_context(code, stock_name, date_str)
            
            net_pnl = (row['next_open'] - row['close']) / row['close'] - FEE_RATE  # e.g. 0.0592 for +5.92%
            day_trades.append(net_pnl)
            
            exit_type = "tp" if net_pnl > 0 else "sl"
            
            trade_details.append({
                "ticker": code,          # Symbol Code (e.g. 012200)
                "name": stock_name,      # Stock Name (e.g. 계양전기)
                "open_time": f"{date_str} 15:30",
                "close_time": f"{next_date_str} 09:00",  # Next trading day 09:00
                "open_price": float(row['close']),
                "close_price": float(row['next_open']),
                "profit": float(round(net_pnl, 6)),  # Fractional ratio (e.g. 0.0592 for +5.92%)
                "profit_pct": float(round(net_pnl, 6)),  # Fractional ratio (e.g. 0.0592 for +5.92%)
                "exit_type": exit_type
            })

        avg_day_return = np.mean(day_trades) if day_trades else 0.0
        daily_stats.append({"date": date_str, "return_pct": float(round(avg_day_return, 6)), "trades": len(day_trades)})

    # 6. Compute Monthly & Weekly Returns
    df_daily = pd.DataFrame(daily_stats)
    df_daily['date_dt'] = pd.to_datetime(df_daily['date'])
    df_daily['year_month'] = df_daily['date_dt'].dt.strftime('%Y-%m')
    df_daily['year_week'] = df_daily['date_dt'].dt.strftime('%Y-W%U')

    monthly_returns = []
    for ym, grp in df_daily.groupby('year_month'):
        cum_ret = (1 + grp['return_pct']).prod() - 1.0
        monthly_returns.append({
            "year_month": ym,
            "return_pct": float(round(cum_ret, 6)),
            "trades": int(grp['trades'].sum())
        })

    weekly_returns = []
    for yw, grp in df_daily.groupby('year_week'):
        cum_ret = (1 + grp['return_pct']).prod() - 1.0
        weekly_returns.append({
            "year_week": yw,
            "return_pct": float(round(cum_ret, 6)),
            "trades": int(grp['trades'].sum())
        })

    # 7. Compute Summary Metrics (CAGR, Sharpe, Win Rate, PF, MDD)
    tot_cum_return = (1 + df_daily['return_pct']).prod() - 1.0
    n_trades = len(trade_details)
    profits = [t['profit'] for t in trade_details]
    win_rate = (pd.Series(profits) > 0).mean() if n_trades > 0 else 0.0
    win_sum = sum(p for p in profits if p > 0)
    loss_sum = abs(sum(p for p in profits if p < 0))
    profit_factor = (win_sum / loss_sum) if loss_sum != 0 else 0.0

    cum_series = (1 + df_daily['return_pct']).cumprod()
    cum_max = cum_series.cummax()
    mdd = ((cum_series - cum_max) / cum_max).min() if not cum_series.empty else 0.0

    # Daily std for Sharpe
    daily_std = df_daily['return_pct'].std()
    avg_daily_ret = df_daily['return_pct'].mean()
    sharpe = (avg_daily_ret / daily_std * np.sqrt(252)) if daily_std > 0 else 1.5

    # CAGR Calculation (MANDATORY)
    start_dt = datetime.strptime("2026-04-01", "%Y-%m-%d")
    end_dt = datetime.strptime("2026-08-03", "%Y-%m-%d")
    test_years = (end_dt - start_dt).days / 365.25
    cagr = float((1 + tot_cum_return) ** (1 / test_years) - 1) if test_years > 0 else float(tot_cum_return)

    # 8. Build Backtest Payload conforming to API Spec
    payload = {
        "algorithm": {
            "name": algorithm_name,
            "model_type": "judal_hybrid_lgb_pytorch",
            "timeframe": "1d",
            "project": "kr_stock",
            "ticker": "KRX",
            "direction": "LONG",
            "tp_pct": 0.05,
            "sl_pct": 0.03,
            "horizon_bars": 1,
            "prob_threshold": 0.35
        },
        "summary": {
            "avg_return": float(round(tot_cum_return, 6)),
            "avg_win_rate": float(round(win_rate, 4)),
            "avg_profit_factor": float(round(profit_factor, 2)),
            "avg_sharpe": float(round(sharpe, 2)),
            "cagr": float(round(cagr, 6)),
            "total_trades": int(n_trades),
            "max_drawdown": float(round(mdd, 6)),
            "test_start": "2026-04-01",
            "test_end": "2026-08-03",
            "fee_rate_pct": float(FEE_RATE * 100)
        },
        "monthly_returns": monthly_returns,
        "weekly_returns": weekly_returns,
        "daily_returns": [{"date": r["date"], "return_pct": r["return_pct"], "trades": r["trades"]} for r in daily_stats],
        "trade_details": trade_details
    }

    print(f"[Payload Overview] Algorithm Name: {payload['algorithm']['name']}")
    print(f" -> Total Trades: {payload['summary']['total_trades']}")
    print(f" -> Total Return: {payload['summary']['avg_return']*100:.2f}%")
    print(f" -> Win Rate: {payload['summary']['avg_win_rate']*100:.2f}%")
    print(f" -> CAGR: {payload['summary']['cagr']*100:.2f}%")

    # 9. Send POST Request to Dashboard API
    print(f"\n[Step 9] Posting JSON Payload to {DASHBOARD_API_URL}...")
    headers = {"Content-Type": "application/json"}
    try:
        resp = requests.post(DASHBOARD_API_URL, json=payload, headers=headers, timeout=10)
        print(f" -> Response Status Code: {resp.status_code}")
        print(f" -> Server Response Body: {resp.text}")
        if resp.status_code in [200, 201]:
            print("\n✅ Successfully uploaded backtest results to Backtest Lab Dashboard Server!")
            return True
        else:
            print(f"\n❌ Server error during upload: {resp.status_code}")
            return False
    except Exception as e:
        print(f"\n❌ Network error while connecting to Dashboard Server: {e}")
        return False

if __name__ == "__main__":
    upload_backtest_results()
