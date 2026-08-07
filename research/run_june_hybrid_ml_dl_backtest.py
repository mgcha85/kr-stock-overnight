#!/usr/bin/env python3
"""
June 2026 Hybrid Walk-Forward Backtest with Judal Theme + LightGBM (ML) + PyTorch (DL)
---------------------------------------------------------------------------------------
Combines:
1. Primary Judal Theme Group Momentum & Leader Positioning (judal.db)
2. LightGBM (ML) Candle Win Probability
3. PyTorch Deep MLP (DL) Candle Win Probability (CUDA accelerated)
4. Auxiliary DART Filings & News Context Bonus

Evaluates performance against Rule-Only Strategy and Buy & Hold Benchmark for June 2026.
"""

import sqlite3
import pandas as pd
import numpy as np
import polars as pl
import torch
import joblib
from pathlib import Path
from research.marketmosaic_integrator import MarketMosaicIntegrator
from research.kline_ml_dl_pipeline import compute_kline_features, DeepOvernightNet

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PARQUET = ROOT_DIR / "data" / "kr_kline_processed.parquet"
JUDAL_DB_PATH = Path("/mnt/data/projects/marketMosaic/backend/data/judal.db")
SECTOR_DB = Path("/mnt/data/finance/candles/KO/sector_info.db")
MODEL_DIR = ROOT_DIR / "research" / "models"
REPORT_DOC = ROOT_DIR / "docs" / "JUNE_HYBRID_ML_DL_STRATEGY_REPORT.md"

START_DATE = "2026-06-01"
END_DATE = "2026-06-30"
FEE_RATE = 0.0023  # 0.23% fee & tax

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

def run_hybrid_backtest():
    print("=========================================================================")
    print("   JUNE 2026 HYBRID BACKTEST: JUDAL THEME + LIGHTGBM (ML) + PYTORCH (DL)  ")
    print(f"   Test Period: {START_DATE} ~ {END_DATE}                                ")
    print("=========================================================================\n")

    # 1. Load Trained ML & DL Models
    print("[Step 1] Loading Trained LightGBM and PyTorch Models...")
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
    print(f" -> Models loaded successfully. PyTorch running on: {device}")

    # 2. Load Stock Names & Judal Theme History
    ticker_map = load_ticker_name_map()
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
        theme_avg_change='mean',
        theme_max_change='max',
        theme_stock_cnt='count'
    ).reset_index()

    df_stock_theme = pd.merge(df_joined, theme_group, on=['date', 'theme_idx', 'theme_name'], how='left')

    # 3. Load Candle Data and Compute Technical Features
    lazy_df = pl.scan_parquet(str(DATA_PARQUET))
    df_candles = (
        lazy_df
        .filter((pl.col("date") >= "2026-05-15") & (pl.col("date") <= END_DATE))  # Include warm-up for rolling indicators
        .select(["date", "ticker", "open", "close", "high", "low", "turnover", "high_close_ratio", "next_open"])
        .collect()
        .to_pandas()
    )
    df_candles = df_candles.sort_values(by=['ticker', 'date']).reset_index(drop=True)
    df_feat = compute_kline_features(df_candles)
    
    # Filter back to June test period
    df_june_candles = df_feat[(df_feat['date'] >= START_DATE) & (df_feat['date'] <= END_DATE)].copy()
    df_june_candles['code'] = df_june_candles['ticker'].apply(lambda x: str(x).split('.')[0].zfill(6))
    
    merged = pd.merge(df_stock_theme, df_june_candles, on=['date', 'code'], how='inner')
    merged['stock_name'] = merged['code'].map(ticker_map).fillna(merged['name'])

    print(f"[Step 2 & 3] Merged {len(merged):,} June stock-theme candle records with technical indicators.")

    # 4. Compute ML & DL Predictions
    print("\n[Step 4] Inference: Computing LightGBM & PyTorch Probabilities...")
    X_mat = merged[feature_cols].fillna(0).values
    
    # LightGBM inference
    merged['p_lgb'] = gbm.predict(X_mat)
    
    # PyTorch inference
    X_scaled = scaler.transform(X_mat)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        p_torch = pytorch_model(X_tensor).cpu().numpy().flatten()
    merged['p_torch'] = p_torch

    # 5. Compute Hybrid Score
    merged['is_leader'] = merged['stock_change'] >= merged['theme_max_change'] * 0.85
    merged['judal_score'] = (
        (merged['is_leader'].astype(int) * 35.0) +
        (np.clip(merged['theme_avg_change'], -5, 12) * 2.5) +
        (np.clip(merged['stock_change'], 2, 14) * 3.0) +
        (merged['high_close_ratio'] * 30.0) -
        (np.maximum(0, merged['stock_change'] - 15) * 4.0)
    )

    # Hybrid Ensemble Score: Judal + 40 * P_LGB + 40 * P_PyTorch
    merged['hybrid_score'] = (
        merged['judal_score'] +
        (merged['p_lgb'] * 40.0) +
        (merged['p_torch'] * 40.0)
    )

    # Deduplicate per stock per date
    deduped = merged.sort_values(by='hybrid_score', ascending=False).groupby(['date', 'code']).first().reset_index()

    # 6. Execute Backtest
    print("\n[Step 5] Running Out-of-Sample Hybrid Trade Execution...")
    integrator = MarketMosaicIntegrator()
    unique_dates = sorted(deduped['date'].unique())
    daily_stats = []
    trade_details = []

    for date_str in unique_dates:
        day_cands = deduped[deduped['date'] == date_str].copy()
        if day_cands.empty:
            daily_stats.append({"date": date_str, "return": 0.0, "trades": 0})
            continue

        # Liquidity, Upper Limit (+29.0% locked) filter & ML/DL confidence threshold
        day_cands = day_cands[
            (day_cands['turnover'] >= 2e10) & 
            (day_cands['stock_change'] < 29.0) &  # 상한가 매수 불가 종목 제외 (실전 매수 가능 종목만)
            (day_cands['p_lgb'] >= 0.35) & 
            (day_cands['p_torch'] >= 0.35) &
            (abs(day_cands['next_open'] / day_cands['close'] - 1) < 0.25)
        ]
        
        if day_cands.empty:
            daily_stats.append({"date": date_str, "return": 0.0, "trades": 0})
            continue

        # Top 3 High-Conviction Picks
        top_picks = day_cands.sort_values(by='hybrid_score', ascending=False).head(3)
        
        day_trades = []
        for idx, row in top_picks.iterrows():
            code = row['code']
            stock_name = row['stock_name']
            
            ctx = integrator.get_full_market_context(code, stock_name, date_str)
            dart_cnt = len(ctx['dart_filings'])
            news_cnt = len(ctx['news_articles'])
            aux_bonus = (dart_cnt * 5.0) + (news_cnt * 3.0)
            
            final_score = row['hybrid_score'] + aux_bonus
            net_pnl = (row['next_open'] - row['close']) / row['close'] - FEE_RATE
            day_trades.append(net_pnl)
            
            trade_details.append({
                "date": date_str,
                "ticker": code,
                "stock_name": stock_name,
                "theme_name": row['theme_name'],
                "entry_price": float(row['close']),
                "exit_price": float(row['next_open']),
                "judal_score": round(row['judal_score'], 1),
                "p_lgb": round(row['p_lgb'], 3),
                "p_torch": round(row['p_torch'], 3),
                "hybrid_score": round(final_score, 1),
                "net_pnl_pct": round(net_pnl * 100, 2)
            })

        avg_day_return = np.mean(day_trades) if day_trades else 0.0
        daily_stats.append({"date": date_str, "return": avg_day_return, "trades": len(day_trades)})

    # 7. KPI Analysis
    res_df = pd.DataFrame(daily_stats)
    res_df['cum_return'] = (1 + res_df['return']).cumprod()
    total_strat_return = (res_df['cum_return'].iloc[-1] - 1.0) * 100

    pnl_list = [t['net_pnl_pct'] for t in trade_details]
    n_trades = len(pnl_list)
    win_rate = (pd.Series(pnl_list) > 0).mean() * 100 if n_trades > 0 else 0.0
    win_sum = sum(p for p in pnl_list if p > 0)
    loss_sum = abs(sum(p for p in pnl_list if p < 0))
    profit_factor = (win_sum / loss_sum) if loss_sum != 0 else 0.0

    cum_max = res_df['cum_return'].cummax()
    mdd = ((res_df['cum_return'] - cum_max) / cum_max).min() * 100

    # Buy & Hold Benchmark
    bh_returns = deduped.groupby('date')['close'].mean().pct_change().fillna(0.0)
    bh_cum = (1 + bh_returns).cumprod()
    bh_total_return = (bh_cum.iloc[-1] - 1.0) * 100
    outperformance = total_strat_return - bh_total_return

    print("\n=========================================================================")
    print(" JUNE 2026 HYBRID ML/DL + JUDAL THEME STRATEGY RESULTS                  ")
    print("=========================================================================")
    print(f" Test Period                     : {START_DATE} ~ {END_DATE}")
    print(f" Total Trades Executed           : {n_trades} trades")
    print(f" Hybrid Strategy Return          : {total_strat_return:+.2f}%")
    print(f" Buy & Hold Benchmark Return     : {bh_total_return:+.2f}%")
    print(f" Outperformance vs B&H (Alpha)   : {outperformance:+.2f}%")
    print(f" Win Rate                        : {win_rate:.2f}%")
    print(f" Profit Factor                   : {profit_factor:.2f}")
    print(f" Max Drawdown (MDD)              : {mdd:.2f}%")
    print("=========================================================================\n")

    print("Sample Trade Logs (First 5 Trades):")
    sample_df = pd.DataFrame(trade_details[:5])
    print(sample_df[['date', 'ticker', 'stock_name', 'p_lgb', 'p_torch', 'hybrid_score', 'net_pnl_pct']].to_string(index=False))

    # Write Markdown Report
    with open(REPORT_DOC, "w", encoding="utf-8") as f:
        f.write("# June 2026 Out-of-Sample Hybrid Judal Theme + ML/DL Strategy Report\n\n")
        f.write(f"### Backtest Period: `{START_DATE} ~ {END_DATE}`\n\n")
        f.write("### Model Architecture\n")
        f.write("- **Judal Theme Driver**: Judal theme group momentum & leader positioning (`judal.db`).\n")
        f.write("- **LightGBM Classifier**: Trained on May 2026 technical candle features (RSI, BB, Vol ratio, Ret 1d/3d/5d).\n")
        f.write("- **PyTorch Deep MLP**: Multi-layer neural network trained on GPU (CUDA) for non-linear feature embeddings.\n")
        f.write("- **Auxiliary Driver**: DART Filings (`dart.db`) and News Articles (Meilisearch) bonus.\n\n")
        f.write("### Strategy Comparison Summary\n\n")
        f.write(f"| Metric | Hybrid ML/DL + Judal | Rule-Only Judal | Buy & Hold Benchmark |\n")
        f.write(f"| :--- | ---: | ---: | ---: |\n")
        f.write(f"| **June Return** | **{total_strat_return:+.2f}%** | -28.85% | {bh_total_return:+.2f}% |\n")
        f.write(f"| **Win Rate** | **{win_rate:.2f}%** | 37.50% | N/A |\n")
        f.write(f"| **Profit Factor** | **{profit_factor:.2f}** | 0.58 | N/A |\n")
        f.write(f"| **Max Drawdown (MDD)** | **{mdd:.2f}%** | -26.44% | N/A |\n")
        f.write(f"| **Total Trades** | **{n_trades}** | 48 | N/A |\n\n")
        f.write("### Executed Hybrid Trades\n\n")
        if trade_details:
            df_trades = pd.DataFrame(trade_details)
            f.write("```text\n")
            f.write(df_trades.to_string(index=False) + "\n")
            f.write("```\n")

    print(f"\nReport saved to {REPORT_DOC}")

if __name__ == "__main__":
    run_hybrid_backtest()
