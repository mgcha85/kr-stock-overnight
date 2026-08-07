#!/usr/bin/env python3
"""
June 2026 Out-of-Sample Walk-Forward Judal Theme Strategy Script (Deduplicated)
--------------------------------------------------------------------------------
Applies the empirical Judal Theme-First Strategy learned from May 2026 (In-Sample)
to June 2026 (Out-of-Sample: 2026-06-01 ~ 2026-06-30).

Strategy Architecture:
1. Primary Driver: Judal Theme Group Momentum & Leader Positioning (judal.db).
2. Auxiliary Driver: DART Filings (dart.db) & MarketMosaic News (Meilisearch) as bonus multipliers.
3. Deduplication: One trade per stock per day.
4. Friction: 0.23% transaction tax & fees.
"""

import sqlite3
import pandas as pd
import numpy as np
import polars as pl
from pathlib import Path
from research.marketmosaic_integrator import MarketMosaicIntegrator

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PARQUET = ROOT_DIR / "data" / "kr_kline_processed.parquet"
JUDAL_DB_PATH = Path("/mnt/data/projects/marketMosaic/backend/data/judal.db")
SECTOR_DB = Path("/mnt/data/finance/candles/KO/sector_info.db")
REPORT_DOC = ROOT_DIR / "docs" / "JUNE_OOS_JUDAL_THEME_STRATEGY_REPORT.md"

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

def run_june_oos_backtest():
    print("=========================================================================")
    print(f"   JUNE 2026 OUT-OF-SAMPLE BACKTEST (JUDAL THEME-FIRST WALK-FORWARD)     ")
    print(f"   Test Period: {START_DATE} ~ {END_DATE}                                ")
    print("=========================================================================\n")

    # Step 1: Load Stock Name Mapping
    ticker_map = load_ticker_name_map()
    print(f"[Step 1] Loaded {len(ticker_map)} stock name mappings.")

    # Step 2: Load Judal Theme History for June 2026
    conn_judal = sqlite3.connect(str(JUDAL_DB_PATH))
    df_hist = pd.read_sql_query("""
        SELECT crawl_date as date, code, name, change_rate as stock_change, neglect_index_52w, expected_return
        FROM stock_history
        WHERE crawl_date LIKE '2026-06%'
    """, conn_judal)
    
    df_ts = pd.read_sql_query("SELECT theme_idx, stock_code as code FROM theme_stocks", conn_judal)
    df_themes = pd.read_sql_query("SELECT theme_idx, name as theme_name FROM themes", conn_judal)
    conn_judal.close()

    # Calculate Theme Group Momentum
    df_joined = pd.merge(df_hist, df_ts, on='code', how='inner')
    df_joined = pd.merge(df_joined, df_themes, on='theme_idx', how='inner')
    
    theme_group = df_joined.groupby(['date', 'theme_idx', 'theme_name'])['stock_change'].agg(
        theme_avg_change='mean',
        theme_max_change='max',
        theme_stock_cnt='count'
    ).reset_index()

    df_stock_theme = pd.merge(df_joined, theme_group, on=['date', 'theme_idx', 'theme_name'], how='left')

    # Step 3: Load Candle Data via Polars
    lazy_df = pl.scan_parquet(str(DATA_PARQUET))
    df_candles = (
        lazy_df
        .filter((pl.col("date") >= START_DATE) & (pl.col("date") <= END_DATE))
        .select(["date", "ticker", "open", "close", "high", "low", "turnover", "high_close_ratio", "next_open"])
        .collect()
        .to_pandas()
    )
    df_candles['code'] = df_candles['ticker'].apply(lambda x: str(x).split('.')[0].zfill(6))
    
    # Merge Judal Theme History & Candles
    merged = pd.merge(df_stock_theme, df_candles, on=['date', 'code'], how='inner')
    merged['stock_name'] = merged['code'].map(ticker_map).fillna(merged['name'])
    
    print(f"[Step 2 & 3] Merged {len(merged):,} June stock-theme candle records.")

    # Step 4: MarketMosaic Integrator (Auxiliary DART & News Bonus)
    print("\n[Step 4] Integrating Auxiliary DART & News Bonus Signals...")
    integrator = MarketMosaicIntegrator()

    merged['is_leader'] = merged['stock_change'] >= merged['theme_max_change'] * 0.85
    merged['judal_theme_score'] = (
        (merged['is_leader'].astype(int) * 35.0) +
        (np.clip(merged['theme_avg_change'], -5, 12) * 2.5) +
        (np.clip(merged['stock_change'], 2, 14) * 3.0) +
        (merged['high_close_ratio'] * 30.0) -
        (np.maximum(0, merged['stock_change'] - 15) * 4.0)
    )

    # Deduplicate per stock per date (take highest scoring theme per stock)
    deduped = merged.sort_values(by='judal_theme_score', ascending=False).groupby(['date', 'code']).first().reset_index()

    unique_dates = sorted(deduped['date'].unique())
    daily_stats = []
    trade_details = []

    print("\n[Step 5] Running Out-of-Sample Trade Execution for June 2026...")
    for date_str in unique_dates:
        day_cands = deduped[deduped['date'] == date_str].copy()
        if day_cands.empty:
            daily_stats.append({"date": date_str, "return": 0.0, "trades": 0})
            continue

        # Liquidity & Price sanity filter
        day_cands = day_cands[(day_cands['turnover'] >= 2e10) & (abs(day_cands['next_open'] / day_cands['close'] - 1) < 0.25)]
        if day_cands.empty:
            daily_stats.append({"date": date_str, "return": 0.0, "trades": 0})
            continue

        # Sort by Judal Theme Score & select Top 3
        top_picks = day_cands.sort_values(by='judal_theme_score', ascending=False).head(3)
        
        day_trades = []
        for idx, row in top_picks.iterrows():
            code = row['code']
            stock_name = row['stock_name']
            
            # Fetch Auxiliary DART & News context
            ctx = integrator.get_full_market_context(code, stock_name, date_str)
            dart_cnt = len(ctx['dart_filings'])
            news_cnt = len(ctx['news_articles'])
            
            aux_bonus = (dart_cnt * 5.0) + (news_cnt * 3.0)
            final_score = row['judal_theme_score'] + aux_bonus
            
            gross_pnl = (row['next_open'] - row['close']) / row['close']
            net_pnl = gross_pnl - FEE_RATE
            day_trades.append(net_pnl)
            
            trade_details.append({
                "date": date_str,
                "ticker": code,
                "stock_name": stock_name,
                "theme_name": row['theme_name'],
                "entry_price": float(row['close']),
                "exit_price": float(row['next_open']),
                "judal_score": round(row['judal_theme_score'], 1),
                "aux_bonus": aux_bonus,
                "final_score": round(final_score, 1),
                "net_pnl_pct": round(net_pnl * 100, 2)
            })

        avg_day_return = np.mean(day_trades) if day_trades else 0.0
        daily_stats.append({"date": date_str, "return": avg_day_return, "trades": len(day_trades)})

    # Step 6: Compute KPI & B&H Benchmark
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

    # Calculate Buy & Hold Benchmark for June 2026
    bh_returns = deduped.groupby('date')['close'].mean().pct_change().fillna(0.0)
    bh_cum = (1 + bh_returns).cumprod()
    bh_total_return = (bh_cum.iloc[-1] - 1.0) * 100
    outperformance = total_strat_return - bh_total_return

    print("\n=========================================================================")
    print(" JUNE 2026 OUT-OF-SAMPLE JUDAL THEME-FIRST STRATEGY RESULTS              ")
    print("=========================================================================")
    print(f" Test Period                     : {START_DATE} ~ {END_DATE}")
    print(f" Total Trades Executed           : {n_trades} trades")
    print(f" Overnight Strategy Return       : {total_strat_return:+.2f}%")
    print(f" Buy & Hold Benchmark Return     : {bh_total_return:+.2f}%")
    print(f" Outperformance vs B&H (Alpha)   : {outperformance:+.2f}%")
    print(f" Win Rate                        : {win_rate:.2f}%")
    print(f" Profit Factor                   : {profit_factor:.2f}")
    print(f" Max Drawdown (MDD)              : {mdd:.2f}%")
    print("=========================================================================\n")

    print("Sample Trade Logs (First 5 Trades):")
    sample_df = pd.DataFrame(trade_details[:5])
    print(sample_df[['date', 'ticker', 'stock_name', 'theme_name', 'entry_price', 'exit_price', 'judal_score', 'net_pnl_pct']].to_string(index=False))

    # Write Markdown Report
    with open(REPORT_DOC, "w", encoding="utf-8") as f:
        f.write("# June 2026 Out-of-Sample Judal Theme-First Strategy Report\n\n")
        f.write(f"### Backtest Period: `{START_DATE} ~ {END_DATE}`\n\n")
        f.write("### Strategy Architecture\n")
        f.write("- **Primary Driver**: Judal Theme Group Momentum & Leader Positioning (`judal.db`).\n")
        f.write("- **Auxiliary Driver**: DART Filings (`dart.db`) and News Articles (Meilisearch) as bonus multipliers.\n")
        f.write("- **Walk-Forward Method**: In-Sample May 2026 Rule Derived -> Applied to June 2026 Out-of-Sample.\n")
        f.write("- **Friction & Time Contract**: Entry at 15:30 close, Exit at T+1 09:00 open, 0.23% fee & tax applied.\n\n")
        f.write("### Performance KPI Summary\n\n")
        f.write(f"| Metric | Strategy | Buy & Hold Benchmark | Delta (Alpha) |\n")
        f.write(f"| :--- | ---: | ---: | ---: |\n")
        f.write(f"| **June Return** | **{total_strat_return:+.2f}%** | {bh_total_return:+.2f}% | **{outperformance:+.2f}%** |\n")
        f.write(f"| **Win Rate** | **{win_rate:.2f}%** | N/A | N/A |\n")
        f.write(f"| **Profit Factor** | **{profit_factor:.2f}** | N/A | N/A |\n")
        f.write(f"| **Max Drawdown (MDD)** | **{mdd:.2f}%** | N/A | N/A |\n")
        f.write(f"| **Total Trades** | **{n_trades}** | N/A | N/A |\n\n")
        f.write("### Trade Execution Details\n\n")
        if trade_details:
            f.write(pd.DataFrame(trade_details).to_markdown(index=False) + "\n\n")

    print(f"\nReport saved to {REPORT_DOC}")

if __name__ == "__main__":
    run_june_oos_backtest()
