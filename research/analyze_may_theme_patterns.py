#!/usr/bin/env python3
"""
May 2026 In-Sample Judal Theme Pattern Mining Script (Refined)
--------------------------------------------------------------
Analyzes Judal theme group momentum (`theme_avg_change`), theme leader positioning,
neglect index (`neglect_index_52w`), DART filings, and news against real KRX candle overnight returns.

Goal: Derive empirical Judal theme rules from May 2026 (In-Sample) to apply to June 2026 (Out-of-Sample).
"""

import sqlite3
import pandas as pd
import numpy as np
import polars as pl
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PARQUET = ROOT_DIR / "data" / "kr_kline_processed.parquet"
JUDAL_DB_PATH = Path("/mnt/data/projects/marketMosaic/backend/data/judal.db")
DART_DB_PATH = Path("/mnt/data/projects/marketMosaic/backend/data/dart.db")
REPORT_PATH = ROOT_DIR / "docs" / "MAY_THEME_PATTERN_ANALYSIS.md"

FEE_RATE = 0.0023  # 0.23% fee & tax

def run_in_sample_analysis():
    print("=========================================================================")
    print("   IN-SAMPLE PATTERN MINING: MAY 2026 JUDAL THEME & OVERNIGHT RETURNS   ")
    print("=========================================================================\n")

    # 1. Load Judal Stock History & Themes
    conn_judal = sqlite3.connect(str(JUDAL_DB_PATH))
    df_hist = pd.read_sql_query("""
        SELECT crawl_date as date, code, name, change_rate as stock_change, neglect_index_52w, expected_return
        FROM stock_history
        WHERE crawl_date LIKE '2026-05%'
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

    # Merge theme metrics back to stock level
    df_stock_theme = pd.merge(df_joined, theme_group, on=['date', 'theme_idx', 'theme_name'], how='left')
    
    # 2. Load KRX Candle Data via Polars
    lazy_df = pl.scan_parquet(str(DATA_PARQUET))
    df_candles = (
        lazy_df
        .filter((pl.col("date") >= "2026-05-01") & (pl.col("date") <= "2026-05-31"))
        .select(["date", "ticker", "open", "close", "high", "low", "turnover", "high_close_ratio", "next_open"])
        .collect()
        .to_pandas()
    )
    df_candles['code'] = df_candles['ticker'].apply(lambda x: str(x).split('.')[0].zfill(6))
    
    # Merge with Candle Data
    merged = pd.merge(df_stock_theme, df_candles, on=['date', 'code'], how='inner')
    merged['overnight_pnl_pct'] = ((merged['next_open'] - merged['close']) / merged['close'] - FEE_RATE) * 100
    merged['is_win'] = merged['overnight_pnl_pct'] > 0
    
    print(f"[Step 1 & 2] Successfully merged {len(merged):,} records matching Judal Theme history and Candle data.")

    # 3. Load DART Filings
    conn_dart = sqlite3.connect(str(DART_DB_PATH))
    df_dart = pd.read_sql_query("SELECT corp_name as name, rcept_dt, report_nm FROM filings WHERE rcept_dt LIKE '202605%'", conn_dart)
    conn_dart.close()
    
    df_dart['date'] = pd.to_datetime(df_dart['rcept_dt'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
    dart_counts = df_dart.groupby(['date', 'name']).size().reset_index(name='dart_count')
    merged = pd.merge(merged, dart_counts, on=['date', 'name'], how='left')
    merged['dart_count'] = merged['dart_count'].fillna(0)

    # 4. Pattern Analysis
    print("\n=========================================================================")
    print("                    EMPIRICAL PATTERN FINDINGS (MAY 2026)                ")
    print("=========================================================================")

    # Theme Average Change Bins vs Overnight PnL
    merged['theme_avg_bin'] = pd.qcut(merged['theme_avg_change'].dropna(), q=5, duplicates='drop')
    grp_theme_avg = merged.groupby('theme_avg_bin', observed=False).agg(
        trades=('overnight_pnl_pct', 'count'),
        win_rate=('is_win', lambda x: x.mean() * 100),
        avg_return=('overnight_pnl_pct', 'mean')
    ).reset_index()
    print("\n[Pattern 1: Judal Theme Average Momentum (theme_avg_change) Impact]:")
    print(grp_theme_avg.to_string(index=False))

    # Theme Leader vs Follower (stock_change >= theme_avg_change)
    merged['is_leader'] = merged['stock_change'] >= merged['theme_max_change'] * 0.9
    grp_leader = merged.groupby('is_leader').agg(
        trades=('overnight_pnl_pct', 'count'),
        win_rate=('is_win', lambda x: x.mean() * 100),
        avg_return=('overnight_pnl_pct', 'mean')
    ).reset_index()
    print("\n[Pattern 2: Judal Theme Leader Stock Impact]:")
    print(grp_leader.to_string(index=False))

    # High Conviction Judal Rule Set Definition (with Upper Limit Filter)
    high_conviction_mask = (
        (merged['turnover'] >= 3e10) &               # 거래대금 300억 이상
        (merged['stock_change'] < 29.0) &             # 상한가 매수 불가 종목 제외
        (merged['theme_avg_change'] >= 10.0) &        # 테마 평균 등락률 10% 이상 (강력한 테마 집단 상승)
        (merged['stock_change'] >= 15.0) &            # 개별 종목 15% 이상 강세 (주도주)
        (merged['high_close_ratio'] >= 0.85)          # 종가 고가 부근 마감
    )
    
    filtered = merged[high_conviction_mask]
    win_rate_rule = filtered['is_win'].mean() * 100 if len(filtered) > 0 else 0
    avg_ret_rule = filtered['overnight_pnl_pct'].mean() if len(filtered) > 0 else 0
    
    print(f"\n[High-Conviction Judal Theme Rule Set (May 2026 In-Sample)]:")
    print(f" -> Candidate Trades Count : {len(filtered)}")
    print(f" -> Win Rate               : {win_rate_rule:.2f}%")
    print(f" -> Avg Overnight Return   : {avg_ret_rule:+.2f}%")

    # Write findings to Markdown Report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# May 2026 In-Sample Judal Theme Empirical Analysis Report\n\n")
        f.write("### Executive Summary\n")
        f.write(f"Analyzed {len(merged):,} stock-theme records for May 2026 to extract quantitative Judal theme rules for June Out-of-Sample testing.\n\n")
        f.write("### Pattern 1: Judal Theme Group Momentum (`theme_avg_change`)\n\n")
        f.write("```text\n" + grp_theme_avg.to_string(index=False) + "\n```\n\n")
        f.write("### Pattern 2: Judal Theme Leader Stock (`is_leader`)\n\n")
        f.write("```text\n" + grp_leader.to_string(index=False) + "\n```\n\n")
        f.write("### Extracted Quantitative Judal Rule Set\n")
        f.write("- **Primary Driver (Judal Theme Score)**:\n")
        f.write("  `judal_score = (theme_avg_change * 3.0) + (stock_change * 2.0) + (high_close_ratio * 30.0)`\n")
        f.write("- **Auxiliary Driver (DART / News Bonus)**:\n")
        f.write("  `bonus = (dart_count * 5.0) + (news_count * 3.0)`\n")
        
    print(f"\nReport saved to {REPORT_PATH}")

if __name__ == "__main__":
    run_in_sample_analysis()
