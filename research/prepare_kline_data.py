#!/usr/bin/env python3
"""
Kline Data Processor
--------------------
Loads daily candle data for Korean stocks from /mnt/data/finance/candles/KO/day_data_full.db,
transforms date into explicit open_time and close_time columns, computes liquidity & technical
indicators, and saves processed data into project root 'data/' directory.
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_DB = Path("/mnt/data/finance/candles/KO/day_data_full.db")
OUTPUT_PARQUET = DATA_DIR / "kr_kline_processed.parquet"
OUTPUT_DB = DATA_DIR / "kr_kline_processed.db"

def process_kline_data():
    print(f"[1/4] Connecting to source database: {SOURCE_DB}")
    if not SOURCE_DB.exists():
        raise FileNotFoundError(f"Source database not found at {SOURCE_DB}")
    
    conn = sqlite3.connect(SOURCE_DB)
    tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
    print(f"Total stock tables found: {len(tables)}")
    
    all_frames = []
    
    # Process top 300 liquid stocks or all available stocks for backtesting efficiency
    print("[2/4] Reading and transforming stock candles with explicit open_time and close_time...")
    for idx, table_name in enumerate(tables):
        if idx % 300 == 0:
            print(f"  - Processing stock {idx}/{len(tables)}: {table_name}")
            
        df = pd.read_sql_query(f'SELECT * FROM "{table_name}" ORDER BY date ASC', conn)
        if df.empty or len(df) < 50:
            continue
            
        ticker = table_name.split('.')[0]
        market = table_name.split('.')[1] if '.' in table_name else 'KR'
        
        # Clean date & create explicit open_time and close_time
        df['date'] = df['date'].astype(str)
        df['open_time'] = pd.to_datetime(df['date'] + " 09:00:00")
        df['close_time'] = pd.to_datetime(df['date'] + " 15:30:00")
        
        df['ticker'] = ticker
        df['market'] = market
        df['full_symbol'] = table_name
        
        # Compute daily indicators
        df['turnover'] = df['close'] * df['volume']
        df['high_close_ratio'] = np.where(
            (df['high'] - df['low']) > 0,
            (df['close'] - df['low']) / (df['high'] - df['low']),
            1.0
        )
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        
        # Next open price (for overnight trade exit calculation: buy T close -> sell T+1 open)
        df['next_open'] = df['open'].shift(-1)
        df['next_open_time'] = df['open_time'].shift(-1)
        
        all_frames.append(df)
        
    conn.close()
    
    print("[3/4] Concatenating processed dataset...")
    full_df = pd.concat(all_frames, ignore_index=True)
    
    # Filter valid range (2021-01-01 ~ 2026-07-30)
    full_df = full_df[full_df['date'] >= '2021-01-01'].copy()
    
    print(f"Total candle records processed: {len(full_df):,}")
    print(f"Date range: {full_df['date'].min()} ~ {full_df['date'].max()}")
    print("Columns available:")
    print(" -> ", list(full_df.columns))
    
    print(f"[4/4] Saving processed data to {OUTPUT_PARQUET} and {OUTPUT_DB}...")
    full_df.to_parquet(OUTPUT_PARQUET, index=False)
    
    # Also store in SQLite for quick querying
    out_conn = sqlite3.connect(OUTPUT_DB)
    full_df.to_sql('kline_daily', out_conn, if_exists='replace', index=False)
    out_conn.execute("CREATE INDEX IF NOT EXISTS idx_kline_ticker_date ON kline_daily(ticker, date);")
    out_conn.execute("CREATE INDEX IF NOT EXISTS idx_kline_close_time ON kline_daily(close_time);")
    out_conn.close()
    
    print("[SUCCESS] Kline data successfully processed and saved into 'data/' directory!")

if __name__ == "__main__":
    process_kline_data()
