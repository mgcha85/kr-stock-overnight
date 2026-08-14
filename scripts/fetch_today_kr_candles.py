#!/usr/bin/env python3
"""
Fetch Today's Daily Candles for All KRX Stocks
----------------------------------------------
Fetches today's OHLCV via FinanceDataReader, upserts day_data_full.db,
and incrementally updates kr_kline_processed.parquet (no full rebuild).
"""

from __future__ import annotations

import datetime
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import polars as pl

DB_PATH = Path("/mnt/data/finance/candles/KO/day_data_full.db")
PARQUET_PATH = Path(__file__).resolve().parent.parent / "data" / "kr_kline_processed.parquet"


def _require_fdr():
    try:
        import FinanceDataReader as fdr  # noqa: F401
        return fdr
    except ImportError as e:
        raise ImportError(
            "FinanceDataReader is required for candle/open-price sync. "
            "Install: uv add finance-datareader"
        ) from e


def parquet_row_count(target_date_str: str) -> int:
    if not PARQUET_PATH.exists():
        return 0
    n = (
        pl.scan_parquet(str(PARQUET_PATH))
        .filter(pl.col("date") == target_date_str)
        .select(pl.len())
        .collect()
        .item()
    )
    return int(n)


def fetch_krx_listing() -> pd.DataFrame:
    fdr = _require_fdr()
    df = fdr.StockListing("KRX")
    if df is None or df.empty:
        raise RuntimeError("FinanceDataReader StockListing('KRX') returned empty")
    return df


def fetch_open_prices(tickers: Iterable[str], target_date_str: Optional[str] = None) -> Dict[str, float]:
    """Today's open from KRX listing. Used at 09:00 SELL — never invent buy_price."""
    wanted = {str(t).split(".")[0].zfill(6) for t in tickers}
    df = fetch_krx_listing()
    out: Dict[str, float] = {}
    for _, row in df.iterrows():
        code = str(row["Code"]).zfill(6)
        if code not in wanted:
            continue
        open_p = float(row.get("Open", 0.0) or 0.0)
        if open_p > 0:
            out[code] = open_p
    return out


def fetch_and_update_today(target_date_str: Optional[str] = None) -> pd.DataFrame:
    if target_date_str is None:
        target_date_str = datetime.date.today().strftime("%Y-%m-%d")

    print("=========================================================================")
    print(f" FETCHING TODAY'S ({target_date_str}) CANDLES FOR ALL KRX STOCKS        ")
    print("=========================================================================")

    df_listing = fetch_krx_listing()
    print(f"[1/3] Retrieved {len(df_listing):,} tickers from KRX.")

    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    existing_tables = {
        row[0]
        for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    }
    print(f" -> Found {len(existing_tables):,} existing stock tables in {DB_PATH}")

    cursor.execute("BEGIN TRANSACTION;")
    updated_count = 0
    print("[2/3] Updating today's candles into day_data_full.db...")
    for _, row in df_listing.iterrows():
        code = str(row["Code"]).zfill(6)
        market = str(row.get("Market", ""))
        table_name = f"{code}.KQ" if "KOSDAQ" in market else f"{code}.KS"
        if table_name not in existing_tables:
            alt_table = f"{code}.KS" if table_name.endswith(".KQ") else f"{code}.KQ"
            if alt_table in existing_tables:
                table_name = alt_table
            else:
                continue

        open_p = float(row.get("Open", 0.0) or 0.0)
        high_p = float(row.get("High", 0.0) or 0.0)
        low_p = float(row.get("Low", 0.0) or 0.0)
        close_p = float(row.get("Close", 0.0) or 0.0)
        volume_v = float(row.get("Volume", 0.0) or 0.0)
        if close_p <= 0 or volume_v < 0:
            continue

        cursor.execute(
            f'''INSERT OR REPLACE INTO "{table_name}" (date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?)''',
            (target_date_str, open_p, high_p, low_p, close_p, volume_v),
        )
        updated_count += 1

    conn.commit()
    conn.close()
    print(f"[3/3] Upserted {updated_count:,} daily bars for {target_date_str}.")
    if updated_count == 0:
        raise RuntimeError(f"No KRX bars written for {target_date_str} (Open/Close empty?)")
    return df_listing


def upsert_parquet_from_listing(df_listing: pd.DataFrame, target_date_str: str) -> int:
    """Replace target_date rows in parquet from KRX listing. Fast incremental path."""
    records: List[dict] = []
    for _, row in df_listing.iterrows():
        code = str(row["Code"]).zfill(6)
        market_raw = str(row.get("Market", ""))
        market = "KQ" if "KOSDAQ" in market_raw else "KS"
        open_p = float(row.get("Open", 0.0) or 0.0)
        high_p = float(row.get("High", 0.0) or 0.0)
        low_p = float(row.get("Low", 0.0) or 0.0)
        close_p = float(row.get("Close", 0.0) or 0.0)
        volume_v = int(float(row.get("Volume", 0.0) or 0.0))
        if close_p <= 0:
            continue
        hl = high_p - low_p
        records.append(
            {
                "date": target_date_str,
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "volume": volume_v,
                "open_time": pd.Timestamp(f"{target_date_str} 09:00:00"),
                "close_time": pd.Timestamp(f"{target_date_str} 15:30:00"),
                "ticker": code,
                "market": market,
                "full_symbol": f"{code}.{market}",
                "turnover": close_p * volume_v,
                "high_close_ratio": ((close_p - low_p) / hl) if hl > 0 else 1.0,
                "sma_5": None,
                "sma_20": None,
                "next_open": None,
                "next_open_time": None,
            }
        )

    if not records:
        raise RuntimeError(f"No valid listing rows to write into parquet for {target_date_str}")

    today_df = pl.from_pandas(pd.DataFrame(records)).with_columns(
        [
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Int64),
            pl.col("turnover").cast(pl.Float64),
            pl.col("high_close_ratio").cast(pl.Float64),
            pl.col("sma_5").cast(pl.Float64),
            pl.col("sma_20").cast(pl.Float64),
            pl.col("next_open").cast(pl.Float64),
        ]
    )
    if PARQUET_PATH.exists():
        hist = pl.read_parquet(str(PARQUET_PATH)).filter(pl.col("date") != target_date_str)
        # Point previous session next_open at today's open (overnight exit).
        prev_date = (
            hist.filter(pl.col("ticker").is_in(today_df["ticker"].to_list()))
            .group_by("ticker")
            .agg(pl.col("date").max().alias("prev_date"))
        )
        open_map = today_df.select(["ticker", "open", "open_time"]).rename(
            {"open": "new_next_open", "open_time": "new_next_open_time"}
        )
        hist = (
            hist.join(prev_date, on="ticker", how="left")
            .join(open_map, on="ticker", how="left")
            .with_columns(
                [
                    pl.when(pl.col("date") == pl.col("prev_date"))
                    .then(pl.col("new_next_open"))
                    .otherwise(pl.col("next_open"))
                    .alias("next_open"),
                    pl.when(pl.col("date") == pl.col("prev_date"))
                    .then(pl.col("new_next_open_time"))
                    .otherwise(pl.col("next_open_time"))
                    .alias("next_open_time"),
                ]
            )
            .drop(["prev_date", "new_next_open", "new_next_open_time"])
        )
        out = pl.concat([hist, today_df], how="diagonal_relaxed").with_columns(
            [
                pl.col("next_open").cast(pl.Float64, strict=False),
                pl.col("close").cast(pl.Float64, strict=False),
                pl.col("open").cast(pl.Float64, strict=False),
            ]
        )
    else:
        out = today_df

    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(str(PARQUET_PATH))
    n = today_df.height
    print(f"[Candle Sync] parquet upserted {n:,} rows for {target_date_str} -> {PARQUET_PATH}")
    return n


def ensure_today_candles_updated(target_date_str: Optional[str] = None, force: bool = True) -> int:
    if target_date_str is None:
        target_date_str = datetime.date.today().strftime("%Y-%m-%d")

    df_listing = fetch_and_update_today(target_date_str)
    if not force:
        existing = parquet_row_count(target_date_str)
        if existing > 0:
            print(f"[Candle Sync] parquet already has {existing} rows for {target_date_str}")
            return existing
    return upsert_parquet_from_listing(df_listing, target_date_str)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch today's candles for KRX stocks")
    parser.add_argument("--date", type=str, default=None, help="Target date YYYY-MM-DD")
    args, unknown = parser.parse_known_args()
    target_date = args.date
    if not target_date and unknown:
        target_date = unknown[0]
    n = ensure_today_candles_updated(target_date, force=True)
    print(f"Done. parquet rows for date: {n}")
