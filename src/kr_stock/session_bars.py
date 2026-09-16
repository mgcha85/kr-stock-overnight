"""Build 15:19 session bars and next-day 09:00 opens from 1-minute tape."""
from __future__ import annotations

from pathlib import Path
from typing import List

import polars as pl

from kr_stock.config import DATA_DIR
from kr_stock.sell_timing import INTRA_1M_DIR

SESSION_1520_PARQUET = DATA_DIR / "kr_kline_1520.parquet"


def _1m_files() -> List[Path]:
    return sorted(INTRA_1M_DIR.glob("year=*/month=*/data_*.parquet"))


def _agg_file(path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    raw = (
        pl.read_parquet(str(path))
        .with_columns(
            [
                pl.col("code").cast(pl.Utf8).str.slice(-6).str.pad_start(6, "0"),
                pl.col("datetime").dt.strftime("%Y-%m-%d").alias("date"),
                pl.col("datetime").dt.hour().alias("hh"),
                pl.col("datetime").dt.minute().alias("mm"),
            ]
        )
    )
    sess = raw.filter(
        (pl.col("hh") < 15) | ((pl.col("hh") == 15) & (pl.col("mm") <= 19))
    ).sort(["code", "datetime"])
    bars = sess.group_by("code").agg(
        [
            pl.col("date").first(),
            pl.col("open").first().cast(pl.Float64),
            pl.col("high").max().cast(pl.Float64),
            pl.col("low").min().cast(pl.Float64),
            pl.col("close").last().cast(pl.Float64),
            pl.col("volume").sum().cast(pl.Float64),
        ]
    )
    open0900 = (
        raw.filter((pl.col("hh") == 9) & (pl.col("mm") == 0))
        .select(
            [
                pl.col("code"),
                pl.col("date"),
                pl.col("open").cast(pl.Float64).alias("open_0900"),
            ]
        )
        .unique(subset=["code", "date"])
    )
    return bars, open0900


def build_session_1520_parquet(force: bool = False) -> Path:
    if SESSION_1520_PARQUET.exists() and not force:
        return SESSION_1520_PARQUET

    files = _1m_files()
    if not files:
        raise FileNotFoundError(f"no 1m files under {INTRA_1M_DIR}")

    bar_parts: List[pl.DataFrame] = []
    open_parts: List[pl.DataFrame] = []
    for i, path in enumerate(files, 1):
        bars, opens = _agg_file(path)
        bar_parts.append(bars)
        open_parts.append(opens)
        if i % 50 == 0 or i == len(files):
            print(f"  [1520 bars] {i}/{len(files)} days")

    bars = pl.concat(bar_parts, how="vertical_relaxed")
    opens = pl.concat(open_parts, how="vertical_relaxed")
    bars = bars.join(opens, on=["code", "date"], how="left")
    bars = bars.sort(["code", "date"])
    bars = bars.with_columns(
        [
            pl.col("open_0900").shift(-1).over("code").alias("next_open"),
            (pl.col("volume") * pl.col("close")).alias("turnover"),
            pl.col("code").alias("ticker"),
        ]
    )
    SESSION_1520_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    bars.write_parquet(str(SESSION_1520_PARQUET))
    print(f"Wrote {SESSION_1520_PARQUET} rows={bars.height} dates={bars['date'].n_unique()}")
    return SESSION_1520_PARQUET


def load_session_1520() -> pl.DataFrame:
    if not SESSION_1520_PARQUET.exists():
        build_session_1520_parquet()
    return pl.read_parquet(str(SESSION_1520_PARQUET))
