"""Overnight sell-time study on 1-minute bars (09:00–09:30).

Trade set matches the validated rolling backtest: Judal + kline features,
turnover / +29% / p_lgb / p_torch filters, Top-3 per buy date.
Sell price at HH:MM is that minute's **open** (executable at that clock, no look-ahead).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import polars as pl
import sqlite3
import torch
import joblib

from kr_stock.config import DATA_PARQUET_PATH, FEE_RATE, JUDAL_DB_PATH, MODEL_DIR
from kr_stock.inference import DAILY_FEATURE_COLS, compute_kline_features, load_ticker_name_map
from research.kline_ml_dl_pipeline import DeepOvernightNet

INTRA_1M_DIR = Path("/mnt/data/finance/candles/KO/interval=1m")
SELL_MINUTES = [f"09:{m:02d}" for m in range(0, 31)]


@dataclass
class OvernightTrade:
    buy_date: str
    sell_date: str
    code: str
    stock_name: str
    buy_close: float
    daily_next_open: float


def _zfill6(code: str) -> str:
    return "".join(ch for ch in str(code) if ch.isdigit())[-6:].zfill(6)


def list_1m_dates() -> List[str]:
    dates = []
    for p in INTRA_1M_DIR.glob("year=*/month=*/data_*.parquet"):
        stem = p.stem  # data_YYYYMMDD
        raw = stem.replace("data_", "")
        if len(raw) == 8 and raw.isdigit():
            dates.append(f"{raw[:4]}-{raw[4:6]}-{raw[6:]}")
    return sorted(set(dates))


def _1m_path(date_str: str) -> Path:
    y, m, d = date_str.split("-")
    return INTRA_1M_DIR / f"year={int(y)}" / f"month={int(m)}" / f"data_{y}{m}{d}.parquet"


def next_trading_day(buy_date: str, calendar: Sequence[str]) -> Optional[str]:
    days = list(calendar)
    try:
        i = days.index(buy_date)
    except ValueError:
        later = [d for d in days if d > buy_date]
        return later[0] if later else None
    if i + 1 < len(days):
        return days[i + 1]
    return None


def _history_from_judal(start_date: str, end_date: str) -> pd.DataFrame:
    conn = sqlite3.connect(str(JUDAL_DB_PATH))
    df = pd.read_sql_query(
        """
        SELECT crawl_date as date, code, name, change_rate as stock_change
        FROM stock_history
        WHERE crawl_date >= ? AND crawl_date <= ?
        """,
        conn,
        params=[start_date, end_date],
    )
    conn.close()
    if df.empty:
        return df
    df["code"] = df["code"].map(_zfill6)
    return df


def _history_from_candles(start_date: str, end_date: str, ticker_map: dict) -> pd.DataFrame:
    """Rebuild date/code/change from daily close when Judal crawl is missing."""
    warmup = (pd.to_datetime(start_date) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    raw = (
        pl.scan_parquet(str(DATA_PARQUET_PATH))
        .filter((pl.col("date") >= warmup) & (pl.col("date") <= end_date))
        .select(["date", "ticker", "close"])
        .collect()
        .to_pandas()
    )
    if raw.empty:
        return pd.DataFrame(columns=["date", "code", "name", "stock_change"])
    raw["code"] = raw["ticker"].map(lambda x: _zfill6(str(x).split(".")[0]))
    raw = raw.sort_values(["code", "date"])
    raw["stock_change"] = raw.groupby("code")["close"].pct_change() * 100.0
    raw = raw[(raw["date"] >= start_date) & (raw["date"] <= end_date) & raw["stock_change"].notna()]
    raw["name"] = raw["code"].map(ticker_map).fillna(raw["code"])
    return raw[["date", "code", "name", "stock_change"]]


def build_overnight_trades(
    start_date: str = "2025-01-02",
    end_date: str = "2026-08-12",
    top_k: int = 3,
    min_turnover: float = 2e10,
    max_stock_change: float = 29.0,
    min_p_lgb: float = 0.35,
    min_p_torch: float = 0.35,
    use_session_1520: bool = False,
    model_tag: str = "",
    open_gap_min: Optional[float] = 0.02,
    open_gap_max: Optional[float] = 0.28,
    min_close: Optional[float] = None,
    max_close: Optional[float] = None,
    require_above_sma20: bool = False,
) -> List[OvernightTrade]:
    """Batch-score the rolling-backtest universe and emit Top-K overnight trades.

    I (close band) and M (close > SMA20) are off by default so HTS can drop them too.
    """
    ticker_map = load_ticker_name_map()
    conn = sqlite3.connect(str(JUDAL_DB_PATH))
    df_ts = pd.read_sql_query("SELECT theme_idx, stock_code as code FROM theme_stocks", conn)
    df_themes = pd.read_sql_query("SELECT theme_idx, name as theme_name FROM themes", conn)
    conn.close()

    warmup = (pd.to_datetime(start_date) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    if use_session_1520:
        from kr_stock.session_bars import load_session_1520

        candles = (
            load_session_1520()
            .filter((pl.col("date") >= warmup) & (pl.col("date") <= end_date))
            .to_pandas()
        )
        candles_hist = candles[["date", "ticker", "close"]].copy()
        candles_hist["code"] = candles_hist["ticker"].map(lambda x: _zfill6(str(x)))
        candles_hist = candles_hist.sort_values(["code", "date"])
        candles_hist["stock_change"] = candles_hist.groupby("code")["close"].pct_change() * 100.0
        candles_hist["name"] = candles_hist["code"].map(ticker_map).fillna(candles_hist["code"])
        df_hist = candles_hist[
            (candles_hist["date"] >= start_date)
            & (candles_hist["date"] <= end_date)
            & candles_hist["stock_change"].notna()
        ][["date", "code", "name", "stock_change"]]
    else:
        judal = _history_from_judal(start_date, end_date)
        candles_hist = _history_from_candles(start_date, end_date, ticker_map)
        if not judal.empty:
            extra = candles_hist[~candles_hist["date"].isin(set(judal["date"]))]
            df_hist = pd.concat([judal, extra], ignore_index=True)
        else:
            df_hist = candles_hist
        candles = (
            pl.scan_parquet(str(DATA_PARQUET_PATH))
            .filter((pl.col("date") >= warmup) & (pl.col("date") <= end_date))
            .select(
                ["date", "ticker", "open", "close", "high", "low", "turnover", "high_close_ratio", "next_open"]
            )
            .collect()
            .to_pandas()
        )
    if df_hist.empty:
        return []

    df_ts["code"] = df_ts["code"].map(_zfill6)
    df_joined = pd.merge(df_hist, df_ts, on="code", how="inner")
    df_joined = pd.merge(df_joined, df_themes, on="theme_idx", how="inner")
    theme_group = (
        df_joined.groupby(["date", "theme_idx", "theme_name"])["stock_change"]
        .agg(theme_avg_change="mean", theme_max_change="max")
        .reset_index()
    )
    df_stock_theme = pd.merge(
        df_joined, theme_group, on=["date", "theme_idx", "theme_name"], how="left"
    )

    candles = candles.sort_values(["ticker", "date"]).reset_index(drop=True)
    feat = compute_kline_features(candles)
    feat["code"] = feat["ticker"].map(lambda x: _zfill6(str(x).split(".")[0]))
    feat = feat.sort_values(["code", "date"])
    feat["prev_close"] = feat.groupby("code")["close"].shift(1)
    feat["open_gap"] = feat["open"] / feat["prev_close"] - 1.0
    feat["sma_20"] = feat.groupby("code")["close"].transform(
        lambda s: s.rolling(20, min_periods=20).mean()
    )
    feat = feat[(feat["date"] >= start_date) & (feat["date"] <= end_date)]

    merged = pd.merge(df_stock_theme, feat, on=["date", "code"], how="inner")
    if merged.empty:
        return []

    tag = model_tag
    gbm = joblib.load(MODEL_DIR / f"lgb_kline{tag}_model.joblib")
    scaler = joblib.load(MODEL_DIR / f"kline_scaler{tag}.joblib")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DeepOvernightNet(input_dim=len(DAILY_FEATURE_COLS)).to(device)
    net.load_state_dict(torch.load(MODEL_DIR / f"pytorch_kline{tag}_model.pt", map_location=device))
    net.eval()

    x = merged[DAILY_FEATURE_COLS].fillna(0).values
    merged["p_lgb"] = gbm.predict(x)
    with torch.no_grad():
        merged["p_torch"] = (
            net(torch.tensor(scaler.transform(x), dtype=torch.float32).to(device))
            .cpu()
            .numpy()
            .flatten()
        )

    merged["is_leader"] = merged["stock_change"] >= merged["theme_max_change"] * 0.85
    merged["judal_score"] = (
        (merged["is_leader"].astype(int) * 35.0)
        + (np.clip(merged["theme_avg_change"], -5, 12) * 2.5)
        + (np.clip(merged["stock_change"], 2, 14) * 3.0)
        + (merged["high_close_ratio"] * 30.0)
        - (np.maximum(0, merged["stock_change"] - 15) * 4.0)
    )
    merged["hybrid_score"] = merged["judal_score"] + merged["p_lgb"] * 40.0 + merged["p_torch"] * 40.0
    deduped = (
        merged.sort_values("hybrid_score", ascending=False)
        .groupby(["date", "code"], as_index=False)
        .first()
    )
    filtered = deduped[
        (deduped["turnover"] >= min_turnover)
        & (deduped["stock_change"] < max_stock_change)
        & (deduped["p_lgb"] >= min_p_lgb)
        & (deduped["p_torch"] >= min_p_torch)
    ].copy()
    if open_gap_min is not None or open_gap_max is not None:
        lo = -np.inf if open_gap_min is None else open_gap_min
        hi = np.inf if open_gap_max is None else open_gap_max
        filtered = filtered[
            filtered["open_gap"].notna()
            & (filtered["open_gap"] >= lo)
            & (filtered["open_gap"] <= hi)
        ]
    if min_close is not None:
        filtered = filtered[filtered["close"] >= min_close]
    if max_close is not None:
        filtered = filtered[filtered["close"] <= max_close]
    if require_above_sma20:
        # Drop only when SMA20 is known and close is at/below it. NaN SMA = incomplete
        # parquet history, not an HTS fail.
        filtered = filtered[filtered["sma_20"].isna() | (filtered["close"] > filtered["sma_20"])]

    daily_cal = sorted(feat["date"].unique().tolist())
    trades: List[OvernightTrade] = []
    for buy_date, day in filtered.groupby("date"):
        later = [d for d in daily_cal if d > str(buy_date)]
        if not later:
            continue
        sell_date = later[0]
        top = day.sort_values("hybrid_score", ascending=False).head(top_k)
        for _, row in top.iterrows():
            close_px = float(row["close"])
            if close_px <= 0:
                continue
            nxt = float(row["next_open"]) if pd.notna(row.get("next_open")) else 0.0
            code = _zfill6(row["code"])
            trades.append(
                OvernightTrade(
                    buy_date=str(buy_date),
                    sell_date=sell_date,
                    code=code,
                    stock_name=ticker_map.get(code, str(row.get("name", code))),
                    buy_close=close_px,
                    daily_next_open=nxt,
                )
            )
    return trades


def load_minute_prices(
    date_str: str,
    codes: Sequence[str],
    hour: int,
    minute_min: int,
    minute_max: int,
    field: str = "open",
) -> pd.DataFrame:
    """1m prices in [hour:minute_min, hour:minute_max] for codes on date_str."""
    path = _1m_path(date_str)
    if not path.exists():
        return pd.DataFrame(columns=["code", "hhmm", "price"])
    want = [_zfill6(c) for c in codes]
    col = pl.col(field).cast(pl.Float64).alias("price")
    df = (
        pl.scan_parquet(str(path))
        .with_columns(pl.col("code").cast(pl.Utf8).str.slice(-6).str.pad_start(6, "0"))
        .filter(pl.col("code").is_in(want))
        .filter(pl.col("datetime").dt.hour() == hour)
        .filter(pl.col("datetime").dt.minute() >= minute_min)
        .filter(pl.col("datetime").dt.minute() <= minute_max)
        .select(
            [
                pl.col("code"),
                pl.col("datetime").dt.strftime("%H:%M").alias("hhmm"),
                col,
            ]
        )
        .collect()
        .to_pandas()
    )
    return df.drop_duplicates(["code", "hhmm"])


def load_buy_1520_prices(buy_date: str, codes: Sequence[str]) -> Dict[str, float]:
    """Last continuous print at/before 15:19 (1m has no 15:20–15:29 auction bars)."""
    df = load_minute_prices(buy_date, codes, hour=15, minute_min=0, minute_max=19, field="close")
    if df.empty:
        return {}
    last = df.sort_values("hhmm").groupby("code", as_index=False).last()
    return {r.code: float(r.price) for r in last.itertuples(index=False) if r.price > 0}


def load_sell_0900_prices(sell_date: str, codes: Sequence[str]) -> Dict[str, float]:
    df = load_minute_prices(sell_date, codes, hour=9, minute_min=0, minute_max=0, field="open")
    return {r.code: float(r.price) for r in df.itertuples(index=False) if r.price > 0}


def load_morning_opens(sell_date: str, codes: Sequence[str]) -> pd.DataFrame:
    path = _1m_path(sell_date)
    if not path.exists():
        return pd.DataFrame(columns=["code", "hhmm", "sell_open"])
    want = [_zfill6(c) for c in codes]
    df = (
        pl.scan_parquet(str(path))
        .with_columns(pl.col("code").cast(pl.Utf8).str.slice(-6).str.pad_start(6, "0"))
        .filter(pl.col("code").is_in(want))
        .filter(pl.col("datetime").dt.hour() == 9)
        .filter(pl.col("datetime").dt.minute() <= 30)
        .select(
            [
                pl.col("code"),
                pl.col("datetime").dt.strftime("%H:%M").alias("hhmm"),
                pl.col("open").cast(pl.Float64).alias("sell_open"),
            ]
        )
        .collect()
        .to_pandas()
    )
    return df.drop_duplicates(["code", "hhmm"])


def evaluate_sell_minutes(
    trades: Sequence[OvernightTrade], fee_rate: float = FEE_RATE
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per-minute summary, per-trade long table)."""
    if not trades:
        return pd.DataFrame(), pd.DataFrame()

    by_sell: Dict[str, List[OvernightTrade]] = {}
    for t in trades:
        by_sell.setdefault(t.sell_date, []).append(t)

    rows = []
    for sell_date, group in by_sell.items():
        morning = load_morning_opens(sell_date, [t.code for t in group])
        if morning.empty:
            continue
        price = {
            (r.code, r.hhmm): float(r.sell_open)
            for r in morning.itertuples(index=False)
        }
        for t in group:
            for hhmm in SELL_MINUTES:
                px = price.get((t.code, hhmm))
                if px is None or px <= 0:
                    continue
                net = px / t.buy_close - 1.0 - fee_rate
                rows.append(
                    {
                        "buy_date": t.buy_date,
                        "sell_date": t.sell_date,
                        "code": t.code,
                        "stock_name": t.stock_name,
                        "hhmm": hhmm,
                        "buy_close": t.buy_close,
                        "sell_open": px,
                        "daily_next_open": t.daily_next_open,
                        "net_pnl": net,
                    }
                )

    long_df = pd.DataFrame(rows)
    if long_df.empty:
        return pd.DataFrame(), long_df

    summaries = []
    baseline = long_df[long_df["hhmm"] == "09:00"][["buy_date", "code", "net_pnl"]].rename(
        columns={"net_pnl": "pnl_0900"}
    )
    for hhmm, g in long_df.groupby("hhmm"):
        pnls = g["net_pnl"]
        pos = pnls[pnls > 0].sum()
        neg = (-pnls[pnls < 0]).sum()
        # equal-weight daily mean, then compound
        daily = g.groupby("buy_date")["net_pnl"].mean().sort_index()
        equity = (1.0 + daily).cumprod()
        total = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0
        merged = g.merge(baseline, on=["buy_date", "code"], how="inner")
        vs_open = float((merged["net_pnl"] - merged["pnl_0900"]).mean()) if not merged.empty else 0.0
        summaries.append(
            {
                "hhmm": hhmm,
                "trades": int(len(g)),
                "mean_pnl_pct": float(pnls.mean() * 100),
                "median_pnl_pct": float(pnls.median() * 100),
                "win_rate": float((pnls > 0).mean() * 100),
                "pf": float(pos / neg) if neg > 0 else 0.0,
                "compound_pct": total * 100,
                "vs_0900_mean_bp": vs_open * 10000,
                "days": int(daily.shape[0]),
            }
        )
    summary = pd.DataFrame(summaries).sort_values("hhmm").reset_index(drop=True)
    return summary, long_df


def pick_best(summary: pd.DataFrame) -> Dict[str, object]:
    if summary.empty:
        return {}
    by_mean = summary.sort_values("mean_pnl_pct", ascending=False).iloc[0]
    by_pf = summary.sort_values("pf", ascending=False).iloc[0]
    by_compound = summary.sort_values("compound_pct", ascending=False).iloc[0]
    open_row = summary[summary["hhmm"] == "09:00"]
    return {
        "best_mean": by_mean.to_dict(),
        "best_pf": by_pf.to_dict(),
        "best_compound": by_compound.to_dict(),
        "open_0900": open_row.iloc[0].to_dict() if not open_row.empty else {},
    }
