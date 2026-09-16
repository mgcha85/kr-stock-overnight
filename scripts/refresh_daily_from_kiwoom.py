#!/usr/bin/env python3
"""Replace one calendar day's OHLCV in parquet + day_data_full.db from Kiwoom ka10081."""
from __future__ import annotations

import datetime
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
import requests

ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
PARQUET_PATH = ROOT / "data" / "kr_kline_processed.parquet"
DB_PATH = Path("/mnt/data/finance/candles/KO/day_data_full.db")


def _load_env() -> None:
    for p in (ROOT / ".env.prod", ROOT / ".env.dev"):
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        break
    os.environ["KIWOOM_TOKEN_PATH"] = str(ROOT / "data" / "kiwoom_access_token.json")


def _parse_f(raw) -> float:
    if raw is None:
        return 0.0
    return float(str(raw).replace(",", "").replace("+", "").replace("-", "").strip() or 0)


def fetch_bar(session: requests.Session, token: str, domain: str, code: str, ymd: str) -> dict | None:
    r = session.post(
        f"{domain}/api/dostk/chart",
        json={"stk_cd": code, "upd_stkpc_tp": "1", "base_dt": ymd},
        headers={
            "authorization": f"Bearer {token}",
            "api-id": "ka10081",
            "cont-yn": "N",
            "next-key": "",
        },
        timeout=25,
    )
    data = r.json()
    rows = data.get("stk_dt_pole_chart_qry") or data.get("output") or []
    if not isinstance(rows, list):
        return None
    for item in rows:
        if str(item.get("dt") or "") == ymd:
            open_p = _parse_f(item.get("open_pric"))
            high_p = _parse_f(item.get("high_pric"))
            low_p = _parse_f(item.get("low_pric"))
            close_p = _parse_f(item.get("cur_prc"))
            vol = int(_parse_f(item.get("trde_qty")))
            prica = _parse_f(item.get("trde_prica"))  # million KRW
            if close_p <= 0:
                return None
            turnover = prica * 1e6 if prica > 0 else close_p * vol
            return {
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "volume": vol,
                "turnover": turnover,
            }
    return None


def upsert_db(code: str, market: str, date_str: str, bar: dict) -> None:
    if not DB_PATH.exists():
        return
    table = f"{code}.{market}"
    conn = sqlite3.connect(str(DB_PATH))
    try:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if table not in names:
            alt = f"{code}.KQ" if market == "KS" else f"{code}.KS"
            if alt not in names:
                conn.close()
                return
            table = alt
        conn.execute(
            f'''INSERT OR REPLACE INTO "{table}" (date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?)''',
            (date_str, bar["open"], bar["high"], bar["low"], bar["close"], bar["volume"]),
        )
        conn.commit()
    finally:
        conn.close()


def patch_next_open(date_str: str) -> None:
    """Point previous session next_open at this day's open; this day next_open at the following session."""
    df = pl.read_parquet(str(PARQUET_PATH))
    day = df.filter(pl.col("date") == date_str)
    if day.is_empty():
        return
    prev_dates = (
        df.filter(pl.col("date") < date_str)
        .group_by("ticker")
        .agg(pl.col("date").max().alias("prev_date"))
    )
    open_map = day.select(["ticker", "open", "open_time"]).rename(
        {"open": "new_next_open", "open_time": "new_next_open_time"}
    )
    df = (
        df.join(prev_dates, on="ticker", how="left")
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
    later = (
        df.filter(pl.col("date") > date_str)
        .group_by("ticker")
        .agg(pl.col("date").min().alias("next_date"))
    )
    nxt = df.join(later, on="ticker", how="left")
    nxt_open = (
        df.select(["ticker", "date", "open", "open_time"])
        .rename({"date": "next_date", "open": "fwd_open", "open_time": "fwd_open_time"})
    )
    df = (
        nxt.join(nxt_open, on=["ticker", "next_date"], how="left")
        .with_columns(
            [
                pl.when(pl.col("date") == date_str)
                .then(pl.col("fwd_open"))
                .otherwise(pl.col("next_open"))
                .alias("next_open"),
                pl.when(pl.col("date") == date_str)
                .then(pl.col("fwd_open_time"))
                .otherwise(pl.col("next_open_time"))
                .alias("next_open_time"),
            ]
        )
        .drop(["next_date", "fwd_open", "fwd_open_time"])
    )
    df.write_parquet(str(PARQUET_PATH))


def listing_day(src: pl.DataFrame, date_str: str) -> pl.DataFrame:
    """Universe from the latest parquet day at or before target, else global max."""
    le = src.filter(pl.col("date") <= date_str)
    if le.height:
        d = le.select(pl.col("date").max()).item()
    else:
        d = src.select(pl.col("date").max()).item()
    return src.filter(pl.col("date") == d)


def parquet_count(src: pl.DataFrame, date_str: str) -> int:
    return int(src.filter(pl.col("date") == date_str).height)


def sync_one_date(
    date_str: str,
    *,
    workers: int,
    min_ok: int,
    token: str,
    domain: str,
    force: bool = False,
) -> int:
    src = pl.read_parquet(str(PARQUET_PATH))
    existing = parquet_count(src, date_str)
    if existing >= min_ok and not force:
        print(f"[kiwoom-daily] skip {date_str}: already {existing} rows")
        return existing

    uni_src = listing_day(src, date_str)
    universe = (
        uni_src.select(["ticker", "market"])
        .with_columns(
            pl.col("ticker").cast(pl.Utf8).str.split(".").list.first().str.zfill(6).alias("ticker")
        )
        .unique(subset=["ticker"])
    )
    codes = universe["ticker"].to_list()
    markets = dict(zip(universe["ticker"].to_list(), universe["market"].to_list()))
    ymd = date_str.replace("-", "")
    print(f"[kiwoom-daily] {date_str} universe={len(codes)} workers={workers}", flush=True)

    import threading

    tls = threading.local()

    def one(code: str):
        if not hasattr(tls, "s"):
            tls.s = requests.Session()
        bar = None
        for attempt in range(4):
            bar = fetch_bar(tls.s, token, domain, code, ymd)
            if bar:
                break
            time.sleep(0.4 * (attempt + 1))
            time.sleep(0.2)
        return code, bar

    ok = 0
    fail = 0
    bars: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, c) for c in codes]
        for i, fut in enumerate(as_completed(futs), 1):
            code, bar = fut.result()
            if bar:
                bars[code] = bar
                ok += 1
            else:
                fail += 1
            if i % 200 == 0:
                print(f"  {i}/{len(codes)} ok={ok} fail={fail}", flush=True)

    print(f"[kiwoom-daily] fetched ok={ok} fail={fail}")
    if ok < min_ok:
        raise SystemExit(
            f"abort write: ok={ok} < min-ok={min_ok} (will not overwrite {date_str})"
        )

    open_t = datetime.datetime.strptime(f"{date_str} 09:00:00", "%Y-%m-%d %H:%M:%S")
    close_t = datetime.datetime.strptime(f"{date_str} 15:30:00", "%Y-%m-%d %H:%M:%S")
    records = []
    for code, bar in bars.items():
        market = str(markets.get(code) or "KS")
        hl = bar["high"] - bar["low"]
        records.append(
            {
                "date": date_str,
                "open": bar["open"],
                "high": bar["high"],
                "low": bar["low"],
                "close": bar["close"],
                "volume": bar["volume"],
                "open_time": open_t,
                "close_time": close_t,
                "ticker": code,
                "market": market,
                "full_symbol": f"{code}.{market}",
                "turnover": bar["turnover"],
                "high_close_ratio": ((bar["close"] - bar["low"]) / hl) if hl > 0 else 1.0,
                "sma_5": None,
                "sma_20": None,
                "next_open": None,
                "next_open_time": None,
            }
        )
        upsert_db(code, market, date_str, bar)

    today_df = pl.DataFrame(records)
    hist = src.filter(pl.col("date") != date_str)
    out = pl.concat([hist, today_df], how="diagonal_relaxed")
    out.write_parquet(str(PARQUET_PATH))
    patch_next_open(date_str)
    print(f"[kiwoom-daily] replaced {date_str} with {today_df.height} Kiwoom bars")
    return int(today_df.height)


def session_dates(start: str, end: str) -> list[str]:
    from datetime import timedelta

    from kr_stock.config import PAPER_DB_PATH
    from kr_stock.krx_calendar import is_session_day

    db = str(PAPER_DB_PATH)
    out: list[str] = []
    cur = datetime.datetime.strptime(start, "%Y-%m-%d").date()
    last = datetime.datetime.strptime(end, "%Y-%m-%d").date()
    while cur <= last:
        d = cur.strftime("%Y-%m-%d")
        if is_session_day(db, d):
            out.append(d)
        cur += timedelta(days=1)
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="YYYY-MM-DD (single day)")
    parser.add_argument("--from-date", help="Inclusive start YYYY-MM-DD")
    parser.add_argument("--to-date", help="Inclusive end YYYY-MM-DD (default: today)")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--min-ok", type=int, default=2500, help="Abort write if fewer bars")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    _load_env()
    from kr_stock.kiwoom_live import request_access_token, _app_domain

    today = datetime.date.today().strftime("%Y-%m-%d")
    src = pl.read_parquet(str(PARQUET_PATH))
    max_d = src.select(pl.col("date").max()).item()
    if args.date:
        dates = [args.date]
    else:
        start = args.from_date
        if not start:
            nxt = datetime.datetime.strptime(max_d, "%Y-%m-%d").date() + datetime.timedelta(days=1)
            start = nxt.strftime("%Y-%m-%d")
        end = args.to_date or today
        dates = session_dates(start, end)
        print(f"[kiwoom-daily] parquet_max={max_d} backfill {start}..{end} -> {dates}")
        if not dates:
            print("[kiwoom-daily] nothing to fetch")
            return

    token = request_access_token()
    domain = _app_domain()
    for d in dates:
        sync_one_date(
            d,
            workers=args.workers,
            min_ok=args.min_ok,
            token=token,
            domain=domain,
            force=args.force,
        )


if __name__ == "__main__":
    main()
