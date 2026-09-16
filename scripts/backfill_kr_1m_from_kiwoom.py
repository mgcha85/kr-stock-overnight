#!/usr/bin/env python3
"""Download missing KRX 1-minute parquets via Kiwoom ka10080 into the finance hive tree."""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import polars as pl
import requests

ROOT = Path("/mnt/data/projects/kr_stock")
OUT_ROOT = Path("/mnt/data/finance/candles/KO/interval=1m")


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


def hive_path(ymd: str) -> Path:
    year, month = ymd[:4], str(int(ymd[4:6]))
    return OUT_ROOT / f"year={year}" / f"month={month}" / f"data_{ymd}.parquet"


def last_codes() -> list[str]:
    files = sorted(OUT_ROOT.glob("year=*/month=*/data_20*.parquet"))
    if not files:
        return []
    return (
        pl.scan_parquet(str(files[-1]))
        .select("code")
        .unique()
        .collect()["code"]
        .to_list()
    )


def parse_bar(item: dict, code: str) -> dict | None:
    tm = str(item.get("cntr_tm") or "")
    if len(tm) < 14:
        return None
    try:
        t = dt.datetime.strptime(tm[:14], "%Y%m%d%H%M%S")
    except ValueError:
        return None

    def num(k):
        return int(float(str(item.get(k) or "0").replace(",", "").replace("+", "").replace("-", "") or 0))

    return {
        "code": code,
        "datetime": t,
        "open": num("open_pric"),
        "high": num("high_pric"),
        "low": num("low_pric"),
        "close": num("cur_prc"),
        "volume": int(num("trde_qty")),
        "year": t.year,
        "month": t.month,
    }


def fetch_day(session, token, domain, code: str, ymd: str) -> list[dict]:
    rows = []
    next_key = ""
    for _ in range(8):
        cont = "N" if not next_key else "Y"
        r = session.post(
            f"{domain}/api/dostk/chart",
            json={"stk_cd": code, "tic_scope": "1", "upd_stkpc_tp": "0", "base_dt": ymd},
            headers={
                "authorization": f"Bearer {token}",
                "api-id": "ka10080",
                "cont-yn": cont,
                "next-key": next_key,
            },
            timeout=30,
        )
        data = r.json()
        items = data.get("stk_min_pole_chart_qry") or data.get("output") or []
        if not isinstance(items, list) or not items:
            break
        for it in items:
            bar = parse_bar(it, code)
            if bar and bar["datetime"].strftime("%Y%m%d") == ymd:
                rows.append(bar)
        next_key = r.headers.get("next-key") or data.get("next_key") or ""
        # If the oldest item is still this date, continue; else stop.
        last_tm = str(items[-1].get("cntr_tm") or "")
        if last_tm[:8] != ymd or not next_key:
            break
    return rows


def main() -> None:
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="20260819")
    parser.add_argument("--end", default="20260901")
    args = parser.parse_args()

    _load_env()
    from kr_stock.kiwoom_live import request_access_token, _app_domain

    token = request_access_token()
    domain = _app_domain()
    codes = [c for c in last_codes() if str(c).isdigit()]
    print(f"[1m] codes={len(codes)} {args.start}..{args.end}")
    start = dt.datetime.strptime(args.start, "%Y%m%d").date()
    end = dt.datetime.strptime(args.end, "%Y%m%d").date()
    session = requests.Session()
    d = start
    while d <= end:
        if d.weekday() >= 5:
            d += dt.timedelta(days=1)
            continue
        ymd = d.strftime("%Y%m%d")
        out = hive_path(ymd)
        if out.exists() and out.stat().st_size > 10_000:
            print(f"[1m] skip existing {ymd}")
            d += dt.timedelta(days=1)
            continue
        print(f"[1m] download {ymd}")
        all_rows: list[dict] = []
        for i, code in enumerate(codes, 1):
            try:
                all_rows.extend(fetch_day(session, token, domain, code, ymd))
            except Exception as e:
                print(f"  {code} err {e}")
            if i % 50 == 0:
                print(f"  {i}/{len(codes)} rows={len(all_rows)}")
            time.sleep(0.08)
        if not all_rows:
            print(f"[1m] empty {ymd}")
            d += dt.timedelta(days=1)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        df = pl.DataFrame(all_rows).unique(subset=["code", "datetime"])
        df.write_parquet(str(out))
        print(f"[1m] wrote {out} rows={df.height} codes={df['code'].n_unique()}")
        d += dt.timedelta(days=1)


if __name__ == "__main__":
    main()
