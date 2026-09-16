#!/usr/bin/env python3
"""Run 15:20/09:00 overnight backtest and upsert to the dashboard API."""
from __future__ import annotations

import sys
from pathlib import Path

import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from kr_stock.config import DASHBOARD_API_URL
from kr_stock.overnight_backtest import dashboard_payload, run_1520_0900_backtest


def main() -> None:
    print("=========================================================================")
    print("   15:20 / 09:00 OVERNIGHT BACKTEST → DASHBOARD")
    print("=========================================================================\n")
    filled, monthly, weekly, sim = run_1520_0900_backtest()
    print(f" -> Filled trades: {len(filled)}")
    print(f" -> Total Return : {sim['total_return']*100:+.2f}%")
    print(f" -> Win Rate     : {sim['win_rate']*100:.2f}%")
    print(f" -> Profit Factor: {sim['pf']:.2f}")
    print(f" -> Sharpe Ratio : {sim['sharpe']:.2f}")
    print(f" -> CAGR         : {sim['cagr']*100:+.2f}%")
    print(f" -> MDD          : {sim['mdd']*100:.2f}%")
    print(f" -> Total Trades : {sim['n_trades']}")
    print(monthly[["label", "return_pct", "win_rate", "pf", "trades"]].to_string(index=False))

    payload = dashboard_payload(sim)
    print(f"\n[Posting to {DASHBOARD_API_URL}]")
    try:
        resp = requests.post(DASHBOARD_API_URL, json=payload, timeout=30)
        print(f" -> status {resp.status_code}")
        if resp.status_code == 200:
            print(" -> uploaded")
        else:
            print(f" -> {resp.text[:500]}")
    except Exception as e:
        print(f" -> HTTP error: {e}")


if __name__ == "__main__":
    main()
