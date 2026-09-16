#!/usr/bin/env python3
"""Build 15:19 bars, retrain LGB/PyTorch, run OOS 15:20/09:00 Top-3 backtest."""
from __future__ import annotations

import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from kr_stock.config import DASHBOARD_API_URL
from kr_stock.overnight_backtest import dashboard_payload, run_1520_0900_backtest
from kr_stock.session_bars import build_session_1520_parquet
from kr_stock.train_1520 import TEST_END, TEST_START, TRAIN_END, TRAIN_START, train_1520_models


def main() -> None:
    print("=========================================================================")
    print("  1) Build 15:19 session bars + next 09:00")
    print("=========================================================================")
    build_session_1520_parquet(force=False)

    print("\n=========================================================================")
    print(f"  2) Train  ({TRAIN_START} .. {TRAIN_END})")
    print("=========================================================================")
    stats = train_1520_models()
    print(stats)

    print("\n=========================================================================")
    print(f"  3) OOS backtest ({TEST_START} .. {TEST_END})")
    print("=========================================================================")
    filled, monthly, weekly, sim = run_1520_0900_backtest(
        start_date=TEST_START,
        end_date=TEST_END,
        use_session_1520=True,
        model_tag="_1520",
    )
    print(f"filled={len(filled)}")
    print(
        f"return={sim['total_return']*100:+.2f}% CAGR={sim['cagr']*100:+.2f}% "
        f"WR={sim['win_rate']*100:.1f}% PF={sim['pf']:.2f} Sharpe={sim['sharpe']:.2f} "
        f"MDD={sim['mdd']*100:.2f}% trades={sim['n_trades']}"
    )
    print(monthly[["label", "return_pct", "win_rate", "pf", "trades"]].to_string(index=False))

    payload = dashboard_payload(sim)
    payload["algorithm"]["name"] = "overnight_1520_retrained_lgb_pytorch_KRX"
    payload["algorithm"]["model_type"] = "lgb_mlp_1520"
    print(f"\n[upload {DASHBOARD_API_URL}]")
    try:
        resp = requests.post(DASHBOARD_API_URL, json=payload, timeout=30)
        print(f" -> {resp.status_code}")
        if resp.status_code != 200:
            print(resp.text[:400])
    except Exception as e:
        print(f" -> {e}")


if __name__ == "__main__":
    main()
