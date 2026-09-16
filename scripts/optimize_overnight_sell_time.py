#!/usr/bin/env python3
"""Find the best fixed 09:00–09:30 sell clock using 1-minute opens."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from kr_stock.sell_timing import build_overnight_trades, evaluate_sell_minutes, pick_best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-04-01")
    ap.add_argument("--end", default="2026-08-12")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("-o", "--out", default="")
    args = ap.parse_args()

    print("=" * 80)
    print(" OVERNIGHT SELL-TIME SWEEP  (1m open, 09:00–09:30)")
    print(f" Buy dates {args.start} .. {args.end} | Top-{args.top_k} | fee 0.23%")
    print("=" * 80)

    trades = build_overnight_trades(start_date=args.start, end_date=args.end, top_k=args.top_k)
    print(f"[trades] {len(trades)} overnight positions with T+1 1m coverage")
    if not trades:
        return

    summary, long_df = evaluate_sell_minutes(trades)
    if summary.empty:
        print("No 1m morning matches.")
        return

    cols = [
        "hhmm",
        "trades",
        "days",
        "mean_pnl_pct",
        "median_pnl_pct",
        "win_rate",
        "pf",
        "compound_pct",
        "vs_0900_mean_bp",
    ]
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
    best = pick_best(summary)
    print("\n[best mean]     {hhmm}  mean={mean_pnl_pct:+.3f}%  PF={pf:.2f}  WR={win_rate:.1f}%".format(**best["best_mean"]))
    print("[best PF]       {hhmm}  mean={mean_pnl_pct:+.3f}%  PF={pf:.2f}  WR={win_rate:.1f}%".format(**best["best_pf"]))
    print("[best compound] {hhmm}  compound={compound_pct:+.2f}%  mean={mean_pnl_pct:+.3f}%".format(**best["best_compound"]))
    if best.get("open_0900"):
        print("[baseline 09:00] mean={mean_pnl_pct:+.3f}%  PF={pf:.2f}  compound={compound_pct:+.2f}%".format(**best["open_0900"]))

    if args.out:
        out = Path(args.out)
        out.write_text(
            json.dumps(
                {
                    "n_trades": len(trades),
                    "summary": summary.to_dict(orient="records"),
                    "best": best,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
