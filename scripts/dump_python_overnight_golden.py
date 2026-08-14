#!/usr/bin/env python3
"""Dump Python overnight golden JSON for Rust parity checks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from kr_stock.kiwoom_condition import KiwoomConditionManager
from kr_stock.inference import OvernightScorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--condition", default="종가베팅")
    args = ap.parse_args()

    cond = KiwoomConditionManager(condition_name=args.condition)
    codes = cond.get_condition_search_codes(args.date)
    scorer = OvernightScorer()
    picks = scorer.get_candidates_for_date(
        target_date=args.date, top_k=args.top_k, candidate_codes=codes
    )

    payload = {
        "date": args.date,
        "condition_name": args.condition,
        "codes": [str(c).zfill(6) for c in codes],
        "top_k": args.top_k,
        "picks": [
            {
                "code": p["code"],
                "ticker": p["ticker"],
                "stock_name": p["stock_name"],
                "theme_name": p.get("theme_name"),
                "close_price": float(p["close_price"]),
                "stock_change": float(p["stock_change"]),
                "p_lgb": float(p["p_lgb"]),
                "p_torch": float(p["p_torch"]),
                "hybrid_score": float(p["hybrid_score"]),
                "news_count": int(p.get("news_count", 0)),
                "dart_count": int(p.get("dart_count", 0)),
            }
            for p in picks
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Wrote {out} codes={len(codes)} picks={len(picks)}")


if __name__ == "__main__":
    main()
