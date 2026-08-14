#!/usr/bin/env python3
import sys
import argparse
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from kr_stock.inference import OvernightScorer
from kr_stock.kiwoom_condition import KiwoomConditionManager

def run_overnight_analysis(target_date: str = "2026-07-30", condition_name: str = "종가베팅", top_k: int = 3):
    print("=" * 80)
    print(f" KIWOOM CONDITION SEARCH ('{condition_name}') -> OVERNIGHT CANDLE ANALYSIS")
    print("=" * 80)

    # 1. Fetch candidate codes from Kiwoom Condition Search ("종가베팅")
    cond_mgr = KiwoomConditionManager(condition_name=condition_name)
    candidate_codes = cond_mgr.get_condition_search_codes(target_date=target_date)

    if not candidate_codes:
        print(f"[Result] No candidate stocks matched condition '{condition_name}' on {target_date}.")
        return

    print(f"\n[Step 1] Condition Search Passed: {len(candidate_codes)} stocks -> {candidate_codes}")
    print(f"[Step 2] Fetching candle data & running AI scoring ONLY for these {len(candidate_codes)} stocks...")

    # 2. Run OvernightScorer ONLY for candidate_codes
    scorer = OvernightScorer()
    candidates = scorer.get_candidates_for_date(
        target_date=target_date,
        top_k=top_k,
        candidate_codes=candidate_codes
    )

    print("\n" + "=" * 80)
    print(f" FINAL TOP-{top_k} OVERNIGHT BUY RECOMMENDATIONS FOR {target_date}")
    print("=" * 80)

    if not candidates:
        print("No stocks met the final AI model probability threshold (P_LGB >= 0.35 & P_MLP >= 0.35).")
        return

    for idx, item in enumerate(candidates, 1):
        print(f"Rank #{idx}: [{item['code']}] {item['stock_name']} ({item['theme_name']})")
        print(f"  - Hybrid Score: {item['hybrid_score']:.2f}")
        print(f"  - Model Conviction: P(LGB)={item['p_lgb']:.4f} | P(MLP)={item['p_torch']:.4f}")
        print(f"  - Price / Change: Close={item['close_price']:,.0f} KRW | Stock Change={item['stock_change']:.2f}%")
        print(f"  - Market Context: News={item.get('news_count', 0)} items | DART Filings={item.get('dart_count', 0)} items")
        print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Kiwoom Condition Search Overnight Analysis")
    parser.add_argument("--date", type=str, default="2026-07-30", help="Target date YYYY-MM-DD")
    parser.add_argument("--condition", type=str, default="종가베팅", help="Kiwoom Condition Search Name")
    parser.add_argument("--top_k", type=int, default=3, help="Top K recommendations")
    args = parser.parse_args()

    run_overnight_analysis(target_date=args.date, condition_name=args.condition, top_k=args.top_k)
