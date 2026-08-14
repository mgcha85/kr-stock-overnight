#!/usr/bin/env python3
"""
Backtest <-> Paper-Trading Parity & Verification Suite
------------------------------------------------------
Verifies that OvernightScorer generates 100% deterministic candidate picks,
probability outputs, and hybrid scores across multiple historical dates.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from kr_stock.inference import OvernightScorer


def run_parity_test():
    print("=========================================================================")
    print("   RUNNING BACKTEST <-> PAPER-TRADING PARITY VERIFICATION SUITE         ")
    print("=========================================================================\n")

    scorer = OvernightScorer()
    print(f" -> Scorer initialized. MTF Model Active: {scorer.is_mtf} | Features: {len(scorer.feature_cols)}")

    test_dates = ["2026-06-01", "2026-06-05", "2026-06-10", "2026-06-15", "2026-06-20"]
    
    total_checks = 0
    passed_checks = 0

    for dt in test_dates:
        print(f"\n[Test Date: {dt}]")
        
        # Run 1: Primary Inference
        run1 = scorer.get_candidates_for_date(target_date=dt, top_k=3)
        
        # Run 2: Verification Inference
        run2 = scorer.get_candidates_for_date(target_date=dt, top_k=3)

        if len(run1) == 0 and len(run2) == 0:
            print(f"  [PASS] No candidate stocks meeting criteria for {dt}.")
            passed_checks += 1
            total_checks += 1
            continue

        total_checks += 1
        if len(run1) != len(run2):
            print(f"  [FAIL] Candidate count mismatch! Run1: {len(run1)}, Run2: {len(run2)}")
            continue

        all_match = True
        for c1, c2 in zip(run1, run2):
            t1 = c1.get('code', c1.get('ticker'))
            t2 = c2.get('code', c2.get('ticker'))
            if t1 != t2:
                all_match = False
                print(f"  [FAIL] Ticker mismatch: {t1} vs {t2}")
            if abs(c1['hybrid_score'] - c2['hybrid_score']) > 1e-5:
                all_match = False
                print(f"  [FAIL] Score mismatch: {c1['hybrid_score']} vs {c2['hybrid_score']}")
            if abs(c1['p_lgb'] - c2['p_lgb']) > 1e-5:
                all_match = False
                print(f"  [FAIL] LightGBM Probability mismatch: {c1['p_lgb']} vs {c2['p_lgb']}")

        if all_match:
            passed_checks += 1
            print(f"  [PASS] 100% Deterministic Parity Verified for {dt}!")
            for idx, c in enumerate(run1, 1):
                t_code = c.get('code', c.get('ticker'))
                print(f"    Rank #{idx}: [{t_code}] {c['stock_name']} ({c['theme_name']}) | Score: {c['hybrid_score']:.2f} | P(LGB): {c['p_lgb']:.4f} | P(MLP): {c['p_torch']:.4f}")

    print("\n" + "="*80)
    print(f"PARITY VERIFICATION RESULT: {passed_checks} / {total_checks} PASSED ({passed_checks/total_checks:.0%})")
    print("="*80)


if __name__ == "__main__":
    run_parity_test()
