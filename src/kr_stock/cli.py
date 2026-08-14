"""
KRX Overnight Paper Trading CLI Entrypoint
------------------------------------------
Usage:
  python3 -m kr_stock.cli --mode buy [--date YYYY-MM-DD]
  python3 -m kr_stock.cli --mode sell [--date YYYY-MM-DD]
  python3 -m kr_stock.cli --mode parity [--date YYYY-MM-DD]
  python3 -m kr_stock.cli --mode full_day [--date YYYY-MM-DD]
"""

import argparse
import sys
from datetime import datetime
from kr_stock.paper_engine import PaperTradingEngine
from kr_stock.config import TRADING_MODE


def main():
    parser = argparse.ArgumentParser(description="KRX Overnight Strategy Paper Trading Engine")
    parser.add_argument(
        "--mode",
        choices=["buy", "sell", "parity", "full_day"],
        required=True,
        help="Trading mode to execute: buy (15:20 close), sell (09:00 open), parity (verification), full_day (simulation)"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Target date in YYYY-MM-DD format (default: today)"
    )

    args = parser.parse_args()
    engine = PaperTradingEngine()

    print(f"=========================================================================")
    print(f"   KRX OVERNIGHT ENGINE | Exec: {TRADING_MODE.upper()} | Mode: {args.mode.upper()} | Date: {args.date}")
    print(f"=========================================================================\n")

    if args.mode == "buy":
        buys = engine.execute_market_close_buy(args.date)
        print(f"[RESULT] Executed {len(buys)} BUY positions for {args.date}.")

    elif args.mode == "sell":
        sells = engine.execute_market_open_sell(args.date)
        print(f"[RESULT] Executed {len(sells)} SELL positions for {args.date}.")

    elif args.mode == "parity":
        is_matched = engine.run_post_market_parity_check(args.date)
        print(f"[RESULT] Parity check for {args.date}: {'100% MATCH ✅' if is_matched else 'MISMATCH ❌'}")

    elif args.mode == "full_day":
        print(f"--- 1. Executing Morning 09:00 SELL ---")
        sells = engine.execute_market_open_sell(args.date)
        print(f"Closed {len(sells)} positions.\n")

        print(f"--- 2. Executing Afternoon 15:20 BUY ---")
        buys = engine.execute_market_close_buy(args.date)
        print(f"Opened {len(buys)} positions.\n")

        print(f"--- 3. Executing Post-Market Parity Verification ---")
        is_matched = engine.run_post_market_parity_check(args.date)
        print(f"Parity Match: {is_matched}\n")

if __name__ == "__main__":
    main()
