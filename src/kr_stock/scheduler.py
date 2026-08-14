"""
Live Daemon Scheduler for KRX Overnight Strategy Paper Trading
--------------------------------------------------------------
Runs as a background daemon inside Podman container.
Executes automated trading cycle on weekdays (Asia/Seoul time):
  - 09:00:00 KST: Market Open SELL & Weekly/Monthly Returns Telegram Alert
  - 15:20:00 KST: Market Close BUY & Overnight Stock Selection Telegram Alert
  - 15:25:00 KST: Post-Market Parity Verification Telegram Alert
"""

import time
import signal
import sys
import logging
from datetime import datetime, timezone, timedelta

from kr_stock.paper_engine import PaperTradingEngine
from kr_stock.config import TRADING_MODE

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("kr_stock_scheduler")

# Asia/Seoul Timezone (UTC+9)
KST = timezone(timedelta(hours=9))

running = True

def signal_handler(signum, frame):
    global running
    logger.info("Termination signal received. Shutting down paper trading scheduler...")
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def run_scheduler():
    logger.info("=========================================================================")
    logger.info("   KRX OVERNIGHT TRADING DAEMON SCHEDULER STARTED")
    logger.info(f"   Execution: {TRADING_MODE.upper()}  (TRADING_MODE=paper|live)")
    logger.info("   Timezone: Asia/Seoul (KST)")
    logger.info("   Schedule: Weekdays Mon-Fri [09:00 SELL | 15:20 BUY | 15:25 PARITY]")
    logger.info("=========================================================================")

    engine = PaperTradingEngine()

    sold_date = ""
    bought_date = ""
    parity_date = ""

    while running:
        now_kst = datetime.now(KST)
        weekday = now_kst.weekday()  # 0: Mon, 1: Tue, 2: Wed, 3: Thu, 4: Fri, 5: Sat, 6: Sun
        today_str = now_kst.strftime("%Y-%m-%d")
        time_str = now_kst.strftime("%H:%M:%S")

        # Only execute on Weekdays (Mon-Fri)
        if weekday < 5:
            # 1. Morning 09:00:00 ~ 09:05:00 Market Open SELL
            if "09:00:00" <= time_str <= "09:05:00" and sold_date != today_str:
                logger.info(f"[{today_str} {time_str}] Executing 09:00 Market Open SELL...")
                try:
                    engine.execute_market_open_sell(today_str)
                    sold_date = today_str
                    logger.info(f"[{today_str}] 09:00 Market Open SELL Completed Successfully.")
                except Exception as e:
                    logger.error(f"[{today_str}] Error during 09:00 Market Open SELL: {e}", exc_info=True)

            # 2. Afternoon 15:20:00 ~ 15:24:00 Market Close BUY
            if "15:20:00" <= time_str <= "15:24:00" and bought_date != today_str:
                logger.info(f"[{today_str} {time_str}] Executing 15:20 Market Close BUY...")
                try:
                    engine.execute_market_close_buy(today_str)
                    bought_date = today_str
                    logger.info(f"[{today_str}] 15:20 Market Close BUY Completed Successfully.")
                except Exception as e:
                    logger.error(f"[{today_str}] Error during 15:20 Market Close BUY: {e}", exc_info=True)

            # 3. Post-Market 15:25:00 ~ 15:29:00 Parity Verification
            if "15:25:00" <= time_str <= "15:29:00" and parity_date != today_str:
                logger.info(f"[{today_str} {time_str}] Executing 15:25 Post-Market Parity Verification...")
                try:
                    engine.run_post_market_parity_check(today_str)
                    parity_date = today_str
                    logger.info(f"[{today_str}] 15:25 Parity Verification Completed Successfully.")
                except Exception as e:
                    logger.error(f"[{today_str}] Error during 15:25 Parity Verification: {e}", exc_info=True)

        time.sleep(15)

    logger.info("Scheduler daemon stopped.")

if __name__ == "__main__":
    run_scheduler()
