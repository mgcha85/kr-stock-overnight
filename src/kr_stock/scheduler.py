"""
Live Daemon Scheduler for KRX Overnight Strategy Paper Trading
--------------------------------------------------------------
Weekdays Asia/Seoul:
  - after 09:00: SELL report (0 positions still Telegram)
  - 15:28-16:00: BUY report (0 candidates still Telegram)
  - 18:00-19:30: parity on completed daily bars
  - after 16:00 if buy never ran: missed-window Telegram, no order
"""
import logging
import signal
import sqlite3
import time
from datetime import datetime, timezone, timedelta

from kr_stock.paper_engine import PaperTradingEngine
from kr_stock.config import TRADING_MODE, PAPER_DB_PATH
from kr_stock.telegram import send_missed_window_alert, send_scheduler_boot_alert
from kr_stock.krx_calendar import ensure_holidays, is_session_day

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("kr_stock_scheduler")

KST = timezone(timedelta(hours=9))
running = True


def signal_handler(signum, frame):
    global running
    logger.info("Termination signal received. Shutting down paper trading scheduler...")
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def _db_path():
    return str(PAPER_DB_PATH)


def ensure_marks():
    conn = sqlite3.connect(_db_path())
    conn.execute(
        """CREATE TABLE IF NOT EXISTS scheduler_marks (
            date TEXT NOT NULL,
            job TEXT NOT NULL,
            ran_at TEXT NOT NULL,
            PRIMARY KEY (date, job)
        )"""
    )
    conn.commit()
    conn.close()


def job_done(date_str: str, job: str) -> bool:
    ensure_marks()
    conn = sqlite3.connect(_db_path())
    row = conn.execute(
        "SELECT 1 FROM scheduler_marks WHERE date = ? AND job = ?",
        (date_str, job),
    ).fetchone()
    conn.close()
    return row is not None


def mark_job(date_str: str, job: str, stamp: str):
    ensure_marks()
    conn = sqlite3.connect(_db_path())
    conn.execute(
        "INSERT OR REPLACE INTO scheduler_marks (date, job, ran_at) VALUES (?, ?, ?)",
        (date_str, job, stamp),
    )
    conn.commit()
    conn.close()


def run_scheduler():
    logger.info("=========================================================================")
    logger.info("   KRX OVERNIGHT TRADING DAEMON SCHEDULER STARTED")
    logger.info(f"   Execution: {TRADING_MODE.upper()}  (TRADING_MODE=paper|live)")
    logger.info("   Timezone: Asia/Seoul (KST)")
    logger.info("   Schedule: KRX session days [09:00 SELL | 15:28 BUY | 18:00 PARITY]")
    logger.info("   Closed: weekends + krx_holidays (2026 seeded)")
    logger.info("=========================================================================")

    engine = PaperTradingEngine()
    ensure_marks()
    ensure_holidays(_db_path())
    boot = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
    send_scheduler_boot_alert(boot)
    last_beat = time.time()

    while running:
        now_kst = datetime.now(KST)
        today_str = now_kst.strftime("%Y-%m-%d")
        time_str = now_kst.strftime("%H:%M:%S")
        stamp = f"{today_str} {time_str}"

        if is_session_day(_db_path(), today_str):
            if time_str >= "09:00:00" and not job_done(today_str, "sell"):
                logger.info(f"[{stamp}] Executing 09:00 Market Open SELL (catch-up ok)...")
                try:
                    engine.execute_market_open_sell(today_str)
                    mark_job(today_str, "sell", stamp)
                    logger.info(f"[{today_str}] 09:00 Market Open SELL Completed Successfully.")
                except Exception as e:
                    logger.error(f"[{today_str}] Error during 09:00 Market Open SELL: {e}", exc_info=True)

            if "15:28:00" <= time_str < "16:00:00" and not job_done(today_str, "buy"):
                logger.info(f"[{stamp}] Executing 15:28 Market Close BUY (until 16:00)...")
                try:
                    engine.execute_market_close_buy(today_str)
                    mark_job(today_str, "buy", stamp)
                    logger.info(f"[{today_str}] 15:28 Market Close BUY Completed Successfully.")
                except Exception as e:
                    logger.error(f"[{today_str}] Error during 15:28 Market Close BUY: {e}", exc_info=True)

            if time_str >= "16:00:00" and not job_done(today_str, "buy"):
                logger.info(f"[{stamp}] BUY window missed; sending 0-fill report, no order.")
                send_missed_window_alert(
                    today_str,
                    "15:28 매수 창 놓침 — 매수 0건",
                    "16:00까지 매수가 실행되지 않아 주문하지 않았습니다. 장중 프로세스가 꺼졌거나 창을 지나쳤습니다.",
                )
                mark_job(today_str, "buy", stamp)

            if "18:00:00" <= time_str < "19:30:00" and not job_done(today_str, "parity"):
                logger.info(f"[{stamp}] Executing 18:00 Post-Market Parity Verification...")
                try:
                    engine.run_post_market_parity_check(today_str)
                    mark_job(today_str, "parity", stamp)
                    logger.info(f"[{today_str}] 18:00 Parity Verification Completed Successfully.")
                except Exception as e:
                    logger.error(f"[{today_str}] Error during 18:00 Parity Verification: {e}", exc_info=True)

        if time.time() - last_beat >= 30 * 60:
            logger.info(
                f"[heartbeat] {stamp} mode={TRADING_MODE} "
                f"sell_done={job_done(today_str, 'sell')} buy_done={job_done(today_str, 'buy')}"
            )
            last_beat = time.time()

        time.sleep(15)

    logger.info("Scheduler daemon stopped.")


if __name__ == "__main__":
    run_scheduler()
