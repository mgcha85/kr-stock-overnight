"""
Live Paper-Trading Engine for KRX Overnight Strategy
---------------------------------------------------
Manages 10,000,000 KRW seed capital, daily position lifecycle,
Telegram alerts, weekly/monthly return tracking, and post-market backtest parity verification.
"""

import sqlite3
import math
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import polars as pl
from pathlib import Path

from kr_stock.config import (
    PAPER_DB_PATH, SEED_CAPITAL, TOP_K_TRADES, FEE_RATE, DATA_PARQUET_PATH
)
from kr_stock.inference import OvernightScorer
from kr_stock.telegram import (
    send_market_close_buy_alert,
    send_market_open_sell_alert,
    send_parity_check_alert
)

logger = logging.getLogger(__name__)


class PaperTradingEngine:
    def __init__(self, db_path: Path = PAPER_DB_PATH):
        self.db_path = Path(db_path)
        self.scorer = OvernightScorer()
        self._init_database()

    def _init_database(self):
        """Initializes SQLite database schemas for paper trading."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Paper Trades Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                stock_name TEXT NOT NULL,
                theme_name TEXT,
                buy_price REAL NOT NULL,
                buy_qty INTEGER NOT NULL,
                buy_amount REAL NOT NULL,
                sell_price REAL,
                sell_amount REAL,
                pnl_krw REAL,
                pnl_pct REAL,
                status TEXT NOT NULL,  -- 'OPEN', 'CLOSED'
                open_time TEXT NOT NULL,
                close_time TEXT,
                hybrid_score REAL,
                p_lgb REAL,
                p_torch REAL
            )
        """)

        # Account Balance History Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_account (
                date TEXT PRIMARY KEY,
                cash_balance REAL NOT NULL,
                invested_amount REAL NOT NULL,
                total_equity REAL NOT NULL,
                daily_pnl_krw REAL DEFAULT 0.0,
                daily_pnl_pct REAL DEFAULT 0.0,
                weekly_pnl_pct REAL DEFAULT 0.0,
                monthly_pnl_pct REAL DEFAULT 0.0,
                updated_at TEXT NOT NULL
            )
        """)

        conn.commit()
        conn.close()

    def get_latest_account_state(self) -> Tuple[float, float, float]:
        """Returns (cash_balance, invested_amount, total_equity). Initialized to SEED_CAPITAL."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        row = cursor.execute("""
            SELECT cash_balance, invested_amount, total_equity 
            FROM paper_account 
            ORDER BY date DESC LIMIT 1
        """).fetchone()

        # Calculate active invested amount from open positions
        open_invested = cursor.execute("""
            SELECT COALESCE(SUM(buy_amount), 0.0) 
            FROM paper_trades 
            WHERE status = 'OPEN'
        """).fetchone()[0]

        conn.close()

        if row:
            cash = row[0]
            total_equity = cash + open_invested
            return cash, open_invested, total_equity
        else:
            return SEED_CAPITAL, 0.0, SEED_CAPITAL

    def _save_account_state(self, date_str: str, cash: float, invested: float, daily_pnl: float = 0.0, daily_pct: float = 0.0):
        """Saves current account snapshot for date_str."""
        total_equity = cash + invested
        weekly_pct, monthly_pct = self.calculate_cumulative_returns(date_str, total_equity)

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO paper_account (
                date, cash_balance, invested_amount, total_equity, 
                daily_pnl_krw, daily_pnl_pct, weekly_pnl_pct, monthly_pnl_pct, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                cash_balance = excluded.cash_balance,
                invested_amount = excluded.invested_amount,
                total_equity = excluded.total_equity,
                daily_pnl_krw = excluded.daily_pnl_krw,
                daily_pnl_pct = excluded.daily_pnl_pct,
                weekly_pnl_pct = excluded.weekly_pnl_pct,
                monthly_pnl_pct = excluded.monthly_pnl_pct,
                updated_at = excluded.updated_at
        """, (
            date_str, cash, invested, total_equity,
            daily_pnl, daily_pct, weekly_pct, monthly_pct,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn.commit()
        conn.close()

    def calculate_cumulative_returns(self, current_date_str: str, current_equity: float) -> Tuple[float, float]:
        """Calculates Weekly (7-day) and Monthly (30-day) cumulative percentage returns."""
        try:
            cur_date = datetime.strptime(current_date_str, "%Y-%m-%d")
        except ValueError:
            return 0.0, 0.0

        week_ago_str = (cur_date - timedelta(days=7)).strftime("%Y-%m-%d")
        month_ago_str = (cur_date - timedelta(days=30)).strftime("%Y-%m-%d")

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Weekly base equity
        row_w = cursor.execute("""
            SELECT total_equity FROM paper_account 
            WHERE date <= ? ORDER BY date DESC LIMIT 1
        """, (week_ago_str,)).fetchone()
        base_weekly = row_w[0] if row_w else SEED_CAPITAL

        # Monthly base equity
        row_m = cursor.execute("""
            SELECT total_equity FROM paper_account 
            WHERE date <= ? ORDER BY date DESC LIMIT 1
        """, (month_ago_str,)).fetchone()
        base_monthly = row_m[0] if row_m else SEED_CAPITAL

        conn.close()

        weekly_pct = ((current_equity / base_weekly) - 1.0) * 100.0 if base_weekly > 0 else 0.0
        monthly_pct = ((current_equity / base_monthly) - 1.0) * 100.0 if base_monthly > 0 else 0.0

        return weekly_pct, monthly_pct

    def execute_market_close_buy(self, target_date: str) -> List[Dict[str, Any]]:
        """
        [15:30 Market Close]
        1. Selects Top-K picks using OvernightScorer.
        2. Allocates available cash.
        3. Executes Paper BUY orders.
        4. Sends Telegram notification.
        """
        cash, invested, total_equity = self.get_latest_account_state()
        picks = self.scorer.get_candidates_for_date(target_date, top_k=TOP_K_TRADES)

        if not picks:
            logger.info(f"[{target_date}] No candidates met scoring threshold. Cash remains 100%.")
            self._save_account_state(target_date, cash, 0.0)
            send_market_close_buy_alert(target_date, [], 0.0, cash, total_equity)
            return []

        # Split available cash equally across picks
        alloc_per_stock = cash / len(picks)
        bought_trades = []
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        total_buy_amount = 0.0
        for p in picks:
            close_price = p['close_price']
            qty = math.floor(alloc_per_stock / close_price)
            if qty <= 0:
                continue

            buy_amount = qty * close_price
            total_buy_amount += buy_amount

            open_time_str = f"{target_date} 15:30:00"

            cursor.execute("""
                INSERT INTO paper_trades (
                    date, ticker, stock_name, theme_name, buy_price, buy_qty, buy_amount,
                    status, open_time, hybrid_score, p_lgb, p_torch
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?)
            """, (
                target_date, p['ticker'], p['stock_name'], p['theme_name'],
                close_price, qty, buy_amount, open_time_str,
                p['hybrid_score'], p['p_lgb'], p['p_torch']
            ))

            trade_info = {
                "ticker": p['ticker'],
                "stock_name": p['stock_name'],
                "theme_name": p['theme_name'],
                "buy_price": close_price,
                "buy_qty": qty,
                "buy_amount": buy_amount,
                "hybrid_score": p['hybrid_score'],
                "p_lgb": p['p_lgb'],
                "p_torch": p['p_torch']
            }
            bought_trades.append(trade_info)

        conn.commit()
        conn.close()

        new_cash = cash - total_buy_amount
        self._save_account_state(target_date, new_cash, total_buy_amount)

        # Telegram Alert
        send_market_close_buy_alert(target_date, bought_trades, alloc_per_stock, new_cash, total_equity)
        return bought_trades

    def execute_market_open_sell(self, target_date: str) -> List[Dict[str, Any]]:
        """
        [09:00 Next Market Open]
        1. Fetches all OPEN trades.
        2. Retrieves Next Open price and computes net PnL (after 0.23% fees/taxes).
        3. Closes positions and updates account balance.
        4. Calculates Weekly & Monthly returns and sends Telegram report.
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        open_trades = cursor.execute("""
            SELECT id, ticker, stock_name, buy_price, buy_qty, buy_amount 
            FROM paper_trades 
            WHERE status = 'OPEN'
        """).fetchall()

        if not open_trades:
            logger.info(f"[{target_date}] No OPEN positions to sell.")
            conn.close()
            cash, invested, equity = self.get_latest_account_state()
            w_pct, m_pct = self.calculate_cumulative_returns(target_date, equity)
            send_market_open_sell_alert(target_date, [], 0.0, 0.0, w_pct, m_pct, equity)
            return []

        # Fetch open prices directly for target_date from candle dataset
        lazy_df = pl.scan_parquet(str(DATA_PARQUET_PATH))
        df_open_candles = (
            lazy_df
            .filter((pl.col("date") == target_date))
            .select(["ticker", "open"])
            .collect()
            .to_pandas()
        )
        open_price_map = {}
        for _, r in df_open_candles.iterrows():
            clean_t = str(r['ticker']).split('.')[0].zfill(6)
            open_price_map[clean_t] = float(r['open'])

        closed_trades = []
        total_pnl_krw = 0.0
        total_returned_cash = 0.0
        close_time_str = f"{target_date} 09:00:00"

        for trade_id, ticker, name, buy_price, qty, buy_amount in open_trades:
            # Look up open price for target_date (fallback to buy_price if missing)
            sell_price = open_price_map.get(ticker, buy_price)

            gross_sell_amount = qty * sell_price
            net_sell_amount = gross_sell_amount * (1.0 - FEE_RATE)
            pnl_krw = net_sell_amount - buy_amount
            pnl_pct = (pnl_krw / buy_amount) * 100.0

            total_pnl_krw += pnl_krw
            total_returned_cash += net_sell_amount

            cursor.execute("""
                UPDATE paper_trades SET
                    sell_price = ?,
                    sell_amount = ?,
                    pnl_krw = ?,
                    pnl_pct = ?,
                    status = 'CLOSED',
                    close_time = ?
                WHERE id = ?
            """, (sell_price, net_sell_amount, pnl_krw, pnl_pct, close_time_str, trade_id))

            closed_trades.append({
                "ticker": ticker,
                "stock_name": name,
                "buy_price": buy_price,
                "sell_price": sell_price,
                "buy_qty": qty,
                "pnl_krw": pnl_krw,
                "pnl_pct": pnl_pct
            })

        conn.commit()

        # Update Account State
        cash, _, _ = self.get_latest_account_state()
        invested_amount = sum(t[5] for t in open_trades)
        prev_equity = cash + invested_amount
        
        new_cash = cash + total_returned_cash
        new_equity = new_cash  # No invested cash after selling everything
        daily_pct = (total_pnl_krw / prev_equity) * 100.0 if prev_equity > 0 else 0.0

        self._save_account_state(target_date, new_cash, 0.0, daily_pnl=total_pnl_krw, daily_pct=daily_pct)
        conn.close()

        # Calculate Weekly & Monthly Returns
        weekly_pct, monthly_pct = self.calculate_cumulative_returns(target_date, new_equity)

        # Telegram Alert
        send_market_open_sell_alert(
            target_date, closed_trades, total_pnl_krw, daily_pct,
            weekly_pct, monthly_pct, new_equity
        )

        return closed_trades

    def run_post_market_parity_check(self, target_date: str) -> bool:
        """
        [15:35 Post-Market Parity Verification]
        1. Runs backtest candidate selection for target_date.
        2. Retrieves paper buys executed at 15:30 on target_date.
        3. Compares paper buy tickers against backtest signals.
        4. Sends Telegram parity verification result.
        """
        # Fetch paper buys for target_date
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        paper_rows = cursor.execute("""
            SELECT ticker FROM paper_trades WHERE date = ? AND open_time LIKE ?
        """, (target_date, f"{target_date}%")).fetchall()
        conn.close()

        paper_tickers = sorted(list(set([r[0] for r in paper_rows])))

        # Fetch backtest signals for target_date
        backtest_picks = self.scorer.get_candidates_for_date(target_date, top_k=TOP_K_TRADES)
        backtest_tickers = sorted([p['ticker'] for p in backtest_picks])

        is_matched = (paper_tickers == backtest_tickers)
        details = (
            f"Paper Buy: {paper_tickers} | Backtest Buy: {backtest_tickers}. "
            f"100% Signal & Parity Match!" if is_matched else
            f"Parity Mismatch! Paper: {paper_tickers} vs Backtest: {backtest_tickers}"
        )

        logger.info(f"[{target_date} Parity Verification] Matched: {is_matched} | {details}")
        send_parity_check_alert(target_date, is_matched, paper_tickers, backtest_tickers, details)
        return is_matched
