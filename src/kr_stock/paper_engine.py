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
    PAPER_DB_PATH, SEED_CAPITAL, TOP_K_TRADES, FEE_RATE, DATA_PARQUET_PATH,
    TRADING_MODE, ACC_NO, is_live_execution,
)
from kr_stock.inference import OvernightScorer
from kr_stock.kiwoom_condition import KiwoomConditionManager
from kr_stock.telegram import (
    send_market_close_buy_alert,
    send_market_open_sell_alert,
    send_parity_check_alert,
    send_ops_error_alert,
)

logger = logging.getLogger(__name__)


class PaperTradingEngine:
    def __init__(self, db_path: Path = PAPER_DB_PATH):
        self.db_path = Path(db_path)
        self.scorer = OvernightScorer()
        self.condition_manager = KiwoomConditionManager(condition_name="종가베팅")
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

        self._ensure_trade_columns(cursor)
        conn.commit()
        conn.close()

    def _ensure_trade_columns(self, cursor: sqlite3.Cursor) -> None:
        cols = {r[1] for r in cursor.execute("PRAGMA table_info(paper_trades)").fetchall()}
        if "buy_ord_no" not in cols:
            cursor.execute("ALTER TABLE paper_trades ADD COLUMN buy_ord_no TEXT")
        if "sell_ord_no" not in cols:
            cursor.execute("ALTER TABLE paper_trades ADD COLUMN sell_ord_no TEXT")
        if "execution_mode" not in cols:
            cursor.execute("ALTER TABLE paper_trades ADD COLUMN execution_mode TEXT")

    def _assert_live_ready(self) -> None:
        if not ACC_NO:
            raise RuntimeError("TRADING_MODE=live requires ACC_NO")
        from kr_stock.kiwoom_live import _app_key, _secret_key
        if not _app_key() or not _secret_key():
            raise RuntimeError("TRADING_MODE=live requires APP_KEY / SECRET_KEY")

    def get_latest_account_state(self) -> Tuple[float, float, float]:
        """Returns (cash_balance, invested_amount, total_equity)."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        row = cursor.execute("""
            SELECT cash_balance, invested_amount, total_equity
            FROM paper_account
            ORDER BY date DESC LIMIT 1
        """).fetchone()

        open_invested = cursor.execute("""
            SELECT COALESCE(SUM(buy_amount), 0.0)
            FROM paper_trades
            WHERE status = 'OPEN'
        """).fetchone()[0]
        conn.close()

        if is_live_execution():
            self._assert_live_ready()
            from kr_stock.kiwoom_broker import get_orderable_cash
            cash = get_orderable_cash()
            return cash, open_invested, cash + open_invested

        if row:
            cash = row[0]
            return cash, open_invested, cash + open_invested
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
        [15:20 Market Close]
        1. Selects Top-K picks using OvernightScorer.
        2. Allocates available cash.
        3. Executes Paper BUY orders.
        4. Sends Telegram notification.
        """
        live = is_live_execution()
        if live:
            try:
                self._assert_live_ready()
            except Exception as e:
                logger.error(f"[{target_date}] LIVE not ready: {e}")
                send_ops_error_alert(target_date, "LIVE 매수 중단 — 계좌/키 미설정", f"<code>{e}</code>")
                return []

        cash, invested, total_equity = self.get_latest_account_state()

        # Idempotent: skip if OPEN trades already exist for target_date
        conn = sqlite3.connect(str(self.db_path))
        existing = conn.execute(
            "SELECT COUNT(*) FROM paper_trades WHERE date = ? AND status = 'OPEN'",
            (target_date,),
        ).fetchone()[0]
        conn.close()
        if existing > 0:
            logger.info(
                f"[{target_date}] Idempotent BUY skip: {existing} OPEN trade(s) already exist."
            )
            return []

        # 0. Today's candles MUST be in parquet before scoring. Missing data is not "no signal".
        try:
            from scripts.fetch_today_kr_candles import (
                ensure_today_candles_updated,
                parquet_row_count,
            )
            ensure_today_candles_updated(target_date, force=True)
            n_bars = parquet_row_count(target_date)
            if n_bars <= 0:
                raise RuntimeError(f"parquet has 0 rows for {target_date} after candle sync")
            logger.info(f"[{target_date}] Candle sync OK: {n_bars} parquet rows")
        except Exception as e:
            logger.error(f"[{target_date}] Candle sync FAILED: {e}", exc_info=True)
            send_ops_error_alert(
                target_date,
                "15:20 캔들 동기화 실패 — 매수 중단",
                f"<code>{e}</code>\n컨테이너에 FinanceDataReader가 없거나 parquet가 비면 "
                "종가 스코어링이 0건이 됩니다. 가짜 '매수 조건 충족 종목 없음'은 보내지 않습니다.",
            )
            return []

        # 1. Fetch candidate codes matching Kiwoom Condition Search ("종가베팅")
        candidate_codes = self.condition_manager.get_condition_search_codes(target_date)
        
        # 2. Score only these matched candidates
        picks = self.scorer.get_candidates_for_date(target_date, top_k=TOP_K_TRADES, candidate_codes=candidate_codes)

        if not picks:
            reason = (
                f"조건검색 {len(candidate_codes)}종목 중 스코어 필터 통과 0 "
                f"(turnover/등락률/p_lgb/p_torch)."
            )
            logger.info(f"[{target_date}] No candidates met scoring threshold. {reason}")
            self._save_account_state(target_date, cash, 0.0)
            send_market_close_buy_alert(target_date, [], 0.0, cash, total_equity, empty_reason=reason)
            return []

        # Split available cash equally across picks
        alloc_per_stock = cash / len(picks)
        bought_trades = []
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        self._ensure_trade_columns(cursor)

        total_buy_amount = 0.0
        rejected = []
        for p in picks:
            close_price = p['close_price']
            qty = math.floor(alloc_per_stock / close_price)
            if qty <= 0:
                continue

            buy_ord_no = ""
            if live:
                from kr_stock.kiwoom_broker import market_buy
                ok, buy_ord_no, raw = market_buy(p['ticker'], int(qty))
                if not ok:
                    msg = raw.get("return_msg") or raw
                    logger.error(f"[{target_date}] LIVE BUY rejected {p['ticker']} qty={qty}: {msg}")
                    rejected.append(f"{p['ticker']} {msg}")
                    continue
                logger.info(f"[{target_date}] LIVE BUY {p['ticker']} qty={qty} ord_no={buy_ord_no}")

            buy_amount = qty * close_price
            total_buy_amount += buy_amount
            open_time_str = f"{target_date} 15:20:00"

            cursor.execute("""
                INSERT INTO paper_trades (
                    date, ticker, stock_name, theme_name, buy_price, buy_qty, buy_amount,
                    status, open_time, hybrid_score, p_lgb, p_torch, buy_ord_no, execution_mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?, ?, ?)
            """, (
                target_date, p['ticker'], p['stock_name'], p['theme_name'],
                close_price, qty, buy_amount, open_time_str,
                p['hybrid_score'], p['p_lgb'], p['p_torch'],
                buy_ord_no or None, TRADING_MODE,
            ))

            bought_trades.append({
                "ticker": p['ticker'],
                "stock_name": p['stock_name'],
                "theme_name": p['theme_name'],
                "buy_price": close_price,
                "buy_qty": qty,
                "buy_amount": buy_amount,
                "hybrid_score": p['hybrid_score'],
                "p_lgb": p['p_lgb'],
                "p_torch": p['p_torch'],
            })

        conn.commit()
        conn.close()

        if live and rejected and not bought_trades:
            send_ops_error_alert(
                target_date,
                "LIVE 매수 전부 거부",
                "키움 주문이 모두 거부되어 DB에 OPEN을 넣지 않았습니다.\n"
                + "<br/>".join(f"<code>{r}</code>" for r in rejected),
            )

        if live:
            try:
                from kr_stock.kiwoom_broker import get_orderable_cash
                new_cash = get_orderable_cash()
            except Exception as e:
                logger.warning(f"[{target_date}] live cash refresh failed, using estimate: {e}")
                new_cash = cash - total_buy_amount
        else:
            new_cash = cash - total_buy_amount
        self._save_account_state(target_date, new_cash, total_buy_amount)

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
        self._ensure_trade_columns(cursor)

        open_trades = cursor.execute("""
            SELECT id, ticker, stock_name, buy_price, buy_qty, buy_amount,
                   COALESCE(execution_mode, 'paper')
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

        live = is_live_execution()
        modes = {row[6] for row in open_trades}
        if live and any(m != "live" for m in modes):
            conn.close()
            send_ops_error_alert(
                target_date,
                "LIVE 매도 중단 — PAPER OPEN 잔존",
                "실주문을 내면 계좌에 없는 종목을 팔 수 있습니다.\n"
                "TRADING_MODE=paper 로 PAPER OPEN을 청산한 뒤 live로 전환하세요.\n"
                f"<code>{[(r[1], r[6]) for r in open_trades]}</code>",
            )
            return []
        if (not live) and any(m == "live" for m in modes):
            conn.close()
            send_ops_error_alert(
                target_date,
                "PAPER 매도 중단 — LIVE OPEN 잔존",
                "DB만 닫으면 키움 잔고와 어긋납니다.\n"
                "TRADING_MODE=live 로 실주문 청산하세요.\n"
                f"<code>{[(r[1], r[6]) for r in open_trades]}</code>",
            )
            return []

        if live:
            try:
                self._assert_live_ready()
            except Exception as e:
                conn.close()
                send_ops_error_alert(target_date, "LIVE 매도 중단 — 계좌/키 미설정", f"<code>{e}</code>")
                return []

        try:
            _pre_cash, _pre_inv, pre_equity = self.get_latest_account_state()
        except Exception as e:
            logger.warning(f"[{target_date}] pre-sell account snapshot failed: {e}")
            pre_equity = 0.0

        # Resolve T+1 OPEN prices. Never fall back to buy_price (that fakes a flat -0.23% fee-only exit).
        tickers = [t[1] for t in open_trades]
        open_price_map: dict = {}
        try:
            from scripts.fetch_today_kr_candles import fetch_open_prices
            open_price_map = fetch_open_prices(tickers, target_date)
            logger.info(f"[{target_date}] FDR open prices fetched for {len(open_price_map)}/{len(tickers)} tickers")
        except Exception as e:
            logger.warning(f"[{target_date}] FDR open fetch failed: {e}")

        if not open_price_map:
            lazy_df = pl.scan_parquet(str(DATA_PARQUET_PATH))
            df_open_candles = (
                lazy_df
                .filter((pl.col("date") == target_date))
                .select(["ticker", "open"])
                .collect()
                .to_pandas()
            )
            for _, r in df_open_candles.iterrows():
                clean_t = str(r["ticker"]).split(".")[0].zfill(6)
                open_price_map[clean_t] = float(r["open"])

        missing = [t for t in tickers if t not in open_price_map or open_price_map[t] <= 0]
        if missing:
            logger.error(
                f"[{target_date}] Missing T+1 open prices for {missing}. Aborting SELL "
                f"(will NOT flatten at buy_price / -0.23% fee)."
            )
            conn.close()
            send_ops_error_alert(
                target_date,
                "09:00 시가 조회 실패 — 매도 중단",
                f"시가가 없어 매수가로 청산하면 무조건 수수료 -0.23%가 됩니다.\n"
                f"미조회 종목: <code>{missing}</code>\n"
                "OPEN 포지션은 그대로 유지합니다.",
            )
            return []

        closed_trades = []
        total_pnl_krw = 0.0
        total_returned_cash = 0.0
        close_time_str = f"{target_date} 09:00:00"
        sell_failed = []

        for trade_id, ticker, name, buy_price, qty, buy_amount, _mode in open_trades:
            sell_price = open_price_map[ticker]
            sell_qty = int(qty)
            sell_ord_no = ""

            if live:
                from kr_stock.kiwoom_broker import holding_qty, market_sell
                held = holding_qty(ticker)
                if held <= 0:
                    sell_failed.append(f"{ticker} broker qty=0")
                    logger.error(f"[{target_date}] LIVE SELL skip {ticker}: no broker holding")
                    continue
                sell_qty = min(sell_qty, held)
                ok, sell_ord_no, raw = market_sell(ticker, sell_qty)
                if not ok:
                    msg = raw.get("return_msg") or raw
                    sell_failed.append(f"{ticker} {msg}")
                    logger.error(f"[{target_date}] LIVE SELL rejected {ticker}: {msg}")
                    continue
                logger.info(f"[{target_date}] LIVE SELL {ticker} qty={sell_qty} ord_no={sell_ord_no}")

            gross_sell_amount = sell_qty * sell_price
            net_sell_amount = gross_sell_amount * (1.0 - FEE_RATE)
            pnl_krw = net_sell_amount - buy_amount
            pnl_pct = (pnl_krw / buy_amount) * 100.0 if buy_amount else 0.0

            total_pnl_krw += pnl_krw
            total_returned_cash += net_sell_amount

            cursor.execute("""
                UPDATE paper_trades SET
                    sell_price = ?,
                    sell_amount = ?,
                    pnl_krw = ?,
                    pnl_pct = ?,
                    status = 'CLOSED',
                    close_time = ?,
                    sell_ord_no = ?
                WHERE id = ?
            """, (sell_price, net_sell_amount, pnl_krw, pnl_pct, close_time_str, sell_ord_no or None, trade_id))

            closed_trades.append({
                "ticker": ticker,
                "stock_name": name,
                "buy_price": buy_price,
                "sell_price": sell_price,
                "buy_qty": sell_qty,
                "pnl_krw": pnl_krw,
                "pnl_pct": pnl_pct
            })

        if live and sell_failed and not closed_trades:
            conn.close()
            send_ops_error_alert(
                target_date,
                "LIVE 매도 전부 실패 — OPEN 유지",
                "키움 매도가 거부되어 DB를 닫지 않았습니다.\n"
                + "\n".join(f"<code>{r}</code>" for r in sell_failed),
            )
            return []

        conn.commit()

        remaining_open = cursor.execute(
            "SELECT COALESCE(SUM(buy_amount), 0.0) FROM paper_trades WHERE status = 'OPEN'"
        ).fetchone()[0]

        if live:
            try:
                from kr_stock.kiwoom_broker import get_orderable_cash
                new_cash = get_orderable_cash()
            except Exception as e:
                logger.warning(f"[{target_date}] live cash refresh failed, using estimate: {e}")
                snap = cursor.execute(
                    "SELECT cash_balance FROM paper_account ORDER BY date DESC LIMIT 1"
                ).fetchone()
                new_cash = (snap[0] if snap else 0.0) + total_returned_cash
        else:
            snap = cursor.execute(
                "SELECT cash_balance FROM paper_account ORDER BY date DESC LIMIT 1"
            ).fetchone()
            leftover = snap[0] if snap else SEED_CAPITAL
            new_cash = leftover + total_returned_cash

        new_equity = new_cash + remaining_open
        daily_pct = (total_pnl_krw / pre_equity) * 100.0 if pre_equity > 0 else 0.0

        self._save_account_state(
            target_date, new_cash, remaining_open, daily_pnl=total_pnl_krw, daily_pct=daily_pct
        )
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

        # Fetch backtest signals for target_date using Kiwoom Condition Search
        candidate_codes = self.condition_manager.get_condition_search_codes(target_date)
        backtest_picks = self.scorer.get_candidates_for_date(target_date, top_k=TOP_K_TRADES, candidate_codes=candidate_codes)
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
