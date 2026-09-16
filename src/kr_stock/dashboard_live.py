"""Sync overnight paper/live book to Backtest Lab (8082) and poll trading mode."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests

from kr_stock import config as kr_config

logger = logging.getLogger(__name__)


def overnight_api_base() -> str:
    if kr_config.OVERNIGHT_API_URL:
        return kr_config.OVERNIGHT_API_URL.rstrip("/")
    parsed = urlparse(kr_config.DASHBOARD_API_URL)
    origin = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else "http://146.56.115.71:8082"
    return f"{origin}/api/overnight"


def apply_remote_trading_mode() -> str:
    """Prefer 8082 FE toggle. On fetch failure keep current env/runtime mode."""
    url = f"{overnight_api_base()}/mode"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        remote = str(data.get("trading_mode") or "").strip().lower()
        if remote in {"paper", "live"}:
            applied = kr_config.set_trading_mode(remote)
            logger.info("[dashboard] trading_mode from 8082: %s", applied)
            return applied
    except Exception as e:
        logger.warning("[dashboard] mode fetch failed, keeping %s: %s", kr_config.TRADING_MODE, e)
    return kr_config.TRADING_MODE


def _trade_rows(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT id, date, ticker, stock_name, theme_name, buy_price, buy_qty, buy_amount,
               sell_price, sell_amount, pnl_krw, pnl_pct, status, open_time, close_time,
               hybrid_score, p_lgb, p_torch, buy_ord_no, sell_ord_no,
               COALESCE(trading_mode, execution_mode, 'paper') AS trading_mode
        FROM paper_trades
        ORDER BY id
        """
    ).fetchall()
    out = []
    for r in rows:
        out.append({k: r[k] for k in r.keys()})
    return out


def _latest_account(conn: sqlite3.Connection) -> Optional[Dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """
        SELECT date, cash_balance, invested_amount, total_equity,
               daily_pnl_krw, daily_pnl_pct, weekly_pnl_pct, monthly_pnl_pct, updated_at
        FROM paper_account
        ORDER BY date DESC LIMIT 1
        """
    ).fetchone()
    if not row:
        return None
    return {k: row[k] for k in row.keys()}


def build_sync_payload(db_path=None) -> Dict[str, Any]:
    path = db_path or kr_config.PAPER_DB_PATH
    conn = sqlite3.connect(str(path))
    try:
        trades = _trade_rows(conn)
        account = _latest_account(conn)
    finally:
        conn.close()
    return {
        "trading_mode": kr_config.TRADING_MODE,
        "engine_mode": kr_config.TRADING_MODE,
        "max_alloc_per_ticker": kr_config.MAX_ALLOC_PER_TICKER,
        "seed_capital": kr_config.SEED_CAPITAL,
        "heartbeat_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "account": account,
        "trades": trades,
    }


def sync_book(db_path=None) -> bool:
    payload = build_sync_payload(db_path)
    url = overnight_api_base()
    try:
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code >= 400:
            logger.warning("[dashboard] sync %s: %s %s", url, resp.status_code, resp.text[:300])
            return False
        logger.info("[dashboard] synced trades=%s", len(payload.get("trades") or []))
        return True
    except Exception as e:
        logger.warning("[dashboard] sync failed: %s", e)
        return False
