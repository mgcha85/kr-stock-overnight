"""Auction expected match at 15:28: 예상체결가 / 예상체결수량.

15:20–15:30 last trade is frozen. Expected price/qty keep moving as auction
orders arrive and cancel.

Sources (paper and live both fetch; paper only skips the order):
1. Websocket 0H 주식예상체결 (realtime ticks)
2. ka10001 snapshot exp_cntr_pric / exp_cntr_qty
3. ka10004 호가: if best bid == best ask, that is the auction price
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import requests

from kr_stock.kiwoom_live import _app_domain, _ws_url, request_access_token

logger = logging.getLogger(__name__)


def _parse_kiwoom_num(raw: Any) -> float:
    if raw is None:
        return 0.0
    s = str(raw).replace(",", "").strip()
    if not s:
        return 0.0
    if s[0] in "+-":
        s = s[1:]
    try:
        return abs(float(s))
    except ValueError:
        return 0.0


def _headers(token: str, api_id: str) -> Dict[str, str]:
    return {
        "authorization": f"Bearer {token}",
        "api-id": api_id,
        "cont-yn": "N",
        "next-key": "",
        "Content-Type": "application/json;charset=UTF-8",
    }


def _post(api_id: str, path: str, body: Dict[str, Any], token: str) -> Dict[str, Any]:
    url = f"{_app_domain()}{path}"
    resp = requests.post(url, json=body, headers=_headers(token, api_id), timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_expected_match(ticker: str, token: Optional[str] = None) -> Dict[str, float]:
    """ka10001 snapshot. During auction exp_* updates; last print stays in cur_prc."""
    code = "".join(ch for ch in str(ticker) if ch.isdigit())[-6:].zfill(6)
    token = token or request_access_token()
    data = _post("ka10001", "/api/dostk/stkinfo", {"stk_cd": code}, token)
    if data.get("return_code") not in (None, 0, "0"):
        logger.warning("[ka10001] %s return=%s %s", code, data.get("return_code"), data.get("return_msg"))
    return {
        "ticker": code,
        "exp_price": _parse_kiwoom_num(data.get("exp_cntr_pric")),
        "exp_qty": _parse_kiwoom_num(data.get("exp_cntr_qty")),
        "cur_prc": _parse_kiwoom_num(data.get("cur_prc")),
        "source": "ka10001",
    }


def get_auction_touch(ticker: str, token: Optional[str] = None) -> Dict[str, float]:
    """ka10004 호가. 동시호가 중 최우선 매수=매도면 그 가격이 예상체결가."""
    code = "".join(ch for ch in str(ticker) if ch.isdigit())[-6:].zfill(6)
    token = token or request_access_token()
    data = _post("ka10004", "/api/dostk/mrkcond", {"stk_cd": code}, token)
    sel = _parse_kiwoom_num(data.get("sel_fpr_bid"))
    buy = _parse_kiwoom_num(data.get("buy_fpr_bid"))
    sel_qty = _parse_kiwoom_num(data.get("sel_fpr_req"))
    buy_qty = _parse_kiwoom_num(data.get("buy_fpr_req"))
    exp_price = sel if sel > 0 and sel == buy else 0.0
    return {
        "ticker": code,
        "exp_price": exp_price,
        "exp_qty": min(sel_qty, buy_qty) if exp_price else 0.0,
        "cur_prc": 0.0,
        "source": "ka10004",
    }


def poll_expected_ws(
    tickers: List[str],
    token: Optional[str] = None,
    listen_sec: float = 2.5,
) -> Dict[str, Dict[str, float]]:
    """Subscribe websocket 0H 주식예상체결 and keep the latest tick per code."""
    try:
        import websocket
    except ImportError:
        logger.warning("[0H] websocket-client not installed; REST only")
        return {}

    codes = ["".join(ch for ch in str(t) if ch.isdigit())[-6:].zfill(6) for t in tickers]
    token = token or request_access_token()
    out: Dict[str, Dict[str, float]] = {}
    ws = websocket.create_connection(_ws_url(), timeout=15)
    deadline = time.time() + listen_sec
    try:
        ws.send(json.dumps({"trnm": "LOGIN", "token": token}))
        logged_in = False
        while time.time() < deadline + 5:
            raw = ws.recv()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            msg = json.loads(raw)
            if msg.get("trnm") == "PING":
                ws.send(raw if isinstance(raw, str) else json.dumps(msg))
                continue
            if msg.get("trnm") == "LOGIN":
                if msg.get("return_code", 0) != 0:
                    raise RuntimeError(f"WS LOGIN failed: {msg.get('return_msg')}")
                logged_in = True
                break
        if not logged_in:
            return {}

        ws.send(
            json.dumps(
                {
                    "trnm": "REG",
                    "grp_no": "1",
                    "refresh": "1",
                    "data": [{"item": codes, "type": ["0H"]}],
                }
            )
        )
        listen_until = time.time() + listen_sec
        while time.time() < listen_until:
            ws.settimeout(max(0.2, listen_until - time.time()))
            try:
                raw = ws.recv()
            except Exception:
                continue
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            if msg.get("trnm") == "PING":
                try:
                    ws.send(raw if isinstance(raw, str) else json.dumps(msg))
                except Exception:
                    pass
                continue
            rows = msg.get("data") if msg.get("trnm") in ("REAL", "REG") else None
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                rtype = str(row.get("type") or row.get("name") or "")
                if rtype not in ("0H", "주식예상체결") and row.get("values") is None and "10" not in row:
                    continue
                item = str(row.get("item") or row.get("9001") or "")
                digits = "".join(ch for ch in item if ch.isdigit())[-6:].zfill(6) if item else ""
                values = row.get("values") if isinstance(row.get("values"), dict) else row
                price = _parse_kiwoom_num(values.get("10") or values.get("27") or values.get("exp_cntr_pric"))
                qty = _parse_kiwoom_num(values.get("15") or values.get("13") or values.get("exp_cntr_qty"))
                if digits and price > 0:
                    out[digits] = {
                        "ticker": digits,
                        "exp_price": price,
                        "exp_qty": qty,
                        "cur_prc": price,
                        "source": "0H",
                    }
    finally:
        try:
            ws.close()
        except Exception:
            pass
    logger.info("[0H] ticks for %s/%s names", len(out), len(codes))
    return out


def overlay_expected_match(
    picks: List[Dict[str, Any]],
    listen_sec: float = 2.5,
) -> List[Dict[str, Any]]:
    """Replace pick close_price with latest 예상체결가. Paper and live both call this."""
    if not picks:
        return picks
    token = request_access_token()
    tickers = [str(p.get("ticker") or "").zfill(6) for p in picks]
    quotes = poll_expected_ws(tickers, token=token, listen_sec=listen_sec)

    for t in tickers:
        if quotes.get(t, {}).get("exp_price", 0) > 0:
            continue
        try:
            q = get_expected_match(t, token=token)
        except Exception as e:
            logger.warning("[ka10001] %s failed: %s", t, e)
            q = {}
        if float(q.get("exp_price") or 0) > 0:
            quotes[t] = q
            continue
        try:
            book = get_auction_touch(t, token=token)
        except Exception as e:
            logger.warning("[ka10004] %s failed: %s", t, e)
            book = {}
        if float(book.get("exp_price") or 0) > 0:
            quotes[t] = book

    overlaid = []
    for p in picks:
        row = dict(p)
        t = str(row.get("ticker") or "").zfill(6)
        row["ticker"] = t
        row["last_print"] = float(row.get("close_price") or 0.0)
        q = quotes.get(t) or {}
        exp = float(q.get("exp_price") or 0.0)
        if exp > 0:
            row["close_price"] = exp
            row["exp_cntr_pric"] = exp
            row["exp_cntr_qty"] = float(q.get("exp_qty") or 0.0)
            row["price_source"] = str(q.get("source") or "exp_cntr")
            logger.info(
                "[auction] %s expected=%s qty=%s src=%s last_print=%s",
                t, exp, row["exp_cntr_qty"], row["price_source"], row["last_print"],
            )
        else:
            row["price_source"] = "last_print_fallback"
            row["exp_cntr_pric"] = 0.0
            row["exp_cntr_qty"] = 0.0
            logger.warning(
                "[auction] %s no 예상체결가 (not in 15:20-15:30 or empty); sizing uses last_print=%.0f",
                t, row["last_print"],
            )
        overlaid.append(row)
    return overlaid
