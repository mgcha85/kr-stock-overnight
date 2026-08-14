"""Kiwoom REST account + order execution (kt00001 / kt00004 / kt10000 / kt10001)."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import requests

from kr_stock.kiwoom_live import _app_domain, request_access_token

logger = logging.getLogger(__name__)


def account_no() -> str:
    return (
        os.getenv("ACC_NO")
        or os.getenv("ACC_ID")
        or os.getenv("KIWOOM_ACC_NO")
        or ""
    ).strip()


def _headers(token: str, api_id: str) -> Dict[str, str]:
    return {
        "authorization": f"Bearer {token}",
        "api-id": api_id,
        "cont-yn": "N",
        "next-key": "",
        "Content-Type": "application/json;charset=UTF-8",
    }


def _post(api_id: str, body: Dict[str, Any], token: Optional[str] = None) -> Dict[str, Any]:
    token = token or request_access_token()
    kind = "acnt" if api_id.startswith("kt000") else "ordr"
    url = f"{_app_domain()}/api/dostk/{kind}"
    resp = requests.post(url, json=body, headers=_headers(token, api_id), timeout=20)
    resp.raise_for_status()
    data = resp.json()
    logger.info("[Kiwoom %s] return_code=%s msg=%s", api_id, data.get("return_code"), data.get("return_msg"))
    return data


def get_orderable_cash() -> float:
    acc = account_no()
    if not acc:
        raise RuntimeError("ACC_NO not set")
    data = _post(
        "kt00001",
        {"canm": acc, "qry_tp": "0", "dmst_stex_tp": "KRX"},
    )
    for key in ("ord_alow_amt", "entr", "pymn_alow_amt"):
        raw = data.get(key)
        if raw is None:
            continue
        try:
            val = float(str(raw).replace(",", "").strip())
            if val >= 0:
                return val
        except ValueError:
            continue
    raise RuntimeError(f"Could not parse orderable cash from kt00001: {data}")


def get_holdings() -> List[Dict[str, Any]]:
    acc = account_no()
    if not acc:
        raise RuntimeError("ACC_NO not set")
    data = _post(
        "kt00004",
        {"canm": acc, "qry_tp": "1", "dmst_stex_tp": "KRX"},
    )
    rows = data.get("stk_acnt_evlt_prst") or data.get("holdings") or []
    out = []
    for r in rows:
        code = str(r.get("stk_cd") or "").replace("A", "")
        digits = "".join(ch for ch in code if ch.isdigit())[-6:].zfill(6) if code else ""
        qty = int(float(str(r.get("rmnd_qty") or "0").replace(",", "") or 0))
        avg = float(str(r.get("avg_prc") or "0").replace(",", "") or 0)
        if digits and qty > 0:
            out.append({"ticker": digits, "qty": qty, "avg_prc": avg, "name": r.get("stk_nm")})
    return out


def holding_qty(ticker: str) -> int:
    want = str(ticker).zfill(6)
    for h in get_holdings():
        if h["ticker"] == want:
            return int(h["qty"])
    return 0


def market_buy(ticker: str, qty: int) -> Tuple[bool, str, Dict[str, Any]]:
    """kt10000 market buy. Returns (ok, ord_no, raw)."""
    if qty <= 0:
        return False, "", {"return_msg": "qty<=0"}
    data = _post(
        "kt10000",
        {
            "dmst_stex_tp": "KRX",
            "stk_cd": str(ticker).zfill(6),
            "ord_qty": str(int(qty)),
            "ord_uv": "",
            "trde_tp": "3",
            "cond_uv": "",
        },
    )
    code = data.get("return_code")
    ord_no = str(data.get("ord_no") or "")
    ok = (code in (0, "0", None) and bool(ord_no)) or bool(ord_no)
    if code not in (None, 0, "0") and not ord_no:
        ok = False
    return ok, ord_no, data


def market_sell(ticker: str, qty: int) -> Tuple[bool, str, Dict[str, Any]]:
    """kt10001 market sell. Returns (ok, ord_no, raw)."""
    if qty <= 0:
        return False, "", {"return_msg": "qty<=0"}
    data = _post(
        "kt10001",
        {
            "dmst_stex_tp": "KRX",
            "stk_cd": str(ticker).zfill(6),
            "ord_qty": str(int(qty)),
            "ord_uv": "",
            "trde_tp": "3",
            "cond_uv": "",
        },
    )
    code = data.get("return_code")
    ord_no = str(data.get("ord_no") or "")
    ok = bool(ord_no) or code in (0, "0", None)
    if code not in (None, 0, "0") and not ord_no:
        ok = False
    return ok, ord_no, data
