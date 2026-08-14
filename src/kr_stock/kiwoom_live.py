"""Kiwoom REST OpenAPI live HTS condition search (OAuth + WebSocket)."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

TOKEN_PATH = Path(
    os.getenv("KIWOOM_TOKEN_PATH", "/mnt/data/projects/kr_stock/data/kiwoom_access_token.json")
)


def _live_enabled() -> bool:
    return os.getenv("KIWOOM_LIVE", "0").strip() in {"1", "true", "True", "yes"}


def _app_key() -> str:
    return os.getenv("KIWOOM_APP_KEY") or os.getenv("APP_KEY") or ""


def _secret_key() -> str:
    return os.getenv("KIWOOM_SECRET_KEY") or os.getenv("SECRET_KEY") or ""


def _app_domain() -> str:
    return os.getenv("APP_DOMAIN", "https://api.kiwoom.com").rstrip("/")


def _ws_url() -> str:
    return os.getenv("WS_URL", "wss://api.kiwoom.com:10000/api/dostk/websocket")


def request_access_token() -> str:
    app_key, secret = _app_key(), _secret_key()
    if not app_key or not secret:
        raise RuntimeError("APP_KEY / SECRET_KEY not set")
    url = f"{_app_domain()}/oauth2/token"
    resp = requests.post(
        url,
        json={
            "grant_type": "client_credentials",
            "appkey": app_key,
            "secretkey": secret,
        },
        headers={"Content-Type": "application/json;charset=UTF-8"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    token = data.get("token")
    if not token:
        raise RuntimeError(f"Kiwoom OAuth failed: {data}")
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(json.dumps({"token": token}))
    return token


def fetch_hts_condition_codes(condition_name: str = "종가베팅") -> List[str]:
    """LOGIN → CNSRLST → CNSRREQ for `condition_name`. Returns 6-digit codes."""
    import websocket

    token = request_access_token()
    ws = websocket.create_connection(_ws_url(), timeout=20)
    codes: List[str] = []
    try:
        ws.send(json.dumps({"trnm": "LOGIN", "token": token}))
        if not _wait_trnm(ws, "LOGIN", ok_code=0):
            raise RuntimeError("Kiwoom WS LOGIN failed")

        ws.send(json.dumps({"trnm": "CNSRLST"}))
        lst = _wait_trnm(ws, "CNSRLST")
        if not lst:
            raise RuntimeError("Kiwoom CNSRLST timeout")

        seq = None
        for item in lst.get("data") or []:
            if isinstance(item, list) and len(item) >= 2 and item[1] == condition_name:
                seq = str(item[0])
                break
        if not seq:
            names = [i[1] for i in (lst.get("data") or []) if isinstance(i, list) and len(i) >= 2]
            raise RuntimeError(f"HTS condition '{condition_name}' not found. available={names[:20]}")

        cont_yn, next_key = "N", ""
        seen = set()
        while True:
            ws.send(
                json.dumps(
                    {
                        "trnm": "CNSRREQ",
                        "seq": seq,
                        "search_type": "0",
                        "stex_tp": "K",
                        "cont_yn": cont_yn,
                        "next_key": next_key,
                    }
                )
            )
            msg = _wait_trnm(ws, "CNSRREQ")
            if not msg:
                break
            for item in msg.get("data") or []:
                raw = None
                if isinstance(item, dict):
                    raw = item.get("9001") or item.get("code") or item.get("stk_cd")
                if not raw:
                    continue
                clean = "".join(ch for ch in str(raw) if ch.isdigit())[-6:].zfill(6)
                if clean not in seen:
                    seen.add(clean)
                    codes.append(clean)
            cont_yn = str(msg.get("cont_yn") or msg.get("cont-yn") or "N").upper()
            next_key = str(msg.get("next_key") or msg.get("next-key") or "")
            if cont_yn != "Y" or not next_key:
                break
        logger.info("[Kiwoom LIVE] '%s' returned %s codes", condition_name, len(codes))
        return codes
    finally:
        try:
            ws.close()
        except Exception:
            pass


def fetch_live_codes_if_enabled(condition_name: str) -> Optional[List[str]]:
    if not _live_enabled():
        return None
    if not _app_key() or not _secret_key():
        logger.warning("KIWOOM_LIVE=1 but APP_KEY/SECRET_KEY missing")
        return None
    try:
        codes = fetch_hts_condition_codes(condition_name)
        print(f"[Kiwoom LIVE] HTS '{condition_name}' returned {len(codes)} candidates: {codes}")
        return codes
    except Exception as e:
        logger.error("Kiwoom LIVE HTS search failed: %s", e, exc_info=True)
        print(f"[Kiwoom LIVE] HTS search failed: {e}")
        return None


def _wait_trnm(ws, trnm: str, ok_code: Optional[int] = None, max_pumps: int = 40):
    for _ in range(max_pumps):
        raw = ws.recv()
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
        if msg.get("trnm") == trnm:
            if ok_code is not None and msg.get("return_code", 0) != ok_code:
                raise RuntimeError(f"{trnm} error: {msg}")
            return msg
    return None
