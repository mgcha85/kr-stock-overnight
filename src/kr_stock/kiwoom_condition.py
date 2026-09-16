import os
from typing import List, Optional

import pandas as pd
import polars as pl

from kr_stock.inference import DATA_PARQUET_PATH
from kr_stock.kiwoom_live import fetch_live_codes_if_enabled

class KiwoomConditionManager:
    """
    Manages Kiwoom Condition Search (키움 조건검색) API integration
    and offline simulation for condition: "종가베팅".
    """
    def __init__(self, condition_name: str = "종가베팅"):
        self.condition_name = condition_name

    def fetch_candidate_codes_from_api(self) -> Optional[List[str]]:
        """
        Attempts to fetch real-time candidate codes from Kiwoom OpenAPI (REST/COM/PyKiwoom).
        Returns None if Kiwoom API service is not running or in offline/backtest mode.
        """
        # 1. Check environment variables or Kiwoom API service endpoint
        kiwoom_api_url = os.getenv("KIWOOM_API_URL", "http://localhost:5000/api/condition")
        try:
            import requests
            resp = requests.get(f"{kiwoom_api_url}?name={self.condition_name}", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("source") == "mock" or _is_mock_endpoint(kiwoom_api_url):
                    if os.getenv("KIWOOM_ACCEPT_MOCK", "0") != "1":
                        print(
                            f"[Kiwoom API] Ignoring mock/stale condition endpoint "
                            f"({kiwoom_api_url}, as_of={data.get('as_of')}). "
                            "Falling back to offline HTS sim."
                        )
                        return None
                codes = data.get("codes", [])
                print(f"[Kiwoom API] Real-time '{self.condition_name}' returned {len(codes)} candidates: {codes}")
                return [str(c).zfill(6) for c in codes]
        except Exception:
            # Offline or API not running, fallback to DB-based condition search
            pass
        return None

    def get_condition_search_codes(self, target_date: str) -> List[str]:
        """Buy universe = parquet G+H (same as backtest). Kiwoom HTS is logged only."""
        env_codes = os.getenv("KIWOOM_CANDIDATE_CODES")
        if env_codes:
            codes = [c.strip().zfill(6) for c in env_codes.split(",") if c.strip()]
            print(f"[Kiwoom Condition] Using environment override candidate codes ({len(codes)}): {codes}")
            return codes

        gh = self._offline_gh_codes(target_date)
        live_codes = fetch_live_codes_if_enabled(self.condition_name)
        if live_codes is None:
            live_codes = self.fetch_candidate_codes_from_api()
        if live_codes:
            hts = {c for c in live_codes if str(c).isdigit() and len(c) == 6}
            gh_set = set(gh)
            print(
                f"[Kiwoom Condition] HTS {len(hts)} vs parquet G+H {len(gh_set)} "
                f"intersect {len(hts & gh_set)} HTS-only {sorted(hts - gh_set)} "
                f"G+H-only {sorted(gh_set - hts)}"
            )
        return gh

    def _offline_gh_codes(self, target_date: str) -> List[str]:
        """G: open/prev_close in [2%, 28%]. H: turnover >= 200억. 6-digit codes only."""
        print(f"[Kiwoom Condition] Simulating '{self.condition_name}' condition for date: {target_date}...")

        try:
            target_dt = pd.to_datetime(target_date)
            start_date = (target_dt - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        except Exception:
            start_date = "2026-01-01"

        df_candles = (
            pl.scan_parquet(str(DATA_PARQUET_PATH))
            .filter((pl.col("date") >= start_date) & (pl.col("date") <= target_date))
            .select(["date", "ticker", "open", "close", "turnover"])
            .collect()
            .to_pandas()
        )
        if df_candles.empty:
            return []

        df_candles["code"] = df_candles["ticker"].apply(lambda x: str(x).split(".")[0].zfill(6))
        df_candles = df_candles.sort_values(by=["code", "date"]).reset_index(drop=True)
        df_candles["prev_close"] = df_candles.groupby("code")["close"].shift(1)

        df_day = df_candles[df_candles["date"] == target_date].copy()
        if df_day.empty:
            return []

        df_day["open_gap"] = df_day["open"] / df_day["prev_close"] - 1.0
        cond_H = df_day["turnover"] >= 2e10
        cond_G = df_day["open_gap"].notna() & (df_day["open_gap"] >= 0.02) & (df_day["open_gap"] <= 0.28)
        candidate_codes = [
            c
            for c in df_day.loc[cond_H & cond_G, "code"].unique().tolist()
            if str(c).isdigit() and len(str(c)) == 6
        ]

        print(
            f"[Kiwoom Condition] '{self.condition_name}' matched {len(candidate_codes)} candidates "
            f"on {target_date}: {candidate_codes}"
        )
        return candidate_codes


def _is_mock_endpoint(url: str) -> bool:
    u = (url or "").lower()
    return ":5000" in u and "/api/condition" in u

