import os
import sqlite3
import pandas as pd
import polars as pl
from pathlib import Path
from typing import List, Optional, Dict, Any

from kr_stock.inference import JUDAL_DB_PATH, DATA_PARQUET_PATH
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
        """
        Fetches candidate stock codes matching the '종가베팅' condition.
        First tries live Kiwoom OpenAPI, then falls back to DB condition simulation (A, B, C, D, E, H).
        """
        # 0. Check custom environment variable / file override if provided
        env_codes = os.getenv("KIWOOM_CANDIDATE_CODES")
        if env_codes:
            codes = [c.strip().zfill(6) for c in env_codes.split(",") if c.strip()]
            print(f"[Kiwoom Condition] Using environment override candidate codes ({len(codes)}): {codes}")
            return codes

        # 1. Real Kiwoom HTS condition search (OAuth + WebSocket)
        live_codes = fetch_live_codes_if_enabled(self.condition_name)
        if live_codes:
            return live_codes

        # 2. Optional HTTP bridge (ignored when it is the localhost mock)
        live_codes = self.fetch_candidate_codes_from_api()
        if live_codes is not None:
            return live_codes

        # 2. Offline / Backtest Simulation of "종가베팅" Condition (Refined HTS Criteria)
        print(f"[Kiwoom Condition] Simulating '{self.condition_name}' condition for date: {target_date}...")
        
        conn_judal = sqlite3.connect(str(JUDAL_DB_PATH))
        df_hist = pd.read_sql_query("""
            SELECT code, change_rate as stock_change, neglect_index_52w
            FROM stock_history
            WHERE crawl_date = ?
        """, conn_judal, params=[target_date])
        conn_judal.close()

        if df_hist.empty:
            return []

        df_hist['code'] = df_hist['code'].apply(lambda x: str(x).split('.')[0].zfill(6))

        # Read candle data for target_date to evaluate Price, Turnover, SMA20
        try:
            target_dt = pd.to_datetime(target_date)
            start_date = (target_dt - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
        except Exception:
            start_date = "2026-01-01"

        lazy_df = pl.scan_parquet(str(DATA_PARQUET_PATH))
        df_candles = (
            lazy_df
            .filter((pl.col("date") >= start_date) & (pl.col("date") <= target_date))
            .select(["date", "ticker", "close", "turnover"])
            .collect()
            .to_pandas()
        )
        if df_candles.empty:
            return []

        df_candles['code'] = df_candles['ticker'].apply(lambda x: str(x).split('.')[0].zfill(6))
        
        # Calculate 20-day SMA & 52-week low
        df_candles = df_candles.sort_values(by=['code', 'date']).reset_index(drop=True)
        df_candles['sma_20'] = df_candles.groupby('code')['close'].transform(lambda x: x.rolling(20, min_periods=5).mean())
        df_candles['low_52w'] = df_candles.groupby('code')['close'].transform(lambda x: x.rolling(250, min_periods=10).min())
        
        # Filter for target_date
        df_day = df_candles[df_candles['date'] == target_date].copy()
        if df_day.empty:
            return []

        # Condition D: Top 150 Turnover Rank
        df_day['turnover_rank'] = df_day['turnover'].rank(ascending=False)

        # Merge with Judal info
        merged = pd.merge(df_day, df_hist, on='code', how='inner')

        # Refined Conditions matching HTS "종가베팅"
        cond_A = merged['turnover'] >= 2e10  # A: 거래대금 >= 200억
        cond_B1 = (merged['stock_change'] >= 10.0) & (merged['stock_change'] <= 28.5) # 강한 등락률
        cond_B2 = (merged['stock_change'] >= 5.0) & (merged['turnover'] >= 5e10)        # 거래대금 500억 이상 유입
        cond_B = cond_B1 | cond_B2
        cond_C = (merged['close'] >= 2000) & (merged['close'] <= 500000)  # C: 주가범위 2,000원 ~ 500,000원
        cond_D = merged['turnover_rank'] <= 150  # D: 거래대금 순위 상위 150위
        cond_E = (merged['close'] > merged['low_52w'])  # E: 52주 신저가 종목 제외
        cond_H = merged['close'] > merged['sma_20']  # H: 종가 > 20일 이동평균선

        # Target Exclusion (우선주 제외, 스팩 제외)
        cond_excl = ~merged['code'].str.endswith(('1', '2', '3', '4', '5', '6', '7', '8', '9', 'K', 'L', 'M'))

        filtered = merged[cond_A & cond_B & cond_C & cond_D & cond_E & cond_H & cond_excl]
        candidate_codes = filtered['code'].unique().tolist()
        
        print(f"[Kiwoom Condition] '{self.condition_name}' matched {len(candidate_codes)} candidates on {target_date}: {candidate_codes}")
        return candidate_codes


def _is_mock_endpoint(url: str) -> bool:
    u = (url or "").lower()
    return ":5000" in u and "/api/condition" in u

