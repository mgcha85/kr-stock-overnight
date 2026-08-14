"""
Single Source of Truth: Scoring & Candidate Selection Inference Module
----------------------------------------------------------------------
Serves both Backtesting and Live Paper-Trading pipelines to guarantee 100% parity.
Uses Intraday MTF (Multi-Timeframe) LightGBM & PyTorch MLP models for maximum accuracy.
"""

import sqlite3
import pandas as pd
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import joblib
from pathlib import Path
from typing import List, Dict, Any, Optional

from kr_stock.config import (
    MODEL_DIR, DATA_PARQUET_PATH, JUDAL_DB_PATH, SECTOR_DB_PATH, FEE_RATE
)
from research.marketmosaic_integrator import MarketMosaicIntegrator

INTRA_15M_DIR = Path("/mnt/data/finance/candles/KO/interval=15m")


class DeepMTFOvernightNet(nn.Module):
    """PyTorch Deep Neural Network for 15-dim MTF Features"""
    def __init__(self, input_dim: int = 15):
        super(DeepMTFOvernightNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


DAILY_FEATURE_COLS = [
    'high_close_ratio', 'body_ratio', 'upper_shadow_ratio',
    'ret_1d', 'ret_3d', 'ret_5d', 'vol_ratio_5d',
    'bb_pct_b', 'bb_width', 'rsi_14'
]

MTF_INTRA_COLS = [
    'intra_ret_15m', 'intra_ret_1h', 'intra_vol_ratio_30m',
    'intra_high_pullback', 'intra_rsi_15m'
]

COMBINED_FEATURE_COLS = DAILY_FEATURE_COLS + MTF_INTRA_COLS


def load_ticker_name_map() -> Dict[str, str]:
    """Loads stock ticker code -> Korean name mapping from sector_info.db."""
    name_map = {}
    if SECTOR_DB_PATH.exists():
        try:
            conn = sqlite3.connect(str(SECTOR_DB_PATH))
            cursor = conn.cursor()
            rows = cursor.execute("SELECT ticker, name FROM sectors").fetchall()
            for t, n in rows:
                clean_t = str(t).split(".")[0].zfill(6)
                name_map[clean_t] = n
            conn.close()
        except Exception:
            pass
    return name_map


def compute_kline_features(df_candles: pd.DataFrame) -> pd.DataFrame:
    """Computes technical indicator features from candle OHLCV data."""
    df = df_candles.copy()
    
    high_low_range = df['high'] - df['low'] + 1e-5
    df['high_close_ratio'] = (df['close'] - df['low']) / high_low_range
    df['body_ratio'] = abs(df['close'] - df['open']) / high_low_range
    df['upper_shadow_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / high_low_range
    
    df['ret_1d'] = df.groupby('ticker')['close'].pct_change(1).fillna(0)
    df['ret_3d'] = df.groupby('ticker')['close'].pct_change(3).fillna(0)
    df['ret_5d'] = df.groupby('ticker')['close'].pct_change(5).fillna(0)
    
    df['vol_ma5'] = df.groupby('ticker')['turnover'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['vol_ratio_5d'] = df['turnover'] / (df['vol_ma5'] + 1e-5)
    
    ma20 = df.groupby('ticker')['close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
    std20 = df.groupby('ticker')['close'].transform(lambda x: x.rolling(20, min_periods=1).std()).fillna(1.0)
    df['bb_pct_b'] = (df['close'] - (ma20 - 2 * std20)) / (4 * std20 + 1e-5)
    df['bb_width'] = (4 * std20) / (ma20 + 1e-5)
    
    delta = df.groupby('ticker')['close'].diff().fillna(0)
    gain = (delta.where(delta > 0, 0)).groupby(df['ticker']).transform(lambda x: x.rolling(14, min_periods=1).mean())
    loss = (-delta.where(delta < 0, 0)).groupby(df['ticker']).transform(lambda x: x.rolling(14, min_periods=1).mean())
    rs = gain / (loss + 1e-5)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    if 'next_open' in df.columns:
        df['overnight_pnl'] = (df['next_open'] - df['close']) / df['close'] - FEE_RATE
        df['is_win'] = (df['overnight_pnl'] > 0).astype(int)

    return df


def load_intraday_15m_features_for_date(target_date: str) -> pd.DataFrame:
    """Loads 15m intraday candles for target_date and computes MTF features per stock."""
    try:
        dt = pd.to_datetime(target_date)
        year, month = dt.year, dt.month
        path_pattern = str(INTRA_15M_DIR / f"year={year}" / f"month={month}" / "*.parquet")
        
        lazy_15m = pl.scan_parquet(path_pattern)
        df_15m = (
            lazy_15m
            .filter(pl.col("datetime").dt.strftime('%Y-%m-%d') == target_date)
            .select(["code", "datetime", "open", "high", "low", "close", "volume"])
            .collect()
            .to_pandas()
        )
    except Exception:
        df_15m = pd.DataFrame()

    if df_15m.empty:
        return pd.DataFrame()

    df_15m['date'] = target_date
    df_15m['time'] = df_15m['datetime'].dt.strftime('%H:%M')
    df_15m['code'] = df_15m['code'].apply(lambda x: str(x).split('.')[0].zfill(6))
    df_15m = df_15m.sort_values(by=['code', 'datetime']).reset_index(drop=True)

    delta = df_15m.groupby('code')['close'].diff().fillna(0)
    gain = (delta.where(delta > 0, 0)).groupby(df_15m['code']).transform(lambda x: x.rolling(14, min_periods=1).mean())
    loss = (-delta.where(delta < 0, 0)).groupby(df_15m['code']).transform(lambda x: x.rolling(14, min_periods=1).mean())
    rs = gain / (loss + 1e-5)
    df_15m['rsi_15m'] = 100 - (100 / (1 + rs))

    records = []
    for (code, date), group in df_15m.groupby(['code', 'date']):
        group = group.sort_values('time')
        if len(group) < 4:
            continue
        
        last_candle = group.iloc[-1]
        last_1h = group.tail(4)
        last_30m = group.tail(2)
        rest_day = group.iloc[:-2]

        intra_ret_15m = (last_candle['close'] - last_candle['open']) / (last_candle['open'] + 1e-5)
        intra_ret_1h = (last_1h.iloc[-1]['close'] - last_1h.iloc[0]['open']) / (last_1h.iloc[0]['open'] + 1e-5)
        
        avg_vol_morning = rest_day['volume'].mean() + 1e-5
        intra_vol_ratio_30m = last_30m['volume'].mean() / avg_vol_morning
        
        day_high = group['high'].max()
        day_close = last_candle['close']
        intra_high_pullback = (day_close - day_high) / (day_high + 1e-5)
        
        intra_rsi_15m = last_candle['rsi_15m']

        records.append({
            'code': code,
            'date': date,
            'intra_ret_15m': float(intra_ret_15m),
            'intra_ret_1h': float(intra_ret_1h),
            'intra_vol_ratio_30m': float(intra_vol_ratio_30m),
            'intra_high_pullback': float(intra_high_pullback),
            'intra_rsi_15m': float(intra_rsi_15m)
        })

    return pd.DataFrame(records)


class OvernightScorer:
    """Loaded Judal + LightGBM + PyTorch Hybrid Scoring Engine."""
    def __init__(self, use_mtf: bool = False):
        self.is_mtf = use_mtf
        if self.is_mtf and (MODEL_DIR / "lgb_mtf_model.joblib").exists():
            self.gbm = joblib.load(MODEL_DIR / "lgb_mtf_model.joblib")
            self.scaler = joblib.load(MODEL_DIR / "mtf_scaler.joblib")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.pytorch_model = DeepMTFOvernightNet(input_dim=15).to(self.device)
            self.pytorch_model.load_state_dict(torch.load(MODEL_DIR / "pytorch_mtf_model.pt", map_location=self.device))
            self.pytorch_model.eval()
            self.feature_cols = COMBINED_FEATURE_COLS
        else:
            self.gbm = joblib.load(MODEL_DIR / "lgb_kline_model.joblib")
            self.scaler = joblib.load(MODEL_DIR / "kline_scaler.joblib")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.pytorch_model = DeepMTFOvernightNet(input_dim=10).to(self.device)
            self.pytorch_model.load_state_dict(torch.load(MODEL_DIR / "pytorch_kline_model.pt", map_location=self.device))
            self.pytorch_model.eval()
            self.feature_cols = DAILY_FEATURE_COLS

        self.ticker_map = load_ticker_name_map()
        self.integrator = MarketMosaicIntegrator()

    def get_candidates_for_date(
        self,
        target_date: str,
        top_k: int = 3,
        min_turnover: float = 2e10,
        max_stock_change: float = 29.0,
        min_p_lgb: float = 0.35,
        min_p_torch: float = 0.35,
        candidate_codes: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Calculates hybrid scores for target_date and returns top_k candidate stocks.
        If candidate_codes is provided (e.g. from Kiwoom Condition Search '종가베팅'),
        only fetches and analyzes candles for those candidate stocks.
        Guarantees exact parity between Backtest and Paper Trading.
        """
        # Clean candidate codes if provided
        clean_candidates = [str(c).split('.')[0].zfill(6) for c in candidate_codes] if candidate_codes is not None else None

        # 1. Judal Theme & Stock History
        conn_judal = sqlite3.connect(str(JUDAL_DB_PATH))
        df_hist = pd.read_sql_query("""
            SELECT crawl_date as date, code, name, change_rate as stock_change, neglect_index_52w, expected_return
            FROM stock_history
            WHERE crawl_date = ?
        """, conn_judal, params=[target_date])
        
        if df_hist.empty:
            conn_judal.close()
            return []

        if clean_candidates is not None:
            df_hist['code'] = df_hist['code'].apply(lambda x: str(x).split('.')[0].zfill(6))
            df_hist = df_hist[df_hist['code'].isin(clean_candidates)].copy()
            if df_hist.empty:
                conn_judal.close()
                return []

        df_ts = pd.read_sql_query("SELECT theme_idx, stock_code as code FROM theme_stocks", conn_judal)
        df_themes = pd.read_sql_query("SELECT theme_idx, name as theme_name FROM themes", conn_judal)
        conn_judal.close()

        df_joined = pd.merge(df_hist, df_ts, on='code', how='inner')
        df_joined = pd.merge(df_joined, df_themes, on='theme_idx', how='inner')
        
        theme_group = df_joined.groupby(['date', 'theme_idx', 'theme_name'])['stock_change'].agg(
            theme_avg_change='mean',
            theme_max_change='max',
            theme_stock_cnt='count'
        ).reset_index()

        df_stock_theme = pd.merge(df_joined, theme_group, on=['date', 'theme_idx', 'theme_name'], how='left')

        # 2. Daily Candle Data & Indicators
        try:
            target_dt = pd.to_datetime(target_date)
            start_date_calc = (target_dt - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
        except Exception:
            start_date_calc = "2026-01-01"

        lazy_df = pl.scan_parquet(str(DATA_PARQUET_PATH))
        
        # Filter candles for specified candidate_codes if provided
        candle_filter = (pl.col("date") >= start_date_calc) & (pl.col("date") <= target_date)
        if clean_candidates is not None:
            # Match 6-digit stock code
            candle_filter = candle_filter & (pl.col("ticker").str.slice(0, 6).is_in(clean_candidates))

        df_candles = (
            lazy_df
            .filter(candle_filter)
            .select(["date", "ticker", "open", "close", "high", "low", "turnover", "high_close_ratio", "next_open"])
            .collect()
            .to_pandas()
        )
        if df_candles.empty:
            return []

        df_candles = df_candles.sort_values(by=['ticker', 'date']).reset_index(drop=True)
        df_feat = compute_kline_features(df_candles)
        
        df_target_candles = df_feat[df_feat['date'] == target_date].copy()
        if df_target_candles.empty:
            return []

        df_target_candles['code'] = df_target_candles['ticker'].apply(lambda x: str(x).split('.')[0].zfill(6))
        merged = pd.merge(df_stock_theme, df_target_candles, on=['date', 'code'], how='inner')

        # 3. Merge Intraday 15m MTF Features if MTF model is enabled
        if self.is_mtf:
            df_intra = load_intraday_15m_features_for_date(target_date)
            if not df_intra.empty:
                merged = pd.merge(merged, df_intra, on=['code', 'date'], how='left')
                for col in MTF_INTRA_COLS:
                    merged[col] = merged[col].fillna(0.0)
            else:
                for col in MTF_INTRA_COLS:
                    merged[col] = 0.0

        merged['stock_name'] = merged['code'].map(self.ticker_map).fillna(merged['name'])

        if merged.empty:
            return []

        # 4. Model Inference
        X_mat = merged[self.feature_cols].fillna(0).values
        merged['p_lgb'] = self.gbm.predict(X_mat)
        
        X_scaled = self.scaler.transform(X_mat)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            p_torch = self.pytorch_model(X_tensor).cpu().numpy().flatten()
        merged['p_torch'] = p_torch

        # 5. Scoring Logic
        merged['is_leader'] = merged['stock_change'] >= merged['theme_max_change'] * 0.85
        merged['judal_score'] = (
            (merged['is_leader'].astype(int) * 35.0) +
            (np.clip(merged['theme_avg_change'], -5, 12) * 2.5) +
            (np.clip(merged['stock_change'], 2, 14) * 3.0) +
            (merged['high_close_ratio'] * 30.0) -
            (np.maximum(0, merged['stock_change'] - 15) * 4.0)
        )

        merged['hybrid_score'] = (
            merged['judal_score'] +
            (merged['p_lgb'] * 40.0) +
            (merged['p_torch'] * 40.0)
        )

        # Deduplicate per code
        deduped = merged.sort_values(by='hybrid_score', ascending=False).groupby('code').first().reset_index()

        # 6. Filtering Rules
        filtered = deduped[
            (deduped['turnover'] >= min_turnover) & 
            (deduped['stock_change'] < max_stock_change) & 
            (deduped['p_lgb'] >= min_p_lgb) & 
            (deduped['p_torch'] >= min_p_torch)
        ].copy()

        if filtered.empty:
            return []

        # Sort & Select Top-K
        top_picks = filtered.sort_values(by='hybrid_score', ascending=False).head(top_k)

        results = []
        for _, row in top_picks.iterrows():
            code = str(row['code']).zfill(6)
            stock_name = str(row['stock_name'])
            
            # Context Bonus
            ctx = self.integrator.get_full_market_context(code, stock_name, target_date)
            dart_cnt = len(ctx['dart_filings'])
            news_cnt = len(ctx['news_articles'])
            aux_bonus = (dart_cnt * 5.0) + (news_cnt * 3.0)
            final_score = float(row['hybrid_score']) + aux_bonus

            next_op = float(row['next_open']) if ('next_open' in row and not pd.isna(row['next_open'])) else float(row['close'])

            results.append({
                "date": target_date,
                "code": code,
                "ticker": code,
                "stock_name": stock_name,
                "theme_name": str(row['theme_name']),
                "close_price": float(row['close']),
                "next_open": next_op,
                "stock_change": float(row['stock_change']),
                "p_lgb": float(row['p_lgb']),
                "p_torch": float(row['p_torch']),
                "hybrid_score": final_score,
                "news_count": news_cnt,
                "dart_count": dart_cnt
            })

        return results
