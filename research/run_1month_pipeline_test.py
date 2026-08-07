#!/usr/bin/env python3
"""
1-Month Pipeline Verification Script (100% REAL MarketMosaic FULL PACK: News + DART + Judal Themes)
---------------------------------------------------------------------------------------------------
Runs an end-to-end backtest for April 16, 2026 ~ May 15, 2026 using:
1. Real KRX candle data from data/kr_kline_processed.parquet.
2. Ticker -> Stock Name resolution via sector_info.db.
3. MarketMosaic Full Data Pack:
   - Meilisearch News Articles (567k index)
   - DART Corporate Filings (83k filings in dart.db)
   - Judal Theme Categories (323 themes & stock mappings in judal.db)
4. OpenRouter multi-model round-robin LLM sentiment scoring (4 free models).
5. Trade execution: Buy at 15:30 close, sell at T+1 09:00 open with 0.23% fee/tax.
6. Buy & Hold (B&H) performance comparison.
"""

import os
import json
import sqlite3
import pandas as pd
import numpy as np
import polars as pl
from pathlib import Path
from research.openrouter_evaluator import OpenRouterEvaluator
from research.marketmosaic_integrator import MarketMosaicIntegrator

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PARQUET = ROOT_DIR / "data" / "kr_kline_processed.parquet"
SECTOR_DB = Path("/mnt/data/finance/candles/KO/sector_info.db")
REPORT_DOC = ROOT_DIR / "docs" / "PIPELINE_1MONTH_VERIFICATION_REPORT.md"

# Target Date Period matching Real MarketMosaic Data Collection (2026-04-16 ~ 2026-05-15)
START_DATE = "2026-04-16"
END_DATE = "2026-05-15"

FEE_RATE = 0.0023  # 0.23% transaction tax & fees

def load_ticker_name_map() -> dict:
    name_map = {}
    if SECTOR_DB.exists():
        try:
            conn = sqlite3.connect(str(SECTOR_DB))
            cursor = conn.cursor()
            rows = cursor.execute("SELECT ticker, name FROM sectors").fetchall()
            for t, n in rows:
                clean_t = t.split(".")[0]
                name_map[clean_t] = n
            conn.close()
        except Exception:
            pass
    return name_map

def run_1month_pipeline():
    print("=========================================================================")
    print(f"  OVERNIGHT AI BACKTEST (REAL MARKETMOSAIC FULL PACK: NEWS+DART+THEMES)  ")
    print(f"  Test Period: {START_DATE} ~ {END_DATE}                                ")
    print("=========================================================================\n")

    # Step 1: Load Stock Name Mapping & Candle Data
    ticker_map = load_ticker_name_map()
    print(f"[Step 1] Loaded {len(ticker_map)} stock name mappings from sector_info.db.")
    
    print(f"Ingesting candle dataset from {DATA_PARQUET} via Polars...")
    lazy_df = pl.scan_parquet(str(DATA_PARQUET))
    df = (
        lazy_df
        .filter((pl.col("date") >= START_DATE) & (pl.col("date") <= END_DATE))
        .collect()
        .to_pandas()
    )
    
    print(f"  -> Total candle records loaded for period: {len(df):,}")
    if df.empty:
        print("ERROR: No candle data found for test period in parquet!")
        return

    # Map stock names
    df['clean_ticker'] = df['ticker'].apply(lambda x: str(x).split('.')[0])
    df['stock_name'] = df['clean_ticker'].map(ticker_map).fillna(df['ticker'])

    # Step 2: Initialize OpenRouter Evaluator & MarketMosaic Integrator
    print("\n[Step 2] Initializing OpenRouter Evaluator & MarketMosaic Full Pack Integrator...")
    evaluator = OpenRouterEvaluator()
    integrator = MarketMosaicIntegrator()
    print(f"  -> Integrated {len(integrator.theme_map)} Judal stock-theme relationships.")
    print("  -> Connected to MarketMosaic DART DB & Meilisearch News DB.")

    # Step 3: Candidate Selection
    print("\n[Step 3] Candidate Selection & Technical Base Scoring...")
    min_turnover = 3e10  # 300억 KRW
    candidates_mask = (
        (df['turnover'] >= min_turnover) &
        (df['high_close_ratio'] >= 0.85) &
        (df['close'] / df['open'] >= 1.02) &
        (df['close'] > df['sma_5']) &
        (df['next_open'].notnull())
    )
    
    cand_df = df[candidates_mask].copy()
    print(f"  -> High-conviction technical candidate setup count: {len(cand_df)}")
    
    # Technical Base Score
    cand_df['tech_score'] = (
        cand_df['high_close_ratio'] * 40 +
        (cand_df['close'] / cand_df['open'] - 1.0) * 100 * 20 +
        np.log1p(cand_df['turnover'] / 1e8) * 2
    )
    
    unique_dates = sorted(df['date'].unique())
    daily_stats = []
    trade_details = []
    
    print("\n[Step 4] Extracting Full Context (News+DART+Themes) & Scoring via OpenRouter LLM...")
    for date_str in unique_dates:
        day_cands = cand_df[cand_df['date'] == date_str]
        if day_cands.empty:
            daily_stats.append({"date": date_str, "return": 0.0, "trades": 0})
            continue
            
        top_cands = day_cands.sort_values(by='tech_score', ascending=False).head(5).copy()
        
        scored_picks = []
        for idx, row in top_cands.iterrows():
            ticker = row['clean_ticker']
            stock_name = row['stock_name']
            
            # Fetch FULL Market Context (News + DART Filings + Judal Themes)
            ctx = integrator.get_full_market_context(ticker, stock_name, date_str)
            composite_text = ctx['composite_summary']
            
            # Evaluate sentiment with Full Market Context
            llm_eval = evaluator.evaluate_sentiment(ticker, stock_name, f"Theme & News Context for {stock_name}", composite_text)
            llm_score = min(max(llm_eval.get("score", 0), -10), 30)
            llm_model = llm_eval.get("model_used", "N/A")
                
            total_score = row['tech_score'] + llm_score
            
            scored_picks.append({
                "row": row,
                "total_score": total_score,
                "llm_score": llm_score,
                "context_summary": composite_text[:50],
                "llm_model": llm_model
            })
            
        # Top 3 picks by final score
        scored_picks.sort(key=lambda x: x['total_score'], reverse=True)
        top_3_picks = scored_picks[:3]
        
        day_trades = []
        for pick in top_3_picks:
            row = pick['row']
            ticker = row['clean_ticker']
            stock_name = row['stock_name']
            gross_pnl = (row['next_open'] - row['close']) / row['close']
            net_pnl = gross_pnl - FEE_RATE
            day_trades.append(net_pnl)
            
            trade_details.append({
                "date": date_str,
                "ticker": ticker,
                "stock_name": stock_name,
                "entry_time": str(row['close_time']),
                "exit_time": str(row['next_open_time']),
                "entry_price": float(row['close']),
                "exit_price": float(row['next_open']),
                "tech_score": round(row['tech_score'], 1),
                "llm_score": pick['llm_score'],
                "total_score": round(pick['total_score'], 1),
                "context": pick['context_summary'],
                "llm_model": pick['llm_model'],
                "net_pnl_pct": round(net_pnl * 100, 2)
            })
            
        avg_day_return = np.mean(day_trades) if day_trades else 0.0
        daily_stats.append({"date": date_str, "return": avg_day_return, "trades": len(day_trades)})

    # Step 5: Compute Strategy & Buy & Hold Metrics
    res_df = pd.DataFrame(daily_stats)
    res_df['cum_return'] = (1 + res_df['return']).cumprod()
    
    total_strat_return = (res_df['cum_return'].iloc[-1] - 1.0) * 100
    
    pnl_list = [t['net_pnl_pct'] for t in trade_details]
    n_trades = len(pnl_list)
    win_rate = (pd.Series(pnl_list) > 0).mean() * 100 if n_trades > 0 else 0.0
    win_sum = sum(p for p in pnl_list if p > 0)
    loss_sum = abs(sum(p for p in pnl_list if p < 0))
    profit_factor = (win_sum / loss_sum) if loss_sum != 0 else 0.0
    
    cum_max = res_df['cum_return'].cummax()
    mdd = ((res_df['cum_return'] - cum_max) / cum_max).min() * 100
    
    # Calculate Buy & Hold Benchmark
    bh_returns = df.groupby('date')['close'].mean().pct_change().fillna(0.0)
    bh_cum = (1 + bh_returns).cumprod()
    bh_total_return = (bh_cum.iloc[-1] - 1.0) * 100
    
    outperformance = total_strat_return - bh_total_return
    
    print("\n=========================================================================")
    print(" REAL MARKETMOSAIC FULL PACK BACKTEST METRICS (NEWS + DART + JUDAL THEMES) ")
    print("=========================================================================")
    print(f" Test Period                     : {START_DATE} ~ {END_DATE}")
    print(f" Total Trades Executed           : {n_trades} trades")
    print(f" Overnight AI Strategy Return    : {total_strat_return:+.2f}%")
    print(f" Buy & Hold Benchmark Return     : {bh_total_return:+.2f}%")
    print(f" Outperformance vs B&H (Alpha)   : {outperformance:+.2f}%")
    print(f" Win Rate                        : {win_rate:.2f}%")
    print(f" Profit Factor                   : {profit_factor:.2f}")
    print(f" Max Drawdown (MDD)              : {mdd:.2f}%")
    print("=========================================================================\n")
    
    # Print sample trade logs with Full Context
    print("Sample Trade Logs with FULL MarketMosaic Pack (First 5 Trades):")
    sample_df = pd.DataFrame(trade_details[:5])
    print(sample_df[['date', 'ticker', 'stock_name', 'entry_price', 'exit_price', 'llm_score', 'context', 'net_pnl_pct']].to_string(index=False))

    # Write Markdown Verification Report
    with open(REPORT_DOC, "w", encoding="utf-8") as f:
        f.write("# Real MarketMosaic Full Pack Strategy Backtest Report (News + DART + Judal Themes)\n\n")
        f.write(f"### Backtest Period: `{START_DATE} ~ {END_DATE}`\n\n")
        f.write("### Data Source & System Architecture\n")
        f.write("- **Candle Data**: `data/kr_kline_processed.parquet` (Polars Engine).\n")
        f.write("- **Stock Names**: Mapped via `sector_info.db`.\n")
        f.write("- **MarketMosaic Full Pack Integration**:\n")
        f.write("  1. **News Articles**: Meilisearch DB (`http://localhost:37700`, 567k articles).\n")
        f.write("  2. **DART Filings**: SQLite DB (`dart.db`, 83k corporate filings).\n")
        f.write("  3. **Judal Themes**: SQLite DB (`judal.db`, 323 Judal theme categories).\n")
        f.write("- **LLM Sentiment Scoring**: OpenRouter 4-Model Round-Robin Evaluator.\n")
        f.write("- **Friction & Time Contract**: Entry at 15:30 close, Exit at T+1 09:00 open, 0.23% fee & tax applied.\n\n")
        f.write("### Performance KPI Summary\n\n")
        f.write(f"| Metric | Strategy | Buy & Hold Benchmark | Delta (Alpha) |\n")
        f.write(f"| :--- | ---: | ---: | ---: |\n")
        f.write(f"| **1-Month Return** | **{total_strat_return:+.2f}%** | {bh_total_return:+.2f}% | **{outperformance:+.2f}%** |\n")
        f.write(f"| **Win Rate** | **{win_rate:.2f}%** | N/A | N/A |\n")
        f.write(f"| **Profit Factor** | **{profit_factor:.2f}** | N/A | N/A |\n")
        f.write(f"| **Max Drawdown (MDD)** | **{mdd:.2f}%** | N/A | N/A |\n")
        f.write(f"| **Total Trades** | **{n_trades}** | N/A | N/A |\n\n")
        f.write("### Trade Execution Details (Full Context LLM Score Included)\n\n")
        if trade_details:
            f.write(pd.DataFrame(trade_details).to_markdown(index=False) + "\n\n")
        f.write("### Conclusion\n")
        f.write("Backtest executed using 100% MarketMosaic Full Pack (News + DART + Judal Themes). LLM scoring receives complete fundamental, news, and thematic context.\n")
        
    print(f"\nReport saved to {REPORT_DOC}")

if __name__ == "__main__":
    run_1month_pipeline()
