#!/usr/bin/env python3
"""
Upload MTF Backtest Results to Backtest Dashboard Server (http://146.56.115.71:8082/api/backtest)
-----------------------------------------------------------------------------------------------
Executes June 2026 Walk-Forward simulation with the updated 15-dim MTF OvernightScorer,
calculates CAGR, Sharpe, Win Rate, MDD, daily/weekly/monthly breakdowns, and trade details,
and uploads payload to the central Backtest Lab API.
"""

import sys
import sqlite3
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from kr_stock.inference import OvernightScorer
from kr_stock.config import DASHBOARD_API_URL, JUDAL_DB_PATH, BACKTEST_DB_PATH

TEST_START = "2026-01-06"
TEST_END = "2026-07-30"


def run_and_upload_backtest():
    print("=========================================================================")
    print("   EXECUTING JUNE 2026 MTF BACKTEST & UPLOADING TO DASHBOARD SERVER     ")
    print("=========================================================================\n")

    scorer = OvernightScorer()
    print(f" -> Scorer initialized (MTF Active: {scorer.is_mtf})")

    # Get trading days in June 2026
    conn_judal = sqlite3.connect(str(JUDAL_DB_PATH))
    trading_days = pd.read_sql_query("""
        SELECT DISTINCT crawl_date as date
        FROM stock_history
        WHERE crawl_date BETWEEN ? AND ?
        ORDER BY crawl_date ASC
    """, conn_judal, params=[TEST_START, TEST_END])['date'].tolist()
    conn_judal.close()

    print(f" -> Found {len(trading_days)} trading days in test window ({TEST_START} to {TEST_END}).")

    seed_capital = 10_000_000.0
    current_cash = seed_capital
    portfolio_history = []
    trade_details = []

    daily_returns_map = {}

    for dt in trading_days:
        candidates = scorer.get_candidates_for_date(target_date=dt, top_k=3, min_p_lgb=0.35, min_p_torch=0.35)
        
        day_pnl = 0.0
        n_picks = len(candidates)

        if n_picks > 0:
            alloc_per_stock = current_cash / 3.0  # Equal allocation among top 3
            
            for pick in candidates:
                close_p = pick['close_price']
                next_op = pick['next_open']
                ret = (next_op - close_p) / close_p - 0.0023  # 0.23% fee & tax
                profit_krw = alloc_per_stock * ret
                day_pnl += profit_krw

                exit_type = "tp" if ret > 0 else "sl"

                trade_details.append({
                    "ticker": pick['ticker'],
                    "open_time": f"{dt} 15:30",
                    "close_time": f"{dt} 09:00",  # Next trading morning open
                    "open_price": float(close_p),
                    "close_price": float(next_op),
                    "profit": float(ret),          # Decimal ratio (+0.0194 for +1.94%)
                    "profit_pct": float(ret),      # Decimal ratio
                    "exit_type": exit_type
                })

        new_cash = current_cash + day_pnl
        daily_ret = (new_cash - current_cash) / current_cash
        current_cash = new_cash

        daily_returns_map[dt] = {
            "date": dt,
            "return_pct": float(daily_ret),
            "trades": n_picks
        }

        portfolio_history.append({
            "date": dt,
            "cash": current_cash,
            "daily_return": daily_ret
        })

    df_port = pd.DataFrame(portfolio_history)
    df_port['cum_max'] = df_port['cash'].cummax()
    df_port['drawdown'] = (df_port['cash'] - df_port['cum_max']) / df_port['cum_max']

    total_return = (current_cash - seed_capital) / seed_capital
    mdd = float(df_port['drawdown'].min())
    
    # Calculate Sharpe
    daily_rets = df_port['daily_return'].values
    sharpe = float((np.mean(daily_rets) / (np.std(daily_rets) + 1e-8)) * np.sqrt(252))

    # Calculate Win Rate & Profit Factor
    if len(trade_details) > 0:
        profits = [t['profit'] for t in trade_details]
        wins = [p for p in profits if p > 0]
        losses = [abs(p) for p in profits if p < 0]
        win_rate = len(wins) / len(profits)
        profit_factor = float(sum(wins) / (sum(losses) + 1e-8))
    else:
        win_rate = 0.0
        profit_factor = 0.0

    # Calculate CAGR (Required)
    days_span = (datetime.fromisoformat(TEST_END) - datetime.fromisoformat(TEST_START)).days
    years_span = days_span / 365.25
    cagr = float((1.0 + total_return) ** (1.0 / max(years_span, 0.08)) - 1.0)

    # Monthly Returns
    df_port['year_month'] = df_port['date'].str.slice(0, 7)
    monthly_list = []
    for ym, group in df_port.groupby('year_month'):
        m_ret = (group['cash'].iloc[-1] - group['cash'].iloc[0]) / group['cash'].iloc[0]
        m_trades = sum(daily_returns_map[d]['trades'] for d in group['date'])
        monthly_list.append({
            "year_month": ym,
            "return_pct": float(m_ret),
            "trades": int(m_trades)
        })

    # Weekly Returns
    df_port['dt_obj'] = pd.to_datetime(df_port['date'])
    df_port['year_week'] = df_port['dt_obj'].dt.strftime('%Y-W%U')
    weekly_list = []
    for yw, group in df_port.groupby('year_week'):
        w_ret = (group['cash'].iloc[-1] - group['cash'].iloc[0]) / group['cash'].iloc[0]
        w_trades = sum(daily_returns_map[d]['trades'] for d in group['date'])
        weekly_list.append({
            "year_week": yw,
            "return_pct": float(w_ret),
            "trades": int(w_trades)
        })

    daily_list = list(daily_returns_map.values())

    payload = {
        "algorithm": {
            "name": "judal_hybrid_lgb_pytorch_KRX",
            "model_type": "lgb_mlp_hybrid",
            "timeframe": "1d",
            "project": "kr-stock-overnight",
            "ticker": "KRX",
            "direction": "LONG",
            "tp_pct": 0.0,
            "sl_pct": 0.0
        },
        "summary": {
            "avg_return": float(total_return),
            "avg_win_rate": float(win_rate),
            "avg_profit_factor": float(profit_factor),
            "avg_sharpe": float(sharpe),
            "cagr": float(cagr),
            "total_trades": int(len(trade_details)),
            "max_drawdown": float(mdd),
            "test_start": TEST_START,
            "test_end": TEST_END,
            "fee_rate_pct": 0.23
        },
        "monthly_returns": monthly_list,
        "weekly_returns": weekly_list,
        "daily_returns": daily_list,
        "trade_details": trade_details
    }

    print("\n[Summary Metrics]")
    print(f" -> Total Return : {total_return*100:+.2f}%")
    print(f" -> Win Rate     : {win_rate*100:.2f}%")
    print(f" -> Profit Factor: {profit_factor:.2f}")
    print(f" -> Sharpe Ratio : {sharpe:.2f}")
    print(f" -> CAGR         : {cagr*100:+.2f}%")
    print(f" -> MDD          : {mdd*100:.2f}%")
    print(f" -> Total Trades : {len(trade_details)}")

    # 1. Local SQLite Upsert
    if BACKTEST_DB_PATH.exists():
        try:
            conn_local = sqlite3.connect(str(BACKTEST_DB_PATH))
            cur = conn_local.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO backtest_summaries 
                (algorithm_name, ticker, timeframe, model_type, avg_return, avg_win_rate, avg_profit_factor, avg_sharpe, cagr, max_drawdown, total_trades, test_start, test_end)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                payload['algorithm']['name'],
                payload['algorithm']['ticker'],
                payload['algorithm']['timeframe'],
                payload['algorithm']['model_type'],
                total_return, win_rate, profit_factor, sharpe, cagr, mdd, len(trade_details), TEST_START, TEST_END
            ))
            conn_local.commit()
            conn_local.close()
            print(" -> Local SQLite backtest DB successfully updated.")
        except Exception as e:
            print(f" -> Local SQLite update error: {e}")

    # 2. Remote Server HTTP API POST
    print(f"\n[Posting Payload to Dashboard API: {DASHBOARD_API_URL}]")
    try:
        resp = requests.post(DASHBOARD_API_URL, json=payload, timeout=10)
        print(f" -> API Response Status Code: {resp.status_code}")
        if resp.status_code == 200:
            print(" -> [SUCCESS] Backtest results uploaded successfully to dashboard server!")
        else:
            print(f" -> [ERROR] Server Response: {resp.text}")
    except Exception as e:
        print(f" -> [HTTP ERROR] Could not reach dashboard server: {e}")


if __name__ == "__main__":
    run_and_upload_backtest()
