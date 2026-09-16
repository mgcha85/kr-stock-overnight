"""Overnight backtest: buy daily close / sell next session daily open.

Selection stays Judal + LGB + PyTorch Top-3.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from kr_stock.config import FEE_RATE, SEED_CAPITAL
from kr_stock.sell_timing import OvernightTrade, build_overnight_trades

SEED = 42
# 1m tape starts 2025-01-02. Judal crawl starts 2026-01-06; earlier dates use daily %change + static themes.
TEST_START = "2026-01-06"
TEST_END = "2026-07-30"
ALGO_NAME = "judal_hybrid_lgb_pytorch_KRX"

MONTHLY_PERIODS = [
    ("2025-01-02", "2025-01-31", "2025-01"),
    ("2025-02-01", "2025-02-28", "2025-02"),
    ("2025-03-01", "2025-03-31", "2025-03"),
    ("2025-04-01", "2025-04-30", "2025-04"),
    ("2025-05-01", "2025-05-31", "2025-05"),
    ("2025-06-01", "2025-06-30", "2025-06"),
    ("2025-07-01", "2025-07-31", "2025-07"),
    ("2025-08-01", "2025-08-31", "2025-08"),
    ("2025-09-01", "2025-09-30", "2025-09"),
    ("2025-10-01", "2025-10-31", "2025-10"),
    ("2025-11-01", "2025-11-30", "2025-11"),
    ("2025-12-01", "2025-12-31", "2025-12"),
    ("2026-01-01", "2026-01-31", "2026-01"),
    ("2026-02-01", "2026-02-28", "2026-02"),
    ("2026-03-01", "2026-03-31", "2026-03"),
    ("2026-04-01", "2026-04-30", "2026-04"),
    ("2026-05-01", "2026-05-31", "2026-05"),
    ("2026-06-01", "2026-06-30", "2026-06"),
    ("2026-07-01", "2026-07-31", "2026-07"),
    ("2026-08-01", "2026-08-12", "2026-08"),
]


@dataclass
class FilledTrade:
    buy_date: str
    sell_date: str
    code: str
    stock_name: str
    buy_price: float
    sell_price: float
    net_pnl: float
    daily_close: float
    daily_next_open: float


def fill_1520_0900(
    trades: Sequence[OvernightTrade], fee_rate: float = FEE_RATE
) -> List[FilledTrade]:
    filled: List[FilledTrade] = []
    for t in trades:
        bp = t.buy_close
        sp = t.daily_next_open
        if not bp or not sp or bp <= 0 or sp <= 0:
            continue
        filled.append(
            FilledTrade(
                buy_date=t.buy_date,
                sell_date=t.sell_date,
                code=t.code,
                stock_name=t.stock_name,
                buy_price=bp,
                sell_price=sp,
                net_pnl=sp / bp - 1.0 - fee_rate,
                daily_close=t.buy_close,
                daily_next_open=t.daily_next_open,
            )
        )
    return filled


def _window_metrics(trades: Sequence[FilledTrade], start_d: str, end_d: str) -> dict:
    sub = [t for t in trades if start_d <= t.buy_date <= end_d]
    dates = sorted({t.buy_date for t in sub})
    if not dates:
        return {
            "period": f"{start_d} ~ {end_d}",
            "trading_days": 0,
            "trades": 0,
            "return_pct": 0.0,
            "win_rate": 0.0,
            "pf": 0.0,
            "mdd": 0.0,
            "mean_trade_pct": 0.0,
        }

    daily = []
    pnls = []
    for d in dates:
        day = [t.net_pnl for t in sub if t.buy_date == d]
        daily.append(float(np.mean(day)) if day else 0.0)
        pnls.extend(day)

    eq = np.cumprod(1.0 + np.array(daily, dtype=float))
    total = float(eq[-1] - 1.0) if len(eq) else 0.0
    peak = np.maximum.accumulate(eq)
    mdd = float(((eq - peak) / np.maximum(peak, 1e-12)).min()) if len(eq) else 0.0
    pos = sum(p for p in pnls if p > 0)
    neg = abs(sum(p for p in pnls if p < 0))
    return {
        "period": f"{start_d} ~ {end_d}",
        "trading_days": len(dates),
        "trades": len(pnls),
        "return_pct": round(total * 100, 2),
        "win_rate": round((sum(1 for p in pnls if p > 0) / len(pnls)) * 100, 2) if pnls else 0.0,
        "pf": round(pos / neg, 2) if neg else 0.0,
        "mdd": round(mdd * 100, 2),
        "mean_trade_pct": round(float(np.mean(pnls)) * 100, 3) if pnls else 0.0,
    }


def evaluate_windows(filled: Sequence[FilledTrade]) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly = []
    for start_d, end_d, label in MONTHLY_PERIODS:
        row = _window_metrics(filled, start_d, end_d)
        row["label"] = label
        monthly.append(row)

    buy_dates = sorted({t.buy_date for t in filled})
    weekly = []
    for i in range(0, len(buy_dates), 5):
        chunk = buy_dates[i : i + 5]
        if len(chunk) < 2:
            continue
        row = _window_metrics(filled, chunk[0], chunk[-1])
        row["week_idx"] = len(weekly) + 1
        weekly.append(row)
    return pd.DataFrame(monthly), pd.DataFrame(weekly)


def simulate_sequential(
    filled: Sequence[FilledTrade],
    seed_capital: float = SEED_CAPITAL,
) -> dict:
    """One overnight book per buy date; cash compounds after T+1 09:00 sell."""
    by_day: Dict[str, List[FilledTrade]] = {}
    for t in filled:
        by_day.setdefault(t.buy_date, []).append(t)
    dates = sorted(by_day)

    cash = seed_capital
    history = []
    trade_details = []
    for d in dates:
        picks = by_day[d]
        alloc = cash / len(picks)
        day_pnl = 0.0
        for t in picks:
            profit = alloc * t.net_pnl
            day_pnl += profit
            trade_details.append(
                {
                    "ticker": t.code,
                    "open_time": f"{t.buy_date} 15:30",
                    "close_time": f"{t.sell_date} 09:00",
                    "open_price": float(t.buy_price),
                    "close_price": float(t.sell_price),
                    "profit": float(t.net_pnl),
                    "profit_pct": float(t.net_pnl),
                    "exit_type": "timeout",
                }
            )
        new_cash = cash + day_pnl
        daily_ret = (new_cash - cash) / cash if cash > 0 else 0.0
        cash = new_cash
        history.append({"date": d, "cash": cash, "daily_return": daily_ret, "trades": len(picks)})

    df = pd.DataFrame(history)
    if df.empty:
        return {
            "seed": seed_capital,
            "final": seed_capital,
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "mdd": 0.0,
            "win_rate": 0.0,
            "pf": 0.0,
            "n_trades": 0,
            "history": df,
            "trade_details": [],
            "daily_returns": [],
            "weekly_returns": [],
            "monthly_returns": [],
        }

    df["cum_max"] = df["cash"].cummax()
    df["drawdown"] = (df["cash"] - df["cum_max"]) / df["cum_max"]
    total_return = (cash - seed_capital) / seed_capital
    mdd = float(df["drawdown"].min())
    rets = df["daily_return"].to_numpy()
    sharpe = float((np.mean(rets) / (np.std(rets) + 1e-8)) * np.sqrt(252))
    profits = [t["profit"] for t in trade_details]
    wins = [p for p in profits if p > 0]
    losses = [abs(p) for p in profits if p < 0]
    win_rate = len(wins) / len(profits) if profits else 0.0
    pf = float(sum(wins) / (sum(losses) + 1e-8)) if profits else 0.0

    start = df["date"].iloc[0]
    end = df["date"].iloc[-1]
    years = max((datetime.fromisoformat(end) - datetime.fromisoformat(start)).days / 365.25, 1 / 365.25)
    cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0)

    daily_returns = [
        {"date": r.date, "return_pct": float(r.daily_return), "trades": int(r.trades)}
        for r in df.itertuples(index=False)
    ]
    df["year_month"] = df["date"].str.slice(0, 7)
    monthly_returns = []
    for ym, g in df.groupby("year_month"):
        m_ret = (g["cash"].iloc[-1] - g["cash"].iloc[0]) / g["cash"].iloc[0]
        monthly_returns.append(
            {"year_month": ym, "return_pct": float(m_ret), "trades": int(g["trades"].sum())}
        )
    df["dt"] = pd.to_datetime(df["date"])
    df["year_week"] = df["dt"].dt.strftime("%Y-W%U")
    weekly_returns = []
    for yw, g in df.groupby("year_week"):
        w_ret = (g["cash"].iloc[-1] - g["cash"].iloc[0]) / g["cash"].iloc[0]
        weekly_returns.append(
            {"year_week": yw, "return_pct": float(w_ret), "trades": int(g["trades"].sum())}
        )

    return {
        "seed": seed_capital,
        "final": float(cash),
        "total_return": float(total_return),
        "cagr": cagr,
        "sharpe": sharpe,
        "mdd": mdd,
        "win_rate": float(win_rate),
        "pf": pf,
        "n_trades": len(trade_details),
        "test_start": start,
        "test_end": end,
        "history": df,
        "trade_details": trade_details,
        "daily_returns": daily_returns,
        "weekly_returns": weekly_returns,
        "monthly_returns": monthly_returns,
    }


def dashboard_payload(sim: dict) -> dict:
    return {
        "algorithm": {
            "name": ALGO_NAME,
            "model_type": "lgb_mlp_hybrid",
            "timeframe": "1d",
            "project": "kr-stock-overnight",
            "ticker": "KRX",
            "direction": "LONG",
            "tp_pct": 0.0,
            "sl_pct": 0.0,
        },
        "summary": {
            "avg_return": float(sim["total_return"]),
            "avg_win_rate": float(sim["win_rate"]),
            "avg_profit_factor": float(sim["pf"]),
            "avg_sharpe": float(sim["sharpe"]),
            "cagr": float(sim["cagr"]),
            "total_trades": int(sim["n_trades"]),
            "max_drawdown": float(sim["mdd"]),
            "test_start": sim["test_start"],
            "test_end": sim["test_end"],
            "fee_rate_pct": FEE_RATE * 100,
        },
        "monthly_returns": sim["monthly_returns"],
        "weekly_returns": sim["weekly_returns"],
        "daily_returns": sim["daily_returns"],
        "trade_details": sim["trade_details"],
    }


def run_1520_0900_backtest(
    start_date: str = TEST_START,
    end_date: str = TEST_END,
    top_k: int = 3,
    use_session_1520: bool = False,
    model_tag: str = "",
    open_gap_min: Optional[float] = 0.02,
    open_gap_max: Optional[float] = 0.28,
    min_close: Optional[float] = None,
    max_close: Optional[float] = None,
    require_above_sma20: bool = False,
) -> tuple[List[FilledTrade], pd.DataFrame, pd.DataFrame, dict]:
    raw = build_overnight_trades(
        start_date=start_date,
        end_date=end_date,
        top_k=top_k,
        use_session_1520=use_session_1520,
        model_tag=model_tag,
        open_gap_min=open_gap_min,
        open_gap_max=open_gap_max,
        min_close=min_close,
        max_close=max_close,
        require_above_sma20=require_above_sma20,
    )
    filled = fill_1520_0900(raw)
    monthly, weekly = evaluate_windows(filled)
    sim = simulate_sequential(filled)
    return filled, monthly, weekly, sim
