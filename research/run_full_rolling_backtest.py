#!/usr/bin/env python3
"""Rolling walk-forward report: 15:20 buy / next 09:00 sell (1m fills)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from kr_stock.overnight_backtest import run_1520_0900_backtest

REPORT_DOC = ROOT_DIR / "docs" / "FULL_ROLLING_BACKTEST_REPORT.md"


def main() -> None:
    print("=========================================================================")
    print("   OVERNIGHT BACKTEST  |  BUY 15:20 (15:19 1m)  /  SELL T+1 09:00")
    print("=========================================================================\n")

    filled, monthly, weekly, sim = run_1520_0900_backtest()
    print(f"[fills] {len(filled)} trades with both 15:19 buy and 09:00 sell prints")
    print(
        f"[sequential 10M] return={sim['total_return']*100:+.2f}%  CAGR={sim['cagr']*100:+.2f}%  "
        f"WR={sim['win_rate']*100:.1f}%  PF={sim['pf']:.2f}  Sharpe={sim['sharpe']:.2f}  "
        f"MDD={sim['mdd']*100:.2f}%  trades={sim['n_trades']}"
    )

    print("\n=========================================================================")
    print("   1. 30-DAY (MONTHLY) ROLLING WINDOW")
    print("=========================================================================")
    for _, r in monthly.iterrows():
        print(
            f"[{r['label']}] Return: {r['return_pct']:+6.2f}% | Win Rate: {r['win_rate']:5.1f}% | "
            f"PF: {r['pf']:5.2f} | MDD: {r['mdd']:6.2f}% | Trades: {int(r['trades']):3d} | "
            f"Mean/trade: {r['mean_trade_pct']:+.2f}%"
        )

    print("\n=========================================================================")
    print("   2. 7-DAY (WEEKLY) ROLLING WINDOW")
    print("=========================================================================")
    for _, r in weekly.iterrows():
        print(
            f"[Week {int(r['week_idx']):02d}: {r['period']}] Return: {r['return_pct']:+6.2f}% | "
            f"Win Rate: {r['win_rate']:5.1f}% | PF: {r['pf']:5.2f} | MDD: {r['mdd']:6.2f}% | "
            f"Trades: {int(r['trades']):2d}"
        )

    with open(REPORT_DOC, "w", encoding="utf-8") as f:
        f.write("# Full Rolling Walk-Forward Strategy Backtest Report\n\n")
        f.write("### Execution (matches live schedule)\n")
        f.write("- **Buy**: 15:20 decision, fill = last continuous 1m close at **15:19** ")
        f.write("(1m tape has no 15:20–15:29; KRX closing auction).\n")
        f.write("- **Sell**: next session **09:00** 1m open.\n")
        f.write("- **Fee**: 0.23% round-trip. Selection: Judal + LGB + PyTorch, Top-3, ")
        f.write("turnover ≥ 200억, change < 29%, p_lgb/p_torch ≥ 0.35.\n")
        f.write(f"- **Scope**: `{sim['test_start']} ~ {sim['test_end']}`.\n\n")
        f.write("## 0. Sequential 10,000,000 KRW\n\n")
        f.write("```text\n")
        f.write(
            f"Total {sim['total_return']*100:+.2f}% | CAGR {sim['cagr']*100:+.2f}% | "
            f"WR {sim['win_rate']*100:.1f}% | PF {sim['pf']:.2f} | "
            f"Sharpe {sim['sharpe']:.2f} | MDD {sim['mdd']*100:.2f}% | "
            f"Trades {sim['n_trades']}\n"
        )
        f.write("```\n\n")
        f.write("## 1. 30-Day (Monthly) Rolling Performance\n\n```text\n")
        cols = ["label", "period", "trading_days", "trades", "return_pct", "win_rate", "pf", "mdd", "mean_trade_pct"]
        f.write(monthly[cols].to_string(index=False) + "\n```\n\n")
        f.write("## 2. 7-Day (Weekly) Rolling Performance\n\n```text\n")
        wcols = ["week_idx", "period", "trades", "return_pct", "win_rate", "pf", "mdd", "mean_trade_pct"]
        f.write(weekly[wcols].to_string(index=False) + "\n```\n")
    print(f"\nReport saved to {REPORT_DOC}")


if __name__ == "__main__":
    main()
