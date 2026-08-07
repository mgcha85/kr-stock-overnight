# Backtest Dashboard API Automatic Integration Guide

## 1. Overview
This module automatically formats and transmits backtest simulation results for the Judal Hybrid Theme Trading Strategy (LightGBM + PyTorch + Judal Driver) to the central Backtest Lab Dashboard API.

- **Target API Endpoint**: `POST http://146.56.115.71:8082/api/backtest`
- **Dashboard Web UI**: `http://146.56.115.71:8082`
- **Registered Algorithm Identifier**: `judal_hybrid_lgb_pytorch_KRX`

---

## 2. API Payload & Schema Compliance

The payload strictly complies with the specifications defined in `AGENTS.md` and SvelteKit API router requirements:

```json
{
  "algorithm": {
    "name": "judal_hybrid_lgb_pytorch_KRX",
    "model_type": "judal_hybrid_lgb_pytorch",
    "timeframe": "1d",
    "project": "kr_stock",
    "ticker": "KRX",
    "direction": "LONG",
    "tp_pct": 0.05,
    "sl_pct": 0.03,
    "horizon_bars": 1,
    "prob_threshold": 0.35
  },
  "summary": {
    "avg_return": 7.0714,
    "avg_win_rate": 0.7222,
    "avg_profit_factor": 4.15,
    "avg_sharpe": 13.51,
    "cagr": 468.3341,
    "total_trades": 144,
    "max_drawdown": -0.0864,
    "test_start": "2026-04-01",
    "test_end": "2026-08-03",
    "fee_rate_pct": 0.23
  },
  "monthly_returns": [ ... ],
  "weekly_returns": [ ... ],
  "daily_returns": [ ... ],
  "trade_details": [
    {
      "ticker": "012200",
      "open_time": "2026-04-03 15:30",
      "close_time": "2026-04-06 09:00",
      "open_price": 6830.0,
      "close_price": 6900.0,
      "profit": 0.007949,
      "profit_pct": 0.007949,
      "exit_type": "tp"
    }
  ]
}
```

---

## 3. Mandatory Rules Implemented
1. **Algorithm Name Upper Case Convention**: `ticker` is set to `"KRX"` (Upper case).
2. **Mandatory CAGR Field**: `cagr` calculated as `(1 + total_return) ^ (1 / test_years) - 1` and passed in summary.
3. **Trade Details Fields**:
   - `ticker`: 종목코드 (e.g. `"012200"`). Web UI에서 Ticker 컬럼으로 표시.
   - `profit` & `profit_pct`: 소수 비율값 (`0.007949` = +0.795%). Web UI에서 `* 100`을 적용해 정상 퍼센트로 표시.
   - `open_time` & `close_time`: 당일 15:30 진입 시 `close_time`은 익영업일 09:00 시가 청산으로 명시.
   - `exit_type`: `"tp"` 또는 `"sl"` (소문자).

---

## 4. Execution Command
To run the automated backtest execution and upload:

```bash
PYTHONPATH=. python3 research/upload_backtest_to_dashboard.py
```

---

## 5. Verification Status
- **API Status**: HTTP `200` Success returned by `http://146.56.115.71:8082/api/backtest`.
- **Database Record ID**: `354` registered in production SQLite DB with 144 individual trade records.
- **Svelte Dashboard UI**: Ticker 컬럼 추가 및 퍼센트 스케일 정상표시 반영 완료.
