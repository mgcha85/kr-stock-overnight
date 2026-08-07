# KRX Overnight Strategy Paper Trading Engine Guide

## 1. Overview & Architecture

The **KRX Overnight Strategy Paper Trading Engine** bridges the gap between quantitative backtesting and real-world trading execution for the `judal_hybrid_lgb_pytorch_KRX` model using **10,000,000 KRW seed capital**.

To eliminate discrepancies between historical simulation and live paper execution, this system enforces **100% Parity** by sharing a single, unified inference module (`src/kr_stock/inference.py`).

```
                              ┌───────────────────────────────────┐
                              │  Single Source of Truth Module    │
                              │     src/kr_stock/inference.py     │
                              └─────────────────┬─────────────────┘
                                                │
                     ┌──────────────────────────┴──────────────────────────┐
                     ▼                                                     ▼
      ┌─────────────────────────────┐                       ┌─────────────────────────────┐
      │      Backtest Engine        │                       │     Paper Trading Engine    │
      │ (research/run_hybrid_bt.py) │                       │ (src/kr_stock/paper_engine) │
      └──────────────┬──────────────┘                       └──────────────┬──────────────┘
                     │                                                     │
                     └──────────────────────────┬──────────────────────────┘
                                                ▼
                               ┌──────────────────────────────────┐
                               │ Daily Post-Market Parity Check   │
                               │  (100% Match Signal Verification) │
                               └──────────────────────────────────┘
```

---

## 2. Daily Execution Schedule

| Time | Action | Description | Telegram Alert |
|------|--------|-------------|----------------|
| **09:00 AM** | **Market Open SELL** | Sell all overnight positions at the market open price. Compute net profit (after 0.23% fees/taxes), update cash & total equity, and calculate **Weekly (7d)** and **Monthly (30d)** cumulative returns. | 🚀 Sell Notification & Returns Report |
| **15:30 PM** | **Market Close BUY** | Run scoring engine across all KRX stocks using Judal theme momentum, LightGBM, PyTorch, and MarketMosaic context. Allocate cash across Top-3 picks and record paper buy positions. | 📈 Buy Notification |
| **15:35 PM** | **Post-Market Parity Check** | Re-run backtest signal extraction on the day's updated candles. Compare paper buy records with backtest candidate selections to verify 100% signal match. | ✅/⚠️ Parity Check Verification Alert |

---

## 3. Telegram Notification Specifications

Telegram alerts are sent automatically using the bot credentials (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`).

### A. 15:30 Market Close BUY Alert
```html
<b>📈 [KRX Overnight Strategy] 장 마감 매수 내역 (15:30)</b>
🗓️ <b>일자:</b> 2026-06-16
💵 <b>설정 시드:</b> 10,000,000 원 | <b>종목당 배정:</b> 3,333,333 원
────────────────────────
<b>1. 한화오션 (042660)</b>
   • 테마: 해양플랜트
   • 매수가(종가): 124,100 원
   • 수량: 26 주 (총 3,226,600 원)
   • 모델점수: 156.0점 (LGB: 0.54 | DL: 0.52)
────────────────────────
💰 <b>매수 후 잔여 예수금:</b> 183,400 원
⏰ <i>익일 09:00 장 시작 시 시가 매도 예정</i>
```

### B. 09:00 Market Open SELL & Returns Alert
```html
<b>🚀 [KRX Overnight Strategy] 장 시작 매도 및 수익률 보고 (09:00)</b>
🗓️ <b>일자:</b> 2026-06-17
────────────────────────
<b>1. 🟢 한화오션 (042660)</b>
   • 매수가: 124,100 원 ➡️ 매도가: 133,100 원
   • 청산 수량: 26 주
   • 손익: <b>+226,041 원 (+7.01%)</b>
────────────────────────
📊 <b>오늘 일간 손익:</b> +440,155 원 (+4.40%)
📅 <b>최근 1주일 누적 수익률:</b> <b>+4.40%</b>
🗓️ <b>최근 1개월 누적 수익률:</b> <b>+4.40%</b>
💎 <b>현재 총 평가 자산:</b> <b>10,440,155 원</b>
```

### C. Post-Market Parity Verification Alert
```html
<b>✅ [100% PARITY MATCH] 장후 백테스트 ↔ 페이퍼트레이딩 검증 보고</b>
🗓️ <b>검증 일자:</b> 2026-06-16
📌 <b>Paper Trading 매수:</b> 042660, 011210, 011200
🔍 <b>Backtest 정답 매수:</b> 042660, 011210, 011200
────────────────────────
📝 <b>결과 요약:</b> Paper Buy: ['042660', '011210', '011200'] | Backtest Buy: ['042660', '011210', '011200']. 100% Signal & Parity Match!
```

---

## 4. CLI Execution & Operation Commands

### Standard CLI Modes (`python3 -m kr_stock.cli`)
```bash
# 1. Execute Afternoon 15:30 BUY
PYTHONPATH=src uv run python -m kr_stock.cli --mode buy --date 2026-06-16

# 2. Execute Morning 09:00 SELL
PYTHONPATH=src uv run python -m kr_stock.cli --mode sell --date 2026-06-17

# 3. Execute Post-Market Parity Verification
PYTHONPATH=src uv run python -m kr_stock.cli --mode parity --date 2026-06-16

# 4. Execute Full-Day Cycle (SELL -> BUY -> PARITY)
PYTHONPATH=src uv run python -m kr_stock.cli --mode full_day --date 2026-06-16
```

---

## 5. Podman Container Management

```bash
# Start paper trading engine (dev mode)
./start.sh

# Start paper trading engine (prod mode)
ENV_TYPE=prod ./start.sh

# Stop services
./stop.sh
```

---

## 6. Database Schema (`data/paper_trading.db`)

### `paper_trades` Table
- `id`: Auto-increment integer primary key
- `date`: Trading date string (`YYYY-MM-DD`)
- `ticker`: 6-digit stock ticker (`042660`)
- `stock_name`: Korean stock name (`한화오션`)
- `theme_name`: Judal theme name (`해양플랜트`)
- `buy_price`: Market close buy price in KRW
- `buy_qty`: Shares purchased
- `buy_amount`: Total KRW allocated to buy
- `sell_price`: Market open sell price in KRW
- `sell_amount`: Net sell return in KRW (after 0.23% fees/taxes)
- `pnl_krw`: Net profit/loss in KRW
- `pnl_pct`: Net profit percentage
- `status`: Position state (`OPEN` / `CLOSED`)
- `open_time`: Timestamp of buy (`YYYY-MM-DD 15:30:00`)
- `close_time`: Timestamp of sell (`YYYY-MM-DD 09:00:00`)
- `hybrid_score`, `p_lgb`, `p_torch`: ML/DL model probability scores

### `paper_account` Table
- `date`: Primary key date string (`YYYY-MM-DD`)
- `cash_balance`: Uninvested cash balance in KRW
- `invested_amount`: Active capital locked in open trades
- `total_equity`: `cash_balance + invested_amount`
- `daily_pnl_krw`, `daily_pnl_pct`: Daily trade gain
- `weekly_pnl_pct`: Rolling 7-day cumulative equity growth %
- `monthly_pnl_pct`: Rolling 30-day cumulative equity growth %
- `updated_at`: Database record modification timestamp
