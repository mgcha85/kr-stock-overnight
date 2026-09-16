# HANDOFF.md — Rust Live Trading Deployment Status

English only. For strategy reproduction details see `PROJECT.md`.

## 1. System Status Summary

- **Engine**: Rust release binary `live-trading/target/release/trader` in container `kr_stock_live_trading`
- **Mode**: `LIVE` (`TRADING_MODE=live`)
- **Env files**: Compose loads `.env.${ENV_TYPE:-prod}`; runtime has often used `.env.dev` — keep `MAX_ALLOC_PER_TICKER` / `TRADING_MODE` aligned in both
- **Schedule (KST, session days)**:
  - **09:00** — market-open SELL
  - **15:28–16:00** — market-close BUY (missed window after 16:00 → Telegram, no order)
  - **18:00–19:30** — parity on **completed daily bars** (was 15:33; changed 2026-09-16)
- **Risk / sizing**:
  - `SEED_CAPITAL`: 300,000 KRW (account seed bookkeeping; live cash comes from Kiwoom)
  - `TOP_K_TRADES`: 3
  - `MAX_ALLOC_PER_TICKER`: **500,000 KRW** (quantity cap only — must not change Top-3 selection)
  - Fee: 0.23% round-trip (`FEE_RATE`)
- **Universe (live)**: HTS condition **`종가베팅`** (user-maintained). Parquet G+H is logged for comparison; live scoring uses HTS.
- **Buy score close**: 15:28 **expected match** (0H / ka10001 exp). Parity close: **ka10081** official daily bar (fallback ka10001 → 1m → Daum/Naver).

---

## 2. Recent Root Causes (2026-09)

### A. Telegram “Parity Mismatch” ≠ parquet overnight backtest

- Alert **Paper** = tickers actually bought that day.
- Alert **Backtest** = **re-score HTS at parity time** with official (now 18:00 daily) bars — not the dashboard G+H overnight study.
- 15:33 mismatches were mostly **broken session bars**, not small expected-vs-official close gaps.

### B. Collapsed OHLC + `vol_ratio=0` at 15:28

- Some names only got 0H expected price → `open=high=low=close`, `turnover=0` → junk ML features → wrong Top-3 (e.g. 삼현, LS에코).
- ka10080 1m charts usually **end ~15:19** (no auction minutes).
- **Fix (deployed)**:
  1. Sort 1m bars by time before open/close (Kiwoom is newest-first).
  2. Day volume ≈ **1m sum(≤15:19) + `exp_cntr_qty`** (auction expected match qty).
  3. If 15:20–15:28 volume missing on chart, **proxy with 09:00–09:10 volume**.
  4. Retry 1m / `_AL`; then Daum → Naver daily OHLC+turnover.
  5. Never treat `exp_cntr_qty` alone as full-day volume (floor only if everything else fails).

### C. 15:33 official 1m “close” wrong

- Using unsorted / incomplete 1m as official close flipped some names (e.g. open used as close).
- **Fix**: Parity moved to **18:00**; prefer **ka10081** completed daily bar.

### D. Verified 2026-09-16 ~18:40 KST

| Stage | Top-3 |
|---|---|
| Live buy 15:28 | `000500`, `001210`, `229640` |
| Old 15:33 (broken bars) | `000500`, `021240`, `126340` |
| New 18:00 ka10081 | `000500`, `054920`, `229640` |

18:00 shares 2/3 with live; remaining gap is real expected-vs-official ranking (`001210` vs `054920`), not missing OHLC. All 28 HTS bars had turnover.

### E. Daily parquet sync

- Weekday **18:00** cron: `scripts/sync_kr_kline_daily.sh` → `scripts/refresh_daily_from_kiwoom.py` (ka10081).
- Parity does **not** wait for full parquet sync; it fetches ka10081 per HTS name.

---

## 3. Technical Notes Still Valid

### Model paths (`config.rs`)

- Do not rely on `CARGO_MANIFEST_DIR` inside the container; resolve `model_dir` at runtime / via env.

### DB isolation (`trading_mode`)

- Filter `paper_trades` by `trading_mode` / `execution_mode` so legacy paper rows never block live buys.

### Deploy

```bash
cd /mnt/data/projects/kr_stock/live-trading && cargo build --release
cd /mnt/data/projects/kr_stock && ENV_TYPE=prod podman-compose up -d --force-recreate --build
```

Env-only changes need recreate; **code changes need `--build`** (binary is copied into the image).

### Logs

```bash
podman logs -f kr_stock_live_trading
```

Expect boot line:

```text
Schedule: KRX session days [09:00 SELL | 15:28 BUY | 18:00 PARITY]
```

### Re-run today’s parity after a fix

```bash
sqlite3 data/paper_trading.db "DELETE FROM scheduler_marks WHERE date='YYYY-MM-DD' AND job='parity';"
# scheduler catch-up within 18:00–19:30, or wait for next loop (~15s)
```

---

## 4. Open / Watch

- Live HTS vs parquet G+H universes still differ by design; do not treat Telegram “Backtest” as the research Top-3.
- Cap is selection-preserving: skip size only if 1 share > `MAX_ALLOC_PER_TICKER`.
- If 15:28 still logs `turnover still 0 after 1m/auction/web`, treat as Kiwoom/API outage for that code and investigate rate limits / empty ka10080.
