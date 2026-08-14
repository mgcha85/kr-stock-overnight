# HANDOFF.md — Kiwoom 종가베팅 Strategy & Automated Pipeline Handoff

**Project**: `kr-stock-overnight` (KRX Stock Overnight Strategy Pipeline)  
**Last Updated**: `2026-08-14 18:40 KST`  
**Status**: `Paper daemon fixed — 15:20 empty-buy and 09:00 fake -0.23% sell were data bugs, not strategy`

---

## 1. Executive Summary & Objective

- Kiwoom HTS **"종가베팅"** → AI hybrid scoring → paper overnight pipeline is production-validated.
- **Rust live crate** (`live-trading/`, package `kr-stock-live`) ports paper functions 1:1 (condition, features, LGB+ONNX scoring, engine, scheduler, telegram).
- **2026-08-13 dual-runtime parity**: Python vs Rust Top3 + scores match within `1e-5` (`PARITY OK`).

---

## 2. Key Architecture & Execution Schedule

### Daily Timeline (Weekdays Mon-Fri KST)

| Time (KST) | Action | Component | Description |
| :---: | :--- | :--- | :--- |
| **09:00:00** | Market Open SELL | `PaperTradingEngine.execute_market_open_sell` | Sells overnight positions at Open. Telegram alert. |
| **15:20:00** | Market Close BUY | `PaperTradingEngine.execute_market_close_buy` | Candle sync → Kiwoom condition → `OvernightScorer` → Top3 BUY (idempotent). |
| **15:25:00** | Parity Check | `PaperTradingEngine.run_post_market_parity_check` | Paper tickers vs rescore. |

**Execution switch** (`TRADING_MODE`, default `paper`):

- `TRADING_MODE=paper` — candles / HTS / scoring / Telegram / DB identical. Fills are SQLite only.
- `TRADING_MODE=live` — same pipeline, but cash from `kt00001`, buy `kt10000`, sell `kt10001` + holdings `kt00004`. Rejected orders are **not** written as OPEN/CLOSED.
- Do **not** flip to live while PAPER OPEN rows exist (and vice versa). Close in the matching mode first.

**Current OPEN (2026-08-14 paper)**: `042700` 한미반도체, `047040` 대우건설, `067310` 하나마이크론. Next session: Monday 2026-08-17 09:00 SELL.

---

## 3. Key Files

### Python paper
- `src/kr_stock/kiwoom_condition.py`, `kiwoom_server.py`, `paper_engine.py`, `scheduler.py`, `inference.py`
- `scripts/fetch_today_kr_candles.py`, `scripts/run_kiwoom_overnight_analysis.py`
- `scripts/dump_python_overnight_golden.py` — golden JSON for Rust parity

### Rust live (`live-trading/`)
- Bins: `analyze`, `parity_check`, `trader` (`--dry-run` default)
- Models: `models/{lgb_kline_model.txt,kline_scaler.json,pytorch_kline_model.onnx}`
- Export: `live-trading/scripts/export_models.py`

---

## 4. Operational Commands

### Python paper daemon (Podman — rebuilt 2026-08-13)
```bash
# Kiwoom mock/API (host)
PYTHONPATH=src python3 -m kr_stock.kiwoom_server &

# Rebuild/restart paper daemon
cd /mnt/data/projects/kr_stock && podman-compose down && podman-compose up -d
podman logs -f kr_stock_paper_trading
# Expect: Schedule ... [09:00 SELL | 15:20 BUY | 15:25 PARITY]
```

### Manual Python
```bash
PYTHONPATH=src python3 -m kr_stock.cli --mode sell --date 2026-08-14
PYTHONPATH=src python3 -m kr_stock.cli --mode buy --date 2026-08-14
PYTHONPATH=src python3 -m kr_stock.cli --mode parity --date 2026-08-14
PYTHONPATH=src:. python3 scripts/run_kiwoom_overnight_analysis.py --date 2026-08-13
```

### Rust analyze + parity (same candidate override)
```bash
export KIWOOM_CANDIDATE_CODES="010060,001450,062040,005930,006400,009830,010170,010950,035720,036930,042700,043260,064760,066970,067310,068270,080220,086520,096770,103590,131290,131970,181710,196170,214450"
export RUST_LOG=kr_stock_live=info,ort=error

PYTHONPATH=src:. python3 scripts/dump_python_overnight_golden.py --date 2026-08-13 -o /tmp/py_golden_2026-08-13.json

cd live-trading
cargo build --release
./target/release/analyze --date 2026-08-13 --ensure-candles --out /tmp/rs_analysis_2026-08-13.json
./target/release/parity_check --golden /tmp/py_golden_2026-08-13.json --rust-out /tmp/rs_analysis_2026-08-13.json --tol 1e-5
# → PARITY OK

./target/release/trader --dry-run true --once --date 2026-08-13
```

---

## 5. Latest Verification Results (2026-08-13)

### Paper OPEN (deduped; residual ids 14/15 removed; equity rebuilt)
| Ticker | Name | Qty | Close | Hybrid |
|--------|------|-----|-------|--------|
| 010170 | 대한광통신 | 522 | 13,240 | 143.40 |
| 103590 | 일진전기 | 95 | 72,400 | 118.52 |
| 196170 | 알테오젠 | 21 | 317,500 | 110.21 |

Account snapshot after fix: cash≈287,359 / invested=20,456,780 / equity≈20,744,139

### Python ↔ Rust parity (`tol=1e-5`)
| Rank | Code | Δp_lgb | Δp_torch | Δhybrid |
|------|------|--------|----------|---------|
| 1 | 010170 | 0 | ~6e-8 | ~1e-6 |
| 2 | 103590 | 0 | 0 | ~1e-6 |
| 3 | 196170 | 0 | ~3e-8 | ~1e-6 |

- codes set: **MATCH**
- Top3 order: **MATCH**
- `parity_check` exit: **0 / PARITY OK**
- Kline spot-check closes: 13240 / 72400 / 317500 **MATCH**

### Ops fixes applied this session
- Idempotent BUY (skip if OPEN exists for date)
- Scheduler logs aligned to 15:20 / 15:25
- Dockerfile copies `scripts/`; compose mounts `src`+`scripts`, finance RW, `KIWOOM_API_URL` → host
- Podman image rebuilt and restarted

---

## 6. Action Items for Next Session

1. **2026-08-17 09:00**: SELL must use FDR 시가 (한미반도체/대우건설/하나마이크론). 시가 없으면 매도 중단 — 매수가 flatten 금지.
2. **15:20 BUY**: 컨테이너에 FinanceDataReader 포함됨. 캔들 실패 시 가짜 '매수 없음' 텔레그램 안 보냄.
3. localhost:5000 mock 조건검색은 무시하고 당일 HTS 오프라인 시뮬 사용. 실 HTS는 `KIWOOM_LIVE`/WS 연결 후에만.

### 2026-08-14 버그 (고침)

| 증상 | 원인 | 수정 |
|------|------|------|
| 15:20 항상 매수 0건 | 컨테이너에 FDR 없음 → parquet에 당일 캔들 0 → 스코어 공집합. 실패를 warning으로 삼키고 '매수 없음' 텔레그램 | FDR 의존성 + 캔들 실패 시 에러 알림/매수 중단 |
| 09:00 고정 -0.23% | 당일 open 없으면 `sell=buy_price` 후 수수료만 차감 | 시가 필수. 없으면 OPEN 유지 |
| 텔레그램 중복/15:30 | 수동 재실행 + 문구 미수정 | idempotent BUY, 문구 15:20, mock 조건식 무시 |

8/14 실제 시가 (어제 포지션, 청산 때 썼어야 할 값): 010170 13,300 / 103590 73,600 / 196170 317,000 (종가 매수 13,240 / 72,400 / 317,500).

8/14 보정 매수 OPEN: 042700 한미반도체, 047040 대우건설, 067310 하나마이크론.
