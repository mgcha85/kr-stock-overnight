# PROJECT.md — KRX Overnight (종가베팅)

이 문서만 보고 **대시보드에 올린 `judal_hybrid_lgb_pytorch_KRX`와 같은 종목**이 나오게 재현한다.  
종목당 한도(`MAX_ALLOC_PER_TICKER`)는 수량만 줄일 수 있고, **선정 종목을 바꾸면 안 된다.**

기준 결과: [http://146.56.115.71:8082/algorithm/354](http://146.56.115.71:8082/algorithm/354)  
알고리즘 이름: `judal_hybrid_lgb_pytorch_KRX`  
기간: **2026-01-06 ~ 2026-07-30**, Top-3, 수수료 0.23%, 종가 매수 / 다음날 시가 매도.

---

## 1. 한 줄 파이프라인

```
일봉 G+H (백테스트와 동일 유니버스)
    → 주달 테마 조인 + LGB/PyTorch 확률
    → hybrid_score Top-3
    → 당일 종가 매수 / 익일 시가 매도
```

키움 「종가베팅」은 **조회해서 차이만 로그**한다. 키움 대금/시가는 일봉 parquet와 달라서 HTS를 유일한 후보로 두면 Top-3가 어긋난다.  
매수 후보는 **항상 일봉 G+H**다. LGB/Torch는 그 안에서 한 번 더 적용한다.

---

## 2. 종목이 같아지려면

| 단계 | 백테스트 | 실매매 |
|---|---|---|
| 1. 후보 | 일봉 G+H | **같은 일봉 G+H** (키움 HTS는 대조 로그) |
| 2. 테마 | 주달 `stock_history` ∩ `theme_stocks` | 동일 |
| 3. ML 컷 | `p_lgb ≥ 0.35`, `p_torch ≥ 0.35` | 동일 |
| 4. 순위 | `hybrid_score` 내림차순 Top-3 | 동일 |
| 5. 체결 | 일봉 종가 → 다음날 일봉 시가 | 종가(15:28 예상체결 ≈ 종가) → 다음날 09:00 시가 |

HTS에 I/K/L/M/보통주/종가등락을 넣으면 후보가 줄어 **백테스트 Top-3와 어긋난다.**  
키움 조건은 **G, H 두 개만** 쓴다.

---

## 3. 키움 HTS 「종가베팅」에 넣을 조건

조건검색 이름: **`종가베팅`** (코드가 이 문자열로 조회한다.)

| # | 항목 | 값 | 백테스트 대응 |
|---|---|---|---|
| **H** | 거래대금 | **≥ 20,000,000,000원 (200억)** | `turnover >= 2e10` |
| **G** | 시가 등락 | 전일 종가 대비 당일 시가 **2~28%** | `0.02 <= open/prev_close-1 <= 0.28` |

넣지 말 것: 종가 등락 하한/상한, 주가 2천~50만(I), 대금 순위(K), 52주 신저가(L), 종가>20일선(M), 보통주만.  
종가 등락 &lt; 29% 와 ML 컷은 **스코어 단계**에서 적용한다.

**검증:** 장 마감 후 HTS 종가베팅 리스트가, 같은 날짜 오프라인 시뮬 리스트와 거의 같아야 한다.

```bash
cd /mnt/data/projects/kr_stock
KIWOOM_LIVE=0 PYTHONPATH=src:. uv run python - <<'PY'
from kr_stock.kiwoom_condition import KiwoomConditionManager
print(KiwoomConditionManager().get_condition_search_codes("2026-07-30"))
PY
```

실시간 조회는 `KIWOOM_LIVE=1` + `.env`의 `KIWOOM_APP_KEY` / `KIWOOM_SECRET_KEY`.

---

## 4. HTS 이후 스코어 (백테스트·라이브 공통)

구현: `src/kr_stock/inference.py` `OvernightScorer`, Rust `live-trading/src/scoring.rs`.

1. 주달 `stock_history` (그날 `crawl_date`)와 `theme_stocks` / `themes` inner join. 테마 없는 종목 탈락.
2. `data/kr_kline_processed.parquet`에서 당일 포함 약 45일 OHLCV → 피처.
3. 피처 10개: `high_close_ratio`, `body_ratio`, `upper_shadow_ratio`, `ret_1d`, `ret_3d`, `ret_5d`, `vol_ratio_5d`, `bb_pct_b`, `bb_width`, `rsi_14`.
4. LightGBM `p_lgb`, PyTorch MLP `p_torch`.
5. `judal_score` (대장 35 + 테마평균 clip + 등락 clip + 종가위치 − 과열 페널티).
6. `hybrid_score = judal_score + p_lgb*40 + p_torch*40`.
7. 컷: 거래대금 ≥ 200억, 등락 < 29%, `p_lgb ≥ 0.35`, `p_torch ≥ 0.35`.
8. 종목당 최고점 1행만 남기고 **Top-3**.

모델 파일 (라이브·대시보드 재현용, **2026-05만 학습**):

| 파일 | 경로 |
|---|---|
| LightGBM | `research/models/lgb_kline_model.joblib` |
| Scaler | `research/models/kline_scaler.joblib` |
| PyTorch | `research/models/pytorch_kline_model.pt` |

학습 코드: `research/kline_ml_dl_pipeline.py` (`2026-05-01` ~ `2026-05-31`).  
이 모델로 5~7월을 돌리면 대시보드 +1170%가 나오지만, 5월은 학습 구간이다.

---

## 5. 체결

- **매수:** 당일 **일봉 종가**. (1분 15:19 사용 금지.)
- **매도:** **다음 거래일 일봉 시가** (`next_open`).
- **수수료:** 왕복 `FEE_RATE=0.0023` (0.23%).
- 손익: `next_open / close - 1 - 0.0023`.
- 포지션: 그날 Top-3에 현금 균등 분할. **캡은 수량만.** 1주도 못 사면 그 종목은 스킵되지만, 스코어 Top-3 자체는 유지하고 패리티는 종목 집합을 본다.

백테스트 체결 구현: `src/kr_stock/overnight_backtest.py` `fill_1520_0900` (이름은 구버전, 내용은 종가/시가).

---

## 6. 데이터

| 이름 | 경로 |
|---|---|
| 일봉 피처 parquet | `/mnt/data/projects/kr_stock/data/kr_kline_processed.parquet` |
| 원본 일봉 SQLite | `/mnt/data/finance/candles/KO/day_data_full.db` |
| 주달 | `JUDAL_DB_PATH` 기본 `/mnt/data/projects/marketMosaic/backend/data/judal.db` |
| 종목명 | `/mnt/data/finance/candles/KO/sector_info.db` |

당일 일봉 갱신:

```bash
PYTHONPATH=src:. uv run python scripts/fetch_today_kr_candles.py --date YYYY-MM-DD
```

FinanceDataReader KRX 리스트 upsert. `next_open`은 다음 날짜 시가로 채운다.  
주달 `stock_history`에 그 날짜 행이 있어야 스코어가 나온다.

---

## 7. 백테스트 재현

환경:

```bash
cd /mnt/data/projects/kr_stock
uv sync
```

대시보드와 같은 창 (모델 tag 빈 문자열 = 위 라이브 모델, 종가/시가):

```bash
PYTHONPATH=src:. uv run python - <<'PY'
from kr_stock.overnight_backtest import run_1520_0900_backtest, simulate_sequential, dashboard_payload

filled, monthly, weekly, _ = run_1520_0900_backtest(
    start_date="2026-01-06",
    end_date="2026-07-30",
    top_k=3,
    model_tag="",  # lgb_kline_model.joblib
)
sim = simulate_sequential(filled, seed_capital=10_000_000)
print(
    f"{sim['test_start']}~{sim['test_end']}  "
    f"return={sim['total_return']*100:+.2f}%  WR={sim['win_rate']*100:.1f}%  "
    f"PF={sim['pf']:.2f}  MDD={sim['mdd']*100:.2f}%  trades={sim['n_trades']}"
)
PY
```

대시보드는 주달이 있는 **56일만** 거래해 +1170% / 168건이다.  
같은 모델로 그 구간 **모든 거래일**을 넣으면 일수·수익률이 더 커진다. 종목 규칙은 같다.

한 날짜 Top-3만 보려면:

```bash
PYTHONPATH=src:. uv run python scripts/run_kiwoom_overnight_analysis.py --date 2026-07-30
```

(`KIWOOM_LIVE=0`이면 오프라인 HTS 시뮬 → 스코어.)

업로드 (선택):

```bash
PYTHONPATH=src:. uv run python scripts/upload_mtf_backtest_to_server.py
```

`POST http://146.56.115.71:8082/api/backtest`  
이름은 `judal_hybrid_lgb_pytorch_KRX`, timeframe `1d`.

---

## 8. 실매매 재현

런타임: Rust `live-trading/` 컨테이너 `kr_stock_live_trading` (`podman-compose.yml`).  
스케줄 (KST 평일): **09:00 매도**, **15:28 매수**, **18:00 패리티(확정 일봉)**.

1. `.env` / `.env.prod`: `TRADING_MODE=live`, `KIWOOM_LIVE=1`, `TOP_K_TRADES=3`, `FEE_RATE=0.0023`, 키움 키.  
   `MAX_ALLOC_PER_TICKER`는 수량 한도일 뿐, Top-3 종목 코드를 바꾸지 말 것.
2. 키움 HTS 조건검색 이름 `종가베팅` = **§3**.
3. 매수: HTS 코드 → `get_candidates_for_date(..., candidate_codes=HTS)` → Top-3.  
   구현: Python `src/kr_stock/paper_engine.py`, Rust `live-trading/src/engine.rs`.
4. 패리티: 같은 날 HTS+스코어 Top-3와 실제 매수 종목 집합이 같으면 성공.

컨테이너:

```bash
cd /mnt/data/projects/kr_stock
podman-compose up -d --build
```

로그: `podman logs -f kr_stock_live_trading`  
HTS 리스트와 Top-3가 찍혀야 한다. 조건검색을 건너뛰고 주달 전체만 돌리면 이 문서의 실매매가 아니다.

---

## 9. 코드 맵

| 역할 | 파일 |
|---|---|
| HTS 실시간 + 오프라인 시뮬 | `src/kr_stock/kiwoom_condition.py`, `src/kr_stock/kiwoom_live.py`, `live-trading/src/condition.rs` |
| 스코어 Top-3 | `src/kr_stock/inference.py`, `live-trading/src/scoring.rs` |
| 백테스트 선정·종가/시가 | `src/kr_stock/sell_timing.py` `build_overnight_trades`, `src/kr_stock/overnight_backtest.py` |
| 모델 학습 (5월) | `research/kline_ml_dl_pipeline.py` |
| 일봉 수집 | `scripts/fetch_today_kr_candles.py` |
| 라이브 스케줄 | `live-trading/src/scheduler.rs` |

---

## 10. 재현 체크리스트

1. parquet에 대상 날짜 일봉이 있다.
2. 주달에 그 날짜 `stock_history`가 있다.
3. 모델 세 파일이 `research/models/lgb_kline*` (태그 없음)이다.
4. 키움 「종가베팅」이 §3 세 조건만 쓴다.
5. 같은 날짜에 `run_kiwoom_overnight_analysis.py` Top-3와 라이브 매수 종목이 같다.
6. 체결은 종가/다음날 시가이다 (15:19 분봉 없음).
)
