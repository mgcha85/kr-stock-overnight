# Development TODO List (개발 로드맵 & 진행 현황)

## Phase 1: 리서치 및 아키텍처 설계 [완료]
- [x] 한국 주식 Overnight (종가 베팅) 매매 기법 및 핵심 알파 요소 리서치
- [x] MarketMosaic 수집 데이터 (뉴스, DART, 펀다멘털, 테마, 캔들/수급) 활용 데이터 파이프라인 설계
- [x] 다중 요소 점수화 모델 (Multi-Factor Overnight Scoring Model) 수립
- [x] 프로젝트 필수 아키텍처 및 환경 설정 파일 구현 (`.env.example`, `.env.dev`, `.env.prod`, `.gitignore`)
- [x] 컨테이너 구동 및 정지 스크립트 작성 (`start.sh`, `stop.sh`)
- [x] 문서화 수립 (`OVERNIGHT_STRATEGY_SPEC.md`, `API_INTERFACE.md`, `ARCHITECTURE.md`, `DATA_ENGINEERING_RULES.md`, `TODO.md`)

---

## Phase 2: Python 백테스팅 & Polars/Parquet/DuckDB 데이터 파이프라인 [완료]
- [x] 데이터 파이프라인 규정 수립 (`docs/DATA_ENGINEERING_RULES.md`: Polars 분석, Parquet 저장, DuckDB 인터페이스)
- [x] `pyproject.toml` 및 `uv sync` 의존성 설정 (Polars, Pandas, PyArrow, Tabulate, Google-GenerativeAI)
- [x] 캔들 타임스탬프 `open_time` (09:00:00) & `close_time` (15:30:00) 커스텀 가공 (`research/prepare_kline_data.py`)
- [x] 프로젝트 루트 `data/` 디렉토리에 `kr_kline_processed.parquet` (92MB) 및 `kr_kline_processed.db` (405MB) 저장
- [x] Train (2021-2023), Validation (2024), Test (2025-2026) 3구간 워크포워드 백테스트 스크립트 작성 (`research/backtest_overnight_splits.py`)
- [x] Buy & Hold (B&H) 벤치마크 대비 초과 수익률(Alpha), Sharpe Ratio, MDD 검증 보고서 작성 (`docs/BACKTEST_WALKFORWARD_RESULTS.md`)
- [x] Judal 테마 + LightGBM + PyTorch 하이브리드 머신러닝/딥러닝 모델 학습 및 검증 완료 (`JUNE_HYBRID_ML_DL_STRATEGY_REPORT.md`)
- [x] 백테스트 대시보드 API (`http://146.56.115.71:8082/api/backtest`) 자동 업로드 연동 완료

---

## Phase 2.5: Paper Trading 파이프라인 & 텔레그램 연동 [완료]
- [x] 1,000만원 시드 머니 기반 페이퍼 트레이딩 엔진 구축 (`src/kr_stock/paper_engine.py`)
- [x] Single Source of Truth 스코어링 모듈 개발 (`src/kr_stock/inference.py` - Backtest ↔ Paper-Trading 100% 로직 공유)
- [x] SQLite 기반 페이퍼 매매 일지 및 계좌 자산 추적 DB 생성 (`data/paper_trading.db`)
- [x] 장 마감(15:30) 매수 알림 텔레그램 봇 자동화 (`src/kr_stock/telegram.py`)
- [x] 장 시작(09:00) 매도 및 1주일/1개월 누적 수익률 보고 텔레그램 알림 구현
- [x] 장후 데이터 다운로드 ↔ 백테스트 ↔ 페이퍼 트레이딩 100% Signal Parity 자동 검증 보고서 구현
- [x] Dockerfile 및 podman-compose 통합 구동 체계 완성
- [x] 알고리즘 명세 및 100% 백테스트 재현 가이드 작성 (`docs/STRATEGY_SPEC_AND_REPRODUCTION_GUIDE.md`)

---

## Phase 3: Go 라이브 매매 엔진 개발 [진행 예정]
- [ ] Go 기반 KIS(한국투자증권) API 클라이언트 모듈 구현 (`backend/internal/kis/`)
- [ ] 15:15 타겟 종목 스코어링 수집 및 15:20 지정가/동시호가 자동 매수 모듈 (`backend/internal/engine/`)
- [ ] 익일 09:00 시가 동시호가 자동 청산 및 트레일링 스탑 모듈
- [ ] KIS 모의투자(Dry-Run) 및 실전 매매 스위칭 기능 구현

---

## Phase 4: Svelte 프론트엔드 대시보드 & 배포 [진행 예정]
- [ ] Svelte + Tailwind CSS 기반 Overnight 실시간 모니터링 UI 구축 (`frontend/`)
- [ ] 실시간 타겟 종목 스코어링 카드, DART 공시/뉴스 하이라이트, 매매 내역 컴포넌트 개발
- [ ] `podman-compose.yml` 서비스 오케스트레이션 패키징
- [ ] GitHub Actions CI/CD pipeline 설정 (`DEPLOY_HOST`, `DEPLOY_SSH_KEY`, `DEPLOY_USER`, `GHCR_TOKEN`)
