# System Architecture (시스템 아키텍처)

## 1. 기술 스택 (Technology Stack)

| 구분 | 기술 스택 | 비고 |
| :--- | :--- | :--- |
| **Language & Backend** | **Go (Golang)** | 실시간 매매 엔진 & API 서버 (Rule 4) |
| **Research & AI** | **Python (uv sync)** | 다중 데이터 파이프라인 & Gemini-3.0-Flash 스코어링 (Rule 8, 10) |
| **Frontend** | **Svelte + Tailwind CSS** | 실시간 오버나이트 대시보드 & 모니터링 UI (Rule 2) |
| **Database** | **Supabase (PostgreSQL) / Meilisearch** | 체결 내역, 종목 스코어링 데이터, 뉴스 검색 (Rule 3) |
| **Container & Orchestration** | **Podman-compose** | `start.sh`, `stop.sh` 구동 제어 (Rule 5, 7) |
| **CI/CD** | **GitHub Actions** | `DEPLOY_HOST`, `DEPLOY_SSH_KEY`, `DEPLOY_USER`, `GHCR_TOKEN` (Rule 6) |

---

## 2. 모듈 구조 (Directory Layout)

```
kr_stock/
├── .env.example              # 환경 변수 템플릿
├── .env.dev                  # 개발 환경 변수
├── .env.prod                 # 운영 환경 변수
├── .gitignore                # .env 및 빌드 결과물 제외
├── podman-compose.yml        # 서비스 오케스트레이션
├── start.sh                  # 구동 스크립트 (ENV_TYPE=dev|prod)
├── stop.sh                   # 정지 스크립트
├── pyproject.toml            # Python uv 패키지 관리
├── docs/                     # 기술 문서 및 명세서
│   ├── OVERNIGHT_STRATEGY_SPEC.md # 전략 상세 명세
│   ├── API_INTERFACE.md           # API 인터페이스 명세
│   ├── ARCHITECTURE.md            # 시스템 아키텍처
│   └── TODO.md                    # 개발 로드맵 및 TODO
├── backend/                  # Go Live Engine
│   ├── cmd/main.go
│   ├── internal/engine/
│   ├── internal/scoring/
│   └── internal/kis/
├── research/                 # Python Backtest & Gemini NLP Pipeline
│   ├── data_loader.py
│   ├── gemini_evaluator.py
│   └── backtest_overnight.py
└── frontend/                 # Svelte Web Dashboard
    ├── src/routes/+page.svelte
    └── src/lib/components/
```

---

## 3. Deployment Flow & Environment Lifecycle

```
[Local Dev / Backtest]
   uv run python research/backtest_overnight.py
   
[Container Run (Podman)]
   ENV_TYPE=dev ./start.sh   --> Loads .env.dev
   ENV_TYPE=prod ./start.sh  --> Loads .env.prod
   ./stop.sh                 --> Brings down podman containers
```
