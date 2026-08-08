# KRX Stock Overnight Strategy (한국 주식 오버나이트 전략)

A research pipeline and execution system for Korean Stock Market Overnight (종가 베팅) trading strategies using multi-source data (News, DART, Themes, Candle/Turnover).

## Core Principles & Engineering Standards
1. **Data Analysis**: `Polars` for fast multi-threaded dataframe processing.
2. **Storage Format**: `Parquet` (.parquet) format stored in `data/` directory.
3. **Analytics DB**: `DuckDB` / SQLite for fast query and interface connection.
4. **Time Contract**: Explicit `open_time` (09:00:00) & `close_time` (15:30:00) to eliminate Look-ahead bias.

Detailed documentation: [docs/DATA_ENGINEERING_RULES.md](docs/DATA_ENGINEERING_RULES.md)

## Features
- Multi-Factor Scoring Model (Technical, Theme, OpenRouter Free AI Model Round-Robin NLP Evaluator via `OPEN_ROUTER_API_KEY`, Risk/Macro)
- Time-partitioned Candle Data Pipeline (`open_time` & `close_time` explicit timestamps)
- Train / Validation / Test Walk-forward Backtesting against Buy & Hold (B&H) benchmark
- Live Trading Engine in Go + Container deployment via Podman-compose

## Execution Guide
```bash
# 1. Install dependencies
uv sync

# 2. Prepare kline data (Extract, transform with open_time/close_time, save to data/)
uv run python research/prepare_kline_data.py

# 3. Run Walk-forward backtest (Train / Validation / Test vs Buy & Hold)
uv run python research/backtest_overnight_splits.py
```

## Documentation
- Strategy Specification: [docs/OVERNIGHT_STRATEGY_SPEC.md](docs/OVERNIGHT_STRATEGY_SPEC.md)
- Walk-forward Results: [docs/BACKTEST_WALKFORWARD_RESULTS.md](docs/BACKTEST_WALKFORWARD_RESULTS.md)
- Data Engineering Rules: [docs/DATA_ENGINEERING_RULES.md](docs/DATA_ENGINEERING_RULES.md)
- Architecture Overview: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Roadmap / TODO: [docs/TODO.md](docs/TODO.md)
