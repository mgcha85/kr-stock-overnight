"""
KRX Stock Overnight Configuration Loader
----------------------------------------
Loads environment variables from .env / .env.dev / .env.prod files.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Determine environment: dev or prod
ENV_TYPE = os.getenv("ENV_TYPE", os.getenv("ENV", "dev"))
env_file = ROOT_DIR / f".env.{ENV_TYPE}"
if not env_file.exists():
    env_file = ROOT_DIR / ".env"

load_dotenv(dotenv_path=env_file)

# Telegram Bot Settings
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8843947924:AAGoW1HAN3XXUG3kLuQ4hp4aMnu7IVJhd18")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "8516370855")

# Seed Capital & Trading Parameters
SEED_CAPITAL = float(os.getenv("SEED_CAPITAL", "10000000"))  # 10,000,000 KRW
TOP_K_TRADES = int(os.getenv("TOP_K_TRADES", "3"))
FEE_RATE = float(os.getenv("FEE_RATE", "0.0023"))  # 0.23% round-trip fee & tax

# Database & Storage Paths
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

DATA_PARQUET_PATH = DATA_DIR / "kr_kline_processed.parquet"
PAPER_DB_PATH = DATA_DIR / "paper_trading.db"
MODEL_DIR = ROOT_DIR / "research" / "models"

JUDAL_DB_PATH = Path(os.getenv("JUDAL_DB_PATH", "/mnt/data/projects/marketMosaic/backend/data/judal.db"))
SECTOR_DB_PATH = Path(os.getenv("SECTOR_DB_PATH", "/mnt/data/finance/candles/KO/sector_info.db"))
BACKTEST_DB_PATH = Path(os.getenv("BACKTEST_DB_PATH", "/mnt/data/finance/backtest_results.db"))
DASHBOARD_API_URL = os.getenv("DASHBOARD_API_URL", "http://146.56.115.71:8082/api/backtest")
