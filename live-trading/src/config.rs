//! Configuration — mirrors `src/kr_stock/config.py`.

use std::env;
use std::path::{Path, PathBuf};

use anyhow::Result;

/// Project root: checked at runtime (supports both host & Docker container).
pub fn root_dir() -> PathBuf {
    if let Ok(dir) = env::current_dir() {
        if dir.join("live-trading").exists() {
            return dir;
        }
        if dir.join("models").exists() {
            return dir.parent().map(Path::to_path_buf).unwrap_or(dir);
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")))
}

pub const FEE_RATE: f64 = 0.0023;
pub const TOP_K: usize = 3;
pub const SEED_CAPITAL: f64 = 10_000_000.0;

#[derive(Debug, Clone)]
pub struct Config {
    pub root_dir: PathBuf,
    pub data_dir: PathBuf,
    pub data_parquet: PathBuf,
    pub paper_db: PathBuf,
    pub model_dir: PathBuf,
    pub judal_db: PathBuf,
    pub sector_db: PathBuf,
    pub dart_db: PathBuf,
    pub day_data_db: PathBuf,
    pub kiwoom_api_url: String,
    pub meili_url: String,
    pub meili_key: String,
    pub telegram_bot_token: String,
    pub telegram_chat_id: String,
    pub seed_capital: f64,
    pub top_k: usize,
    pub fee_rate: f64,
    pub max_alloc_per_ticker: f64,
    pub overnight_api_url: String,
    pub env_type: String,
    /// `paper` (DB fills) or `live` (Kiwoom kt10000/kt10001).
    pub trading_mode: String,
    pub acc_no: String,
}

impl Config {
    /// Load `.env.{ENV_TYPE}` (default `dev`) then `.env`, matching Python.
    pub fn load() -> Result<Self> {
        let root = root_dir();
        let env_type = env::var("ENV_TYPE")
            .or_else(|_| env::var("ENV"))
            .unwrap_or_else(|_| "dev".into());
        let env_file = root.join(format!(".env.{env_type}"));
        if env_file.exists() {
            let _ = dotenvy::from_path(&env_file);
        } else {
            let _ = dotenvy::from_path(root.join(".env"));
        }
        // Also allow live-trading/.env
        let _ = dotenvy::dotenv();

        let data_dir = root.join("data");
        let model_dir = if root.join("live-trading/models").exists() {
            root.join("live-trading/models")
        } else {
            root.join("models")
        };

        Ok(Self {
            root_dir: root.clone(),
            data_dir: data_dir.clone(),
            data_parquet: env_path(
                "DATA_PARQUET_PATH",
                data_dir.join("kr_kline_processed.parquet"),
            ),
            paper_db: env_path("PAPER_DB_PATH", data_dir.join("paper_trading.db")),
            model_dir: env_path("MODEL_DIR", model_dir),
            judal_db: env_path(
                "JUDAL_DB_PATH",
                PathBuf::from("/mnt/data/projects/marketMosaic/backend/data/judal.db"),
            ),
            sector_db: env_path(
                "SECTOR_DB_PATH",
                PathBuf::from("/mnt/data/finance/candles/KO/sector_info.db"),
            ),
            dart_db: env_path(
                "DART_DB_PATH",
                PathBuf::from("/mnt/data/projects/marketMosaic/backend/data/dart.db"),
            ),
            day_data_db: env_path(
                "DAY_DATA_DB_PATH",
                PathBuf::from("/mnt/data/finance/candles/KO/day_data_full.db"),
            ),
            kiwoom_api_url: env::var("KIWOOM_API_URL")
                .unwrap_or_else(|_| "http://localhost:5000/api/condition".into()),
            meili_url: env::var("MEILI_URL").unwrap_or_else(|_| "http://localhost:37700".into()),
            meili_key: env::var("MEILI_KEY").unwrap_or_else(|_| "masterKey".into()),
            telegram_bot_token: env::var("TELEGRAM_BOT_TOKEN").unwrap_or_else(|_| {
                "8843947924:AAGoW1HAN3XXUG3kLuQ4hp4aMnu7IVJhd18".into()
            }),
            telegram_chat_id: env::var("TELEGRAM_CHAT_ID")
                .unwrap_or_else(|_| "8516370855".into()),
            seed_capital: env::var("SEED_CAPITAL")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(SEED_CAPITAL),
            top_k: env::var("TOP_K_TRADES")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(TOP_K),
            fee_rate: env::var("FEE_RATE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(FEE_RATE),
            max_alloc_per_ticker: env::var("MAX_ALLOC_PER_TICKER")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(500_000.0),
            overnight_api_url: overnight_api_url(),
            env_type,
            trading_mode: parse_trading_mode(),
            acc_no: env::var("ACC_NO")
                .or_else(|_| env::var("ACC_ID"))
                .or_else(|_| env::var("KIWOOM_ACC_NO"))
                .unwrap_or_default(),
        })
    }

    pub fn is_live(&self) -> bool {
        self.trading_mode == "live"
    }

    pub fn lgb_model_path(&self) -> PathBuf {
        env_path("LGB_MODEL_PATH", self.model_dir.join("lgb_kline_model.txt"))
    }

    pub fn scaler_path(&self) -> PathBuf {
        env_path("SCALER_PATH", self.model_dir.join("kline_scaler.json"))
    }

    pub fn onnx_model_path(&self) -> PathBuf {
        env_path("ONNX_MODEL_PATH", self.model_dir.join("pytorch_kline_model.onnx"))
    }

    pub fn roundtrip_path(&self) -> PathBuf {
        env_path("ROUNDTRIP_VECTORS_PATH", self.model_dir.join("roundtrip_vectors.json"))
    }
}

fn overnight_api_url() -> String {
    if let Ok(v) = env::var("OVERNIGHT_API_URL") {
        let trimmed = v.trim().trim_end_matches('/').to_string();
        if !trimmed.is_empty() {
            return trimmed;
        }
    }
    let dash = env::var("DASHBOARD_API_URL")
        .unwrap_or_else(|_| "http://146.56.115.71:8082/api/backtest".into());
    if let Some(idx) = dash.find("/api/") {
        format!("{}/api/overnight", &dash[..idx])
    } else {
        "http://146.56.115.71:8082/api/overnight".into()
    }
}

fn parse_trading_mode() -> String {
    let exec_live = env::var("EXECUTION_LIVE").unwrap_or_default();
    let raw = if matches!(
        exec_live.trim().to_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    ) {
        "live".to_string()
    } else {
        env::var("TRADING_MODE").unwrap_or_else(|_| "paper".into())
    };
    match raw.trim().to_lowercase().as_str() {
        "live" | "on" | "1" | "true" => "live".into(),
        _ => "paper".into(),
    }
}

fn env_path(key: &str, default: PathBuf) -> PathBuf {
    match env::var(key) {
        Ok(v) => {
            let p = PathBuf::from(&v);
            if p.exists() {
                p
            } else {
                default
            }
        }
        Err(_) => default,
    }
}
