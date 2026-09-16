//! Kiwoom condition search manager — mirrors `kiwoom_condition.py`.

use std::collections::HashSet;
use std::future::Future;

use anyhow::Result;
use chrono::{Duration, NaiveDate};
use polars::prelude::*;
use tracing::{info, warn};

use crate::config::Config;
use crate::kiwoom::{fetch_condition_stocks, KiwoomAuth, KiwoomClient};

pub struct KiwoomConditionManager {
    pub condition_name: String,
    cfg: Config,
}

fn block_on<T>(fut: impl Future<Output = T>) -> T {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => tokio::task::block_in_place(|| handle.block_on(fut)),
        Err(_) => tokio::runtime::Runtime::new()
            .expect("tokio runtime")
            .block_on(fut),
    }
}

fn kiwoom_live_enabled() -> bool {
    matches!(
        std::env::var("KIWOOM_LIVE")
            .unwrap_or_default()
            .to_lowercase()
            .as_str(),
        "1" | "true" | "yes"
    )
}

impl KiwoomConditionManager {
    pub fn new(cfg: Config, condition_name: impl Into<String>) -> Self {
        Self {
            condition_name: condition_name.into(),
            cfg,
        }
    }

    fn fetch_hts_condition_codes(&self) -> Option<Vec<String>> {
        if !kiwoom_live_enabled() {
            return None;
        }
        let app_key = std::env::var("KIWOOM_APP_KEY")
            .or_else(|_| std::env::var("APP_KEY"))
            .unwrap_or_default();
        let secret = std::env::var("KIWOOM_SECRET_KEY")
            .or_else(|_| std::env::var("SECRET_KEY"))
            .unwrap_or_default();
        if app_key.is_empty() || secret.is_empty() {
            warn!("KIWOOM_LIVE=1 but APP_KEY/SECRET_KEY missing");
            return None;
        }
        let client = KiwoomClient::from_env();
        let token_path = std::env::var("KIWOOM_TOKEN_PATH").unwrap_or_else(|_| {
            self.cfg
                .paper_db
                .with_file_name("kiwoom_access_token.json")
                .display()
                .to_string()
        });
        let auth = KiwoomAuth::new(client, app_key, secret).with_token_file(&token_path);
        let token = match block_on(auth.ensure_token()) {
            Ok(t) => t,
            Err(e) => {
                warn!("[Kiwoom LIVE] token: {e}");
                return None;
            }
        };
        let ws = KiwoomAuth::ws_url();
        let names = [self.condition_name.clone()];
        match block_on(fetch_condition_stocks(&names, &token, &ws)) {
            Ok(codes) if !codes.is_empty() => {
                let codes: Vec<String> = codes.into_iter().map(|c| zfill6(&c)).collect();
                info!(
                    "[Kiwoom LIVE] HTS '{}' returned {} candidates: {:?}",
                    self.condition_name,
                    codes.len(),
                    codes
                );
                Some(codes)
            }
            Ok(_) => {
                warn!("[Kiwoom LIVE] HTS '{}' returned 0 codes", self.condition_name);
                None
            }
            Err(e) => {
                warn!("[Kiwoom LIVE] HTS search failed: {e}");
                None
            }
        }
    }

    pub fn fetch_candidate_codes_from_api(&self) -> Option<Vec<String>> {
        let url = format!(
            "{}?name={}",
            self.cfg.kiwoom_api_url,
            urlencoding_simple(&self.condition_name)
        );
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .ok()?;
        let resp = client.get(&url).send().ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let data: serde_json::Value = resp.json().ok()?;
        let accept_mock = std::env::var("KIWOOM_ACCEPT_MOCK").unwrap_or_default() == "1";
        let is_mock = data.get("source").and_then(|v| v.as_str()) == Some("mock")
            || (self.cfg.kiwoom_api_url.contains(":5000")
                && self.cfg.kiwoom_api_url.contains("/api/condition"));
        if is_mock && !accept_mock {
            info!(
                "[Kiwoom API] Ignoring mock/stale condition endpoint {}. Falling back to offline HTS sim.",
                self.cfg.kiwoom_api_url
            );
            return None;
        }
        let codes = data.get("codes")?.as_array()?;
        let out: Vec<String> = codes
            .iter()
            .filter_map(|c| {
                if let Some(s) = c.as_str() {
                    Some(zfill6(s))
                } else if let Some(n) = c.as_u64() {
                    Some(format!("{n:06}"))
                } else if let Some(n) = c.as_i64() {
                    Some(format!("{n:06}"))
                } else {
                    None
                }
            })
            .collect();
        info!(
            "[Kiwoom API] Real-time '{}' returned {} candidates: {:?}",
            self.condition_name,
            out.len(),
            out
        );
        Some(out)
    }

    pub fn get_condition_search_codes(&self, target_date: &str) -> Result<Vec<String>> {
        if let Ok(env_codes) = std::env::var("KIWOOM_CANDIDATE_CODES") {
            let codes: Vec<String> = env_codes
                .split(',')
                .map(|c| c.trim())
                .filter(|c| !c.is_empty())
                .map(zfill6)
                .collect();
            info!(
                "[Kiwoom Condition] Using environment override candidate codes ({}): {:?}",
                codes.len(),
                codes
            );
            return Ok(codes);
        }

        let gh = {
            info!(
                "[Kiwoom Condition] Simulating '{}' condition for date: {}...",
                self.condition_name, target_date
            );
            self.offline_fallback(target_date)?
        };

        let live = self
            .fetch_hts_condition_codes()
            .or_else(|| self.fetch_candidate_codes_from_api());
        if let Some(live) = live {
            if !live.is_empty() {
                info!(
                    "[Kiwoom Condition] using live HTS {} codes (parquet G+H had {})",
                    live.len(),
                    gh.len()
                );
                return Ok(live);
            }
        }
        Ok(gh)
    }

    /// Offline HTS 「종가베팅」= G (open gap 2~28%) + H (turnover >= 200억).
    fn offline_fallback(&self, target_date: &str) -> Result<Vec<String>> {
        let start_date = NaiveDate::parse_from_str(target_date, "%Y-%m-%d")
            .ok()
            .and_then(|d| d.checked_sub_signed(Duration::days(10)))
            .map(|d| d.format("%Y-%m-%d").to_string())
            .unwrap_or_else(|| "2026-01-01".into());

        let candles = LazyFrame::scan_parquet(
            self.cfg.data_parquet.to_string_lossy().as_ref(),
            ScanArgsParquet::default(),
        )?
        .filter(
            col("date")
                .gt_eq(lit(start_date.as_str()))
                .and(col("date").lt_eq(lit(target_date))),
        )
        .select([
            col("date"),
            col("ticker"),
            col("open"),
            col("close"),
            col("turnover"),
        ])
        .collect()?;

        if candles.height() == 0 {
            return Ok(vec![]);
        }

        let day = candles
            .lazy()
            .with_columns([col("ticker").str().slice(lit(0), lit(6)).alias("code")])
            .sort(
                ["code", "date"],
                SortMultipleOptions::default().with_nulls_last(true),
            )
            .with_columns([col("close")
                .shift(lit(1))
                .over([col("code")])
                .alias("prev_close")])
            .filter(col("date").eq(lit(target_date)))
            .with_columns([((col("open") / col("prev_close")) - lit(1.0)).alias("open_gap")])
            .filter(
                col("turnover")
                    .gt_eq(lit(2e10))
                    .and(col("open_gap").is_not_null())
                    .and(col("open_gap").gt_eq(lit(0.02)))
                    .and(col("open_gap").lt_eq(lit(0.28))),
            )
            .collect()?;

        if day.height() == 0 {
            return Ok(vec![]);
        }

        let codes_col = day.column("code")?.str()?;
        let mut out: Vec<String> = Vec::new();
        let mut seen = HashSet::new();
        for i in 0..day.height() {
            let code = codes_col.get(i).unwrap_or("");
            if code.len() == 6
                && code.chars().all(|ch| ch.is_ascii_digit())
                && seen.insert(code.to_string())
            {
                out.push(code.to_string());
            }
        }

        info!(
            "[Kiwoom Condition] '{}' matched {} candidates on {}: {:?}",
            self.condition_name,
            out.len(),
            target_date,
            out
        );
        Ok(out)
    }
}

pub fn zfill6(s: &str) -> String {
    let clean = s.split('.').next().unwrap_or(s).trim();
    if clean.len() >= 6 {
        clean.to_string()
    } else {
        format!("{clean:0>6}")
    }
}

fn urlencoding_simple(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}
