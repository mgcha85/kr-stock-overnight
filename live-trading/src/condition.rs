//! Kiwoom condition search manager — mirrors `kiwoom_condition.py`.

use std::collections::HashSet;

use anyhow::{Context, Result};
use chrono::{Duration, NaiveDate};
use polars::prelude::*;
use tracing::info;

use crate::config::Config;

pub struct KiwoomConditionManager {
    pub condition_name: String,
    cfg: Config,
}

impl KiwoomConditionManager {
    pub fn new(cfg: Config, condition_name: impl Into<String>) -> Self {
        Self {
            condition_name: condition_name.into(),
            cfg,
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

        if let Some(live) = self.fetch_candidate_codes_from_api() {
            return Ok(live);
        }

        info!(
            "[Kiwoom Condition] Simulating '{}' condition for date: {}...",
            self.condition_name, target_date
        );
        self.offline_fallback(target_date)
    }

    /// Offline A/B/C/D/E/H filters — identical to Python lines 108–124.
    fn offline_fallback(&self, target_date: &str) -> Result<Vec<String>> {
        let conn = rusqlite::Connection::open(&self.cfg.judal_db)
            .with_context(|| format!("open judal {}", self.cfg.judal_db.display()))?;
        let mut stmt = conn.prepare(
            "SELECT code, change_rate as stock_change, neglect_index_52w
             FROM stock_history WHERE crawl_date = ?",
        )?;
        let hist_rows: Vec<(String, f64)> = stmt
            .query_map([target_date], |r| {
                Ok((zfill6(&r.get::<_, String>(0)?), r.get::<_, f64>(1)?))
            })?
            .filter_map(|r| r.ok())
            .collect();
        if hist_rows.is_empty() {
            return Ok(vec![]);
        }

        let start_date = NaiveDate::parse_from_str(target_date, "%Y-%m-%d")
            .ok()
            .and_then(|d| d.checked_sub_signed(Duration::days(45)))
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
        .select([col("date"), col("ticker"), col("close"), col("turnover")])
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
            .with_columns([
                col("close")
                    .rolling_mean(RollingOptionsFixedWindow {
                        window_size: 20,
                        min_periods: 5,
                        weights: None,
                        center: false,
                        fn_params: None,
                    })
                    .over([col("code")])
                    .alias("sma_20"),
                col("close")
                    .rolling_min(RollingOptionsFixedWindow {
                        window_size: 250,
                        min_periods: 10,
                        weights: None,
                        center: false,
                        fn_params: None,
                    })
                    .over([col("code")])
                    .alias("low_52w"),
            ])
            .filter(col("date").eq(lit(target_date)))
            .with_columns([col("turnover")
                .rank(
                    RankOptions {
                        method: RankMethod::Average,
                        descending: true,
                    },
                    None,
                )
                .alias("turnover_rank")])
            .collect()?;

        if day.height() == 0 {
            return Ok(vec![]);
        }

        let codes: Vec<String> = hist_rows.iter().map(|(c, _)| c.clone()).collect();
        let changes: Vec<f64> = hist_rows.iter().map(|(_, ch)| *ch).collect();
        let hist_df = DataFrame::new(vec![
            Series::new("code".into(), codes).into(),
            Series::new("stock_change".into(), changes).into(),
        ])?;

        let merged = day
            .lazy()
            .join(
                hist_df.lazy(),
                [col("code")],
                [col("code")],
                JoinArgs::new(JoinType::Inner),
            )
            .collect()?;

        let codes_col = merged.column("code")?.str()?;
        let turnover = merged.column("turnover")?.f64()?;
        let stock_change = merged.column("stock_change")?.f64()?;
        let close = merged.column("close")?.f64()?;
        let turnover_rank = merged.column("turnover_rank")?.f64()?;
        let low_52w = merged.column("low_52w")?.f64()?;
        let sma_20 = merged.column("sma_20")?.f64()?;

        let excl_suffixes: HashSet<char> =
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'K', 'L', 'M']
                .into_iter()
                .collect();

        let mut out: Vec<String> = Vec::new();
        let mut seen = HashSet::new();
        for i in 0..merged.height() {
            let code = codes_col.get(i).unwrap_or("");
            let t = turnover.get(i).unwrap_or(f64::NAN);
            let ch = stock_change.get(i).unwrap_or(f64::NAN);
            let c = close.get(i).unwrap_or(f64::NAN);
            let tr = turnover_rank.get(i).unwrap_or(f64::NAN);
            let lo = low_52w.get(i).unwrap_or(f64::NAN);
            let sma = sma_20.get(i).unwrap_or(f64::NAN);

            let cond_a = t >= 2e10;
            let cond_b1 = (10.0..=28.5).contains(&ch);
            let cond_b2 = ch >= 5.0 && t >= 5e10;
            let cond_b = cond_b1 || cond_b2;
            let cond_c = (2000.0..=500_000.0).contains(&c);
            let cond_d = tr <= 150.0;
            let cond_e = c > lo;
            let cond_h = c > sma;
            let last = code.chars().last().unwrap_or('\0');
            let cond_excl = !excl_suffixes.contains(&last);

            if cond_a && cond_b && cond_c && cond_d && cond_e && cond_h && cond_excl
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
