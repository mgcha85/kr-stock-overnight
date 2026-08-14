//! Overnight hybrid scorer — mirrors `inference.OvernightScorer`.

use std::collections::HashMap;

use anyhow::{Context, Result};
use chrono::{Duration, NaiveDate};
use ndarray::Array2;
use polars::prelude::*;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use crate::condition::zfill6;
use crate::config::Config;
use crate::features::{compute_kline_features, DAILY_FEATURE_COLS};
use crate::models::{LgbModel, OnnxModel, StandardScaler};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pick {
    pub date: String,
    pub code: String,
    pub ticker: String,
    pub stock_name: String,
    pub theme_name: String,
    pub close_price: f64,
    pub next_open: f64,
    pub stock_change: f64,
    pub p_lgb: f64,
    pub p_torch: f64,
    pub hybrid_score: f64,
    pub news_count: usize,
    pub dart_count: usize,
}

pub struct OvernightScorer {
    cfg: Config,
    gbm: LgbModel,
    scaler: StandardScaler,
    onnx: OnnxModel,
    ticker_map: HashMap<String, String>,
}

impl OvernightScorer {
    pub fn new(cfg: Config) -> Result<Self> {
        let gbm = LgbModel::load(cfg.lgb_model_path())?;
        let scaler = StandardScaler::load(cfg.scaler_path())?;
        let onnx = OnnxModel::load(cfg.onnx_model_path())?;
        let ticker_map = load_ticker_name_map(&cfg.sector_db);
        Ok(Self {
            cfg,
            gbm,
            scaler,
            onnx,
            ticker_map,
        })
    }

    pub fn get_candidates_for_date(
        &mut self,
        target_date: &str,
        top_k: usize,
        min_turnover: f64,
        max_stock_change: f64,
        min_p_lgb: f64,
        min_p_torch: f64,
        candidate_codes: Option<&[String]>,
    ) -> Result<Vec<Pick>> {
        let clean_candidates: Option<Vec<String>> =
            candidate_codes.map(|cs| cs.iter().map(|c| zfill6(c)).collect());

        let conn = Connection::open(&self.cfg.judal_db)
            .with_context(|| format!("open {}", self.cfg.judal_db.display()))?;

        let hist: Vec<(String, String, String, f64)> = {
            let mut stmt = conn.prepare(
                "SELECT crawl_date as date, code, name, change_rate as stock_change,
                        neglect_index_52w, expected_return
                 FROM stock_history WHERE crawl_date = ?",
            )?;
            let rows = stmt.query_map([target_date], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    zfill6(&r.get::<_, String>(1)?),
                    r.get::<_, String>(2)?,
                    r.get::<_, f64>(3)?,
                ))
            })?;
            rows.filter_map(|r| r.ok()).collect()
        };

        if hist.is_empty() {
            return Ok(vec![]);
        }

        let hist: Vec<_> = if let Some(ref cands) = clean_candidates {
            let set: std::collections::HashSet<_> = cands.iter().cloned().collect();
            hist.into_iter()
                .filter(|(_, code, _, _)| set.contains(code))
                .collect()
        } else {
            hist
        };
        if hist.is_empty() {
            return Ok(vec![]);
        }

        let theme_stocks: Vec<(i64, String)> = {
            let mut ts_stmt =
                conn.prepare("SELECT theme_idx, stock_code as code FROM theme_stocks")?;
            let rows =
                ts_stmt.query_map([], |r| Ok((r.get(0)?, zfill6(&r.get::<_, String>(1)?))))?;
            rows.filter_map(|r| r.ok()).collect()
        };

        let themes: HashMap<i64, String> = {
            let mut th_stmt = conn.prepare("SELECT theme_idx, name as theme_name FROM themes")?;
            let rows = th_stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
            rows.filter_map(|r| r.ok()).collect()
        };
        drop(conn);

        // Join hist × theme_stocks × themes
        #[derive(Clone)]
        struct Joined {
            date: String,
            code: String,
            name: String,
            stock_change: f64,
            theme_idx: i64,
            theme_name: String,
        }

        let mut joined: Vec<Joined> = Vec::new();
        for (date, code, name, stock_change) in &hist {
            for (theme_idx, scode) in &theme_stocks {
                if scode == code {
                    if let Some(theme_name) = themes.get(theme_idx) {
                        joined.push(Joined {
                            date: date.clone(),
                            code: code.clone(),
                            name: name.clone(),
                            stock_change: *stock_change,
                            theme_idx: *theme_idx,
                            theme_name: theme_name.clone(),
                        });
                    }
                }
            }
        }
        if joined.is_empty() {
            return Ok(vec![]);
        }

        // theme_avg / max / count per (date, theme_idx, theme_name)
        let mut theme_stats: HashMap<(String, i64, String), (f64, f64, usize)> = HashMap::new();
        for j in &joined {
            let key = (j.date.clone(), j.theme_idx, j.theme_name.clone());
            let e = theme_stats.entry(key).or_insert((0.0, f64::NEG_INFINITY, 0));
            e.0 += j.stock_change;
            e.1 = e.1.max(j.stock_change);
            e.2 += 1;
        }
        for v in theme_stats.values_mut() {
            v.0 /= v.2 as f64;
        }

        // Candles last 45 days
        let start_date = NaiveDate::parse_from_str(target_date, "%Y-%m-%d")
            .ok()
            .and_then(|d| d.checked_sub_signed(Duration::days(45)))
            .map(|d| d.format("%Y-%m-%d").to_string())
            .unwrap_or_else(|| "2026-01-01".into());

        let mut candle_filter = col("date")
            .gt_eq(lit(start_date.as_str()))
            .and(col("date").lt_eq(lit(target_date)));
        if let Some(ref cands) = clean_candidates {
            candle_filter =
                candle_filter.and(
                    col("ticker")
                        .str()
                        .slice(lit(0), lit(6))
                        .is_in(lit(Series::new("cands".into(), cands.clone()))),
                );
        }

        let candles = LazyFrame::scan_parquet(
            self.cfg.data_parquet.to_string_lossy().as_ref(),
            ScanArgsParquet::default(),
        )?
        .filter(candle_filter)
        .select([
            col("date"),
            col("ticker"),
            col("open"),
            col("close"),
            col("high"),
            col("low"),
            col("turnover"),
            col("high_close_ratio"),
            col("next_open"),
        ])
        .collect()?;

        if candles.height() == 0 {
            return Ok(vec![]);
        }

        let feat = compute_kline_features(candles)?;
        let target = feat
            .lazy()
            .filter(col("date").eq(lit(target_date)))
            .with_columns([col("ticker").str().slice(lit(0), lit(6)).alias("code")])
            .collect()?;
        if target.height() == 0 {
            return Ok(vec![]);
        }

        // Index candles by code
        let mut candle_by_code: HashMap<String, usize> = HashMap::new();
        let code_col = target.column("code")?.str()?;
        for i in 0..target.height() {
            if let Some(c) = code_col.get(i) {
                candle_by_code.insert(c.to_string(), i);
            }
        }

        #[derive(Clone)]
        struct Row {
            code: String,
            stock_name: String,
            theme_name: String,
            stock_change: f64,
            theme_avg: f64,
            theme_max: f64,
            high_close_ratio: f64,
            close: f64,
            next_open: f64,
            turnover: f64,
            features: [f64; 10],
        }

        let mut rows: Vec<Row> = Vec::new();
        for j in &joined {
            let Some(&ci) = candle_by_code.get(&j.code) else {
                continue;
            };
            let (theme_avg, theme_max, _) = theme_stats
                .get(&(j.date.clone(), j.theme_idx, j.theme_name.clone()))
                .copied()
                .unwrap_or((0.0, 0.0, 0));

            let mut features = [0.0_f64; 10];
            for (fi, name) in DAILY_FEATURE_COLS.iter().enumerate() {
                features[fi] = col_f64(&target, name, ci).unwrap_or(0.0);
            }

            let stock_name = self
                .ticker_map
                .get(&j.code)
                .cloned()
                .unwrap_or_else(|| j.name.clone());

            rows.push(Row {
                code: j.code.clone(),
                stock_name,
                theme_name: j.theme_name.clone(),
                stock_change: j.stock_change,
                theme_avg,
                theme_max,
                high_close_ratio: col_f64(&target, "high_close_ratio", ci).unwrap_or(0.0),
                close: col_f64(&target, "close", ci).unwrap_or(0.0),
                next_open: col_f64(&target, "next_open", ci)
                    .unwrap_or_else(|| col_f64(&target, "close", ci).unwrap_or(0.0)),
                turnover: col_f64(&target, "turnover", ci).unwrap_or(0.0),
                features,
            });
        }

        if rows.is_empty() {
            return Ok(vec![]);
        }

        let n = rows.len();
        let mut x = Array2::<f64>::zeros((n, 10));
        for (i, r) in rows.iter().enumerate() {
            for j in 0..10 {
                let v = r.features[j];
                x[[i, j]] = if v.is_finite() { v } else { 0.0 };
            }
        }

        let p_lgb = self.gbm.predict(&x)?;
        let x_scaled = self.scaler.transform(&x)?;
        let p_torch = self.onnx.predict(&x_scaled)?;

        #[derive(Clone)]
        struct Scored {
            row: Row,
            p_lgb: f64,
            p_torch: f64,
            hybrid: f64,
        }

        let mut scored: Vec<Scored> = Vec::with_capacity(n);
        for i in 0..n {
            let r = &rows[i];
            let is_leader = r.stock_change >= r.theme_max * 0.85;
            let judal = (if is_leader { 35.0 } else { 0.0 })
                + clip(r.theme_avg, -5.0, 12.0) * 2.5
                + clip(r.stock_change, 2.0, 14.0) * 3.0
                + r.high_close_ratio * 30.0
                - (r.stock_change - 15.0).max(0.0) * 4.0;
            let hybrid = judal + p_lgb[i] * 40.0 + p_torch[i] * 40.0;
            scored.push(Scored {
                row: r.clone(),
                p_lgb: p_lgb[i],
                p_torch: p_torch[i],
                hybrid,
            });
        }

        // Deduplicate per code keeping max hybrid
        scored.sort_by(|a, b| {
            b.hybrid
                .partial_cmp(&a.hybrid)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut best: HashMap<String, Scored> = HashMap::new();
        for s in scored {
            best.entry(s.row.code.clone()).or_insert(s);
        }
        let mut deduped: Vec<Scored> = best.into_values().collect();

        // Filters
        deduped.retain(|s| {
            s.row.turnover >= min_turnover
                && s.row.stock_change < max_stock_change
                && s.p_lgb >= min_p_lgb
                && s.p_torch >= min_p_torch
        });
        if deduped.is_empty() {
            return Ok(vec![]);
        }

        deduped.sort_by(|a, b| {
            b.hybrid
                .partial_cmp(&a.hybrid)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        deduped.truncate(top_k);

        // After top_k: add dart*5 + news*3 bonus to hybrid_score
        let mut results = Vec::new();
        for s in deduped {
            let (dart_cnt, news_cnt) =
                market_context_counts(&self.cfg, &s.row.code, &s.row.stock_name, target_date);
            let aux_bonus = (dart_cnt as f64) * 5.0 + (news_cnt as f64) * 3.0;
            let final_score = s.hybrid + aux_bonus;
            results.push(Pick {
                date: target_date.to_string(),
                code: s.row.code.clone(),
                ticker: s.row.code.clone(),
                stock_name: s.row.stock_name,
                theme_name: s.row.theme_name,
                close_price: s.row.close,
                next_open: s.row.next_open,
                stock_change: s.row.stock_change,
                p_lgb: s.p_lgb,
                p_torch: s.p_torch,
                hybrid_score: final_score,
                news_count: news_cnt,
                dart_count: dart_cnt,
            });
        }
        Ok(results)
    }
}

fn clip(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

fn col_f64(df: &DataFrame, name: &str, idx: usize) -> Option<f64> {
    let c = df.column(name).ok()?;
    if let Ok(a) = c.f64() {
        a.get(idx)
    } else if let Ok(a) = c.cast(&DataType::Float64).ok()?.f64() {
        a.get(idx)
    } else {
        None
    }
}

fn load_ticker_name_map(sector_db: &std::path::Path) -> HashMap<String, String> {
    let mut map = HashMap::new();
    if !sector_db.exists() {
        return map;
    }
    let Ok(conn) = Connection::open(sector_db) else {
        return map;
    };
    let Ok(mut stmt) = conn.prepare("SELECT ticker, name FROM sectors") else {
        return map;
    };
    let Ok(rows) = stmt.query_map([], |r| {
        Ok((zfill6(&r.get::<_, String>(0)?), r.get::<_, String>(1)?))
    }) else {
        return map;
    };
    for row in rows.flatten() {
        map.insert(row.0, row.1);
    }
    map
}

/// DART + Meilisearch counts — same logic as `marketmosaic_integrator.py`.
fn market_context_counts(
    cfg: &Config,
    _ticker: &str,
    stock_name: &str,
    target_date: &str,
) -> (usize, usize) {
    let dart_cnt = get_dart_count(cfg, stock_name, target_date);
    let news_cnt = get_news_count(cfg, stock_name, target_date, 2);
    (dart_cnt, news_cnt)
}

fn get_dart_count(cfg: &Config, stock_name: &str, target_date: &str) -> usize {
    let clean_date = target_date.replace('-', "");
    if !cfg.dart_db.exists() {
        return 0;
    }
    let Ok(conn) = Connection::open(&cfg.dart_db) else {
        return 0;
    };
    conn.query_row(
        "SELECT COUNT(*) FROM filings WHERE corp_name = ? AND rcept_dt = ?",
        rusqlite::params![stock_name, clean_date],
        |r| r.get::<_, i64>(0),
    )
    .unwrap_or(0) as usize
}

fn get_news_count(cfg: &Config, stock_name: &str, target_date: &str, limit: usize) -> usize {
    let url = format!("{}/indexes/articles/search", cfg.meili_url);
    let cutoff = format!("{target_date} 15:30:00");
    let client = match reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
    {
        Ok(c) => c,
        Err(_) => return 0,
    };
    let payload = serde_json::json!({ "q": stock_name, "limit": 20 });
    let resp = match client
        .post(&url)
        .header("Authorization", format!("Bearer {}", cfg.meili_key))
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
    {
        Ok(r) if r.status().is_success() => r,
        Ok(_) | Err(_) => return 0,
    };
    let Ok(body) = resp.json::<serde_json::Value>() else {
        return 0;
    };
    let Some(hits) = body.get("hits").and_then(|h| h.as_array()) else {
        return 0;
    };
    let valid: Vec<&serde_json::Value> = hits
        .iter()
        .filter(|h| {
            let pub_at = h.get("published_at").and_then(|v| v.as_str()).unwrap_or("");
            pub_at.contains(target_date) && pub_at <= cutoff.as_str()
        })
        .collect();
    let title_matches: Vec<_> = valid
        .iter()
        .filter(|h| {
            h.get("title")
                .and_then(|v| v.as_str())
                .map(|t| t.contains(stock_name))
                .unwrap_or(false)
        })
        .collect();
    let chosen = if !title_matches.is_empty() {
        title_matches
    } else {
        valid.iter().collect()
    };
    chosen.len().min(limit)
}
