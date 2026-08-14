//! Candle availability check for parity — FDR sync remains Python-side.

use anyhow::{bail, Context, Result};
use polars::prelude::*;
use tracing::{info, warn};

use crate::config::Config;

/// Ensure parquet contains rows for `date`.
///
/// FDR/KRX candle sync is intentionally Python
/// (`scripts/fetch_today_kr_candles.py` → rebuilds `kr_kline_processed.parquet`).
/// Rust validates presence for parity; optionally probes `day_data_full.db`.
pub fn ensure_today_updated(cfg: &Config, date: &str) -> Result<()> {
    if !cfg.data_parquet.exists() {
        bail!(
            "parquet missing at {}. Run: PYTHONPATH=src python scripts/fetch_today_kr_candles.py --date {}",
            cfg.data_parquet.display(),
            date
        );
    }

    let count = LazyFrame::scan_parquet(
        cfg.data_parquet.to_string_lossy().as_ref(),
        ScanArgsParquet::default(),
    )?
    .filter(col("date").eq(lit(date)))
    .select([len().alias("n")])
    .collect()?
    .column("n")?
    .u32()?
    .get(0)
    .unwrap_or(0);

    if count > 0 {
        info!("[candles] parquet has {count} rows for {date}");
        return Ok(());
    }

    // Fallback probe: day_data_full.db may have today's bars even if parquet is stale.
    if cfg.day_data_db.exists() {
        match probe_day_data_db(&cfg.day_data_db, date) {
            Ok(n) if n > 0 => {
                warn!(
                    "[candles] parquet missing {date} but day_data_full.db has ~{n} tables with that date. \
                     Rebuild parquet via: PYTHONPATH=src python scripts/fetch_today_kr_candles.py --date {date}"
                );
                bail!(
                    "parquet missing rows for {date} (day_data_full.db appears populated). \
                     Run Python fetch script to rebuild kr_kline_processed.parquet."
                );
            }
            Ok(_) => {}
            Err(e) => warn!("[candles] day_data_full.db probe failed: {e}"),
        }
    }

    bail!(
        "No candle rows for {date} in {}. \
         FDR sync is Python-only — run: PYTHONPATH=src python scripts/fetch_today_kr_candles.py --date {date}",
        cfg.data_parquet.display()
    );
}

fn probe_day_data_db(path: &std::path::Path, date: &str) -> Result<usize> {
    let conn = rusqlite::Connection::open(path)
        .with_context(|| format!("open {}", path.display()))?;
    let mut stmt = conn.prepare(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%.%' LIMIT 50",
    )?;
    let tables: Vec<String> = stmt
        .query_map([], |r| r.get(0))?
        .filter_map(|r| r.ok())
        .collect();
    let mut hits = 0usize;
    for t in tables {
        let sql = format!(r#"SELECT 1 FROM "{t}" WHERE date = ? LIMIT 1"#);
        let found: Result<i64, _> = conn.query_row(&sql, [date], |r| r.get(0));
        if found.is_ok() {
            hits += 1;
        }
    }
    Ok(hits)
}
