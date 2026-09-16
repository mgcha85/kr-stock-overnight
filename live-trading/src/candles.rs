//! Candle availability check. Live scoring uses Kiwoom session bars, not FDR.

use anyhow::{bail, Context, Result};
use polars::prelude::*;
use tracing::info;

use crate::config::Config;

/// History parquet must exist. Today's bar is injected from Kiwoom at score time.
pub fn ensure_today_updated(cfg: &Config, date: &str) -> Result<()> {
    if !cfg.data_parquet.exists() {
        bail!(
            "parquet missing at {}. History required for kline features.",
            cfg.data_parquet.display()
        );
    }

    let hist = LazyFrame::scan_parquet(
        cfg.data_parquet.to_string_lossy().as_ref(),
        ScanArgsParquet::default(),
    )?
    .select([len().alias("n")])
    .collect()?
    .column("n")?
    .u32()?
    .get(0)
    .unwrap_or(0);
    if hist == 0 {
        bail!("parquet is empty at {}", cfg.data_parquet.display());
    }

    let today_n = LazyFrame::scan_parquet(
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
    info!(
        "[candles] parquet history ok ({hist} rows); today={date} parquet_rows={today_n} (ignored for live score)"
    );
    Ok(())
}

/// Open from `day_data_full.db` when parquet has no row for the ticker.
pub fn open_from_day_db(db: &std::path::Path, ticker: &str, date: &str) -> Option<f64> {
    if !db.exists() {
        return None;
    }
    let conn = rusqlite::Connection::open(db)
        .with_context(|| format!("open {}", db.display()))
        .ok()?;
    let code = crate::condition::zfill6(ticker);
    for suffix in [".KQ", ".KS"] {
        let table = format!("{code}{suffix}");
        let sql = format!(
            r#"SELECT open FROM "{table}" WHERE date = ? ORDER BY rowid DESC LIMIT 1"#
        );
        if let Ok(px) = conn.query_row(&sql, [date], |r| r.get::<_, f64>(0)) {
            if px > 0.0 {
                return Some(px);
            }
        }
    }
    None
}
