//! Weekday KST scheduler — 09:00 sell / 15:28 buy / 18:00 parity.
//!
//! Jobs run once per calendar day after the scheduled time (catch-up).
//! Empty sell/buy still send Telegram. A 4-minute exclusive window is not used.

use std::path::Path;
use std::time::Duration;

use chrono::{FixedOffset, Timelike, Utc};
use rusqlite::Connection;
use tracing::{error, info};

use crate::calendar::{ensure_holidays, is_session_day};
use crate::dashboard::{apply_remote_trading_mode, sync_book};
use crate::engine::PaperTradingEngine;
use crate::telegram::{send_missed_window_alert, send_scheduler_boot_alert};

/// Asia/Seoul fixed offset UTC+9.
pub fn kst_offset() -> FixedOffset {
    FixedOffset::east_opt(9 * 3600).expect("KST")
}

fn ensure_marks(db: &Path) {
    if let Ok(conn) = Connection::open(db) {
        let _ = conn.execute(
            "CREATE TABLE IF NOT EXISTS scheduler_marks (
                date TEXT NOT NULL,
                job TEXT NOT NULL,
                ran_at TEXT NOT NULL,
                PRIMARY KEY (date, job)
            )",
            [],
        );
    }
}

fn job_done(db: &Path, date: &str, job: &str) -> bool {
    ensure_marks(db);
    Connection::open(db)
        .ok()
        .and_then(|c| {
            c.query_row(
                "SELECT 1 FROM scheduler_marks WHERE date = ?1 AND job = ?2",
                [date, job],
                |_| Ok(()),
            )
            .ok()
        })
        .is_some()
}

fn mark_job(db: &Path, date: &str, job: &str, now: &str) {
    ensure_marks(db);
    if let Ok(conn) = Connection::open(db) {
        let _ = conn.execute(
            "INSERT OR REPLACE INTO scheduler_marks (date, job, ran_at) VALUES (?1, ?2, ?3)",
            [date, job, now],
        );
    }
}

pub async fn run_scheduler(mut engine: PaperTradingEngine) {
    apply_remote_trading_mode(&mut engine.cfg);
    info!("=========================================================================");
    info!("   KRX OVERNIGHT TRADING DAEMON SCHEDULER STARTED (Rust)");
    info!(
        "   Execution: {}  (TRADING_MODE=paper|live)",
        engine.cfg.trading_mode.to_uppercase()
    );
    info!("   Timezone: Asia/Seoul (KST)");
    info!("   Schedule: KRX session days [09:00 SELL | 15:28 BUY | 18:00 PARITY]");
    info!("   Closed: weekends + krx_holidays table (2026 seeded)");
    info!("   Catch-up: sell after 09:00; buy 15:28-16:00; parity after 18:00 (completed daily bars)");
    info!("   Empty reports always Telegram");
    info!(
        "   Max alloc/ticker: {:.0} KRW",
        engine.cfg.max_alloc_per_ticker
    );
    info!("=========================================================================");

    let db = engine.db_path.clone();
    ensure_marks(&db);
    let _ = ensure_holidays(&db);
    let _ = sync_book(&engine.cfg, &engine.db_path);

    let boot = Utc::now()
        .with_timezone(&kst_offset())
        .format("%Y-%m-%d %H:%M:%S")
        .to_string();
    let _ = send_scheduler_boot_alert(&engine.cfg, &boot, engine.dry_run);

    let mut last_beat = Utc::now();

    loop {
        let now = Utc::now().with_timezone(&kst_offset());
        let today = now.format("%Y-%m-%d").to_string();
        let time_str = format!(
            "{:02}:{:02}:{:02}",
            now.hour(),
            now.minute(),
            now.second()
        );
        let stamp = format!("{today} {time_str}");
        let weekday_ok = is_session_day(&db, &today);

        if weekday_ok {
            if time_str.as_str() >= "09:00:00" && !job_done(&db, &today, "sell") {
                info!("[{stamp}] Executing 09:00 Market Open SELL (catch-up ok)...");
                match tokio::task::block_in_place(|| engine.execute_market_open_sell(&today)) {
                    Ok(_) => {
                        mark_job(&db, &today, "sell", &stamp);
                        info!("[{today}] 09:00 Market Open SELL Completed Successfully.");
                    }
                    Err(e) => error!("[{today}] Error during 09:00 SELL: {e:?}"),
                }
            }

            if time_str.as_str() >= "15:28:00"
                && time_str.as_str() < "16:00:00"
                && !job_done(&db, &today, "buy")
            {
                info!("[{stamp}] Executing 15:28 Market Close BUY (until 16:00)...");
                match tokio::task::block_in_place(|| engine.execute_market_close_buy(&today)) {
                    Ok(_) => {
                        mark_job(&db, &today, "buy", &stamp);
                        info!("[{today}] 15:28 Market Close BUY Completed Successfully.");
                    }
                    Err(e) => error!("[{today}] Error during 15:28 BUY: {e:?}"),
                }
            }

            if time_str.as_str() >= "16:00:00" && !job_done(&db, &today, "buy") {
                info!("[{stamp}] BUY window missed; sending 0-fill report, no order.");
                let _ = send_missed_window_alert(
                    &engine.cfg,
                    &today,
                    "15:28 매수 창 놓침 — 매수 0건",
                    "16:00까지 매수가 실행되지 않아 주문하지 않았습니다. 장중 프로세스가 꺼졌거나 창을 지나쳤습니다.",
                    engine.dry_run,
                );
                mark_job(&db, &today, "buy", &stamp);
            }

            if time_str.as_str() >= "18:00:00"
                && time_str.as_str() < "19:30:00"
                && !job_done(&db, &today, "parity")
            {
                info!("[{stamp}] Executing 18:00 Post-Market Parity Verification...");
                match tokio::task::block_in_place(|| engine.run_post_market_parity_check(&today)) {
                    Ok(_) => {
                        mark_job(&db, &today, "parity", &stamp);
                        info!("[{today}] 18:00 Parity Verification Completed Successfully.");
                    }
                    Err(e) => error!("[{today}] Error during 18:00 Parity: {e:?}"),
                }
            }
        }

        if Utc::now().signed_duration_since(last_beat) >= chrono::Duration::minutes(30) {
            info!(
                "[heartbeat] {stamp} mode={} sell_done={} buy_done={}",
                engine.cfg.trading_mode,
                job_done(&db, &today, "sell"),
                job_done(&db, &today, "buy")
            );
            last_beat = Utc::now();
        }

        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                info!("Received shutdown signal (Ctrl+C / SIGINT / SIGTERM). Exiting scheduler...");
                break;
            }
            _ = tokio::time::sleep(Duration::from_secs(15)) => {}
        }
    }
}

/// Run one full cycle for `date` (sell → buy → parity) — useful for `--once`.
pub fn run_once(engine: &mut PaperTradingEngine, date: &str) -> anyhow::Result<()> {
    info!("[once] sell {date}");
    let _ = engine.execute_market_open_sell(date)?;
    info!("[once] buy {date}");
    let _ = engine.execute_market_close_buy(date)?;
    info!("[once] parity {date}");
    let _ = engine.run_post_market_parity_check(date)?;
    Ok(())
}
