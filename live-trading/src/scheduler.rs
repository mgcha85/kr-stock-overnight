//! Weekday KST scheduler — mirrors `scheduler.py` (09:00 / 15:20 / 15:25).

use std::time::Duration;

use chrono::{Datelike, FixedOffset, Timelike, Utc, Weekday};
use tracing::{error, info};

use crate::engine::PaperTradingEngine;

/// Asia/Seoul fixed offset UTC+9.
pub fn kst_offset() -> FixedOffset {
    FixedOffset::east_opt(9 * 3600).expect("KST")
}

pub async fn run_scheduler(mut engine: PaperTradingEngine) {
    info!("=========================================================================");
    info!("   KRX OVERNIGHT TRADING DAEMON SCHEDULER STARTED (Rust)");
    info!("   Execution: {}  (TRADING_MODE=paper|live)", engine.cfg.trading_mode.to_uppercase());
    info!("   Timezone: Asia/Seoul (KST)");
    info!("   Schedule: Weekdays Mon-Fri [09:00 SELL | 15:20 BUY | 15:25 PARITY]");
    info!("=========================================================================");

    let mut sold_date = String::new();
    let mut bought_date = String::new();
    let mut parity_date = String::new();

    loop {
        let now = Utc::now().with_timezone(&kst_offset());
        let weekday = now.weekday();
        let today = now.format("%Y-%m-%d").to_string();
        let time_str = format!(
            "{:02}:{:02}:{:02}",
            now.hour(),
            now.minute(),
            now.second()
        );

        if matches!(
            weekday,
            Weekday::Mon | Weekday::Tue | Weekday::Wed | Weekday::Thu | Weekday::Fri
        ) {
            if time_str.as_str() >= "09:00:00"
                && time_str.as_str() <= "09:05:00"
                && sold_date != today
            {
                info!("[{today} {time_str}] Executing 09:00 Market Open SELL...");
                match engine.execute_market_open_sell(&today) {
                    Ok(_) => {
                        sold_date = today.clone();
                        info!("[{today}] 09:00 Market Open SELL Completed Successfully.");
                    }
                    Err(e) => error!("[{today}] Error during 09:00 SELL: {e:?}"),
                }
            }

            if time_str.as_str() >= "15:20:00"
                && time_str.as_str() <= "15:24:00"
                && bought_date != today
            {
                info!("[{today} {time_str}] Executing 15:20 Market Close BUY...");
                match engine.execute_market_close_buy(&today) {
                    Ok(_) => {
                        bought_date = today.clone();
                        info!("[{today}] 15:20 Market Close BUY Completed Successfully.");
                    }
                    Err(e) => error!("[{today}] Error during 15:20 BUY: {e:?}"),
                }
            }

            if time_str.as_str() >= "15:25:00"
                && time_str.as_str() <= "15:29:00"
                && parity_date != today
            {
                info!("[{today} {time_str}] Executing 15:25 Post-Market Parity Verification...");
                match engine.run_post_market_parity_check(&today) {
                    Ok(_) => {
                        parity_date = today.clone();
                        info!("[{today}] 15:25 Parity Verification Completed Successfully.");
                    }
                    Err(e) => error!("[{today}] Error during 15:25 Parity: {e:?}"),
                }
            }
        }

        tokio::time::sleep(Duration::from_secs(15)).await;
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
