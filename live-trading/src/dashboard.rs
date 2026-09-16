//! Sync overnight book to Backtest Lab 8082 and poll live/paper mode.

use anyhow::Result;
use chrono::Local;
use rusqlite::Connection;
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::{info, warn};

use crate::config::Config;

#[derive(Debug, Deserialize)]
struct ModeResponse {
    trading_mode: Option<String>,
}

pub fn apply_remote_trading_mode(cfg: &mut Config) {
    let url = format!("{}/mode", cfg.overnight_api_url.trim_end_matches('/'));
    let client = match reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            warn!("[dashboard] mode client: {e}");
            return;
        }
    };
    match client.get(&url).send() {
        Ok(resp) if resp.status().is_success() => match resp.json::<ModeResponse>() {
            Ok(body) => {
                if let Some(raw) = body.trading_mode {
                    let mode = match raw.trim().to_lowercase().as_str() {
                        "live" | "on" | "1" | "true" => "live",
                        _ => "paper",
                    };
                    if cfg.trading_mode != mode {
                        info!("[dashboard] trading_mode {} -> {}", cfg.trading_mode, mode);
                    }
                    cfg.trading_mode = mode.to_string();
                }
            }
            Err(e) => warn!("[dashboard] mode json: {e}"),
        },
        Ok(resp) => warn!("[dashboard] mode HTTP {}", resp.status()),
        Err(e) => warn!("[dashboard] mode fetch failed, keeping {}: {e}", cfg.trading_mode),
    }
}

pub fn sync_book(cfg: &Config, db_path: &std::path::Path) -> Result<()> {
    let conn = Connection::open(db_path)?;
    let mut trades = Vec::new();
    {
        let mut stmt = conn.prepare(
            "SELECT id, date, ticker, stock_name, theme_name, buy_price, buy_qty, buy_amount,
                    sell_price, sell_amount, pnl_krw, pnl_pct, status, open_time, close_time,
                    hybrid_score, p_lgb, p_torch, buy_ord_no, sell_ord_no,
                    COALESCE(trading_mode, execution_mode, 'paper')
             FROM paper_trades ORDER BY id",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(json!({
                "id": r.get::<_, i64>(0)?,
                "date": r.get::<_, String>(1)?,
                "ticker": r.get::<_, String>(2)?,
                "stock_name": r.get::<_, String>(3)?,
                "theme_name": r.get::<_, Option<String>>(4)?,
                "buy_price": r.get::<_, f64>(5)?,
                "buy_qty": r.get::<_, i64>(6)?,
                "buy_amount": r.get::<_, f64>(7)?,
                "sell_price": r.get::<_, Option<f64>>(8)?,
                "sell_amount": r.get::<_, Option<f64>>(9)?,
                "pnl_krw": r.get::<_, Option<f64>>(10)?,
                "pnl_pct": r.get::<_, Option<f64>>(11)?,
                "status": r.get::<_, String>(12)?,
                "open_time": r.get::<_, String>(13)?,
                "close_time": r.get::<_, Option<String>>(14)?,
                "hybrid_score": r.get::<_, Option<f64>>(15)?,
                "p_lgb": r.get::<_, Option<f64>>(16)?,
                "p_torch": r.get::<_, Option<f64>>(17)?,
                "buy_ord_no": r.get::<_, Option<String>>(18)?,
                "sell_ord_no": r.get::<_, Option<String>>(19)?,
                "trading_mode": r.get::<_, String>(20)?,
            }))
        })?;
        for row in rows {
            trades.push(row?);
        }
    }
    let account: Option<Value> = conn
        .query_row(
            "SELECT date, cash_balance, invested_amount, total_equity,
                    daily_pnl_krw, daily_pnl_pct, weekly_pnl_pct, monthly_pnl_pct, updated_at
             FROM paper_account ORDER BY date DESC LIMIT 1",
            [],
            |r| {
                Ok(json!({
                    "date": r.get::<_, String>(0)?,
                    "cash_balance": r.get::<_, f64>(1)?,
                    "invested_amount": r.get::<_, f64>(2)?,
                    "total_equity": r.get::<_, f64>(3)?,
                    "daily_pnl_krw": r.get::<_, f64>(4)?,
                    "daily_pnl_pct": r.get::<_, f64>(5)?,
                    "weekly_pnl_pct": r.get::<_, f64>(6)?,
                    "monthly_pnl_pct": r.get::<_, f64>(7)?,
                    "updated_at": r.get::<_, String>(8)?,
                }))
            },
        )
        .ok();

    let payload = json!({
        "trading_mode": cfg.trading_mode,
        "engine_mode": cfg.trading_mode,
        "max_alloc_per_ticker": cfg.max_alloc_per_ticker,
        "seed_capital": cfg.seed_capital,
        "heartbeat_at": Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
        "account": account,
        "trades": trades,
    });
    let url = cfg.overnight_api_url.trim_end_matches('/').to_string();
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()?;
    match client.post(&url).json(&payload).send() {
        Ok(resp) if resp.status().is_success() => {
            info!("[dashboard] synced trades={}", trades.len());
        }
        Ok(resp) => {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            warn!("[dashboard] sync HTTP {status}: {}", body.chars().take(300).collect::<String>());
        }
        Err(e) => warn!("[dashboard] sync failed: {e}"),
    }
    Ok(())
}
