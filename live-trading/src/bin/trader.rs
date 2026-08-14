//! Paper trader daemon / one-shot runner.

use anyhow::Result;
use chrono::Utc;
use clap::Parser;
use kr_stock_live::config::Config;
use kr_stock_live::engine::PaperTradingEngine;
use kr_stock_live::scheduler::{kst_offset, run_once, run_scheduler};
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "trader")]
struct Args {
    /// When true (default), skip DB writes / Telegram side-effects where marked dry-run
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    dry_run: bool,

    /// Run a single sell/buy/parity cycle then exit
    #[arg(long, default_value_t = false)]
    once: bool,

    /// Date for --once (default: today KST)
    #[arg(long)]
    date: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    let args = Args::parse();
    let cfg = Config::load()?;
    tracing::info!(
        "trader start execution={} dry_run={}",
        cfg.trading_mode,
        args.dry_run
    );
    let mut engine = PaperTradingEngine::new(cfg, args.dry_run)?;

    if args.once {
        let date = args.date.unwrap_or_else(|| {
            Utc::now()
                .with_timezone(&kst_offset())
                .format("%Y-%m-%d")
                .to_string()
        });
        run_once(&mut engine, &date)?;
    } else {
        run_scheduler(engine).await;
    }
    Ok(())
}
