//! CLI: overnight analysis for a target date.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use kr_stock_live::config::Config;
use kr_stock_live::engine::PaperTradingEngine;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "analyze", about = "Kiwoom 종가베팅 overnight analysis")]
struct Args {
    /// Target date YYYY-MM-DD
    #[arg(long)]
    date: String,

    /// Write JSON result to this path
    #[arg(long)]
    out: Option<PathBuf>,

    /// Verify parquet has candles for date
    #[arg(long, default_value_t = false)]
    ensure_candles: bool,

    /// Top-K picks
    #[arg(long, default_value_t = 3)]
    top_k: usize,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    let args = Args::parse();
    let cfg = Config::load()?;
    let mut engine = PaperTradingEngine::new(cfg, true)?;
    let (codes, picks) = engine.analyze(&args.date, args.top_k, args.ensure_candles)?;

    println!("========================================================================");
    println!(" KIWOOM CONDITION SEARCH -> OVERNIGHT ANALYSIS ({})", args.date);
    println!("========================================================================");
    println!("[Step 1] Condition candidates ({}): {:?}", codes.len(), codes);
    println!("[Step 2] Top-{} picks:", args.top_k);
    if picks.is_empty() {
        println!("No stocks met P_LGB/P_MLP >= 0.35 filters.");
    } else {
        for (i, p) in picks.iter().enumerate() {
            println!(
                "Rank #{}: [{}] {} ({})",
                i + 1,
                p.code,
                p.stock_name,
                p.theme_name
            );
            println!(
                "  - Hybrid Score: {:.2} | P(LGB)={:.4} | P(MLP)={:.4}",
                p.hybrid_score, p.p_lgb, p.p_torch
            );
            println!(
                "  - Close={:.0} | Change={:.2}% | News={} | DART={}",
                p.close_price, p.stock_change, p.news_count, p.dart_count
            );
        }
    }

    let doc = serde_json::json!({
        "date": args.date,
        "codes": codes,
        "picks": picks,
    });
    println!("{}", serde_json::to_string_pretty(&doc)?);
    if let Some(out) = args.out {
        if let Some(parent) = out.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(&out, serde_json::to_string_pretty(&doc)?)?;
        println!("Wrote {}", out.display());
    }
    Ok(())
}
