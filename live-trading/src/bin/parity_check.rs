//! Compare Rust analysis output against a Python golden JSON.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Result;
use clap::Parser;
use kr_stock_live::config::Config;
use kr_stock_live::engine::PaperTradingEngine;
use kr_stock_live::parity::{assert_parity_or_bail, compare_golden, load_json_file};
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "parity_check")]
struct Args {
    /// Golden JSON from Python analysis
    #[arg(long)]
    golden: PathBuf,

    /// Existing Rust analysis JSON (if omitted, run analysis for golden.date)
    #[arg(long)]
    rust_out: Option<PathBuf>,

    /// Score absolute tolerance
    #[arg(long, default_value_t = 1e-4)]
    tol: f64,

    #[arg(long, default_value_t = 3)]
    top_k: usize,
}

fn main() -> Result<ExitCode> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    let args = Args::parse();
    let golden = load_json_file(&args.golden)?;

    let (codes, picks) = if let Some(ref path) = args.rust_out {
        let doc = load_json_file(path)?;
        let codes: Vec<String> = doc["codes"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect();
        let picks: Vec<kr_stock_live::Pick> =
            serde_json::from_value(doc["picks"].clone()).unwrap_or_default();
        (codes, picks)
    } else {
        let date = golden["date"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("golden JSON missing date; provide --rust-out"))?;
        let cfg = Config::load()?;
        let mut engine = PaperTradingEngine::new(cfg, true)?;
        engine.analyze(date, args.top_k, false)?
    };

    let report = compare_golden(&golden, &codes, &picks, args.tol)?;
    println!(
        "codes_match={} top3_order_match={} scores_within_tol={}",
        report.codes_match, report.top3_order_match, report.scores_within_tol
    );
    for d in &report.details {
        println!(" - {d}");
    }
    match assert_parity_or_bail(&report) {
        Ok(()) => {
            println!("PARITY OK");
            Ok(ExitCode::SUCCESS)
        }
        Err(e) => {
            eprintln!("{e}");
            Ok(ExitCode::FAILURE)
        }
    }
}
