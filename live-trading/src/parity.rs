//! Golden JSON parity comparison.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::Value;

use crate::scoring::Pick;

#[derive(Debug, Deserialize)]
pub struct GoldenDoc {
    pub date: Option<String>,
    pub codes: Option<Vec<String>>,
    pub picks: Option<Vec<Value>>,
}

#[derive(Debug, Clone)]
pub struct ParityReport {
    pub codes_match: bool,
    pub top3_order_match: bool,
    pub scores_within_tol: bool,
    pub details: Vec<String>,
}

impl ParityReport {
    pub fn ok(&self) -> bool {
        self.codes_match && self.top3_order_match && self.scores_within_tol
    }
}

pub fn compare_golden(
    golden: &Value,
    rust_codes: &[String],
    rust_picks: &[Pick],
    score_tol: f64,
) -> Result<ParityReport> {
    let g_codes: Vec<String> = golden
        .get("codes")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|c| c.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let mut rust_set = rust_codes.to_vec();
    rust_set.sort();
    let mut gold_set = g_codes.clone();
    gold_set.sort();
    let codes_match = rust_set == gold_set;

    let g_picks = golden
        .get("picks")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let gold_top: Vec<String> = g_picks
        .iter()
        .take(3)
        .filter_map(|p| {
            p.get("code")
                .or_else(|| p.get("ticker"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
        .collect();
    let rust_top: Vec<String> = rust_picks.iter().take(3).map(|p| p.code.clone()).collect();
    let top3_order_match = gold_top == rust_top;

    let mut details = Vec::new();
    if !codes_match {
        details.push(format!(
            "codes set mismatch: golden={gold_set:?} rust={rust_set:?}"
        ));
    }
    if !top3_order_match {
        details.push(format!(
            "top3 order mismatch: golden={gold_top:?} rust={rust_top:?}"
        ));
    }

    let mut scores_within_tol = true;
    for (i, gp) in g_picks.iter().take(3).enumerate() {
        let Some(rp) = rust_picks.get(i) else {
            scores_within_tol = false;
            details.push(format!("missing rust pick at rank {}", i + 1));
            continue;
        };
        for key in ["hybrid_score", "p_lgb", "p_torch"] {
            let gv = gp.get(key).and_then(|v| v.as_f64()).unwrap_or(f64::NAN);
            let rv = match key {
                "hybrid_score" => rp.hybrid_score,
                "p_lgb" => rp.p_lgb,
                "p_torch" => rp.p_torch,
                _ => f64::NAN,
            };
            if (gv - rv).abs() > score_tol {
                scores_within_tol = false;
                details.push(format!(
                    "rank{} {key}: golden={gv:.6} rust={rv:.6} tol={score_tol}",
                    i + 1
                ));
            }
        }
    }

    Ok(ParityReport {
        codes_match,
        top3_order_match,
        scores_within_tol,
        details,
    })
}

pub fn load_json_file(path: &std::path::Path) -> Result<Value> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&text).context("parse golden/rust json")
}

pub fn assert_parity_or_bail(report: &ParityReport) -> Result<()> {
    if report.ok() {
        Ok(())
    } else {
        bail!("parity failed: {}", report.details.join("; "))
    }
}
