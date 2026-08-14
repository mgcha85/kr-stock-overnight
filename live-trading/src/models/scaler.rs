//! StandardScaler loaded from JSON (exported from sklearn).

use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};
use ndarray::{Array2, Axis};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct StandardScaler {
    pub mean: Vec<f64>,
    pub scale: Vec<f64>,
    #[serde(default)]
    pub feature_names: Vec<String>,
}

impl StandardScaler {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let text = fs::read_to_string(path.as_ref())
            .with_context(|| format!("read scaler {}", path.as_ref().display()))?;
        let s: Self = serde_json::from_str(&text).context("parse kline_scaler.json")?;
        if s.mean.len() != s.scale.len() {
            bail!(
                "scaler mean/scale length mismatch: {} vs {}",
                s.mean.len(),
                s.scale.len()
            );
        }
        Ok(s)
    }

    pub fn n_features(&self) -> usize {
        self.mean.len()
    }

    /// sklearn-compatible: `(X - mean) / scale`
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.mean.len() {
            bail!(
                "feature dim {} != scaler dim {}",
                x.ncols(),
                self.mean.len()
            );
        }
        let mut out = x.clone();
        for (mut row, _) in out.axis_iter_mut(Axis(0)).zip(0..) {
            for (j, v) in row.iter_mut().enumerate() {
                *v = (*v - self.mean[j]) / self.scale[j];
            }
        }
        Ok(out)
    }
}
