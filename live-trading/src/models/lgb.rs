//! LightGBM text-model loader via `lightgbm3` (native Booster, matches Python).

use std::path::Path;

use anyhow::{Context, Result};
use lightgbm3::Booster;
use ndarray::Array2;

pub struct LgbModel {
    booster: Booster,
    n_features: i32,
}

impl LgbModel {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let booster = Booster::from_file(path.to_str().context("lgb path utf-8")?)
            .map_err(|e| anyhow::anyhow!("lightgbm3 load {}: {e}", path.display()))?;
        let n_features = booster.num_features();
        Ok(Self {
            booster,
            n_features,
        })
    }

    pub fn n_features(&self) -> usize {
        self.n_features as usize
    }

    /// Row-major predict; returns sigmoid probabilities for binary objective.
    pub fn predict(&self, x: &Array2<f64>) -> Result<Vec<f64>> {
        if x.ncols() as i32 != self.n_features {
            anyhow::bail!(
                "LGB feature dim {} != model {}",
                x.ncols(),
                self.n_features
            );
        }
        let flat: Vec<f64> = x.iter().copied().collect();
        let pred = self
            .booster
            .predict(&flat, self.n_features, true)
            .map_err(|e| anyhow::anyhow!("lgb predict: {e}"))?;
        Ok(pred)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use serde_json::Value;

    #[test]
    fn roundtrip_within_1e6() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let model_path = root.join("models/lgb_kline_model.txt");
        let vectors = root.join("models/roundtrip_vectors.json");
        if !model_path.exists() || !vectors.exists() {
            eprintln!("skip: models not present");
            return;
        }
        let model = LgbModel::load(&model_path).unwrap();
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&vectors).unwrap()).unwrap();
        let rows = v["X"].as_array().unwrap();
        let expected: Vec<f64> = v["p_lgb"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        let mut data = Array2::<f64>::zeros((rows.len(), 10));
        for (i, row) in rows.iter().enumerate() {
            for (j, cell) in row.as_array().unwrap().iter().enumerate() {
                data[[i, j]] = cell.as_f64().unwrap();
            }
        }
        let pred = model.predict(&data).unwrap();
        assert_eq!(pred.len(), expected.len());
        for (a, b) in pred.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-6);
        }
    }
}
