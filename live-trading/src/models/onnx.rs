//! PyTorch MLP via ONNX Runtime (`ort`).

use std::path::Path;

use anyhow::{Context, Result};
use ndarray::Array2;
use ort::session::Session;
use ort::{inputs, value::TensorRef};

pub struct OnnxModel {
    session: Session,
}

impl OnnxModel {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let session = Session::builder()
            .context("ort Session::builder")?
            .commit_from_file(path)
            .with_context(|| format!("load onnx {}", path.display()))?;
        Ok(Self { session })
    }

    /// Predict sigmoid outputs shape (batch, 1) → flat Vec.
    pub fn predict(&mut self, x_scaled: &Array2<f64>) -> Result<Vec<f64>> {
        let rows = x_scaled.nrows();
        let cols = x_scaled.ncols();
        let flat_f32: Vec<f32> = x_scaled.iter().map(|&v| v as f32).collect();
        let array = Array2::from_shape_vec((rows, cols), flat_f32).context("onnx input shape")?;
        let input_tensor = TensorRef::from_array_view(array.view()).context("TensorRef")?;

        let outputs = self
            .session
            .run(inputs![input_tensor])
            .context("onnx session.run")?;

        let (_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("extract f32 output")?;
        Ok(data.iter().map(|&v| v as f64).collect())
    }
}
