#!/usr/bin/env python3
"""Export Python paper-trading models to Rust-loadable artifacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "research" / "models"
OUT_DIR = ROOT / "live-trading" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))
from kr_stock.inference import DeepMTFOvernightNet, DAILY_FEATURE_COLS  # noqa: E402


def export_scaler():
    scaler = joblib.load(MODEL_DIR / "kline_scaler.joblib")
    payload = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "feature_names": list(DAILY_FEATURE_COLS),
    }
    path = OUT_DIR / "kline_scaler.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {path}")
    return scaler


def export_lgb():
    gbm = joblib.load(MODEL_DIR / "lgb_kline_model.joblib")
    # sklearn LGBMRegressor/Classifier or raw booster
    booster = getattr(gbm, "booster_", None) or getattr(gbm, "_Booster", None) or gbm
    txt_path = OUT_DIR / "lgb_kline_model.txt"
    if hasattr(booster, "save_model"):
        booster.save_model(str(txt_path))
    elif hasattr(gbm, "booster_") and hasattr(gbm.booster_, "save_model"):
        gbm.booster_.save_model(str(txt_path))
    else:
        # fallback: dump via lightgbm
        import lightgbm as lgb

        if isinstance(gbm, lgb.Booster):
            gbm.save_model(str(txt_path))
        else:
            raise TypeError(f"Unsupported LGB type: {type(gbm)}")
    print(f"Wrote {txt_path} ({type(gbm)})")
    return gbm


def export_pytorch_onnx():
    model = DeepMTFOvernightNet(input_dim=10)
    state = torch.load(MODEL_DIR / "pytorch_kline_model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    dummy = torch.randn(1, 10, dtype=torch.float32)
    onnx_path = OUT_DIR / "pytorch_kline_model.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print(f"Wrote {onnx_path}")
    return model


def roundtrip_check(scaler, gbm, model):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(8, 10)).astype(np.float64)
    p_lgb = np.asarray(gbm.predict(X), dtype=np.float64).reshape(-1)
    Xs = scaler.transform(X).astype(np.float32)
    with torch.no_grad():
        p_torch = model(torch.from_numpy(Xs)).numpy().reshape(-1)
    payload = {
        "X": X.tolist(),
        "p_lgb": p_lgb.tolist(),
        "p_torch": p_torch.tolist(),
        "X_scaled": Xs.tolist(),
    }
    path = OUT_DIR / "roundtrip_vectors.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {path}")


def main():
    scaler = export_scaler()
    gbm = export_lgb()
    model = export_pytorch_onnx()
    roundtrip_check(scaler, gbm, model)
    print("Export complete.")


if __name__ == "__main__":
    main()
