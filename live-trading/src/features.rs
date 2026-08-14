//! Kline feature engineering — identical to Python `inference.compute_kline_features`.

use anyhow::{Context, Result};
use polars::prelude::*;
use polars::series::ops::NullBehavior;

pub const DAILY_FEATURE_COLS: &[&str] = &[
    "high_close_ratio",
    "body_ratio",
    "upper_shadow_ratio",
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "vol_ratio_5d",
    "bb_pct_b",
    "bb_width",
    "rsi_14",
];

/// Computes technical indicator features from candle OHLCV data (per-ticker rolling).
pub fn compute_kline_features(df_candles: DataFrame) -> Result<DataFrame> {
    let lf = df_candles
        .lazy()
        .sort(
            ["ticker", "date"],
            SortMultipleOptions::default().with_nulls_last(true),
        )
        .with_columns([(col("high") - col("low") + lit(1e-5)).alias("_hl_range")])
        .with_columns([
            ((col("close") - col("low")) / col("_hl_range")).alias("high_close_ratio"),
            ((col("close") - col("open")).abs() / col("_hl_range")).alias("body_ratio"),
            ((col("high")
                - when(col("open").gt_eq(col("close")))
                    .then(col("open"))
                    .otherwise(col("close")))
                / col("_hl_range"))
            .alias("upper_shadow_ratio"),
        ])
        .with_columns([
            col("close")
                .pct_change(lit(1))
                .over([col("ticker")])
                .fill_null(0.0)
                .alias("ret_1d"),
            col("close")
                .pct_change(lit(3))
                .over([col("ticker")])
                .fill_null(0.0)
                .alias("ret_3d"),
            col("close")
                .pct_change(lit(5))
                .over([col("ticker")])
                .fill_null(0.0)
                .alias("ret_5d"),
        ])
        .with_columns([col("turnover")
            .rolling_mean(RollingOptionsFixedWindow {
                window_size: 5,
                min_periods: 1,
                weights: None,
                center: false,
                fn_params: None,
            })
            .over([col("ticker")])
            .alias("vol_ma5")])
        .with_columns([
            (col("turnover") / (col("vol_ma5") + lit(1e-5))).alias("vol_ratio_5d"),
            col("close")
                .rolling_mean(RollingOptionsFixedWindow {
                    window_size: 20,
                    min_periods: 1,
                    weights: None,
                    center: false,
                    fn_params: None,
                })
                .over([col("ticker")])
                .alias("ma20"),
            col("close")
                .rolling_std(RollingOptionsFixedWindow {
                    window_size: 20,
                    min_periods: 1,
                    weights: None,
                    center: false,
                    fn_params: None,
                })
                .over([col("ticker")])
                .fill_null(1.0)
                .alias("std20"),
        ])
        .with_columns([
            ((col("close") - (col("ma20") - lit(2.0) * col("std20")))
                / (lit(4.0) * col("std20") + lit(1e-5)))
            .alias("bb_pct_b"),
            ((lit(4.0) * col("std20")) / (col("ma20") + lit(1e-5))).alias("bb_width"),
        ])
        .with_columns([col("close")
            .diff(1, NullBehavior::Ignore)
            .over([col("ticker")])
            .fill_null(0.0)
            .alias("_delta")])
        .with_columns([
            when(col("_delta").gt(lit(0.0)))
                .then(col("_delta"))
                .otherwise(lit(0.0))
                .alias("_gain"),
            when(col("_delta").lt(lit(0.0)))
                .then(-col("_delta"))
                .otherwise(lit(0.0))
                .alias("_loss"),
        ])
        .with_columns([
            col("_gain")
                .rolling_mean(RollingOptionsFixedWindow {
                    window_size: 14,
                    min_periods: 1,
                    weights: None,
                    center: false,
                    fn_params: None,
                })
                .over([col("ticker")])
                .alias("_avg_gain"),
            col("_loss")
                .rolling_mean(RollingOptionsFixedWindow {
                    window_size: 14,
                    min_periods: 1,
                    weights: None,
                    center: false,
                    fn_params: None,
                })
                .over([col("ticker")])
                .alias("_avg_loss"),
        ])
        .with_columns([
            (lit(100.0)
                - (lit(100.0)
                    / (lit(1.0) + (col("_avg_gain") / (col("_avg_loss") + lit(1e-5))))))
            .alias("rsi_14"),
        ]);

    let out = lf
        .collect()
        .context("compute_kline_features collect failed")?;
    Ok(out)
}
