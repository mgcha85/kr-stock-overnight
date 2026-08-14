//! Rust 1:1 port of the Python `kr_stock` overnight paper-trading pipeline.

pub mod broker;
pub mod candles;
pub mod condition;
pub mod config;
pub mod engine;
pub mod features;
pub mod kiwoom;
pub mod models;
pub mod parity;
pub mod scheduler;
pub mod scoring;
pub mod telegram;

pub use config::Config;
pub use engine::PaperTradingEngine;
pub use scoring::{OvernightScorer, Pick};
