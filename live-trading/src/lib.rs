//! Rust 1:1 port of the Python `kr_stock` overnight paper-trading pipeline.

pub mod broker;
pub mod calendar;
pub mod candles;
pub mod condition;
pub mod config;
pub mod dashboard;
pub mod engine;
pub mod ext_quotes;
pub mod features;
pub mod kiwoom;
pub mod models;
pub mod parity;
pub mod quotes;
pub mod scheduler;
pub mod scoring;
pub mod sizing;
pub mod telegram;

pub use config::Config;
pub use engine::PaperTradingEngine;
pub use scoring::{OvernightScorer, Pick};
