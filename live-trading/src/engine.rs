//! Paper trading engine — mirrors `paper_engine.py` schema & semantics.

use std::path::PathBuf;

use anyhow::{Context, Result};
use chrono::Local;
use polars::prelude::*;
use rusqlite::{params, Connection};
use serde_json::json;
use tracing::info;

use crate::broker::LiveBroker;
use crate::candles::ensure_today_updated;
use crate::condition::KiwoomConditionManager;
use crate::config::{Config, SEED_CAPITAL};
use crate::scoring::{OvernightScorer, Pick};
use crate::telegram::{
    send_market_close_buy_alert, send_market_open_sell_alert, send_ops_error_alert,
    send_parity_check_alert,
};

pub struct PaperTradingEngine {
    pub cfg: Config,
    pub db_path: PathBuf,
    pub scorer: OvernightScorer,
    pub condition_manager: KiwoomConditionManager,
    pub dry_run: bool,
}

impl PaperTradingEngine {
    pub fn new(cfg: Config, dry_run: bool) -> Result<Self> {
        let db_path = cfg.paper_db.clone();
        if let Some(parent) = db_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let engine = Self {
            scorer: OvernightScorer::new(cfg.clone())?,
            condition_manager: KiwoomConditionManager::new(cfg.clone(), "종가베팅"),
            cfg,
            db_path,
            dry_run,
        };
        engine.init_database()?;
        Ok(engine)
    }

    fn conn(&self) -> Result<Connection> {
        Connection::open(&self.db_path).with_context(|| format!("open {}", self.db_path.display()))
    }

    fn init_database(&self) -> Result<()> {
        let conn = self.conn()?;
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                stock_name TEXT NOT NULL,
                theme_name TEXT,
                buy_price REAL NOT NULL,
                buy_qty INTEGER NOT NULL,
                buy_amount REAL NOT NULL,
                sell_price REAL,
                sell_amount REAL,
                pnl_krw REAL,
                pnl_pct REAL,
                status TEXT NOT NULL,
                open_time TEXT NOT NULL,
                close_time TEXT,
                hybrid_score REAL,
                p_lgb REAL,
                p_torch REAL
            );
            CREATE TABLE IF NOT EXISTS paper_account (
                date TEXT PRIMARY KEY,
                cash_balance REAL NOT NULL,
                invested_amount REAL NOT NULL,
                total_equity REAL NOT NULL,
                daily_pnl_krw REAL DEFAULT 0.0,
                daily_pnl_pct REAL DEFAULT 0.0,
                weekly_pnl_pct REAL DEFAULT 0.0,
                monthly_pnl_pct REAL DEFAULT 0.0,
                updated_at TEXT NOT NULL
            );
            "#,
        )?;
        let _ = conn.execute("ALTER TABLE paper_trades ADD COLUMN buy_ord_no TEXT", []);
        let _ = conn.execute("ALTER TABLE paper_trades ADD COLUMN sell_ord_no TEXT", []);
        let _ = conn.execute("ALTER TABLE paper_trades ADD COLUMN execution_mode TEXT", []);
        Ok(())
    }

    /// Returns (cash_balance, invested_amount, total_equity).
    pub fn get_latest_account_state(&self) -> Result<(f64, f64, f64)> {
        let conn = self.conn()?;
        let row: Option<(f64, f64, f64)> = conn
            .query_row(
                "SELECT cash_balance, invested_amount, total_equity
                 FROM paper_account ORDER BY date DESC LIMIT 1",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .ok();
        let open_invested: f64 = conn.query_row(
            "SELECT COALESCE(SUM(buy_amount), 0.0) FROM paper_trades WHERE status = 'OPEN'",
            [],
            |r| r.get(0),
        )?;
        if let Some((cash, _, _)) = row {
            if self.cfg.is_live() && !self.dry_run {
                let broker = LiveBroker::from_cfg(&self.cfg)?;
                let live_cash = broker.orderable_cash()?;
                return Ok((live_cash, open_invested, live_cash + open_invested));
            }
            Ok((cash, open_invested, cash + open_invested))
        } else if self.cfg.is_live() && !self.dry_run {
            let broker = LiveBroker::from_cfg(&self.cfg)?;
            let live_cash = broker.orderable_cash()?;
            Ok((live_cash, open_invested, live_cash + open_invested))
        } else {
            Ok((self.cfg.seed_capital, 0.0, self.cfg.seed_capital))
        }
    }

    fn save_account_state(
        &self,
        date_str: &str,
        cash: f64,
        invested: f64,
        daily_pnl: f64,
        daily_pct: f64,
    ) -> Result<()> {
        let total_equity = cash + invested;
        let (weekly_pct, monthly_pct) = self.calculate_cumulative_returns(date_str, total_equity)?;
        let updated_at = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
        if self.dry_run {
            info!(
                "[dry-run] account {date_str}: cash={cash:.0} invested={invested:.0} equity={total_equity:.0}"
            );
            return Ok(());
        }
        let conn = self.conn()?;
        conn.execute(
            "INSERT INTO paper_account (
                date, cash_balance, invested_amount, total_equity,
                daily_pnl_krw, daily_pnl_pct, weekly_pnl_pct, monthly_pnl_pct, updated_at
             ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)
             ON CONFLICT(date) DO UPDATE SET
                cash_balance=excluded.cash_balance,
                invested_amount=excluded.invested_amount,
                total_equity=excluded.total_equity,
                daily_pnl_krw=excluded.daily_pnl_krw,
                daily_pnl_pct=excluded.daily_pnl_pct,
                weekly_pnl_pct=excluded.weekly_pnl_pct,
                monthly_pnl_pct=excluded.monthly_pnl_pct,
                updated_at=excluded.updated_at",
            params![
                date_str,
                cash,
                invested,
                total_equity,
                daily_pnl,
                daily_pct,
                weekly_pct,
                monthly_pct,
                updated_at
            ],
        )?;
        Ok(())
    }

    pub fn calculate_cumulative_returns(
        &self,
        current_date_str: &str,
        current_equity: f64,
    ) -> Result<(f64, f64)> {
        let cur = match chrono::NaiveDate::parse_from_str(current_date_str, "%Y-%m-%d") {
            Ok(d) => d,
            Err(_) => return Ok((0.0, 0.0)),
        };
        let week_ago = (cur - chrono::Duration::days(7))
            .format("%Y-%m-%d")
            .to_string();
        let month_ago = (cur - chrono::Duration::days(30))
            .format("%Y-%m-%d")
            .to_string();
        let conn = self.conn()?;
        let base_weekly: f64 = conn
            .query_row(
                "SELECT total_equity FROM paper_account WHERE date <= ? ORDER BY date DESC LIMIT 1",
                [&week_ago],
                |r| r.get(0),
            )
            .unwrap_or(SEED_CAPITAL);
        let base_monthly: f64 = conn
            .query_row(
                "SELECT total_equity FROM paper_account WHERE date <= ? ORDER BY date DESC LIMIT 1",
                [&month_ago],
                |r| r.get(0),
            )
            .unwrap_or(SEED_CAPITAL);
        let weekly_pct = if base_weekly > 0.0 {
            ((current_equity / base_weekly) - 1.0) * 100.0
        } else {
            0.0
        };
        let monthly_pct = if base_monthly > 0.0 {
            ((current_equity / base_monthly) - 1.0) * 100.0
        } else {
            0.0
        };
        Ok((weekly_pct, monthly_pct))
    }

    pub fn execute_market_close_buy(&mut self, target_date: &str) -> Result<Vec<serde_json::Value>> {
        let (cash, _invested, total_equity) = self.get_latest_account_state()?;

        let existing: i64 = self.conn()?.query_row(
            "SELECT COUNT(*) FROM paper_trades WHERE date = ? AND status = 'OPEN'",
            [target_date],
            |r| r.get(0),
        )?;
        if existing > 0 {
            info!("[{target_date}] Idempotent BUY skip: {existing} OPEN trade(s) already exist.");
            return Ok(vec![]);
        }

        ensure_today_updated(&self.cfg, target_date).map_err(|e| {
            anyhow::anyhow!("candle sync failed for {target_date}: {e}")
        })?;

        let candidate_codes = self
            .condition_manager
            .get_condition_search_codes(target_date)?;
        let picks = self.scorer.get_candidates_for_date(
            target_date,
            self.cfg.top_k,
            2e10,
            29.0,
            0.35,
            0.35,
            Some(&candidate_codes),
        )?;

        if picks.is_empty() {
            info!("[{target_date}] No candidates met scoring threshold. Cash remains 100%.");
            self.save_account_state(target_date, cash, 0.0, 0.0, 0.0)?;
            let _ = send_market_close_buy_alert(
                &self.cfg,
                target_date,
                &[],
                0.0,
                cash,
                total_equity,
                self.dry_run,
            );
            return Ok(vec![]);
        }

        let live = self.cfg.is_live();
        if live && !self.dry_run {
            if let Err(e) = LiveBroker::from_cfg(&self.cfg) {
                let _ = send_ops_error_alert(
                    &self.cfg,
                    target_date,
                    "LIVE 매수 중단 — 계좌/키 미설정",
                    &format!("<code>{e}</code>"),
                    self.dry_run,
                );
                return Ok(vec![]);
            }
        }

        let alloc_per_stock = cash / picks.len() as f64;
        let mut bought = Vec::new();
        let mut total_buy_amount = 0.0;
        let open_time = format!("{target_date} 15:30:00");

        if !self.dry_run {
            let broker = if live {
                Some(LiveBroker::from_cfg(&self.cfg)?)
            } else {
                None
            };
            let conn = self.conn()?;
            for p in &picks {
                let qty = (alloc_per_stock / p.close_price).floor() as i64;
                if qty <= 0 {
                    continue;
                }
                let mut buy_ord_no = String::new();
                if let Some(ref broker) = broker {
                    match broker.market_buy(&p.ticker, qty as i32) {
                        Ok((true, ord_no, _)) => buy_ord_no = ord_no,
                        Ok((false, _, msg)) => {
                            tracing::error!("LIVE BUY rejected {} qty={qty}: {msg}", p.ticker);
                            continue;
                        }
                        Err(e) => {
                            tracing::error!("LIVE BUY error {}: {e}", p.ticker);
                            continue;
                        }
                    }
                }
                let buy_amount = qty as f64 * p.close_price;
                total_buy_amount += buy_amount;
                conn.execute(
                    "INSERT INTO paper_trades (
                        date, ticker, stock_name, theme_name, buy_price, buy_qty, buy_amount,
                        status, open_time, hybrid_score, p_lgb, p_torch, buy_ord_no, execution_mode
                     ) VALUES (?1,?2,?3,?4,?5,?6,?7,'OPEN',?8,?9,?10,?11,?12,?13)",
                    params![
                        target_date,
                        p.ticker,
                        p.stock_name,
                        p.theme_name,
                        p.close_price,
                        qty,
                        buy_amount,
                        open_time,
                        p.hybrid_score,
                        p.p_lgb,
                        p.p_torch,
                        buy_ord_no,
                        self.cfg.trading_mode,
                    ],
                )?;
                bought.push(json!({
                    "ticker": p.ticker,
                    "stock_name": p.stock_name,
                    "theme_name": p.theme_name,
                    "buy_price": p.close_price,
                    "buy_qty": qty,
                    "buy_amount": buy_amount,
                    "hybrid_score": p.hybrid_score,
                    "p_lgb": p.p_lgb,
                    "p_torch": p.p_torch,
                }));
            }
        } else {
            for p in &picks {
                let qty = (alloc_per_stock / p.close_price).floor() as i64;
                if qty <= 0 {
                    continue;
                }
                let buy_amount = qty as f64 * p.close_price;
                total_buy_amount += buy_amount;
                bought.push(json!({
                    "ticker": p.ticker,
                    "stock_name": p.stock_name,
                    "theme_name": p.theme_name,
                    "buy_price": p.close_price,
                    "buy_qty": qty,
                    "buy_amount": buy_amount,
                    "hybrid_score": p.hybrid_score,
                    "p_lgb": p.p_lgb,
                    "p_torch": p.p_torch,
                }));
            }
            info!("[dry-run] would buy {} names, total={total_buy_amount:.0}", bought.len());
        }

        let new_cash = if live && !self.dry_run {
            match LiveBroker::from_cfg(&self.cfg).and_then(|b| b.orderable_cash()) {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!("live cash refresh failed, using estimate: {e}");
                    cash - total_buy_amount
                }
            }
        } else {
            cash - total_buy_amount
        };
        self.save_account_state(target_date, new_cash, total_buy_amount, 0.0, 0.0)?;
        let _ = send_market_close_buy_alert(
            &self.cfg,
            target_date,
            &bought,
            alloc_per_stock,
            new_cash,
            total_equity,
            self.dry_run,
        );
        Ok(bought)
    }

    pub fn execute_market_open_sell(&mut self, target_date: &str) -> Result<Vec<serde_json::Value>> {
        let conn = self.conn()?;
        let open_trades: Vec<(i64, String, String, f64, i64, f64, String)> = {
            let mut stmt = conn.prepare(
                "SELECT id, ticker, stock_name, buy_price, buy_qty, buy_amount,
                        COALESCE(execution_mode, 'paper')
                 FROM paper_trades WHERE status = 'OPEN'",
            )?;
            let mapped = stmt.query_map([], |r| {
                Ok((
                    r.get(0)?,
                    r.get(1)?,
                    r.get(2)?,
                    r.get(3)?,
                    r.get(4)?,
                    r.get(5)?,
                    r.get(6)?,
                ))
            })?;
            let collected: Result<Vec<_>, _> = mapped.collect();
            collected?
        };

        if open_trades.is_empty() {
            info!("[{target_date}] No OPEN positions to sell.");
            let (cash, invested, equity) = self.get_latest_account_state()?;
            let _ = (cash, invested);
            let (w, m) = self.calculate_cumulative_returns(target_date, equity)?;
            let _ = send_market_open_sell_alert(
                &self.cfg,
                target_date,
                &[],
                0.0,
                0.0,
                w,
                m,
                equity,
                self.dry_run,
            );
            return Ok(vec![]);
        }

        let live = self.cfg.is_live();
        let has_paper = open_trades.iter().any(|t| t.6 != "live");
        let has_live = open_trades.iter().any(|t| t.6 == "live");
        if live && has_paper {
            let detail = format!("{open_trades:?}");
            let _ = send_ops_error_alert(
                &self.cfg,
                target_date,
                "LIVE 매도 중단 — PAPER OPEN 잔존",
                &format!(
                    "실주문을 내면 계좌에 없는 종목을 팔 수 있습니다. TRADING_MODE=paper 로 청산하세요.\n<code>{detail}</code>"
                ),
                self.dry_run,
            );
            return Ok(vec![]);
        }
        if !live && has_live {
            let detail = format!("{open_trades:?}");
            let _ = send_ops_error_alert(
                &self.cfg,
                target_date,
                "PAPER 매도 중단 — LIVE OPEN 잔존",
                &format!(
                    "DB만 닫으면 키움 잔고와 어긋납니다. TRADING_MODE=live 로 청산하세요.\n<code>{detail}</code>"
                ),
                self.dry_run,
            );
            return Ok(vec![]);
        }

        let pre_equity = self.get_latest_account_state().map(|s| s.2).unwrap_or(0.0);

        let open_candles = LazyFrame::scan_parquet(
            self.cfg.data_parquet.to_string_lossy().as_ref(),
            ScanArgsParquet::default(),
        )?
        .filter(col("date").eq(lit(target_date)))
        .select([col("ticker"), col("open")])
        .collect()?;

        let mut open_price_map = std::collections::HashMap::new();
        let tickers = open_candles.column("ticker")?.str()?;
        let opens = open_candles.column("open")?.f64()?;
        for i in 0..open_candles.height() {
            if let (Some(t), Some(o)) = (tickers.get(i), opens.get(i)) {
                let code = crate::condition::zfill6(t.split('.').next().unwrap_or(t));
                open_price_map.insert(code, o);
            }
        }

        let broker = if live && !self.dry_run {
            Some(LiveBroker::from_cfg(&self.cfg)?)
        } else {
            None
        };

        let mut closed = Vec::new();
        let mut total_pnl_krw = 0.0;
        let mut total_returned_cash = 0.0;
        let close_time = format!("{target_date} 09:00:00");
        let fee = self.cfg.fee_rate;

        for (trade_id, ticker, name, buy_price, qty, buy_amount, _mode) in &open_trades {
            let Some(&sell_price) = open_price_map.get(ticker) else {
                anyhow::bail!(
                    "[{target_date}] Missing T+1 open for {ticker}. Refusing fee-only flatten at buy_price."
                );
            };
            if sell_price <= 0.0 {
                anyhow::bail!("[{target_date}] Invalid open {sell_price} for {ticker}");
            }
            let mut sell_qty = *qty;
            let mut sell_ord_no = String::new();
            if let Some(ref broker) = broker {
                let held = broker.holding_qty(ticker).unwrap_or(0);
                if held <= 0 {
                    tracing::error!("LIVE SELL skip {ticker}: no broker holding");
                    continue;
                }
                sell_qty = sell_qty.min(held);
                match broker.market_sell(ticker, sell_qty as i32) {
                    Ok((true, ord_no, _)) => sell_ord_no = ord_no,
                    Ok((false, _, msg)) => {
                        tracing::error!("LIVE SELL rejected {ticker}: {msg}");
                        continue;
                    }
                    Err(e) => {
                        tracing::error!("LIVE SELL error {ticker}: {e}");
                        continue;
                    }
                }
            }
            let gross = sell_qty as f64 * sell_price;
            let net = gross * (1.0 - fee);
            let pnl_krw = net - buy_amount;
            let pnl_pct = if *buy_amount > 0.0 {
                (pnl_krw / buy_amount) * 100.0
            } else {
                0.0
            };
            total_pnl_krw += pnl_krw;
            total_returned_cash += net;

            if !self.dry_run {
                conn.execute(
                    "UPDATE paper_trades SET
                        sell_price=?, sell_amount=?, pnl_krw=?, pnl_pct=?,
                        status='CLOSED', close_time=?, sell_ord_no=?
                     WHERE id=?",
                    params![sell_price, net, pnl_krw, pnl_pct, close_time, sell_ord_no, trade_id],
                )?;
            }

            closed.push(json!({
                "ticker": ticker,
                "stock_name": name,
                "buy_price": buy_price,
                "sell_price": sell_price,
                "buy_qty": sell_qty,
                "pnl_krw": pnl_krw,
                "pnl_pct": pnl_pct,
            }));
        }

        if live && !self.dry_run && closed.is_empty() {
            let _ = send_ops_error_alert(
                &self.cfg,
                target_date,
                "LIVE 매도 전부 실패 — OPEN 유지",
                "키움 매도가 거부되어 DB를 닫지 않았습니다.",
                self.dry_run,
            );
            return Ok(vec![]);
        }

        let remaining_open: f64 = conn.query_row(
            "SELECT COALESCE(SUM(buy_amount), 0.0) FROM paper_trades WHERE status = 'OPEN'",
            [],
            |r| r.get(0),
        )?;
        let leftover: f64 = conn
            .query_row(
                "SELECT cash_balance FROM paper_account ORDER BY date DESC LIMIT 1",
                [],
                |r| r.get(0),
            )
            .unwrap_or(self.cfg.seed_capital);
        let new_cash = if live && !self.dry_run {
            match LiveBroker::from_cfg(&self.cfg).and_then(|b| b.orderable_cash()) {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!("live cash refresh failed, using estimate: {e}");
                    leftover + total_returned_cash
                }
            }
        } else {
            leftover + total_returned_cash
        };
        let new_equity = new_cash + remaining_open;
        let daily_pct = if pre_equity > 0.0 {
            (total_pnl_krw / pre_equity) * 100.0
        } else {
            0.0
        };
        self.save_account_state(target_date, new_cash, remaining_open, total_pnl_krw, daily_pct)?;
        let (weekly_pct, monthly_pct) = self.calculate_cumulative_returns(target_date, new_equity)?;
        let _ = send_market_open_sell_alert(
            &self.cfg,
            target_date,
            &closed,
            total_pnl_krw,
            daily_pct,
            weekly_pct,
            monthly_pct,
            new_equity,
            self.dry_run,
        );
        Ok(closed)
    }

    pub fn run_post_market_parity_check(&mut self, target_date: &str) -> Result<bool> {
        let conn = self.conn()?;
        let like = format!("{target_date}%");
        let mut stmt =
            conn.prepare("SELECT ticker FROM paper_trades WHERE date = ? AND open_time LIKE ?")?;
        let mut paper: Vec<String> = stmt
            .query_map(params![target_date, like], |r| r.get(0))?
            .filter_map(|r| r.ok())
            .collect();
        paper.sort();
        paper.dedup();

        let candidate_codes = self
            .condition_manager
            .get_condition_search_codes(target_date)?;
        let picks = self.scorer.get_candidates_for_date(
            target_date,
            self.cfg.top_k,
            2e10,
            29.0,
            0.35,
            0.35,
            Some(&candidate_codes),
        )?;
        let mut backtest: Vec<String> = picks.into_iter().map(|p| p.ticker).collect();
        backtest.sort();

        let is_matched = paper == backtest;
        let details = if is_matched {
            format!(
                "Paper Buy: {paper:?} | Backtest Buy: {backtest:?}. 100% Signal & Parity Match!"
            )
        } else {
            format!("Parity Mismatch! Paper: {paper:?} vs Backtest: {backtest:?}")
        };
        info!("[{target_date} Parity Verification] Matched: {is_matched} | {details}");
        let _ = send_parity_check_alert(
            &self.cfg,
            target_date,
            is_matched,
            &paper,
            &backtest,
            &details,
            self.dry_run,
        );
        Ok(is_matched)
    }

    /// Standalone analysis helper used by `analyze` binary.
    pub fn analyze(
        &mut self,
        target_date: &str,
        top_k: usize,
        ensure_candles: bool,
    ) -> Result<(Vec<String>, Vec<Pick>)> {
        if ensure_candles {
            ensure_today_updated(&self.cfg, target_date)?;
        }
        let codes = self
            .condition_manager
            .get_condition_search_codes(target_date)?;
        let picks = self.scorer.get_candidates_for_date(
            target_date,
            top_k,
            2e10,
            29.0,
            0.35,
            0.35,
            Some(&codes),
        )?;
        Ok((codes, picks))
    }
}
