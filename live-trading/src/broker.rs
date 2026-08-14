//! Blocking wrappers around Kiwoom account + order APIs for the sync engine.

use std::future::Future;

use anyhow::{anyhow, Context, Result};
use tracing::{info, warn};

use crate::config::Config;
use crate::kiwoom::{
    AccountService, HoldingItem, KiwoomAuth, KiwoomClient, OrderApi, OrderResponse,
};

fn block_on<T>(fut: impl Future<Output = T>) -> T {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => tokio::task::block_in_place(|| handle.block_on(fut)),
        Err(_) => tokio::runtime::Runtime::new()
            .expect("tokio runtime")
            .block_on(fut),
    }
}

fn parse_f64(raw: Option<&String>) -> Option<f64> {
    raw.and_then(|s| s.replace(',', "").trim().parse::<f64>().ok())
}

fn parse_i64(raw: Option<&String>) -> i64 {
    raw.and_then(|s| s.replace(',', "").trim().parse::<f64>().ok())
        .map(|v| v as i64)
        .unwrap_or(0)
}

fn zfill6(code: &str) -> String {
    let digits: String = code.chars().filter(|c| c.is_ascii_digit()).collect();
    format!("{digits:0>6}")
}

pub struct LiveBroker {
    auth: KiwoomAuth,
    account: AccountService,
    orders: OrderApi,
    acc_no: String,
}

impl LiveBroker {
    pub fn from_cfg(cfg: &Config) -> Result<Self> {
        if cfg.acc_no.trim().is_empty() {
            anyhow::bail!("TRADING_MODE=live requires ACC_NO");
        }
        let app_key = std::env::var("KIWOOM_APP_KEY")
            .or_else(|_| std::env::var("APP_KEY"))
            .unwrap_or_default();
        let secret = std::env::var("KIWOOM_SECRET_KEY")
            .or_else(|_| std::env::var("SECRET_KEY"))
            .unwrap_or_default();
        if app_key.is_empty() || secret.is_empty() {
            anyhow::bail!("TRADING_MODE=live requires APP_KEY / SECRET_KEY");
        }
        let client = KiwoomClient::from_env();
        let token_path = std::env::var("KIWOOM_TOKEN_PATH")
            .unwrap_or_else(|_| "access_token.txt".into());
        Ok(Self {
            auth: KiwoomAuth::new(client.clone(), app_key, secret).with_token_file(&token_path),
            account: AccountService::new(client.clone()),
            orders: OrderApi::new(client),
            acc_no: cfg.acc_no.clone(),
        })
    }

    fn token(&self) -> Result<String> {
        block_on(self.auth.ensure_token()).map_err(|e| anyhow!("{e}"))
    }

    pub fn orderable_cash(&self) -> Result<f64> {
        let token = self.token()?;
        let deposit = block_on(self.account.get_deposit(&token, &self.acc_no))
            .map_err(|e| anyhow!("kt00001: {e}"))?;
        parse_f64(deposit.orderable_cash.as_ref())
            .or_else(|| parse_f64(deposit.deposit.as_ref()))
            .or_else(|| parse_f64(deposit.withdrawable_cash.as_ref()))
            .context("Could not parse orderable cash from kt00001")
    }

    pub fn holdings(&self) -> Result<Vec<HoldingItem>> {
        let token = self.token()?;
        block_on(self.account.get_holdings(&token, &self.acc_no))
            .map_err(|e| anyhow!("kt00004: {e}"))
    }

    pub fn holding_qty(&self, ticker: &str) -> Result<i64> {
        let want = zfill6(ticker);
        for h in self.holdings()? {
            let code = h.stk_cd.as_deref().unwrap_or("");
            if zfill6(code) == want {
                return Ok(parse_i64(h.rmnd_qty.as_ref()));
            }
        }
        Ok(0)
    }

    fn order_ok(resp: &OrderResponse) -> (bool, String) {
        let ord_no = resp.ord_no.clone().unwrap_or_default();
        let code = resp.return_code.unwrap_or(0);
        ( !ord_no.is_empty() && code == 0, ord_no )
    }

    pub fn market_buy(&self, ticker: &str, qty: i32) -> Result<(bool, String, String)> {
        if qty <= 0 {
            return Ok((false, String::new(), "qty<=0".into()));
        }
        let token = self.token()?;
        let code = zfill6(ticker);
        let resp = block_on(self.orders.market_buy(&token, &code, qty))
            .map_err(|e| anyhow!("kt10000 {code}: {e}"))?;
        let (ok, ord_no) = Self::order_ok(&resp);
        let msg = resp.return_msg.clone().unwrap_or_default();
        if ok {
            info!("LIVE BUY {code} qty={qty} ord_no={ord_no}");
        } else {
            warn!("LIVE BUY rejected {code} qty={qty} code={:?} msg={msg}", resp.return_code);
        }
        Ok((ok, ord_no, msg))
    }

    pub fn market_sell(&self, ticker: &str, qty: i32) -> Result<(bool, String, String)> {
        if qty <= 0 {
            return Ok((false, String::new(), "qty<=0".into()));
        }
        let token = self.token()?;
        let code = zfill6(ticker);
        let resp = block_on(self.orders.market_sell(&token, &code, qty))
            .map_err(|e| anyhow!("kt10001 {code}: {e}"))?;
        let (ok, ord_no) = Self::order_ok(&resp);
        let msg = resp.return_msg.clone().unwrap_or_default();
        if ok {
            info!("LIVE SELL {code} qty={qty} ord_no={ord_no}");
        } else {
            warn!("LIVE SELL rejected {code} qty={qty} code={:?} msg={msg}", resp.return_code);
        }
        Ok((ok, ord_no, msg))
    }
}
