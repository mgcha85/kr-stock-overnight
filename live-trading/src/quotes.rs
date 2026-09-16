//! 15:28 auction overlay: websocket 0H + ka10001 + ka10004 touch.

use std::collections::HashMap;
use std::future::Future;
use std::time::{Duration, Instant};

use anyhow::Result;
use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use tokio_tungstenite::{connect_async, tungstenite::client::IntoClientRequest, tungstenite::protocol::Message};
use tracing::{info, warn};

use crate::config::Config;
use crate::kiwoom::{KiwoomAuth, KiwoomClient, MarketDataService};
use crate::scoring::Pick;

fn block_on<T>(fut: impl Future<Output = T>) -> T {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => tokio::task::block_in_place(|| handle.block_on(fut)),
        Err(_) => tokio::runtime::Runtime::new()
            .expect("tokio runtime")
            .block_on(fut),
    }
}

fn quote_auth(cfg: &Config) -> Result<(KiwoomAuth, MarketDataService)> {
    let app_key = std::env::var("KIWOOM_APP_KEY")
        .or_else(|_| std::env::var("APP_KEY"))
        .unwrap_or_default();
    let secret = std::env::var("KIWOOM_SECRET_KEY")
        .or_else(|_| std::env::var("SECRET_KEY"))
        .unwrap_or_default();
    if app_key.is_empty() || secret.is_empty() {
        anyhow::bail!("auction quote requires APP_KEY / SECRET_KEY");
    }
    let client = KiwoomClient::from_env();
    let token_path = std::env::var("KIWOOM_TOKEN_PATH").unwrap_or_else(|_| {
        cfg.paper_db
            .with_file_name("kiwoom_access_token.json")
            .display()
            .to_string()
    });
    let auth = KiwoomAuth::new(client.clone(), app_key, secret).with_token_file(&token_path);
    Ok((auth, MarketDataService::new(client)))
}

fn parse_num(raw: Option<&Value>) -> f64 {
    let Some(v) = raw else {
        return 0.0;
    };
    let s = v
        .as_str()
        .map(|s| s.to_string())
        .or_else(|| v.as_f64().map(|n| n.to_string()))
        .or_else(|| v.as_i64().map(|n| n.to_string()))
        .unwrap_or_default();
    let s = s.replace(',', "");
    let s = s.trim().trim_start_matches(['+', '-']);
    s.parse::<f64>().unwrap_or(0.0).abs()
}

fn first_num(json: &Value, src: &Value, keys: &[&str]) -> f64 {
    let mut best: f64 = 0.0;
    for k in keys {
        best = best.max(parse_num(src.get(*k))).max(parse_num(json.get(*k)));
        if let Some(out) = json.get("output") {
            if let Some(arr) = out.as_array() {
                for item in arr {
                    best = best.max(parse_num(item.get(*k)));
                }
            } else {
                best = best.max(parse_num(out.get(*k)));
            }
        }
    }
    best
}

fn ingest_0h_row(row: &Value, out: &mut HashMap<String, (f64, f64, &'static str)>) {
    let rtype = row
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if rtype != "0H" && rtype != "주식예상체결" && row.get("values").is_none() {
        return;
    }
    let item = row
        .get("item")
        .or(row.get("9001"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let digits: String = item.chars().filter(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return;
    }
    let code = format!("{digits:0>6}");
    let values = row.get("values").unwrap_or(row);
    let price = parse_num(values.get("10"))
        .max(parse_num(values.get("27")))
        .max(parse_num(values.get("exp_cntr_pric")));
    let qty = parse_num(values.get("15"))
        .max(parse_num(values.get("13")))
        .max(parse_num(values.get("exp_cntr_qty")));
    if price > 0.0 {
        out.insert(code, (price, qty, "0H"));
    }
}

async fn poll_0h(
    token: &str,
    tickers: &[String],
    listen: Duration,
) -> HashMap<String, (f64, f64, &'static str)> {
    let mut out = HashMap::new();
    let ws_url = KiwoomAuth::ws_url();
    let request = match ws_url.as_str().into_client_request() {
        Ok(r) => r,
        Err(e) => {
            warn!("[0H] ws request: {e}");
            return out;
        }
    };
    let (ws_stream, _) = match connect_async(request).await {
        Ok(v) => v,
        Err(e) => {
            warn!("[0H] connect: {e}");
            return out;
        }
    };
    let (mut write, mut read) = ws_stream.split();
    if write
        .send(Message::Text(
            serde_json::json!({"trnm": "LOGIN", "token": token}).to_string(),
        ))
        .await
        .is_err()
    {
        return out;
    }
    let login_deadline = Instant::now() + Duration::from_secs(8);
    let mut logged_in = false;
    while Instant::now() < login_deadline {
        match tokio::time::timeout(Duration::from_secs(2), read.next()).await {
            Ok(Some(Ok(Message::Text(text)))) => {
                let Ok(json) = serde_json::from_str::<Value>(&text) else {
                    continue;
                };
                if json.get("trnm").and_then(|v| v.as_str()) == Some("PING") {
                    let _ = write.send(Message::Text(text)).await;
                    continue;
                }
                if json.get("trnm").and_then(|v| v.as_str()) == Some("LOGIN") {
                    if json.get("return_code").and_then(|v| v.as_i64()).unwrap_or(-1) == 0 {
                        logged_in = true;
                    } else {
                        warn!("[0H] LOGIN failed");
                    }
                    break;
                }
            }
            _ => break,
        }
    }
    if !logged_in {
        warn!("[0H] LOGIN timeout");
        let _ = write.send(Message::Close(None)).await;
        return out;
    }
    let reg = serde_json::json!({
        "trnm": "REG",
        "grp_no": "1",
        "refresh": "1",
        "data": [{"item": tickers, "type": ["0H"]}],
    });
    if write.send(Message::Text(reg.to_string())).await.is_err() {
        return out;
    }
    let until = Instant::now() + listen;
    while Instant::now() < until {
        let remain = until.saturating_duration_since(Instant::now());
        match tokio::time::timeout(remain.max(Duration::from_millis(50)), read.next()).await {
            Ok(Some(Ok(Message::Text(text)))) => {
                let Ok(json) = serde_json::from_str::<Value>(&text) else {
                    continue;
                };
                if json.get("trnm").and_then(|v| v.as_str()) == Some("PING") {
                    let _ = write.send(Message::Text(text)).await;
                    continue;
                }
                if let Some(rows) = json.get("data").and_then(|v| v.as_array()) {
                    for row in rows {
                        ingest_0h_row(row, &mut out);
                    }
                }
            }
            Ok(Some(Ok(_))) => {}
            _ => break,
        }
    }
    let _ = write.send(Message::Close(None)).await;
    info!("[0H] ticks for {}/{} names", out.len(), tickers.len());
    out
}

async fn fetch_quotes(
    auth: &KiwoomAuth,
    market: &MarketDataService,
    tickers: &[String],
) -> HashMap<String, (f64, f64, &'static str)> {
    let token = match auth.ensure_token().await {
        Ok(t) => t,
        Err(e) => {
            warn!("[auction] token: {e}");
            return HashMap::new();
        }
    };
    let mut quotes = poll_0h(&token, tickers, Duration::from_millis(2500)).await;
    for t in tickers {
        if quotes.get(t).map(|(px, _, _)| *px > 0.0).unwrap_or(false) {
            continue;
        }
        match market.get_expected_match(&token, t).await {
            Ok((px, qty, _)) if px > 0.0 => {
                quotes.insert(t.clone(), (px, qty, "ka10001"));
                continue;
            }
            Ok(_) => {}
            Err(e) => warn!("[ka10001] {t}: {e}"),
        }
        match market.get_auction_touch(&token, t).await {
            Ok((px, qty)) if px > 0.0 => {
                quotes.insert(t.clone(), (px, qty, "ka10004"));
            }
            Ok(_) => {}
            Err(e) => warn!("[ka10004] {t}: {e}"),
        }
    }
    quotes
}

#[derive(Debug, Clone)]
pub struct SessionBar {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub last_print: f64,
    pub turnover: f64,
    pub exp_cntr_qty: f64,
    pub price_source: String,
}

fn turnover_from_prica(raw: f64) -> f64 {
    if raw <= 0.0 {
        return 0.0;
    }
    // Kiwoom trde_prica is usually 백만원; already-won values are much larger.
    if raw < 1e8 {
        raw * 1e6
    } else {
        raw
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionCloseMode {
    /// 15:28 buy: expected match is today's close for score and size.
    Expected,
    /// 18:00 parity: official daily bar (ka10081), never expected.
    Official,
}

fn session_from_ka10001(
    code: &str,
    json: &Value,
    exp_overlay: Option<(f64, f64, &str)>,
    mode: SessionCloseMode,
) -> Option<SessionBar> {
    let src = json.get("output").unwrap_or(json);
    let open = first_num(json, src, &["open_pric"]);
    let high = first_num(json, src, &["high_pric"]);
    let low = first_num(json, src, &["low_pric"]);
    let last = first_num(json, src, &["cur_prc"]);
    let exp = first_num(json, src, &["exp_cntr_pric"]);
    let exp_qty = first_num(json, src, &["exp_cntr_qty"]);
    // Daily accumulated volume/amount to 15:28 — not 0H expected match qty.
    let prica = first_num(json, src, &["trde_prica", "acc_trde_prica"]);
    let trde_qty = first_num(json, src, &["trde_qty", "acc_trde_qty"]);
    let mut close = last;
    let mut src_name = "ka10001_last".to_string();
    let mut qty = exp_qty;
    match mode {
        SessionCloseMode::Official => {
            if last <= 0.0 {
                warn!("[session] {code} no official last print");
                return None;
            }
            close = last;
            src_name = "ka10001_close".into();
        }
        SessionCloseMode::Expected => {
            if let Some((px, q, s)) = exp_overlay {
                if px > 0.0 {
                    close = px;
                    qty = q;
                    src_name = s.to_string();
                }
            } else if exp > 0.0 {
                close = exp;
                src_name = "ka10001_exp".into();
            }
            if close <= 0.0 {
                warn!("[session] {code} no close/expected");
                return None;
            }
        }
    }
    let high = high.max(close).max(open);
    let low = if low > 0.0 {
        low.min(close).min(if open > 0.0 { open } else { close })
    } else {
        close.min(if open > 0.0 { open } else { close })
    };
    let mut turnover = turnover_from_prica(prica);
    if turnover <= 0.0 && trde_qty > 0.0 && close > 0.0 {
        turnover = trde_qty * close;
    }
    if turnover <= 0.0 {
        warn!("[session] {code} daily volume missing (trde_qty/prica=0); vol_ratio will be 0");
    }
    Some(SessionBar {
        open: if open > 0.0 { open } else { close },
        high,
        low,
        close,
        last_print: last,
        turnover,
        exp_cntr_qty: qty,
        price_source: src_name,
    })
}

fn bar_needs_1m(bar: &SessionBar) -> bool {
    let range = (bar.high - bar.low).abs();
    bar.turnover <= 0.0 || range < 1.0 || (bar.open == bar.high && bar.high == bar.low)
}

fn merge_web(
    mut bar: SessionBar,
    web: &crate::ext_quotes::PublicDailyBar,
    mode: SessionCloseMode,
) -> SessionBar {
    if web.open > 0.0 {
        bar.open = web.open;
    }
    if web.high > 0.0 {
        bar.high = web.high.max(bar.close).max(bar.open);
    }
    if web.low > 0.0 {
        bar.low = web.low.min(bar.close).min(if bar.open > 0.0 { bar.open } else { bar.close });
    }
    if mode == SessionCloseMode::Official && web.close > 0.0 {
        bar.close = web.close;
        bar.last_print = web.close;
    } else if bar.last_print <= 0.0 && web.close > 0.0 {
        bar.last_print = web.close;
    }
    if bar.turnover <= 0.0 {
        bar.turnover = if web.turnover > 0.0 {
            web.turnover
        } else if web.volume > 0.0 {
            web.volume * if bar.close > 0.0 { bar.close } else { web.close }
        } else {
            0.0
        };
    }
    if !bar.price_source.contains(web.source) {
        bar.price_source = format!("{}_{}", bar.price_source, web.source);
    }
    bar
}

fn merge_1m(
    mut bar: SessionBar,
    ohlcv: (f64, f64, f64, f64, f64),
    add_auction_qty: bool,
) -> SessionBar {
    let (open, high, low, last, volume) = ohlcv;
    if open > 0.0 {
        bar.open = open;
    }
    if high > 0.0 {
        bar.high = high.max(bar.close).max(bar.open);
    }
    if low > 0.0 {
        bar.low = low.min(bar.close).min(if bar.open > 0.0 { bar.open } else { bar.close });
    }
    if bar.last_print <= 0.0 && last > 0.0 {
        bar.last_print = last;
    }
    let px = if bar.close > 0.0 { bar.close } else { last };
    // 1m chart ends ~15:19. Day volume ≈ 1m sum + closing-auction expected qty.
    let mut shares = volume;
    if add_auction_qty && bar.exp_cntr_qty > 0.0 {
        shares += bar.exp_cntr_qty;
    }
    if bar.turnover <= 0.0 && shares > 0.0 && px > 0.0 {
        bar.turnover = shares * px;
    } else if add_auction_qty && bar.exp_cntr_qty > 0.0 && px > 0.0 && volume > 0.0 {
        // ka10001 day qty usually freezes before auction — append expected match.
        bar.turnover += bar.exp_cntr_qty * px;
    }
    if !bar.price_source.contains("1m") {
        bar.price_source = format!("{}_1m", bar.price_source);
    }
    bar
}

fn apply_auction_qty_to_turnover(mut bar: SessionBar) -> SessionBar {
    if bar.exp_cntr_qty <= 0.0 || bar.close <= 0.0 {
        return bar;
    }
    // Floor when day turnover is still missing: auction qty alone beats vol_ratio=0.
    if bar.turnover <= 0.0 {
        bar.turnover = bar.exp_cntr_qty * bar.close;
        if !bar.price_source.contains("expqty") {
            bar.price_source = format!("{}_expqty", bar.price_source);
        }
    }
    bar
}

/// Kiwoom 당일 봉. Expected: 예상체결가 as close. Official: ka10081 확정 일봉 우선.
pub fn fetch_session_bars(
    cfg: &Config,
    tickers: &[String],
    mode: SessionCloseMode,
) -> HashMap<String, SessionBar> {
    if tickers.is_empty() {
        return HashMap::new();
    }
    let (auth, market) = match quote_auth(cfg) {
        Ok(v) => v,
        Err(e) => {
            warn!("[session] skip: {e}");
            return HashMap::new();
        }
    };
    block_on(async {
        let token = match auth.ensure_token().await {
            Ok(t) => t,
            Err(e) => {
                warn!("[session] token: {e}");
                return HashMap::new();
            }
        };
        let exp_map = if mode == SessionCloseMode::Expected {
            poll_0h(&token, tickers, Duration::from_millis(2500)).await
        } else {
            HashMap::new()
        };
        let mut out = HashMap::new();
        for t in tickers {
            let overlay = exp_map.get(t).map(|(px, q, s)| (*px, *q, *s));
            let mut bar: Option<SessionBar> = None;

            if mode == SessionCloseMode::Official {
                match market.get_today_daily_bar(&token, t).await {
                    Ok(Some((open, high, low, close, _vol, turnover))) => {
                        bar = Some(SessionBar {
                            open,
                            high,
                            low,
                            close,
                            last_print: close,
                            turnover,
                            exp_cntr_qty: 0.0,
                            price_source: "ka10081".into(),
                        });
                    }
                    Ok(None) => {}
                    Err(e) => warn!("[session] ka10081 {t}: {e}"),
                }
            }

            if bar.is_none() {
                bar = match market.get_session_snapshot(&token, t).await {
                    Ok(json) => session_from_ka10001(t, &json, overlay, mode),
                    Err(e) => {
                        warn!("[session] ka10001 {t}: {e}");
                        None
                    }
                };
            }

            if bar.as_ref().map(bar_needs_1m).unwrap_or(true) {
                match market.get_today_minute_agg(&token, t).await {
                    Ok(Some(ohlcv)) => {
                        let add_auction = mode == SessionCloseMode::Expected;
                        if let Some(b) = bar.take() {
                            bar = Some(merge_1m(b, ohlcv, add_auction));
                        } else if let Some((px, q, s)) = overlay {
                            if px > 0.0 {
                                let (open, high, low, last, _volume) = ohlcv;
                                bar = Some(merge_1m(
                                    SessionBar {
                                        open,
                                        high,
                                        low,
                                        close: px,
                                        last_print: last,
                                        turnover: 0.0,
                                        exp_cntr_qty: q,
                                        price_source: format!("{s}_1m"),
                                    },
                                    ohlcv,
                                    true,
                                ));
                            }
                        } else if mode == SessionCloseMode::Official {
                            let (open, high, low, last, _volume) = ohlcv;
                            if last > 0.0 {
                                bar = Some(merge_1m(
                                    SessionBar {
                                        open,
                                        high,
                                        low,
                                        close: last,
                                        last_print: last,
                                        turnover: 0.0,
                                        exp_cntr_qty: 0.0,
                                        price_source: "1m_close".into(),
                                    },
                                    ohlcv,
                                    false,
                                ));
                            }
                        }
                    }
                    Ok(None) => warn!("[session] {t} 1m fallback empty"),
                    Err(e) => warn!("[session] {t} 1m fallback: {e}"),
                }
            }

            if let Some(b) = bar.take() {
                bar = Some(apply_auction_qty_to_turnover(b));
            }

            if bar.as_ref().map(bar_needs_1m).unwrap_or(true) {
                if let Some(web) = crate::ext_quotes::fetch_public_daily(t).await {
                    if let Some(b) = bar.take() {
                        info!(
                            "[session] {t} web fallback src={} ohlc={:.0}/{:.0}/{:.0}/{:.0} turn={:.0}",
                            web.source, web.open, web.high, web.low, web.close, web.turnover
                        );
                        bar = Some(merge_web(b, &web, mode));
                    } else if let Some((px, q, s)) = overlay {
                        if px > 0.0 {
                            bar = Some(merge_web(
                                SessionBar {
                                    open: web.open,
                                    high: web.high,
                                    low: web.low,
                                    close: px,
                                    last_print: web.close,
                                    turnover: 0.0,
                                    exp_cntr_qty: q,
                                    price_source: format!("{s}_{}", web.source),
                                },
                                &web,
                                mode,
                            ));
                        }
                    } else if mode == SessionCloseMode::Official && web.close > 0.0 {
                        bar = Some(merge_web(
                            SessionBar {
                                open: web.open,
                                high: web.high,
                                low: web.low,
                                close: web.close,
                                last_print: web.close,
                                turnover: 0.0,
                                exp_cntr_qty: 0.0,
                                price_source: web.source.into(),
                            },
                            &web,
                            mode,
                        ));
                    }
                }
            }
            if let Some(bar) = bar {
                if bar.turnover <= 0.0 {
                    warn!(
                        "[session] {t} turnover still 0 after 1m/auction/web; vol_ratio will be 0"
                    );
                }
                info!(
                    "[session] {t} mode={mode:?} open={:.0} high={:.0} low={:.0} close={:.0} src={} turn={:.0}",
                    bar.open, bar.high, bar.low, bar.close, bar.price_source, bar.turnover
                );
                out.insert(t.clone(), bar);
            }
        }
        info!("[session] bars {}/{} mode={mode:?}", out.len(), tickers.len());
        out
    })
}

/// Replace pick.close_price with latest 예상체결가. Paper and live both fetch.
pub fn overlay_expected_match(cfg: &Config, picks: &mut [Pick]) {
    if picks.is_empty() {
        return;
    }
    let (auth, market) = match quote_auth(cfg) {
        Ok(v) => v,
        Err(e) => {
            warn!("[auction] skip overlay: {e}");
            return;
        }
    };
    let tickers: Vec<String> = picks.iter().map(|p| p.ticker.clone()).collect();
    let quotes = block_on(fetch_quotes(&auth, &market, &tickers));
    for p in picks.iter_mut() {
        p.last_print = p.close_price;
        if let Some((px, qty, src)) = quotes.get(&p.ticker) {
            if *px > 0.0 {
                p.close_price = *px;
                p.exp_cntr_qty = *qty;
                p.price_source = (*src).to_string();
                info!(
                    "[auction] {} expected={} qty={} src={} last_print={}",
                    p.ticker, px, qty, src, p.last_print
                );
                continue;
            }
        }
        p.price_source = "last_print_fallback".into();
        warn!(
            "[auction] {} no 예상체결가; sizing uses last_print={:.0}",
            p.ticker, p.last_print
        );
    }
}
