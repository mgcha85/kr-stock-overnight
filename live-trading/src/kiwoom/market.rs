use super::client::KiwoomClient;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
struct StockInfoRequest {
    stk_cd: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct CandleData {
    pub date: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: i64,
}

pub struct MarketDataService {
    client: KiwoomClient,
}

impl MarketDataService {
    pub fn new(client: KiwoomClient) -> Self {
        Self { client }
    }

    pub async fn get_stock_price(&self, token: &str, code: &str) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        let payload = StockInfoRequest {
            stk_cd: code.to_string(),
        };
        
        let endpoint = "/api/dostk/stkinfo";
        let auth = format!("Bearer {}", token);
        let headers = vec![
            ("authorization", auth.as_str()),
            ("api-id", "ka10001"),
            ("cont-yn", "N"),
            ("next-key", ""),
        ];
        
        let resp_json: serde_json::Value = self.client.post(
            endpoint,
            &payload,
            headers
        ).await?;
        
        let parse_price = |v: &serde_json::Value| -> Option<f64> {
            v.as_str()
             .and_then(|s| s.replace(",", "").trim().parse::<f64>().ok())
             .map(|price| price.abs())
        };
        
        if let Some(output) = resp_json.get("output") {
            if let Some(cur_prc) = output.get("cur_prc") {
                if let Some(p) = parse_price(cur_prc) { return Ok(p); }
            }
        }
        
        if let Some(cur_prc) = resp_json.get("cur_prc") {
            if let Some(p) = parse_price(cur_prc) { return Ok(p); }
        }

        Err("Failed to parse current price".into())
    }

    pub async fn get_open_price(&self, token: &str, code: &str) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        let payload = StockInfoRequest {
            stk_cd: code.to_string(),
        };
        let endpoint = "/api/dostk/stkinfo";
        let auth = format!("Bearer {token}");
        let headers = vec![
            ("authorization", auth.as_str()),
            ("api-id", "ka10001"),
            ("cont-yn", "N"),
            ("next-key", ""),
        ];
        let resp_json: serde_json::Value = self.client.post(endpoint, &payload, headers).await?;
        let parse_price = |v: &serde_json::Value| -> Option<f64> {
            v.as_str().and_then(|s| {
                s.replace(',', "")
                    .trim()
                    .trim_start_matches(['+', '-'])
                    .parse::<f64>()
                    .ok()
            })
            .filter(|p| *p > 0.0)
        };
        let src = resp_json.get("output").unwrap_or(&resp_json);
        if let Some(p) = src.get("open_pric").and_then(parse_price) {
            return Ok(p);
        }
        Err("Failed to parse open_pric".into())
    }

    pub async fn get_daily_candles(&self, token: &str, code: &str, start_date: &str) -> Result<Vec<CandleData>, Box<dyn std::error::Error + Send + Sync>> {
        let auth = format!("Bearer {}", token);
        let endpoint = "/api/dostk/chart";
        
        let today = chrono::Local::now().format("%Y%m%d").to_string();
        let payload = serde_json::json!({
            "stk_cd": code.to_string(),
            "upd_stkpc_tp": "1",
            "base_dt": today,
        });

        let headers = vec![
            ("authorization", auth.as_str()),
            ("api-id", "ka10081"),
            ("cont-yn", "N"),
            ("next-key", ""),
        ];

        let resp_json: serde_json::Value = self.client.post(
            endpoint,
            &payload,
            headers
        ).await?;

        let mut candles = Vec::new();
        
        let output = resp_json.get("stk_dt_pole_chart_qry")
            .or(resp_json.get("output"))
            .or(resp_json.get("stk_pole_chart_qry"));
        
        if let Some(items) = output.and_then(|v| v.as_array()) {
            for item in items {
                let date = item["dt"].as_str()
                    .or(item["cntr_tm"].as_str())
                    .unwrap_or("").to_string();
                if date.is_empty() || date < start_date.replace("-", "") {
                    continue;
                }

                let open = item["open_pric"].as_str().unwrap_or("0").replace(['+', '-'], "").parse::<f64>().unwrap_or(0.0);
                let high = item["high_pric"].as_str().unwrap_or("0").replace(['+', '-'], "").parse::<f64>().unwrap_or(0.0);
                let low = item["low_pric"].as_str().unwrap_or("0").replace(['+', '-'], "").parse::<f64>().unwrap_or(0.0);
                let close = item["cur_prc"].as_str().unwrap_or("0").replace(['+', '-'], "").parse::<f64>().unwrap_or(0.0);
                let volume = item["trde_qty"].as_str().unwrap_or("0").parse::<i64>().unwrap_or(0);

                candles.push(CandleData {
                    date: format!("{}-{}-{}", &date[0..4], &date[4..6], &date[6..8]),
                    open,
                    high,
                    low,
                    close,
                    volume,
                });
            }
        }

        candles.sort_by(|a, b| a.date.cmp(&b.date));

        Ok(candles)
    }

    /// Today's completed daily bar via ka10081 (available after market close).
    /// Returns (open, high, low, close, volume_shares, turnover_krw).
    pub async fn get_today_daily_bar(
        &self,
        token: &str,
        code: &str,
    ) -> Result<Option<(f64, f64, f64, f64, f64, f64)>, Box<dyn std::error::Error + Send + Sync>> {
        let ymd = chrono::Local::now().format("%Y%m%d").to_string();
        let auth = format!("Bearer {token}");
        let endpoint = "/api/dostk/chart";
        let payload = serde_json::json!({
            "stk_cd": code,
            "upd_stkpc_tp": "1",
            "base_dt": ymd,
        });
        let headers = vec![
            ("authorization", auth.as_str()),
            ("api-id", "ka10081"),
            ("cont-yn", "N"),
            ("next-key", ""),
        ];
        let resp_json: serde_json::Value = self.client.post(endpoint, &payload, headers).await?;
        let parse = |v: &serde_json::Value| -> f64 {
            let s = v
                .as_str()
                .map(|s| s.to_string())
                .or_else(|| v.as_f64().map(|n| n.to_string()))
                .unwrap_or_default();
            s.replace(',', "")
                .trim()
                .trim_start_matches(['+', '-'])
                .parse::<f64>()
                .unwrap_or(0.0)
                .abs()
        };
        let items = resp_json
            .get("stk_dt_pole_chart_qry")
            .or(resp_json.get("output"))
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        for item in items {
            let date = item
                .get("dt")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if date != ymd {
                continue;
            }
            let open = parse(&item["open_pric"]);
            let high = parse(&item["high_pric"]);
            let low = parse(&item["low_pric"]);
            let close = parse(&item["cur_prc"]);
            let volume = parse(&item["trde_qty"]);
            let mut turnover = parse(&item["trde_prica"]);
            // ka10081 trde_prica is usually 백만원.
            if turnover > 0.0 && turnover < 1e8 {
                turnover *= 1e6;
            }
            if turnover <= 0.0 && volume > 0.0 && close > 0.0 {
                turnover = volume * close;
            }
            if open <= 0.0 || close <= 0.0 {
                continue;
            }
            return Ok(Some((
                open,
                high.max(open).max(close),
                if low > 0.0 {
                    low.min(open).min(close)
                } else {
                    open.min(close)
                },
                close,
                volume,
                turnover,
            )));
        }
        Ok(None)
    }

    /// Aggregate today's ka10080 1-minute bars up to 15:28.
    /// Kiwoom returns newest-first; open/close must be taken after sorting by time.
    /// Chart usually ends ~15:19; missing 15:20–15:28 volume is proxied by 09:00–09:10.
    pub async fn get_today_minute_agg(
        &self,
        token: &str,
        code: &str,
    ) -> Result<Option<(f64, f64, f64, f64, f64)>, Box<dyn std::error::Error + Send + Sync>> {
        let ymd = chrono::Local::now().format("%Y%m%d").to_string();
        for attempt in 0..2 {
            if let Some(agg) = self.minute_agg_for_code(token, code, &ymd).await? {
                return Ok(Some(agg));
            }
            let al = format!("{code}_AL");
            if let Some(agg) = self.minute_agg_for_code(token, &al, &ymd).await? {
                return Ok(Some(agg));
            }
            if attempt == 0 {
                tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            }
        }
        Ok(None)
    }

    async fn minute_agg_for_code(
        &self,
        token: &str,
        stk_cd: &str,
        ymd: &str,
    ) -> Result<Option<(f64, f64, f64, f64, f64)>, Box<dyn std::error::Error + Send + Sync>> {
        let auth = format!("Bearer {token}");
        let endpoint = "/api/dostk/chart";
        let payload = serde_json::json!({
            "stk_cd": stk_cd,
            "tic_scope": "1",
            "upd_stkpc_tp": "0",
            "base_dt": ymd,
        });
        let parse = |v: &serde_json::Value| -> f64 {
            let s = v
                .as_str()
                .map(|s| s.to_string())
                .or_else(|| v.as_f64().map(|n| n.to_string()))
                .unwrap_or_default();
            s.replace(',', "")
                .trim()
                .trim_start_matches(['+', '-'])
                .parse::<f64>()
                .unwrap_or(0.0)
                .abs()
        };

        let mut next_key = String::new();
        let mut bars: Vec<(u32, f64, f64, f64, f64, f64)> = Vec::new();
        for _ in 0..8 {
            let cont = if next_key.is_empty() { "N" } else { "Y" };
            let headers = vec![
                ("authorization", auth.as_str()),
                ("api-id", "ka10080"),
                ("cont-yn", cont),
                ("next-key", next_key.as_str()),
            ];
            let (hdrs, json): (reqwest::header::HeaderMap, serde_json::Value) =
                self.client
                    .post_json_with_headers(endpoint, &payload, headers)
                    .await?;
            let items = json
                .get("stk_min_pole_chart_qry")
                .or(json.get("output"))
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();
            if items.is_empty() {
                if bars.is_empty() {
                    if let Some(msg) = json.get("return_msg").and_then(|v| v.as_str()) {
                        tracing::warn!("[ka10080] {stk_cd} empty: {msg}");
                    }
                }
                break;
            }
            for item in &items {
                let tm = item.get("cntr_tm").map(value_as_text).unwrap_or_default();
                let Some(hhmm) = parse_minute_hhmm(&tm, ymd) else {
                    continue;
                };
                let o = parse(&item["open_pric"]);
                let h = parse(&item["high_pric"]);
                let l = parse(&item["low_pric"]);
                let c = parse(&item["cur_prc"]);
                let v = parse(&item["trde_qty"]);
                bars.push((hhmm, o, h, l, c, v));
            }
            next_key = hdrs
                .get("next-key")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string())
                .or_else(|| {
                    json.get("next_key")
                        .or(json.get("next-key"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                })
                .unwrap_or_default();
            if next_key.is_empty() {
                break;
            }
        }
        Ok(aggregate_minute_bars(bars, 1528))
    }

    pub async fn get_session_snapshot(
        &self,
        token: &str,
        code: &str,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>> {
        let payload = StockInfoRequest {
            stk_cd: code.to_string(),
        };
        let endpoint = "/api/dostk/stkinfo";
        let auth = format!("Bearer {token}");
        let headers = vec![
            ("authorization", auth.as_str()),
            ("api-id", "ka10001"),
            ("cont-yn", "N"),
            ("next-key", ""),
        ];
        self.client
            .post(endpoint, &payload, headers)
            .await
            .map_err(|e| e.into())
    }

    pub async fn get_expected_match(
        &self,
        token: &str,
        code: &str,
    ) -> Result<(f64, f64, f64), Box<dyn std::error::Error + Send + Sync>> {
        let payload = StockInfoRequest {
            stk_cd: code.to_string(),
        };
        let endpoint = "/api/dostk/stkinfo";
        let auth = format!("Bearer {token}");
        let headers = vec![
            ("authorization", auth.as_str()),
            ("api-id", "ka10001"),
            ("cont-yn", "N"),
            ("next-key", ""),
        ];
        let resp_json: serde_json::Value = self.client.post(endpoint, &payload, headers).await?;
        Ok((
            read_kiwoom_num(&resp_json, "exp_cntr_pric"),
            read_kiwoom_num(&resp_json, "exp_cntr_qty"),
            read_kiwoom_num(&resp_json, "cur_prc"),
        ))
    }

    /// During 동시호가 best bid == best ask == 예상체결가.
    pub async fn get_auction_touch(
        &self,
        token: &str,
        code: &str,
    ) -> Result<(f64, f64), Box<dyn std::error::Error + Send + Sync>> {
        let payload = StockInfoRequest {
            stk_cd: code.to_string(),
        };
        let endpoint = "/api/dostk/mrkcond";
        let auth = format!("Bearer {token}");
        let headers = vec![
            ("authorization", auth.as_str()),
            ("api-id", "ka10004"),
            ("cont-yn", "N"),
            ("next-key", ""),
        ];
        let resp_json: serde_json::Value = self.client.post(endpoint, &payload, headers).await?;
        let sel = read_kiwoom_num(&resp_json, "sel_fpr_bid");
        let buy = read_kiwoom_num(&resp_json, "buy_fpr_bid");
        if sel > 0.0 && (sel - buy).abs() < 1e-9 {
            let qty = read_kiwoom_num(&resp_json, "sel_fpr_req")
                .min(read_kiwoom_num(&resp_json, "buy_fpr_req"));
            return Ok((sel, qty));
        }
        Ok((0.0, 0.0))
    }
}

fn value_as_text(v: &serde_json::Value) -> String {
    v.as_str()
        .map(|s| s.to_string())
        .or_else(|| v.as_i64().map(|n| n.to_string()))
        .or_else(|| v.as_u64().map(|n| n.to_string()))
        .unwrap_or_default()
}

/// `cntr_tm` is usually YYYYMMDDHHMM[SS]; some payloads are HHMMSS only.
pub(crate) fn parse_minute_hhmm(tm: &str, ymd: &str) -> Option<u32> {
    let digits: String = tm.chars().filter(|c| c.is_ascii_digit()).collect();
    if digits.len() >= 12 && digits.starts_with(ymd) {
        return digits.get(8..12)?.parse().ok();
    }
    if digits.len() == 6 {
        return digits.get(0..4)?.parse().ok();
    }
    if digits.len() == 4 {
        return digits.parse().ok();
    }
    None
}

/// `(hhmm, open, high, low, close, volume)` — order in `bars` does not matter.
/// Kiwoom 1m charts usually stop ~15:19 (no auction bars). If 15:20–15:28 volume
/// is missing, add 09:00–09:10 volume as a proxy for that window.
pub(crate) fn aggregate_minute_bars(
    mut bars: Vec<(u32, f64, f64, f64, f64, f64)>,
    max_hhmm: u32,
) -> Option<(f64, f64, f64, f64, f64)> {
    bars.retain(|(tm, o, _, _, c, _)| *tm <= max_hhmm && (*o > 0.0 || *c > 0.0));
    if bars.is_empty() {
        return None;
    }
    bars.sort_by_key(|(tm, ..)| *tm);
    let first = bars[0];
    let last = *bars.last().unwrap();
    let open = if first.1 > 0.0 { first.1 } else { first.4 };
    let close = if last.4 > 0.0 { last.4 } else { last.1 };
    if open <= 0.0 || close <= 0.0 {
        return None;
    }
    let mut high = 0.0_f64;
    let mut low = 0.0_f64;
    let mut volume = 0.0_f64;
    let mut early_vol = 0.0_f64;
    let mut auction_vol = 0.0_f64;
    for (tm, o, h, l, c, v) in &bars {
        high = high.max(*h).max(*c).max(*o);
        if *l > 0.0 {
            low = if low <= 0.0 { *l } else { low.min(*l) };
        }
        volume += *v;
        if (900..=910).contains(tm) {
            early_vol += *v;
        }
        if (1520..=1528).contains(tm) {
            auction_vol += *v;
        }
    }
    if auction_vol <= 0.0 && early_vol > 0.0 {
        volume += early_vol;
    }
    if low <= 0.0 {
        low = open.min(close);
    }
    Some((open, high, low, close, volume))
}

fn read_kiwoom_num(v: &serde_json::Value, key: &str) -> f64 {
    let parse = |x: &serde_json::Value| -> f64 {
        let s = x.as_str().map(|s| s.to_string()).unwrap_or_else(|| {
            x.as_f64()
                .map(|n| n.to_string())
                .or_else(|| x.as_i64().map(|n| n.to_string()))
                .unwrap_or_default()
        });
        let s = s.replace(',', "");
        let s = s.trim().trim_start_matches(['+', '-']);
        s.parse::<f64>().unwrap_or(0.0).abs()
    };
    if let Some(x) = v.get(key) {
        let n = parse(x);
        if n > 0.0 {
            return n;
        }
    }
    if let Some(out) = v.get("output") {
        if let Some(x) = out.get(key) {
            return parse(x);
        }
    }
    0.0
}

#[cfg(test)]
mod tests {
    use super::{aggregate_minute_bars, parse_minute_hhmm};

    #[test]
    fn newest_first_uses_session_open_and_1528_close() {
        // Same order Kiwoom sends: 15:28 first, 09:00 last.
        let bars = vec![
            (1528, 39400.0, 39450.0, 39300.0, 39350.0, 100.0),
            (1200, 41000.0, 41900.0, 38850.0, 40000.0, 200.0),
            (900, 40800.0, 40900.0, 40700.0, 40800.0, 50.0),
        ];
        let (open, high, low, last, vol) = aggregate_minute_bars(bars, 1528).unwrap();
        assert_eq!(open, 40800.0);
        assert_eq!(last, 39350.0);
        assert_eq!(high, 41900.0);
        assert_eq!(low, 38850.0);
        assert_eq!(vol, 350.0);
    }

    #[test]
    fn skips_after_1528() {
        let bars = vec![
            (1530, 1.0, 1.0, 1.0, 999.0, 1.0),
            (900, 10.0, 11.0, 9.0, 10.5, 1.0),
        ];
        let (_, _, _, last, _) = aggregate_minute_bars(bars, 1528).unwrap();
        assert_eq!(last, 10.5);
    }

    #[test]
    fn proxies_missing_auction_window_with_open() {
        // Chart ends 15:19; no 15:20–15:28 bars → add 09:00–09:10 volume once.
        let bars = vec![
            (1519, 10.0, 11.0, 9.0, 10.5, 100.0),
            (910, 9.0, 9.5, 8.5, 9.0, 40.0),
            (900, 8.0, 8.5, 7.5, 8.0, 60.0),
        ];
        let (_, _, _, last, vol) = aggregate_minute_bars(bars, 1528).unwrap();
        assert_eq!(last, 10.5);
        assert_eq!(vol, 100.0 + 40.0 + 60.0 + 40.0 + 60.0);
    }

    #[test]
    fn parses_kiwoom_cntr_tm() {
        assert_eq!(parse_minute_hhmm("20260911152800", "20260911"), Some(1528));
        assert_eq!(parse_minute_hhmm("202609111528", "20260911"), Some(1528));
        assert_eq!(parse_minute_hhmm("152800", "20260911"), Some(1528));
        assert_eq!(parse_minute_hhmm("0900", "20260911"), Some(900));
        assert_eq!(parse_minute_hhmm("20260910152800", "20260911"), None);
    }
}
