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
}
