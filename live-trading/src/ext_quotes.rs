//! Public web OHLC fallback when Kiwoom ka10001/ka10080 miss a name.
//! Daum (Kakao) first — includes turnover in KRW — then Naver daily chart.

use std::time::Duration;

use reqwest::Client;
use tracing::warn;

#[derive(Debug, Clone)]
pub struct PublicDailyBar {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub turnover: f64,
    pub source: &'static str,
}

fn http() -> Client {
    Client::builder()
        .timeout(Duration::from_secs(8))
        .user_agent("Mozilla/5.0 (compatible; kr-stock-live/0.1)")
        .build()
        .unwrap_or_else(|_| Client::new())
}

/// Today's completed (or last-printed) daily bar from Daum, then Naver.
pub async fn fetch_public_daily(code: &str) -> Option<PublicDailyBar> {
    let digits: String = code.chars().filter(|c| c.is_ascii_digit()).collect();
    if digits.len() < 6 {
        return None;
    }
    let code = format!("{:0>6}", &digits[digits.len() - 6..]);
    let ymd = chrono::Local::now().format("%Y%m%d").to_string();
    match fetch_daum_daily(&code, &ymd).await {
        Ok(Some(bar)) => return Some(bar),
        Ok(None) => {}
        Err(e) => warn!("[web] daum {code}: {e}"),
    }
    match fetch_naver_daily(&code, &ymd).await {
        Ok(Some(bar)) => Some(bar),
        Ok(None) => {
            warn!("[web] {code} no public daily bar for {ymd}");
            None
        }
        Err(e) => {
            warn!("[web] naver {code}: {e}");
            None
        }
    }
}

async fn fetch_daum_daily(
    code: &str,
    ymd: &str,
) -> Result<Option<PublicDailyBar>, reqwest::Error> {
    let url = format!("https://finance.daum.net/api/quote/A{code}/days?page=1&perPage=5&pagination=true");
    let json: serde_json::Value = http()
        .get(url)
        .header("Referer", format!("https://finance.daum.net/quotes/A{code}"))
        .send()
        .await?
        .json()
        .await?;
    let Some(rows) = json.get("data").and_then(|v| v.as_array()) else {
        return Ok(None);
    };
    let want = format!("{}-{}-{}", &ymd[0..4], &ymd[4..6], &ymd[6..8]);
    for row in rows {
        let date = row
            .get("date")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if !date.starts_with(&want) {
            continue;
        }
        let open = json_f64(row.get("openingPrice"));
        let high = json_f64(row.get("highPrice"));
        let low = json_f64(row.get("lowPrice"));
        let close = json_f64(row.get("tradePrice"));
        let volume = json_f64(row.get("accTradeVolume"));
        let turnover = json_f64(row.get("accTradePrice"));
        if open <= 0.0 || close <= 0.0 {
            continue;
        }
        return Ok(Some(PublicDailyBar {
            open,
            high: high.max(open).max(close),
            low: if low > 0.0 {
                low.min(open).min(close)
            } else {
                open.min(close)
            },
            close,
            volume,
            turnover,
            source: "daum",
        }));
    }
    Ok(None)
}

async fn fetch_naver_daily(
    code: &str,
    ymd: &str,
) -> Result<Option<PublicDailyBar>, reqwest::Error> {
    let url = format!(
        "https://api.finance.naver.com/siseJson.naver?symbol={code}&requestType=1&startTime={ymd}&endTime={ymd}&timeframe=day"
    );
    let text = http()
        .get(url)
        .header("Referer", "https://finance.naver.com/")
        .send()
        .await?
        .text()
        .await?;
    Ok(parse_naver_sise_json(&text, ymd))
}

pub(crate) fn parse_naver_sise_json(text: &str, ymd: &str) -> Option<PublicDailyBar> {
    for line in text.lines() {
        let line = line.trim().trim_end_matches(',');
        if !line.contains(ymd) {
            continue;
        }
        let inner = line.trim_start_matches('[').trim_end_matches(']');
        let parts: Vec<&str> = inner
            .split(',')
            .map(|p| p.trim().trim_matches('"').trim_matches('\''))
            .collect();
        if parts.len() < 6 || parts[0] != ymd {
            continue;
        }
        let open: f64 = parts[1].parse().ok()?;
        let high: f64 = parts[2].parse().ok()?;
        let low: f64 = parts[3].parse().ok()?;
        let close: f64 = parts[4].parse().ok()?;
        let volume: f64 = parts[5].parse().ok()?;
        if open <= 0.0 || close <= 0.0 {
            continue;
        }
        return Some(PublicDailyBar {
            open,
            high: high.max(open).max(close),
            low: if low > 0.0 {
                low.min(open).min(close)
            } else {
                open.min(close)
            },
            close,
            volume,
            turnover: if volume > 0.0 { volume * close } else { 0.0 },
            source: "naver",
        });
    }
    None
}

fn json_f64(v: Option<&serde_json::Value>) -> f64 {
    let Some(v) = v else {
        return 0.0;
    };
    v.as_f64()
        .or_else(|| v.as_i64().map(|n| n as f64))
        .or_else(|| {
            v.as_str().and_then(|s| {
                s.replace(',', "")
                    .trim()
                    .trim_start_matches(['+', '-'])
                    .parse()
                    .ok()
            })
        })
        .unwrap_or(0.0)
        .abs()
}

#[cfg(test)]
mod tests {
    use super::parse_naver_sise_json;

    #[test]
    fn parses_naver_daily_row() {
        let raw = r#"
 [['날짜', '시가', '고가', '저가', '종가', '거래량', '외국인소진율'],
["20260911", 39100, 42950, 38400, 42300, 678198, 1.67]
]
"#;
        let bar = parse_naver_sise_json(raw, "20260911").unwrap();
        assert_eq!(bar.open, 39100.0);
        assert_eq!(bar.high, 42950.0);
        assert_eq!(bar.low, 38400.0);
        assert_eq!(bar.close, 42300.0);
        assert_eq!(bar.volume, 678198.0);
        assert_eq!(bar.source, "naver");
    }
}
