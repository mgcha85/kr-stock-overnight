//! Telegram alerts — mirrors `telegram.py`.

use anyhow::Result;
use tracing::{error, info, warn};

use crate::config::Config;

fn mode_tag(cfg: &Config) -> &'static str {
    if cfg.is_live() {
        "LIVE"
    } else {
        "PAPER"
    }
}

fn fmt_int(v: f64) -> String {
    let n = v.round() as i64;
    let neg = n < 0;
    let s = n.abs().to_string();
    let mut out = String::new();
    for (i, ch) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    let rev: String = out.chars().rev().collect();
    if neg {
        format!("-{rev}")
    } else {
        rev
    }
}

fn fmt_signed_int(v: f64) -> String {
    let base = fmt_int(v.abs());
    if v >= 0.0 {
        format!("+{base}")
    } else {
        format!("-{base}")
    }
}

pub fn send_telegram_message(cfg: &Config, message: &str, dry_run: bool) -> Result<bool> {
    if dry_run {
        info!("[TELEGRAM dry-run]\n{message}");
        return Ok(false);
    }
    if cfg.telegram_bot_token.is_empty() || cfg.telegram_chat_id.is_empty() {
        warn!("Telegram Bot Token or Chat ID not configured. Skipping.");
        println!("[TELEGRAM MOCK MESSAGE]\n{message}\n");
        return Ok(false);
    }
    let url = format!(
        "https://api.telegram.org/bot{}/sendMessage",
        cfg.telegram_bot_token
    );
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;
    let payload = serde_json::json!({
        "chat_id": cfg.telegram_chat_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": true,
    });
    match client.post(&url).json(&payload).send() {
        Ok(res) if res.status().is_success() => {
            info!("Telegram message sent successfully.");
            Ok(true)
        }
        Ok(res) => {
            error!(
                "Failed to send Telegram message: {} - {:?}",
                res.status(),
                res.text()
            );
            Ok(false)
        }
        Err(e) => {
            error!("Error sending Telegram message: {e}");
            Ok(false)
        }
    }
}

pub fn send_ops_error_alert(cfg: &Config, date_str: &str, title: &str, details: &str, dry_run: bool) -> Result<bool> {
    let msg = format!(
        "<b>🚨 [{}] [KRX Overnight] {title}</b>\n\
         🗓️ <b>일자:</b> {date_str}\n\
         ────────────────────────\n\
         {details}\n\
         ────────────────────────\n\
         <i>가짜 매수없음/가짜 -0.23% 청산은 하지 않았습니다. 원인 해결 후 재실행하세요.</i>",
        mode_tag(cfg)
    );
    send_telegram_message(cfg, &msg, dry_run)
}

pub fn send_market_close_buy_alert(
    cfg: &Config,
    date_str: &str,
    buys: &[serde_json::Value],
    capital_per_stock: f64,
    cash_remaining: f64,
    total_equity: f64,
    dry_run: bool,
) -> Result<bool> {
    let mut lines = vec![
        format!(
            "<b>📈 [{}] [KRX Overnight Strategy] 장 마감 매수 내역 (15:20)</b>",
            mode_tag(cfg)
        ),
        format!("🗓️ <b>일자:</b> {date_str}"),
        format!(
            "💵 <b>설정 시드:</b> {} 원 | <b>종목당 배정:</b> {} 원",
            fmt_int(total_equity),
            fmt_int(capital_per_stock)
        ),
        "────────────────────────".into(),
    ];
    if buys.is_empty() {
        lines.push("⚠️ <b>매수 조건 충족 종목 없음 (Cash 100% 보유)</b>".into());
    } else {
        for (idx, b) in buys.iter().enumerate() {
            lines.push(format!(
                "<b>{}. {} ({})</b>\n   • 테마: {}\n   • 매수가(종가): {} 원\n   • 수량: {} 주 (총 {} 원)\n   • 모델점수: {:.1}점 (LGB: {:.2} | DL: {:.2})",
                idx + 1,
                b["stock_name"].as_str().unwrap_or(""),
                b["ticker"].as_str().unwrap_or(""),
                b.get("theme_name").and_then(|v| v.as_str()).unwrap_or("N/A"),
                fmt_int(b["buy_price"].as_f64().unwrap_or(0.0)),
                b["buy_qty"].as_i64().unwrap_or(0),
                fmt_int(b["buy_amount"].as_f64().unwrap_or(0.0)),
                b.get("hybrid_score").and_then(|v| v.as_f64()).unwrap_or(0.0),
                b.get("p_lgb").and_then(|v| v.as_f64()).unwrap_or(0.0),
                b.get("p_torch").and_then(|v| v.as_f64()).unwrap_or(0.0),
            ));
        }
    }
    lines.push("────────────────────────".into());
    lines.push(format!(
        "💰 <b>매수 후 잔여 예수금:</b> {} 원",
        fmt_int(cash_remaining)
    ));
    lines.push("⏰ <i>익일 09:00 장 시작 시 시가 매도 예정</i>".into());
    send_telegram_message(cfg, &lines.join("\n"), dry_run)
}

pub fn send_market_open_sell_alert(
    cfg: &Config,
    date_str: &str,
    sells: &[serde_json::Value],
    daily_pnl_krw: f64,
    daily_pnl_pct: f64,
    weekly_pnl_pct: f64,
    monthly_pnl_pct: f64,
    total_equity: f64,
    dry_run: bool,
) -> Result<bool> {
    let pnl_icon = if daily_pnl_krw >= 0.0 { "🚀" } else { "🔻" };
    let mut lines = vec![
        format!(
            "<b>{pnl_icon} [{}] [KRX Overnight Strategy] 장 시작 매도 및 수익률 보고 (09:00)</b>",
            mode_tag(cfg)
        ),
        format!("🗓️ <b>일자:</b> {date_str}"),
        "────────────────────────".into(),
    ];
    if sells.is_empty() {
        lines.push("ℹ️ <b>오늘 청산 대상 보유 종목이 없습니다.</b>".into());
    } else {
        for (idx, s) in sells.iter().enumerate() {
            let pnl = s["pnl_krw"].as_f64().unwrap_or(0.0);
            let item_icon = if pnl >= 0.0 { "🟢" } else { "🔴" };
            lines.push(format!(
                "<b>{}. {} {} ({})</b>\n   • 매수가: {} 원 ➡️ 매도가: {} 원\n   • 청산 수량: {} 주\n   • 손익: <b>{} 원 ({:+.2}%)</b>",
                idx + 1,
                item_icon,
                s["stock_name"].as_str().unwrap_or(""),
                s["ticker"].as_str().unwrap_or(""),
                fmt_int(s["buy_price"].as_f64().unwrap_or(0.0)),
                fmt_int(s["sell_price"].as_f64().unwrap_or(0.0)),
                s["buy_qty"].as_i64().unwrap_or(0),
                fmt_signed_int(pnl),
                s["pnl_pct"].as_f64().unwrap_or(0.0),
            ));
        }
    }
    lines.push("────────────────────────".into());
    lines.push(format!(
        "📊 <b>오늘 일간 손익:</b> {} 원 ({daily_pnl_pct:+.2}%)",
        fmt_signed_int(daily_pnl_krw)
    ));
    lines.push(format!(
        "📅 <b>최근 1주일 누적 수익률:</b> <b>{weekly_pnl_pct:+.2}%</b>"
    ));
    lines.push(format!(
        "🗓️ <b>최근 1개월 누적 수익률:</b> <b>{monthly_pnl_pct:+.2}%</b>"
    ));
    lines.push(format!(
        "💎 <b>현재 총 평가 자산:</b> <b>{} 원</b>",
        fmt_int(total_equity)
    ));
    send_telegram_message(cfg, &lines.join("\n"), dry_run)
}

pub fn send_parity_check_alert(
    cfg: &Config,
    date_str: &str,
    is_matched: bool,
    paper_tickers: &[String],
    backtest_tickers: &[String],
    details: &str,
    dry_run: bool,
) -> Result<bool> {
    let status = if is_matched {
        "✅ [100% PARITY MATCH]"
    } else {
        "⚠️ [PARITY MISMATCH DETECTED]"
    };
    let paper = if paper_tickers.is_empty() {
        "없음".into()
    } else {
        paper_tickers.join(", ")
    };
    let back = if backtest_tickers.is_empty() {
        "없음".into()
    } else {
        backtest_tickers.join(", ")
    };
    let summary = if details.is_empty() {
        if is_matched {
            "매수 종목이 100% 일치합니다.".into()
        } else {
            "매수 내역 불일치 발생! 검증 필요.".into()
        }
    } else {
        details.to_string()
    };
    let msg = format!(
        "<b>{status} [{}] 장후 백테스트 ↔ 페이퍼트레이딩 검증 보고</b>\n\
         🗓️ <b>검증 일자:</b> {date_str}\n\
         📌 <b>Paper Trading 매수:</b> {paper}\n\
         🔍 <b>Backtest 정답 매수:</b> {back}\n\
         ────────────────────────\n\
         📝 <b>결과 요약:</b> {summary}",
        mode_tag(cfg)
    );
    send_telegram_message(cfg, &msg, dry_run)
}
