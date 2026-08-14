"""
Telegram Bot Messaging Helper for KRX Paper-Trading Engine
----------------------------------------------------------
Sends real-time alerts for:
- Market Close BUY entries (15:20)
- Market Open SELL exits (09:00)
- Post-Trade Weekly & Monthly Returns Summary
- Daily Backtest Parity Verification Report
"""

import requests
import logging
from typing import List, Dict, Any
from kr_stock.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, execution_mode_tag

logger = logging.getLogger(__name__)


def send_telegram_message(message: str) -> bool:
    """Sends a Telegram message using Bot API."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram Bot Token or Chat ID not configured. Skipping message.")
        print(f"[TELEGRAM MOCK MESSAGE]\n{message}\n")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }

    try:
        res = requests.post(url, json=payload, timeout=10)
        if res.status_code == 200:
            logger.info("Telegram message sent successfully.")
            return True
        else:
            logger.error(f"Failed to send Telegram message: {res.status_code} - {res.text}")
            return False
    except Exception as e:
        logger.error(f"Error sending Telegram message: {e}")
        return False


def send_ops_error_alert(date_str: str, title: str, details: str) -> bool:
    """Failure that must not be reported as a normal empty-buy / fee-only sell."""
    msg = (
        f"<b>🚨 [{execution_mode_tag()}] [KRX Overnight] {title}</b>\n"
        f"🗓️ <b>일자:</b> {date_str}\n"
        "────────────────────────\n"
        f"{details}\n"
        "────────────────────────\n"
        "<i>가짜 매수없음/가짜 -0.23% 청산은 하지 않았습니다. 원인 해결 후 재실행하세요.</i>"
    )
    return send_telegram_message(msg)


def send_market_close_buy_alert(
    date_str: str,
    buys: List[Dict[str, Any]],
    capital_per_stock: float,
    cash_remaining: float,
    total_equity: float,
    empty_reason: str = "",
) -> bool:
    """Formats and sends the 15:20 Market Close BUY notification."""
    lines = [
        f"<b>📈 [{execution_mode_tag()}] [KRX Overnight Strategy] 장 마감 매수 내역 (15:20)</b>",
        f"🗓️ <b>일자:</b> {date_str}",
        f"💵 <b>설정 시드:</b> {total_equity:,.0f} 원 | <b>종목당 배정:</b> {capital_per_stock:,.0f} 원",
        "────────────────────────"
    ]

    if not buys:
        if empty_reason:
            lines.append(f"⚠️ <b>매수 없음</b>\n   {empty_reason}")
        else:
            lines.append("⚠️ <b>매수 조건 충족 종목 없음 (필터 통과 종목 0)</b>")
    else:
        for idx, b in enumerate(buys, 1):
            lines.append(
                f"<b>{idx}. {b['stock_name']} ({b['ticker']})</b>\n"
                f"   • 테마: {b.get('theme_name', 'N/A')}\n"
                f"   • 매수가(종가): {b['buy_price']:,.0f} 원\n"
                f"   • 수량: {b['buy_qty']:,} 주 (총 {b['buy_amount']:,.0f} 원)\n"
                f"   • 모델점수: {b.get('hybrid_score', 0):.1f}점 (LGB: {b.get('p_lgb', 0):.2f} | DL: {b.get('p_torch', 0):.2f})"
            )

    lines.extend([
        "────────────────────────",
        f"💰 <b>매수 후 잔여 예수금:</b> {cash_remaining:,.0f} 원",
        "⏰ <i>익일 09:00 장 시작 시 시가 매도 예정</i>"
    ])

    msg = "\n".join(lines)
    return send_telegram_message(msg)


def send_market_open_sell_alert(
    date_str: str,
    sells: List[Dict[str, Any]],
    daily_pnl_krw: float,
    daily_pnl_pct: float,
    weekly_pnl_pct: float,
    monthly_pnl_pct: float,
    total_equity: float
) -> bool:
    """Formats and sends the 09:00 Market Open SELL & Returns notification."""
    pnl_icon = "🚀" if daily_pnl_krw >= 0 else "🔻"
    lines = [
        f"<b>{pnl_icon} [{execution_mode_tag()}] [KRX Overnight Strategy] 장 시작 매도 및 수익률 보고 (09:00)</b>",
        f"🗓️ <b>일자:</b> {date_str}",
        "────────────────────────"
    ]

    if not sells:
        lines.append("ℹ️ <b>오늘 청산 대상 보유 종목이 없습니다.</b>")
    else:
        for idx, s in enumerate(sells, 1):
            item_icon = "🟢" if s['pnl_krw'] >= 0 else "🔴"
            lines.append(
                f"<b>{idx}. {item_icon} {s['stock_name']} ({s['ticker']})</b>\n"
                f"   • 매수가: {s['buy_price']:,.0f} 원 ➡️ 매도가: {s['sell_price']:,.0f} 원\n"
                f"   • 청산 수량: {s['buy_qty']:,} 주\n"
                f"   • 손익: <b>{s['pnl_krw']:+,.0f} 원 ({s['pnl_pct']:+.2f}%)</b>"
            )

    lines.extend([
        "────────────────────────",
        f"📊 <b>오늘 일간 손익:</b> {daily_pnl_krw:+,.0f} 원 ({daily_pnl_pct:+.2f}%)",
        f"📅 <b>최근 1주일 누적 수익률:</b> <b>{weekly_pnl_pct:+.2f}%</b>",
        f"🗓️ <b>최근 1개월 누적 수익률:</b> <b>{monthly_pnl_pct:+.2f}%</b>",
        f"💎 <b>현재 총 평가 자산:</b> <b>{total_equity:,.0f} 원</b>"
    ])

    msg = "\n".join(lines)
    return send_telegram_message(msg)


def send_parity_check_alert(
    date_str: str,
    is_matched: bool,
    paper_tickers: List[str],
    backtest_tickers: List[str],
    details: str = ""
) -> bool:
    """Formats and sends the post-market Backtest vs Paper Trading parity verification report."""
    both_empty = (not paper_tickers) and (not backtest_tickers)
    if both_empty:
        status_icon = "ℹ️ [양쪽 매수 없음]"
    elif is_matched:
        status_icon = "✅ [100% PARITY MATCH]"
    else:
        status_icon = "⚠️ [PARITY MISMATCH DETECTED]"
    
    lines = [
        f"<b>{status_icon} [{execution_mode_tag()}] 장후 백테스트 ↔ 페이퍼트레이딩 검증 보고</b>",
        f"🗓️ <b>검증 일자:</b> {date_str}",
        f"📌 <b>Paper Trading 매수:</b> {', '.join(paper_tickers) if paper_tickers else '없음'}",
        f"🔍 <b>Backtest 정답 매수:</b> {', '.join(backtest_tickers) if backtest_tickers else '없음'}",
        "────────────────────────",
        f"📝 <b>결과 요약:</b> {details if details else ('매수 종목이 100% 일치합니다.' if is_matched else '매수 내역 불일치 발생! 검증 필요.')}"
    ]

    msg = "\n".join(lines)
    return send_telegram_message(msg)
