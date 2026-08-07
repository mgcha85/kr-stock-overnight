#!/usr/bin/env python3
"""
MarketMosaic Unified Data Integrator
-------------------------------------
Integrates ALL THREE core MarketMosaic datasets into the KRX Overnight AI pipeline:
1. Meilisearch News Articles (567,595 articles index on port 37700)
2. DART Corporate Filings (83,092 filings in dart.db)
3. Judal Theme Categories & Stock Mappings (323 themes in judal.db)
"""

import sqlite3
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional

DART_DB_PATH = Path("/mnt/data/projects/marketMosaic/backend/data/dart.db")
JUDAL_DB_PATH = Path("/mnt/data/projects/marketMosaic/backend/data/judal.db")
MEILI_URL = "http://localhost:37700"
MEILI_KEY = "masterKey"

class MarketMosaicIntegrator:
    def __init__(self):
        self.meili_headers = {
            "Authorization": f"Bearer {MEILI_KEY}",
            "Content-Type": "application/json"
        }
        self.theme_map: Dict[str, List[str]] = self._load_judal_themes()

    def _load_judal_themes(self) -> Dict[str, List[str]]:
        """Loads Judal Stock Code -> Theme Names mapping."""
        theme_map = {}
        if JUDAL_DB_PATH.exists():
            try:
                conn = sqlite3.connect(str(JUDAL_DB_PATH))
                c = conn.cursor()
                rows = c.execute("""
                    SELECT s.code, t.name 
                    FROM themes t 
                    JOIN theme_stocks ts ON t.theme_idx = ts.theme_idx 
                    JOIN stocks s ON ts.stock_code = s.code
                """).fetchall()
                for code, theme_name in rows:
                    clean_code = str(code).zfill(6)
                    if clean_code not in theme_map:
                        theme_map[clean_code] = []
                    theme_map[clean_code].append(theme_name)
                conn.close()
            except Exception:
                pass
        return theme_map

    def get_dart_filings(self, stock_name: str, target_date: str) -> List[Dict[str, Any]]:
        """Fetches DART filings for a stock on target_date (YYYY-MM-DD or YYYYMMDD)."""
        clean_date = target_date.replace("-", "")
        filings = []
        if DART_DB_PATH.exists():
            try:
                conn = sqlite3.connect(str(DART_DB_PATH))
                c = conn.cursor()
                rows = c.execute("""
                    SELECT corp_name, report_nm, rcept_dt, rm 
                    FROM filings 
                    WHERE corp_name = ? AND rcept_dt = ?
                """, (stock_name, clean_date)).fetchall()
                for r in rows:
                    filings.append({
                        "corp_name": r[0],
                        "report_nm": r[1].strip(),
                        "rcept_dt": r[2],
                        "rm": r[3]
                    })
                conn.close()
            except Exception:
                pass
        return filings

    def get_stock_themes(self, ticker: str) -> List[str]:
        """Gets Judal themes associated with a stock ticker."""
        clean_ticker = str(ticker).split(".")[0].zfill(6)
        return self.theme_map.get(clean_ticker, [])

    def get_news_articles(self, stock_name: str, target_date: str, limit: int = 2) -> List[Dict[str, Any]]:
        """Fetches news articles published on target_date BEFORE 15:30:00 close to prevent look-ahead bias."""
        url = f"{MEILI_URL}/indexes/articles/search"
        payload = {"q": stock_name, "limit": 20}
        cutoff_str = f"{target_date} 15:30:00"
        try:
            res = requests.post(url, headers=self.meili_headers, json=payload, timeout=3)
            if res.status_code == 200:
                hits = res.json().get("hits", [])
                valid_hits = []
                for h in hits:
                    pub_at = h.get("published_at", "")
                    if target_date in pub_at and pub_at <= cutoff_str:
                        valid_hits.append(h)
                
                title_matches = [h for h in valid_hits if stock_name in h.get("title", "")]
                if title_matches:
                    return title_matches[:limit]
                if valid_hits:
                    return valid_hits[:limit]
        except Exception:
            pass
        return []

    def get_full_market_context(self, ticker: str, stock_name: str, target_date: str) -> Dict[str, Any]:
        """
        Combines News + DART Filings + Judal Themes into a unified context packet.
        """
        news = self.get_news_articles(stock_name, target_date)
        dart = self.get_dart_filings(stock_name, target_date)
        themes = self.get_stock_themes(ticker)

        # Build composite text summary for OpenRouter LLM
        summary_parts = []
        if themes:
            summary_parts.append(f"[소속 테마]: {', '.join(themes)}")
        if dart:
            dart_titles = [d['report_nm'] for d in dart]
            summary_parts.append(f"[DART 공시]: {'; '.join(dart_titles)}")
        if news:
            news_titles = [n.get('title', '') for n in news]
            summary_parts.append(f"[주요 뉴스]: {'; '.join(news_titles)}")

        composite_text = " | ".join(summary_parts) if summary_parts else "특이 공시/뉴스 없음"

        return {
            "ticker": ticker,
            "stock_name": stock_name,
            "target_date": target_date,
            "themes": themes,
            "dart_filings": dart,
            "news_articles": news,
            "composite_summary": composite_text
        }

def test_integrator():
    integrator = MarketMosaicIntegrator()
    print(f"Loaded Judal stock-theme mappings for {len(integrator.theme_map)} stocks.")
    
    context = integrator.get_full_market_context("005930", "삼성전자", "2026-04-17")
    print("\n[Sample Market Context - 삼성전자 2026-04-17]:")
    print(f" - Themes: {context['themes']}")
    print(f" - DART Filings Count: {len(context['dart_filings'])}")
    print(f" - News Count: {len(context['news_articles'])}")
    print(f" - Composite Summary: {context['composite_summary']}")

if __name__ == "__main__":
    test_integrator()
