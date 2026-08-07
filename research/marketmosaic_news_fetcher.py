#!/usr/bin/env python3
"""
MarketMosaic Real News Fetcher
------------------------------
Queries local MarketMosaic Meilisearch instance (port 37700) for real historical news
matching candidate stock names/tickers on specific target dates (YYYY-MM-DD).
"""

import requests
from typing import Dict, Any, List, Optional

MEILI_URL = "http://localhost:37700"
MEILI_KEY = "masterKey"

class MarketMosaicNewsFetcher:
    def __init__(self, host: str = MEILI_URL, master_key: str = MEILI_KEY):
        self.host = host
        self.headers = {
            "Authorization": f"Bearer {master_key}",
            "Content-Type": "application/json"
        }

    def get_news_for_stock(self, stock_name: str, target_date: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Fetches real news for a stock name published on target_date (YYYY-MM-DD).
        """
        url = f"{self.host}/indexes/articles/search"
        payload = {
            "q": stock_name,
            "limit": 10
        }
        
        try:
            res = requests.post(url, headers=self.headers, json=payload, timeout=3)
            if res.status_code == 200:
                hits = res.json().get("hits", [])
                # Filter for articles containing stock_name in title AND published on target_date
                date_hits = [
                    h for h in hits 
                    if target_date in h.get("published_at", "") and stock_name in h.get("title", "")
                ]
                if date_hits:
                    return date_hits[:limit]
                    
                # Secondary fallback: articles published on target_date matching query
                date_any_hits = [h for h in hits if target_date in h.get("published_at", "")]
                if date_any_hits:
                    return date_any_hits[:limit]
        except Exception:
            pass
            
        return []

def test_fetcher():
    fetcher = MarketMosaicNewsFetcher()
    news = fetcher.get_news_for_stock("한화에어로스페이스", "2026-04-17")
    print(f"Fetched {len(news)} real news articles for 한화에어로스페이스 on 2026-04-17:")
    for n in news:
        print(f" - [{n.get('published_at')}] {n.get('title')}")

if __name__ == "__main__":
    test_fetcher()
