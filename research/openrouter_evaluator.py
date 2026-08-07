#!/usr/bin/env python3
"""
OpenRouter Multi-Model Round-Robin Evaluator
---------------------------------------------
Evaluates stock news, DART announcements, and theme sentiment for Korean Overnight trading.
Rotates through OpenRouter's top 4 verified free models in a round-robin cycle with automatic fallback:
1. google/gemma-4-26b-a4b-it:free
2. nvidia/nemotron-3-nano-30b-a3b:free
3. openai/gpt-oss-20b:free
4. google/gemma-4-31b-it:free
"""

import os
import re
import json
import requests
from typing import Dict, Any, Optional

# Verified Active OpenRouter Free Models
FREE_MODELS = [
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "openai/gpt-oss-20b:free",
    "google/gemma-4-31b-it:free",
]

class OpenRouterEvaluator:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPEN_ROUTER_API_KEY")
        if not self.api_key:
            env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("OPEN_ROUTER_API_KEY="):
                            self.api_key = line.strip().split("=", 1)[1]
                            break
                            
        self.model_idx = 0
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://kr-stock-overnight.internal",
            "X-Title": "KRX Overnight Strategy Evaluator"
        }
        self.cache: Dict[str, Any] = {}
        
    def _get_next_model(self) -> str:
        model = FREE_MODELS[self.model_idx % len(FREE_MODELS)]
        self.model_idx += 1
        return model

    def _parse_llm_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Extracts and safely parses JSON from LLM text output."""
        try:
            # Match first json block or object pattern
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                clean_json_str = json_match.group(0)
                # Fix trailing commas or common formatting glitches
                clean_json_str = re.sub(r',\s*\}', '}', clean_json_str)
                clean_json_str = re.sub(r',\s*\]', ']', clean_json_str)
                return json.loads(clean_json_str)
        except Exception:
            pass
        return None

    def evaluate_sentiment(self, ticker: str, stock_name: str, news_title: str, content_snippet: str = "") -> Dict[str, Any]:
        """
        Evaluates stock news/announcement sentiment returning score between -10 and +30.
        """
        cache_key = f"{ticker}_{news_title}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        prompt = f"""
종목: {stock_name} ({ticker})
제목: {news_title}
내용: {content_snippet}

익일 시가 상승 호재 정도를 평가하여 JSON으로만 출력하세요.
형식: {{"sentiment": "BULLISH", "score": 20, "reason": "호재 사유"}}
"""

        for attempt in range(len(FREE_MODELS)):
            current_model = self._get_next_model()
            payload = {
                "model": current_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 120
            }
            
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=self.headers,
                    data=json.dumps(payload),
                    timeout=5
                )
                
                if response.status_code == 200:
                    res_json = response.json()
                    content = res_json['choices'][0]['message']['content'].strip()
                    parsed = self._parse_llm_json(content)
                    if parsed and "score" in parsed:
                        parsed["model_used"] = current_model
                        self.cache[cache_key] = parsed
                        return parsed
            except Exception:
                pass
                
        # Heuristic fallback if rate-limited or JSON parse fails
        fallback_score = 15 if any(kw in news_title for kw in ["공급", "수주", "실적", "흑자", "특허", "계약", "상한가"]) else 0
        fallback_res = {
            "sentiment": "BULLISH" if fallback_score > 0 else "NEUTRAL",
            "score": fallback_score,
            "reason": "Rule-based Fallback",
            "model_used": "rule_fallback"
        }
        self.cache[cache_key] = fallback_res
        return fallback_res
