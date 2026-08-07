# API Interface Specification (API 인터페이스 명세서)

이 문서에는 `kr_stock` Overnight 전략 엔진과 Svelte 프론트엔드 및 외부 데이터 시스템(MarketMosaic, KIS API, Supabase) 간의 REST & WebSocket API 명세를 기술합니다.

---

## 1. Internal Engine Endpoints (Go Backend / Engine)

Backend Base URL: `http://localhost:8080` (or `http://kr_stock_engine:8080`)

### 1.1 Strategy Status & Dashboard Summary
- **GET `/api/v1/overnight/status`**
- **Description**: 현재 엔진 구동 상태 및 당일 Overnight 타겟 종목 스코어링 현황 조회.
- **Response**:
```json
{
  "status": "RUNNING",
  "env": "prod",
  "server_time": "2026-08-08T15:15:00+09:00",
  "active_candidates": [
    {
      "code": "005930",
      "name": "삼성전자",
      "market": "KOSPI",
      "total_score": 88.5,
      "breakdown": {
        "technical_score": 32.0,
        "theme_score": 25.0,
        "news_dart_score": 21.5,
        "macro_score": 10.0
      },
      "trade_status": "TARGET_SELECTED",
      "news_highlights": ["대규모 반도체 공급 계약 체결"],
      "theme_name": "반도체 대표주"
    }
  ]
}
```

### 1.2 Execution Logs & Trade History
- **GET `/api/v1/overnight/trades`**
- **Query Params**: `page` (default 1), `limit` (default 20), `start_date`, `end_date`
- **Description**: 최근 집행된 Overnight 매매 내역 및 수익률 조회.
- **Response**:
```json
{
  "total": 45,
  "page": 1,
  "trades": [
    {
      "id": "trade_20260807_01",
      "code": "005930",
      "name": "삼성전자",
      "buy_time": "2026-08-07 15:20:00",
      "buy_price": 75000,
      "sell_time": "2026-08-08 09:00:15",
      "sell_price": 76800,
      "profit_pct": 2.40,
      "net_profit_pct": 2.15,
      "status": "CLOSED",
      "exit_reason": "GAP_OPEN_PROFIT"
    }
  ]
}
```

### 1.3 Trigger Strategy Analysis Manually
- **POST `/api/v1/overnight/analyze`**
- **Description**: 15:15 정규 스케줄링 전 수동으로 마켓모자이크 다중 데이터 기반 종목 분석 실행.
- **Payload**:
```json
{
  "min_turnover_krw": 50000000000,
  "score_threshold": 80.0
}
```
- **Response**:
```json
{
  "message": "Analysis started successfully",
  "job_id": "job_20260808_151500"
}
```

---

## 2. Integration Endpoints (External Services)

### 2.1 MarketMosaic API
- `POST http://localhost:38080/candle/data` (Candle Ingestion)
- `POST http://localhost:38080/news/migration` (News Ingestion)
- `POST http://localhost:38080/dart/migration/filings` (DART Filings)
- `POST http://localhost:38080/judal/migration/themes` (Theme Ingestion)

### 2.2 KIS (한국투자증권) OpenAPI
- **주식 동시호가/지정가 주문**: `/uapi/domestic-stock/v1/trading/order-cash`
- **잔고 조회**: `/uapi/domestic-stock/v1/trading/inquire-balance`
- **시가/종가 호가 조회**: `/uapi/domestic-stock/v1/quotations/inquire-price`

---

## 3. Gemini 3.0 Flash AI Integration

- **Model**: `gemini-3.0-flash`
- **API Endpoint**: `https://generativelanguage.googleapis.com/v1beta/models/gemini-3.0-flash:generateContent`
- **Prompt Structure**:
```json
{
  "contents": [
    {
      "parts": [
        {
          "text": "다음 뉴스와 공시가 장 마감 후 익일 아침 주가 상승에 미치는 호재 영향도를 0~25점으로 평가하고 이유를 요약하세요.\n뉴스: 대기업 대규모 AI 칩 공급 체결\n공시: 제3자배정 유상증자 500억원 납입 완료"
        }
      ]
    }
  ]
}
```
