# Data Engineering & Analysis System Rules (데이터 엔지니어링 및 분석 표준 규정)

본 문서는 `kr_stock` 프로젝트에서 수행하는 모든 데이터 수집, 가공, 분석, 백테스팅 및 DB 인터페이스 구축에 적용되는 **필수 개발 규정**입니다.

---

## 1. 데이터 분석 엔진 규정 (Data Analysis)
* **원칙**: **모든 데이터 가공 및 분석 작업에는 `Polars` 라이브러리를 최우선으로 사용한다.**
* **사유**: `Pandas` 대비 압도적인 메모리 효율성과 멀티스레드 쿼리 최적화(Lazy Evaluation)를 제공하여 290만 건 이상의 일봉/분봉 데이터를 초고속 처리하기 위함.
* **적용**:
  * DataFrame 쿼리, 집계(Aggregation), 이평선/지표 계산, 타임프레임 리샘플링 시 `polars` 사용.
  * 복잡한 변환 연산 시 `lazy()` 모드 및 `scan_parquet()` 적극 활용.

---

## 2. 데이터 저장 포맷 규정 (Data Storage)
* **원칙**: **모든 가공 데이터 및 중간 분석 결과물의 표준 저장 포맷은 `Parquet` (.parquet)을 사용한다.**
* **사유**: Columnar 압축 포맷으로 디스크 사용량을 극도로 절감하고, schema type 유지 및 Polars/DuckDB와의 zero-copy 읽기 성능을 보장함.
* **적용**:
  * 원본 DB에서 추출 및 가공된 데이터는 프로젝트 루트 `data/*.parquet` 경로에 저장.
  * 대용량 데이터는 날짜(`date`) 또는 심볼(`ticker`) 파티셔닝 구조로 저장.

---

## 3. 분석 및 인터페이스 DB 규정 (Analytics & Interface DB)
* **원칙**: **분석용 쿼리, 실시간 조회 인터페이스, 대용량 SQL 연산이 필요한 경우 `DuckDB`를 사용한다.**
* **사유**: Parquet 파일과 별도의 데이터 복사 없이 직접 SQL 쿼리가 가능한 최적의 인메모리 OLAP 데이터베이스 엔진임.
* **적용**:
  * 백테스터, 대시보드 API, 복잡한 필터링/조인 쿼리 실행 시 DuckDB 커넥션 활용.
  * 필요 시 SQLite 대신 `data/kr_stock.duckdb` 파일 기반 인덱스 및 뷰 구성.

---

## 4. 시계열 타임스탬프 및 Look-ahead Bias 방지 규정 (Time Contract)
* **원칙**: **모든 캔들(Kline) 데이터는 `open_time`과 `close_time`을 명시적으로 분리하여 저장하고 연산한다.**
* **시간 규격**:
  * `open_time`: 당일 장 개장 시각 (`YYYY-MM-DD 09:00:00`)
  * `close_time`: 당일 장 마감 시각 (`YYYY-MM-DD 15:30:00`)
  * `next_open_time`: 익일 장 개장 시각 (`YYYY-MM-DD(T+1) 09:00:00`)
* **적용**:
  * 종가 베팅(Overnight) 진입 타점은 반드시 당일 `close_time` (15:30) 기준.
  * 청산 타점은 반드시 익일 `next_open_time` (09:00) 기준.
  * 미래 데이터 참조(Look-ahead bias) 발생 가능성을 엄격히 통제.

---

## 5. 프로젝트 데이터 관리 규정 (Directory & Versioning)
* **원칙**: **모든 가공된 데이터 파일은 프로젝트 루트의 `data/` 폴더에 일관되게 위치시킨다.**
* **규칙**:
  * `data/` 디렉토리는 대용량 바이너리 데이터 저장을 담당하며, `.gitignore`에 등록하여 Git 추적 대상에서 제외함.
  * 데이터 파이프라인 재현을 위해 가공 스크립트(`research/prepare_kline_data.py` 등)를 자동화하여 관리함.

---

### 체크리스트 (Checklist for Developers & AI Agents)
- [ ] 데이터프레임 조작 시 `import polars as pl`을 표준으로 사용하는가?
- [ ] 파일 입출력 시 `to_parquet()` / `read_parquet()` / `scan_parquet()`을 사용하는가?
- [ ] 분석 DB 조회가 필요한 경우 `import duckdb`로 쿼리하는가?
- [ ] 캔들 데이터에 `open_time` 및 `close_time` 컬럼이 명시되어 있는가?
- [ ] `data/` 디렉토리에 데이터가 생성되고 저장되는가?
