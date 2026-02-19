# 뉴스 모니터링 시스템

네이버 뉴스 API로 호텔 브랜드 관련 기사를 수집하고, OpenAI 기반 LLM으로 분류/검증한 뒤 Google Sheets에 저장하는 파이프라인입니다.

![architecture](resources/news_monitoring_architecture.png)

## 개요

- **핵심 흐름**: Collection -> Processing -> LLM Classification -> Reporting -> Logger Flush
- **분석 방식**: LLM-only
- **저장소**: Google Sheets
- **예외 모드**: Sheets 미연결 시 `raw.csv`만 비상 저장 후 파이프라인 종료

## 주요 기능

- 네이버 API 페이지네이션 수집 (`--max_api_pages`)
- 보도자료/유사주제 탐지 (TF-IDF + 코사인 + Jaccard)
- OpenAI 기반 언론사 분류 (도메인 -> 언론사명/그룹/유형)
- LLM 구조화 분류 (brand/sentiment/danger/category/keyword)
- 소스 검증 및 토픽 그룹핑 (LLM 경계선 검증 포함)
- 실행 메트릭 로깅 (`run_history`, `errors`, `events`)

## 빠른 시작

### 1) 의존성 설치

```bash
pip install -r requirements.txt
```

### 2) `.env` 설정

```bash
# Required
NAVER_CLIENT_ID=your_id
NAVER_CLIENT_SECRET=your_secret
OPENAI_API_KEY=sk-your_key

# Recommended (primary data store)
GOOGLE_SHEETS_CREDENTIALS_PATH=.secrets/service-account.json
GOOGLE_SHEET_ID=your_sheet_id
```

### 3) 실행

```bash
# 기본 실행 (브랜드당 100건)
python main.py

# 권장: 페이지네이션 수집 (최대 9페이지)
python main.py --max_api_pages 9

# 테스트 실행 (LLM 비용 없음)
python main.py --dry_run --display 10
```

## 파이프라인 구조

### STEP 1: Collection

- 네이버 API 수집
- 기존 `raw_data` 링크를 읽어 중복 스킵
- 신규 기사 즉시 `raw_data` 탭 동기화
- `--recheck_only` 시 API를 건너뛰고 Sheets의 `raw_data` 로드

### STEP 1.5a: Sheets Deduplication

- `raw_data`, `total_result` 탭에서 `link` 기준 중복 행 제거

### STEP 1.5: Reprocess Check

- `total_result` 누락/불완전 데이터 재처리 대상 탐지
- 필드 점검: `brand_relevance`, `sentiment_stage`, `source`, `media_domain`, `date_only`

### STEP 1.6: Date Filtering

- `2026-02-01` 이후 기사만 처리 (코드 하드코딩)

### STEP 2: Processing

- 정규화 (HTML 제거, 날짜 변환, ID 생성)
- 링크 중복 제거 + 쿼리 병합
- 보도자료 클러스터 탐지/요약
- 언론사 도메인 분류

### STEP 3: LLM Classification & Source Verification

- 보도자료 대표 기사 기반 클러스터 분류 후 결과 전파
- 일반 기사 LLM 분류
- 소스 검증 + 교차 쿼리 클러스터 병합 + 토픽 그룹핑

### STEP 4: Reporting

- 콘솔 요약 리포트
- 카테고리별 키워드 추출 후 `keywords` 탭 저장

### STEP 5: Logger Flush

- `run_history`, `errors`, `events` 탭 동기화

## 프로젝트 구조

```text
main.py                  # 5-step 파이프라인 진입점
src/cli.py               # CLI 파싱 + run_mode 판별

src/pipeline/
├── context.py
├── setup.py
├── step1_collection.py
├── step2_processing.py
├── step3_classification.py
├── step4_reporting.py
├── step5_sheets_sync.py
└── finalize.py

src/modules/
├── collection/collect.py
├── processing/
│   ├── process.py
│   ├── press_release_detector.py
│   ├── media_classify.py
│   ├── reprocess_checker.py
│   └── looker_prep.py
├── analysis/
│   ├── classify_llm.py
│   ├── classify_press_releases.py
│   ├── llm_engine.py
│   ├── llm_orchestrator.py
│   ├── source_verifier.py
│   ├── result_writer.py
│   ├── classification_stats.py
│   ├── keyword_extractor.py
│   ├── prompts.yaml
│   └── source_verifier_prompts.yaml
├── monitoring/logger.py
└── export/
    ├── report.py
    └── sheets.py

src/utils/
├── openai_client.py
├── sheets_helpers.py
└── text_cleaning.py
```

## 주요 CLI 옵션

```bash
python main.py --max_api_pages 9
python main.py --raw_only
python main.py --preprocess_only
python main.py --recheck_only
python main.py --recheck_only --dry_run
python main.py --chunk_size 50
python main.py --max_workers 5
python main.py --max_competitor_classify 20
python main.py --keyword_top_k 30
python main.py --clean_bom
python main.py --outdir reports
```

## 데이터 저장 구조 (Google Sheets)

- `raw_data`: 원본 수집 데이터
- `total_result`: LLM 분류 + 처리 컬럼 포함 결과
- `run_history`: 실행 메트릭 (고정 스키마)
- `errors`: ERROR 로그
- `events`: INFO 로그
- `keywords`: 카테고리별 키워드

## 핵심 컬럼

- 수집: `title`, `description`, `link`, `originallink`, `pubDate`, `query`, `group`
- 처리: `pub_datetime`, `article_id`, `article_no`, `source`, `cluster_id`, `cluster_summary`, `media_domain`, `media_name`, `media_group`, `media_type`
- 분류: `brand_relevance`, `brand_relevance_query_keywords`, `sentiment_stage`, `danger_level`, `issue_category`, `news_category`, `news_keyword_summary`, `classified_at`

## 문제 해결

- `401 Unauthorized`: `.env` API 키 확인
- 분류 타임아웃: `--chunk_size 50` 또는 `30`
- Sheets 동기화 실패: 인증 JSON 경로/Sheet ID/권한 확인
- BOM 문자 정리: `python main.py --clean_bom`

## 테스트

```bash
python main.py --dry_run --display 10
python main.py --display 10 --raw_only
python main.py --display 20 --chunk_size 5 --max_workers 1
pytest tests/test_press_release_detector.py
pytest tests/test_reprocess_checker.py
pytest tests/test_sheets_sync.py
```

## 참고

- 현재 아키텍처의 기준 문서는 `CLAUDE.md`입니다.
- `README.md`는 사용자 실행/운영 관점으로 요약되어 있습니다.
