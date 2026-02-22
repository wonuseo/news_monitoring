# 뉴스 모니터링 시스템

네이버 뉴스 API로 호텔 브랜드 관련 기사를 수집하고, OpenAI 기반 LLM으로 분류/검증한 뒤 Google Sheets에 저장하는 파이프라인입니다.

![architecture_dev](/Users/wonuseo/PycharmProjects/news_monitoring/resources/news_monitoring_architecture_0222A.PNG)
![architecture_users](/Users/wonuseo/PycharmProjects/news_monitoring/resources/news_monitoring_architecture_0222B.jpeg)

## 개요

- **핵심 흐름**: Collection → Processing → LLM Classification → Reporting → Logger Flush
- **분석 방식**: Two-pass LLM (일반 분류 + 부정기사 심층 분석)
- **저장소**: Google Sheets (sole data store)
- **예외 모드**: Sheets 미연결 시 `raw.csv`만 비상 저장 후 파이프라인 종료

## 주요 기능

- 네이버 API 페이지네이션 수집 (`--max_api_pages`, 브랜드당 최대 900건)
- 보도자료/유사주제 탐지 (TF-IDF + 코사인 + Jaccard)
- OpenAI 기반 언론사 분류 (도메인 → 언론사명/그룹/유형)
- **Two-pass LLM 분류**
  - Pass 1: brand_relevance, sentiment_stage, news_category, news_keyword_summary (`config/prompts.yaml`)
  - Pass 2: 부정 후보/부정 확정 기사만 → danger_level, issue_category chain-of-thought (`config/negative_prompts.yaml`)
- LLM 추론 과정 reasoning 탭 저장
- 소스 검증 및 토픽 그룹핑 (LLM 경계선 검증 포함)
- 카테고리 발견 — `issue_category="기타"` 기사 분석 → 신규 카테고리 후보 제안
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
# 권장: 페이지네이션 수집 (최대 9페이지)
python main.py --max_api_pages 9

# 테스트 실행 (LLM 비용 없음)
python main.py --dry_run --display 10

# 수집만 (분류 건너뜀)
python main.py --raw_only

# 수집 + 전처리만 (LLM 분류 건너뜀)
python main.py --preprocess_only

# Sheets 누락/불완전 기사 재처리
python main.py --recheck_only
```

## 파이프라인 구조

### STEP 1: Collection

- 네이버 API 수집 (브랜드당 최대 9페이지 × 100건)
- 기존 `raw_data` 링크를 읽어 중복 스킵
- 신규 기사 즉시 `raw_data` 탭 동기화
- `--recheck_only` 시 API를 건너뛰고 Sheets의 `raw_data` 로드

### STEP 1.5a: Sheets Deduplication

- `raw_data`, `total_result` 탭에서 `link` 기준 중복 행 제거

### STEP 1.5: Reprocess Check

- `total_result` 누락/불완전 데이터 재처리 대상 탐지
- 필드 점검: `brand_relevance`, `sentiment_stage`, `source`, `media_domain`, `date_only`

### STEP 1.6: Date Filtering

- `config/pipeline.yaml`의 `date_filter_start` 이후 기사만 처리 (매월 업데이트)

### STEP 2: Processing

- 정규화 (HTML 제거, ISO 날짜 변환, article_id MD5 해시, article_no 순번)
- 링크 중복 제거 + 쿼리 병합 (pipe-separated)
- 보도자료 클러스터 탐지 (TF-IDF + Jaccard, 임계값: `config/thresholds.yaml`)
- OpenAI 클러스터 요약 (cluster_summary, 3단어)
- 언론사 도메인 분류 (batch API)

### STEP 3: LLM Classification & Source Verification

- **3-1**: 보도자료 클러스터 — 대표 기사 1건 LLM 분류 → 클러스터 전체 전파
- **3-2**: 일반 기사 — 병렬 chunked LLM (`ThreadPoolExecutor`)
  - Pass 1 (`prompts.yaml`): brand_relevance, sentiment_stage, news_category, news_keyword_summary
  - Pass 2 (`negative_prompts.yaml`): 부정 기사만 → danger_level, issue_category (chain-of-thought)
  - 추론 과정은 `reasoning` 탭에 저장 (`reasoning_writer.py`)
  - 각 chunk 완료 후 incremental Sheets 동기화
- **3-3**: 소스 검증
  - Part A: 클러스터별 LLM 검증 → 보도자료 / 유사주제
  - Part A-2: 교차 쿼리 클러스터 병합 (TF-IDF cosine + Jaccard)
  - Part B: 비클러스터 기사 중 토픽 그룹 발견

### STEP 4: Reporting

- 콘솔 요약 리포트
- **STEP 4.5**: 카테고리별 키워드 추출 (kiwipiepy + Log-odds → `keywords` 탭)
- **STEP 4.6**: 카테고리 발견 — `issue_category="기타"` 기사 분석 → 신규 카테고리 후보 제안 (콘솔 + `events` 탭)

### STEP 5: Logger Flush

- `run_history`, `errors`, `events` 탭 동기화

## 설정 파일 (`config/`)

모든 운영 설정은 `config/` 폴더에 집중. Python 코드 수정 없이 운영 가능.

| 파일 | 용도 |
|------|------|
| `brands.yaml` | 모니터링 브랜드 목록 (our_brands, competitors) |
| `pipeline.yaml` | 수집 기간 필터 (`date_filter_start`) — 매월 업데이트 |
| `thresholds.yaml` | 유사도 임계값 (보도자료 탐지, cross-query 병합, 주제 그룹화) |
| `models.yaml` | 기능별 OpenAI 모델 선택 |
| `prompts.yaml` | LLM 1차 분류 기준 + 카테고리 목록 + few-shot 예시 |
| `negative_prompts.yaml` | 부정기사 2차 분석 프롬프트 (chain-of-thought) |
| `source_verifier_prompts.yaml` | 클러스터 검증 + 주제 유사도 판단 프롬프트 |

## 주요 CLI 옵션

```bash
python main.py --max_api_pages 9          # 수집 페이지 수 (기본: 9)
python main.py --raw_only                  # 수집만
python main.py --preprocess_only           # 수집 + 전처리만
python main.py --recheck_only              # Sheets 누락 기사 재처리
python main.py --recheck_only --dry_run    # 재처리 대상 확인 (LLM 비용 없음)
python main.py --dry_run --display 10      # 테스트 실행
python main.py --chunk_size 50             # LLM 배치 크기 (기본: 100)
python main.py --max_workers 5             # 병렬 스레드 수 (기본: 3)
python main.py --max_competitor_classify 20 # 경쟁사 분류 상한
python main.py --sort sim                  # 정렬: date(기본) / sim
python main.py --keyword_top_k 30          # 키워드 추출 수 (기본: 20)
python main.py --outdir reports            # 출력 디렉토리 (기본: data)
python main.py --sheets_id SHEET_ID        # Sheet ID 오버라이드
python main.py --clean_bom                 # BOM 문자 정리
```

## 데이터 저장 구조 (Google Sheets)

| 탭 | 용도 |
|----|------|
| `raw_data` | 원본 수집 데이터 |
| `total_result` | LLM 분류 + 처리 컬럼 포함 최종 결과 |
| `reasoning` | LLM 추론 과정 (brand_role, sentiment_rationale, severity_analysis 등 11 컬럼) |
| `run_history` | 실행 메트릭 (고정 스키마, 55 컬럼) |
| `errors` | ERROR 로그 |
| `events` | INFO 로그 + 카테고리 발견 제안 |
| `keywords` | 카테고리별 키워드 |

## 핵심 컬럼

- **수집**: `title`, `description`, `link`, `originallink`, `pubDate`, `query`, `group`
- **처리**: `pub_datetime`, `article_id`, `article_no`, `source`, `cluster_id`, `cluster_summary`, `media_domain`, `media_name`, `media_group`, `media_type`
- **분류**: `brand_relevance`, `sentiment_stage`, `danger_level`, `issue_category`, `news_category`, `news_keyword_summary`, `classified_at`

## 문제 해결

| 문제 | 해결 |
|------|------|
| `401 Unauthorized` | `.env` API 키 확인 |
| 분류 타임아웃 | `--chunk_size 30` |
| Rate limit (429) | 자동 재시도 내장; `--chunk_size` 줄이기 |
| Sheets 동기화 실패 | 인증 JSON 경로/Sheet ID/권한 확인 |
| BOM 문자 | `python main.py --clean_bom` |
| 카테고리 발견 미실행 | `issue_category="기타"` 기사 3건 이상 필요 |
| 잘못된 분류 재처리 | Sheets에서 `brand_relevance` 비우기 → `--recheck_only` |

## 테스트

```bash
# pytest (별도 설치 필요)
pytest tests/test_press_release_detector.py
pytest tests/test_reprocess_checker.py
pytest tests/test_sheets_sync.py

# LLM 품질 테스트 (실제 OpenAI API 호출 — 비용 발생)
python tests/test_llm_quality.py
```

## 참고

- 아키텍처 기준 문서: `CLAUDE.md`
- 카테고리 추가: `config/prompts.yaml`의 `labels` 섹션에 추가 → 프로세스 재시작
- `config/pipeline.yaml`의 `date_filter_start`를 매월 업데이트
