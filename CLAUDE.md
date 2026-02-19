# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**News Monitoring System** — Automated pipeline for hotel brand news monitoring. Collects articles from Naver API, performs LLM-based analysis (sentiment/risk/categorization), and stores everything in Google Sheets as the sole data store.

Brands monitored are defined in `config/brands.yaml` (`our_brands`, `competitors`).

## Common Commands

```bash
# Full run (recommended)
python main.py --max_api_pages 9

# Quick test without AI costs
python main.py --dry_run --display 10

# Collection only (no classification)
python main.py --raw_only

# Reprocess missing/incomplete articles from Sheets (no new API collection)
python main.py --recheck_only

# Reprocess inspection without LLM cost
python main.py --recheck_only --dry_run

# Debug classification
python main.py --display 20 --chunk_size 5 --max_workers 1

# BOM cleanup in Sheets
python main.py --clean_bom
```

Tests (pytest must be installed manually — not in current venv):
```bash
python -m pytest tests/test_press_release_detector.py
python -m pytest tests/test_reprocess_checker.py
```

## Pipeline Architecture

`main.py` is a thin orchestrator. All logic lives in `src/pipeline/step*.py`. State is shared via `PipelineContext` (`ctx`), a dataclass defined in `src/pipeline/context.py`.

```
STEP 1: Collection
├─ Naver API pagination (9 pages × 100 = 900 articles/brand)
├─ Load existing links from Sheets → skip duplicates
├─ Sync new articles to Sheets raw_data tab
├─ STEP 1.5a: Dedup raw_data + total_result tabs (by link)
└─ STEP 1.5: Reprocess check — find articles missing from total_result
              OR with empty fields (brand_relevance, sentiment_stage,
              source, media_domain, date_only)
   ⚠️  Only targets MISSING/EMPTY fields. Articles with wrong
       classifications are NOT flagged. To force re-classify: clear
       brand_relevance (or any REPROCESS_RULES field) in Sheets first.

STEP 2: Processing
├─ Normalize (HTML strip, ISO dates, article_id MD5-hash, article_no)
├─ Deduplicate by link (query field pipe-merged: "롯데호텔|호텔롯데")
├─ TF-IDF + Jaccard similarity → detect press releases (thresholds in config/thresholds.yaml)
├─ OpenAI cluster summarization (cluster_summary, 3-word)
└─ Media classification (domain → media_name/group/type, batch API call)

STEP 3: LLM Classification
├─ 3-1: Press release clusters — 1 representative → LLM → share across cluster
├─ 3-2: General articles — parallel chunked LLM (ThreadPoolExecutor)
│        → brand_relevance, sentiment_stage, danger_level,
│           issue_category, news_category, news_keyword_summary
│        → incremental Sheets sync after each chunk
└─ 3-3: Source verification
        Part A:  LLM verifies each cluster → 보도자료 / 유사주제
        Part A-2: Cross-query cluster merge (TF-IDF cosine + Jaccard)
        Part B:  Discover topic groups among unclustered articles

STEP 4: Reporting
├─ Console summary report
├─ STEP 4.5: Keyword extraction (kiwipiepy + Log-odds → Sheets keywords tab)
└─ STEP 4.6: Category discovery — LLM analyzes issue_category="기타" articles
              → suggests new category candidates (console + Sheets events tab)

STEP 5: Logger flush
└─ Append run_history row + errors/events to Sheets (data already synced in Steps 1 & 3)
```

## Module Structure

```
config/                           # ★ All human-editable settings (see Key Configuration)
main.py                           # Thin orchestrator
src/cli.py                        # argparse + run_mode detection
src/pipeline/
├── context.py                    # PipelineContext dataclass (ctx)
├── setup.py                      # Env setup, Sheets connection
├── step1_collection.py
├── step2_processing.py
├── step3_classification.py
├── step4_reporting.py            # Includes STEP 4.6 category discovery
├── step5_sheets_sync.py          # Logger flush only
└── finalize.py

src/modules/analysis/
├── llm_engine.py                 # OpenAI Responses API, build_response_schema() (YAML-driven)
├── classify_llm.py               # Parallel classification orchestrator
├── classify_press_releases.py
├── llm_orchestrator.py           # run_chunked_parallel() via ThreadPoolExecutor
├── source_verifier.py            # Step 3-3: cluster verification + topic grouping
├── category_discovery.py         # STEP 4.6: "기타" pattern analysis → new category suggestions
├── classification_stats.py
├── result_writer.py              # Thread-safe incremental Sheets sync
└── keyword_extractor.py          # kiwipiepy + Log-odds

src/modules/processing/
├── process.py                    # normalize_df(), dedupe_df()
├── press_release_detector.py     # TF-IDF detection + OpenAI summarization
├── media_classify.py             # Batch domain → media info
├── reprocess_checker.py          # REPROCESS_RULES field-level check
└── looker_prep.py

src/modules/export/
├── sheets.py                     # sync_raw_and_processed(), deduplicate_sheet()
└── report.py

src/modules/monitoring/
└── logger.py                     # RUN_HISTORY_SCHEMA (55 cols), _NumpyEncoder for int64 safety

src/utils/
├── config.py                     # load_config("name") → config/<name>.yaml
├── openai_client.py              # Direct HTTP to OpenAI (no SDK), retry/backoff
├── sheets_helpers.py
└── text_cleaning.py

scripts/                          # One-off maintenance scripts (not part of main pipeline)
```

## Key Configuration

### Human-in-the-Loop 설정 파일 (`config/`)

사람이 직접 수정하는 모든 설정은 **`config/`** 폴더에 집중되어 있다. Python 코드 수정 없이 운영 가능.

| 파일 | 용도 |
|------|------|
| `config/brands.yaml` | 모니터링 브랜드 목록 (our_brands, competitors) |
| `config/pipeline.yaml` | 수집 기간 필터 (`date_filter_start`) — 매월 업데이트 |
| `config/thresholds.yaml` | 유사도 임계값 (보도자료 탐지, cross-query 병합, 주제 그룹화) |
| `config/models.yaml` | 기능별 OpenAI 모델 선택 |
| `config/prompts.yaml` | ★ LLM 분류 기준 + 카테고리 목록 + few-shot 예시 |
| `config/source_verifier_prompts.yaml` | 클러스터 검증 + 주제 유사도 판단 프롬프트 |

설정값은 `src/utils/config.py`의 `load_config(name)` 으로 로드된다. 파일이 없으면 하드코딩 fallback으로 동작한다.

### Category Management (YAML-driven)

`config/prompts.yaml` is the **single source of truth** for all categories. Editing it is sufficient — no Python code changes needed.

```yaml
# config/prompts.yaml
labels:
  issue_category_kr:
    - "안전/사고"
    - "위생/식음"
    - "상품/서비스 철수"   # ← add new categories here
  news_category_kr:
    - "PR/보도자료"
    - ...
```

`build_response_schema()` in `llm_engine.py` reads `labels` at first call and caches in memory (`_schema_cache`). The OpenAI structured output enum is built from these lists automatically.

**Adding a new category workflow**:
1. Add to `config/prompts.yaml` `labels` section + add definition in system prompt
2. Optionally add a few-shot example
3. Restart process (cache is per-process)

### Category Discovery (STEP 4.6)

After each run, `category_discovery.py` automatically analyzes `issue_category="기타"` articles (min 3 articles) and suggests new categories via LLM. Results appear on console and in Sheets `events` tab (`category="category_suggestion"`).

### Environment (`.env`)

```bash
NAVER_CLIENT_ID=...
NAVER_CLIENT_SECRET=...
OPENAI_API_KEY=sk-...
GOOGLE_SHEETS_CREDENTIALS_PATH=.secrets/service-account.json
GOOGLE_SHEET_ID=...
```

## Data Columns

**raw_data tab**: `title`, `description`, `link`, `originallink`, `pubDate`, `query` (pipe-separated if multi-brand), `group` (OUR/COMPETITOR)

**total_result tab** (adds after processing + classification):
- `article_id` (MD5 12-char, permanent ID), `article_no` (sequential, human-readable)
- `source`: "보도자료" / "유사주제" / "일반기사" (set by Step 3-3)
- `cluster_id`, `cluster_summary`
- `media_domain`, `media_name`, `media_group`, `media_type`
- `brand_relevance`: "관련" / "언급" / "무관" / "판단 필요"
- `sentiment_stage`: "긍정" / "중립" / "부정 후보" / "부정 확정"
- `danger_level`: "상" / "중" / "하" / null
- `issue_category`: 12 categories or null (see `config/prompts.yaml` labels)
- `news_category`: 13 categories (see `config/prompts.yaml` labels); "비관련" when brand_relevance="무관"
- `news_keyword_summary`: 5-word Korean summary
- `classified_at`: ISO timestamp — **controls whether article is re-classified** (empty = will be processed)

**Sheets tabs**: `raw_data`, `total_result`, `run_history` (55 cols), `errors`, `events`, `keywords`

## Critical Patterns

### Re-classifying Already-Classified Articles

`--recheck_only` only targets articles with **empty** fields (see `REPROCESS_RULES` in `reprocess_checker.py`). To force re-classification of wrongly-classified articles:
1. In Sheets `total_result`, clear `brand_relevance` (or any REPROCESS_RULES field) for target rows
2. Run `python main.py --recheck_only`

### OpenAI Integration

Direct HTTP (no openai SDK) via `src/utils/openai_client.py`. Uses the Responses API format. All structured output calls go through `call_openai_structured()` in `llm_engine.py`.

### Press Release Cluster Classification

One representative article per cluster is LLM-analyzed; results propagate to all cluster members. Reduces API calls significantly. `classified_at` is set for all cluster members after propagation, so they are skipped in the general classification pass.

### Thread Safety

`result_writer.py` uses a threading lock for incremental Sheets sync. `classify_llm.py` runs `ThreadPoolExecutor` with `max_workers` (default 3). Each successful chunk triggers a Sheets sync callback.

### Post-process Rules (`_post_process_result` in `llm_engine.py`)

After every LLM call:
- `brand_relevance="무관"` → `sentiment_stage="중립"`, `danger_level=null`, `issue_category=null`, `news_category="비관련"`
- `danger_level` only valid when `brand_relevance` ∈ {관련, 언급} AND `sentiment_stage` ∈ {부정 후보, 부정 확정}

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Timeout during classification | `--chunk_size 30` |
| Rate limit (429) | Built-in retry; also reduce `--chunk_size` |
| Sheets sync fails | Check `.env` credentials path and Sheet ID |
| BOM characters in Sheets | `python main.py --clean_bom` |
| Category discovery not running | Need ≥3 `issue_category="기타"` articles in df_result |
| Wrong classifications (already classified) | Clear `brand_relevance` in Sheets → `--recheck_only` |

## Notes

- `README.md` is outdated (describes removed hybrid system). This file is authoritative.
- Date filter is set in `config/pipeline.yaml` (`date_filter_start`). Update monthly.
- `logger.py` uses `_NumpyEncoder` (custom `json.JSONEncoder`) to handle pandas `int64`/`float64` in metrics serialization.
- No linting enforced. DataFrame naming convention: `df_raw`, `df_normalized`, `df_processed`, `df_result`.
