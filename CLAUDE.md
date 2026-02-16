# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**News Monitoring System** - Automated pipeline for hotel brand news monitoring that collects articles from Naver API, performs LLM-based analysis for sentiment/risk/categorization, and outputs results to Google Sheets + CSV + Word reports.

**Core Flow**: Collection → Processing → LLM Classification → Reporting → Sheets Sync

**Key Features**:
- LLM-only classification (OpenAI GPT with structured output via prompts.yaml)
- TF-IDF similarity detection for press release grouping
- OpenAI-powered media outlet classification
- Automatic Google Sheets sync (primary data store)
- CSV backups for troubleshooting

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure .env with API keys (required)
# NAVER_CLIENT_ID, NAVER_CLIENT_SECRET, OPENAI_API_KEY
# GOOGLE_SHEETS_CREDENTIALS_PATH, GOOGLE_SHEET_ID (recommended)

# Run full pipeline (default: 100 articles/brand)
python main.py

# Recommended: API pagination (Sheets sync is automatic if configured)
python main.py --max_api_pages 9

# Test run without AI costs
python main.py --dry_run --display 10
```

## Architecture

Linear 5-step pipeline:

```
STEP 1: Collection
├─ Naver API (pagination: 9 pages × 100 = 900 articles/brand)
├─ Load existing links from Google Sheets (skip duplicates)
├─ Save to raw.csv + append to Sheets
└─ --recheck_only: Skip API, load raw_data from Sheets

STEP 1.5: Reprocess Check
├─ Load total_result (Sheets → CSV fallback)
├─ Rule 1: raw links missing from total_result
├─ Rules 2-5: Field-level empty check (brand_relevance, sentiment_stage, source, media_domain, date_only)
├─ Union all targets → clear classified_at → merge with new articles
└─ Module: reprocess_checker.py

STEP 1.6: Date Filtering
└─ Filter articles to 2026-02-01+ only (hard-coded in main.py and sheets.py)

STEP 2: Processing
├─ Normalize (HTML strip, ISO dates, article_id, article_no)
├─ Deduplicate by link
├─ TF-IDF similarity (detect press releases, cosine ≥ 0.8)
├─ OpenAI group summarization (press_release_group)
└─ Media classification (domain → name/group/type via OpenAI batch)

STEP 3: LLM Classification & Source Verification
├─ Step 3-1: Press release LLM classification (representative per cluster → share results)
│   └─ Module: classify_press_releases.py
├─ Step 3-2: General LLM classification (non-press-release articles)
│   └─ LLM Analysis (OpenAI gpt-4o-mini, 우리 브랜드 all + 경쟁사 all)
│   └─ Outputs: brand_relevance, sentiment_stage, danger_level,
│       issue_category, news_category, news_keyword_summary
├─ Step 3-3: Source verification & topic grouping (LLM cluster verification)
│   ├─ Part A: Verify press release clusters via LLM (1 API call/cluster)
│   │   └─ LLM judges 보도자료/유사주제 per cluster; rule-based fallback on failure
│   ├─ Part A-2: Cross-query cluster merge (TF-IDF cosine + Jaccard, no date constraint)
│   │   └─ Merges clusters/articles across different queries; LLM borderline verification
│   ├─ Part B: Discover topic groups among unclustered articles
│   │   └─ Jaccard similarity + news_category match + LLM borderline verification
│   ├─ Prompts: source_verifier_prompts.yaml (external, editable)
│   └─ Module: source_verifier.py
└─ Structured output via prompts.yaml (no retraining needed)

STEP 4: Reporting
├─ Console report (summary statistics)
└─ Keyword extraction (automatic, saved to Google Sheets + CSV)

STEP 5: Sheets Sync
└─ Incremental upload to 2 tabs (raw_data, result)
```

## Module Structure

```
src/modules/
├── collection/
│   └── collect.py       # Naver API pagination
├── processing/
│   ├── process.py       # Normalize, dedupe, CSV I/O
│   ├── press_release_detector.py # Press release detection & summarization
│   ├── media_classify.py # OpenAI media outlet classification
│   ├── reprocess_checker.py # Reprocess target detection (missing/incomplete fields)
│   └── looker_prep.py   # Time-series columns (Looker Studio)
├── analysis/
│   ├── classify_llm.py  # LLM classification orchestrator (parallel processing)
│   ├── classify_press_releases.py # Press release cluster LLM classification
│   ├── llm_engine.py    # OpenAI Responses API engine (structured output)
│   ├── llm_orchestrator.py # Parallel chunked task runner (ThreadPoolExecutor)
│   ├── classification_stats.py # Statistics generation & reporting
│   ├── result_writer.py # CSV/Sheets incremental saving (thread-safe)
│   ├── keyword_extractor.py # Category-specific keyword extraction (kiwipiepy + Log-odds)
│   ├── source_verifier.py # Source verification & topic grouping (LLM cluster verification)
│   ├── source_verifier_prompts.yaml # Source verification prompts (cluster_verification + topic_similarity)
│   └── prompts.yaml     # LLM prompts & schemas
├── monitoring/
│   └── logger.py        # Fixed-schema run metrics (55 cols, 3-sheet: run_history/errors/events)
└── export/
    ├── report.py        # CSV + Word generation
    └── sheets.py        # Google Sheets sync

src/utils/
├── openai_client.py     # OpenAI API wrapper with retry/backoff (direct HTTP, no SDK)
├── sheets_helpers.py    # Sheet creation & intermediate sync helpers
└── text_cleaning.py     # BOM/invisible character cleaning
```

**Note**: System uses LLM-only classification. Legacy files (`classify.py`, `hybrid.py`, `rule_engine.py`, `rules.yaml`, `preset_pr.py`) have been removed.

## Key Configuration Files

**Brand Definitions** (`src/modules/collection/collect.py:14-15`):
```python
OUR_BRANDS = ["롯데호텔", "호텔롯데", "L7", "시그니엘"]
COMPETITORS = ["신라호텔", "조선호텔"]
```

**LLM Prompts** (`src/modules/analysis/prompts.yaml`):
- System prompt for classification
- Output schema (JSON)
- Decision rules for sentiment, danger, issue categories
- No retraining needed - edit YAML to change logic

**Source Verifier Prompts** (`src/modules/analysis/source_verifier_prompts.yaml`):
- cluster_verification: LLM 보도자료/유사주제 클러스터 판단
- topic_similarity: LLM 경계선 주제 유사도 판단

**API Model Config** (`src/api_models.yaml`):
- Per-task model selection (article_classification, media_classification, press_release_summary, source_verification)
- All default to gpt-4o-mini

**Environment** (`.env`):
```bash
# Required
NAVER_CLIENT_ID=your_id
NAVER_CLIENT_SECRET=your_secret
OPENAI_API_KEY=sk-your_key

# Recommended (primary data store)
GOOGLE_SHEETS_CREDENTIALS_PATH=.secrets/service-account.json
GOOGLE_SHEET_ID=your_sheet_id
```

## Common Commands

```bash
# Standard run (API only, 100 articles/brand)
python main.py

# With pagination (900 articles/brand, recommended)
python main.py --max_api_pages 9

# Raw collection only (no classification/reporting, auto-syncs to Sheets)
python main.py --raw_only

# Preprocess only (no classification, but with Sheets sync)
python main.py --preprocess_only

# Reduce chunk size if timeouts occur
python main.py --chunk_size 50

# Limit competitor analysis (default is unlimited)
python main.py --max_competitor_classify 20

# Adjust keyword extraction count (default: 20, auto-runs every time)
python main.py --keyword_top_k 30

# BOM cleanup in Google Sheets
python main.py --clean_bom

# Custom output directory
python main.py --outdir reports

# Parallel processing (adjust workers, default: 3)
python main.py --max_workers 5

# Recheck only (no API collection, reprocess missing/incomplete from Sheets)
python main.py --recheck_only

# Recheck only + dry run (inspect targets without LLM cost)
python main.py --recheck_only --dry_run
```

## Data Flow & Key Columns

**Collection (raw.csv)**:
- From API: `title`, `description`, `link`, `originallink`, `pubDate`
- Added: `query` (brand, 복수 브랜드 수집 시 파이프 구분 예: "롯데호텔|호텔롯데"), `group` (OUR/COMPETITOR)

**Processing (intermediate)**:
- `pub_datetime` (ISO 8601)
- `article_id` (MD5 hash, 12-char, 영구 식별자, 시스템용)
- `article_no` (순차 번호, 사람이 읽는 번호, 검토용)
- `source` ("보도자료" / "유사주제" / "일반기사" — verified by LLM results in Step 3-3)
- `cluster_id` (press release cluster ID)
- `press_release_group` (OpenAI 3-word summary)
- `media_domain`, `media_name`, `media_group`, `media_type`

**LLM Classification (result.csv)**:
- `brand_relevance`: "관련" / "언급" / "무관" / "판단 필요"
- `brand_relevance_query_keywords`: [array of keywords]
- `sentiment_stage`: "긍정" / "중립" / "부정 후보" / "부정 확정"
- `danger_level`: "상" / "중" / "하" / null (only if brand_relevance + negative)
- `issue_category`: One of 11 Korean categories (안전/사고, 법무/규제, etc.) or null
- `news_category`: One of 13 Korean categories (PR/보도자료, 사업/실적, 브랜드/마케팅, 상품/오퍼링, 제휴/파트너십, 이벤트/프로모션, 시설/오픈, 고객 경험, 운영/기술, 인사/조직, 리스크/위기, ESG/사회, 기타)
- `news_keyword_summary`: 5-word Korean summary
- `classified_at`: ISO timestamp

**Google Sheets** (primary data store):
- `raw_data` tab: Raw collected articles
- `total_result` tab: Classified articles with all LLM columns
- `run_history` tab: Fixed-schema run metrics (55 columns)
- `errors` tab: ERROR-level logs only
- `events` tab: INFO-level logs only
- `keywords` tab: Category-specific keywords (automatic)
- Incremental append (deduplicates by link)

## Critical Implementation Details

### Press Release Detection & Summarization
- **Detection**: Character-level n-grams (3-5) TF-IDF + Cosine + Jaccard similarity
- **Date-based rules**: Δdays ≤ 3 (standard) vs Δdays ≥ 4 (super-similar)
- **Clustering**: Query-specific BFS Connected Components
- **Summarization**: OpenAI 3-word summaries per cluster
- **Non-destructive**: Preserves all articles, labels as "보도자료"
- **Module**: `press_release_detector.py` (all-in-one: detection + summarization)

### Deduplication with Query Merging
- Duplicate links merged by `originallink` or `link`
- Query field merged with pipe separator (e.g., "롯데호텔|호텔롯데")
- Preserves multi-brand collection information
- Function: `dedupe_df()` in `process.py`

### Media Classification (OpenAI Batch)
- Extracts domains from URLs
- Batch classifies unknown domains (1 API call for all)
- Persistent CSV directory (`media_directory.csv`)
- Cost: ~$0.001 per 100 domains

### LLM Classification Strategy
- **Model**: OpenAI gpt-4o-mini (as configured in prompts.yaml and api_models.yaml)
- **Full analysis by default**: OUR_BRANDS (all) + COMPETITORS (all)
  - Use `--max_competitor_classify N` to limit competitor analysis
- **Press releases**: Cluster-based LLM classification via `classify_press_releases.py`
  - Selects 1 representative article per cluster → LLM analyzes → shares result across cluster
  - Significantly reduces API calls vs individual classification
- **Chunking**: Default 100 articles/chunk (reduce for timeouts)
- **Parallel**: ThreadPoolExecutor via `llm_orchestrator.py` with max_workers (default 3)
- **Incremental save**: Appends to result.csv after each successful chunk
- **Configuration**: All prompts and rules in `prompts.yaml` - no code changes needed
- **OpenAI integration**: Direct HTTP via `src/utils/openai_client.py` (no openai SDK)

### Keyword Extraction (Automatic)
- **Method**: kiwipiepy (Korean morphological analysis) + Log-odds ratio with Laplace smoothing
- **POS tags**: NNG (common noun), NNP (proper noun), VV (verb), VA (adjective)
- **Statistics**: Log-odds ratio = log(P(word|category)) - log(P(word|other categories))
- **Output**: Top-K keywords per category (default: 20)
- **Categories**: sentiment_stage, danger_level, issue_category, news_category, brand_relevance
- **Usage**: Runs automatically after classification, adjust count with `--keyword_top_k 30`
- **Storage**:
  - Primary: Google Sheets `keywords` tab
  - Backup: `data/keywords/keywords_{category}.csv`

### Google Sheets Integration
- **Primary data store** (CSV = backup for troubleshooting)
- **Automatic sync**: No `--sheets` flag needed - syncs automatically if credentials configured
- **Incremental collection**: Loads existing links at start, skips duplicates
- **Sync function**: `sync_raw_and_processed()` in `sheets.py`
- **Graceful fallback**: Continues with CSV-only mode if credentials missing
- **Configuration**: Set `GOOGLE_SHEETS_CREDENTIALS_PATH` and `GOOGLE_SHEET_ID` in `.env`

### Error Handling
- All steps non-blocking (pipeline always completes)
- API failures: retry with exponential backoff
- Sheets failures: continues with CSV-only mode
- LLM failures: records error, continues with remaining articles

## Logging System

Fixed-schema 3-sheet logging system. Tracks all pipeline metrics per run.

**3-Sheet Structure** (Google Sheets):
- `run_history`: Fixed 55-column metrics per run (full replace on sync)
- `errors`: ERROR-level logs with stage context (append)
- `events`: INFO-level logs with stage context (append)

**Tracked Metrics** (55 columns in `RUN_HISTORY_SCHEMA`):
- **Basic**: run_id, timestamp, run_mode, cli_args
- **Collection**: articles_collected_total, articles_collected_per_query, existing_links_skipped, duration_collection
- **Reprocess**: reprocess_targets_total, reprocess_missing_from_result, reprocess_field_missing
- **Processing**: articles_processed, duplicates_removed, articles_filtered_by_date, press_releases_detected, press_release_groups, press_release_avg_cluster_size, media_domains_total, media_domains_new, media_domains_cached, duration_processing
- **PR Classification**: pr_clusters_analyzed, pr_articles_propagated, pr_llm_success, pr_llm_failed, pr_cost_estimated, duration_pr_classification
- **General Classification**: articles_classified_llm, llm_api_calls, classification_errors, press_releases_skipped, llm_cost_estimated, duration_general_classification
- **Source Verification**: sv_clusters_verified, sv_kept_press_release, sv_reclassified_similar_topic, sv_cross_merged_groups, sv_cross_merged_articles, sv_new_topic_groups, sv_new_topic_articles, duration_source_verification
- **Results**: our_brands_relevant, our_brands_negative, danger_high, danger_medium, competitor_articles, total_result_count
- **Sheets Sync**: sheets_sync_enabled, sheets_rows_uploaded_raw, sheets_rows_uploaded_result, duration_sheets_sync
- **Errors/Total**: errors_total, duration_total

**Error/Event Schema** (6 columns): run_id, timestamp, category, stage, message, data_json

**Log Storage**:
- CSV: `data/logs/run_history.csv` (append mode, auto-backup on schema mismatch)
- Google Sheets: `run_history` tab (full replace), `errors`/`events` tabs (append)

**Usage**:
- Automatic: No configuration needed, logs saved after every run
- Monitor costs: Check `llm_cost_estimated` column (USD)
- Track performance: Compare `duration_*` columns across runs (6 stages)

## Output Files

**Location**: `data/` directory (or `--outdir`)

- `raw.csv` - Raw API collection (UTF-8 BOM, troubleshooting backup)
- `result.csv` - LLM classified results (UTF-8 BOM, troubleshooting backup)
- `media_directory.csv` - Media outlet directory (persistent)
- `keywords/` - Category-specific keyword CSV files (automatic, troubleshooting backup)
- `logs/run_history.csv` - Run metrics history (persistent, append mode)

**Google Sheets** (primary data store, if configured):
- `raw_data` tab: Raw collected articles
- `total_result` tab: Classified articles with all LLM columns
- `run_history` tab: Fixed-schema run metrics (55 columns)
- `errors` tab: ERROR-level logs with stage context
- `events` tab: INFO-level logs with stage context
- `keywords` tab: Category-specific keywords (automatic)
- Auto-deduplication by link
- CSV files are backups for troubleshooting

## Debugging & Troubleshooting

**Common Issues**:

| Problem | Solution |
|---------|----------|
| `401 Unauthorized` | Check API keys in `.env` |
| No articles collected | Verify brand names (Korean), try `--display 200` |
| Timeout during classification | Reduce `--chunk_size` (try 50 or 30) |
| Rate limit (429) | Built-in retry; reduce `--chunk_size` if persistent |
| Sheets sync fails | Check credentials path and Sheet ID in `.env` |
| BOM characters in Sheets | Run `python main.py --clean_bom` |

**Test Commands**:
```bash
# Quick test (10 articles, no AI)
python main.py --dry_run --display 10

# Test Sheets integration (auto-syncs if credentials configured)
python main.py --display 10 --raw_only

# Debug classification
python main.py --display 20 --chunk_size 5 --max_workers 1

# Recheck only (inspect reprocess targets without API collection)
python main.py --recheck_only --dry_run
```

**Reprocess Check** (STEP 1.5, `reprocess_checker.py`):
- Loads total_result from Sheets (fallback: result.csv)
- Rule 1: Finds raw links missing from total_result
- Rules 2-5: Checks field-level completeness (brand_relevance, sentiment_stage, source, media_domain, date_only)
- Merges all targets, clears classified_at, combines with new articles
- `--recheck_only`: Skips API collection, loads raw_data from Sheets, runs full pipeline on targets

## Development Notes

**Code Style**:
- DataFrame naming: `df_raw`, `df_normalized`, `df_processed`, `df_result`
- Use f-strings for formatting
- Keep functions under 50 lines
- No strict linting enforced

**Testing Strategy**:
- `--dry_run` for pipeline testing without AI costs
- `--raw_only` for collection testing
- `--display 10` for small dataset tests
- `--chunk_size 5` for debugging classification
- Tests: `tests/test_llm_quality.py`, `tests/test_press_release_detector.py`, `tests/test_reprocess_checker.py`

**Important Functions**:
- Collection: `collect_all_news()` in `collect.py`
- Processing: `normalize_df()`, `dedupe_df()`, `detect_similar_articles()` in `process.py`
- Classification:
  - `classify_llm()` in `classify_llm.py` - Main orchestrator with parallel processing
  - `classify_press_releases()` in `classify_press_releases.py` - Cluster-based press release classification
  - `analyze_article_llm()` in `llm_engine.py` - Single article LLM analysis
  - `run_chunked_parallel()` in `llm_orchestrator.py` - Parallel chunked task runner
  - `save_result_to_csv_incremental()` in `result_writer.py` - Thread-safe CSV saving
  - `get_classification_stats()` in `classification_stats.py` - Statistics generation
- Source Verification: `verify_and_regroup_sources()`, `verify_press_release_clusters()`, `llm_verify_cluster()`, `merge_cross_query_clusters()`, `discover_topic_groups()`, `llm_verify_topic_similarity()` in `source_verifier.py`
- Source Verifier Engine: `load_source_verifier_prompts()`, `render_prompt()`, `call_openai_structured()` in `llm_engine.py`
- Reprocess: `check_reprocess_targets()`, `load_raw_data_from_sheets()`, `clear_classified_at_for_targets()` in `reprocess_checker.py`
- Reporting: `generate_console_report()` in `report.py`
- Sheets: `sync_raw_and_processed()` in `sheets.py`, `sync_result_to_sheets()` in `result_writer.py`

**Documentation Notes**:
- ⚠️ README.md is outdated - describes removed hybrid system (rules.yaml, hybrid.py)
- ✅ CLAUDE.md (this file) reflects current LLM-only architecture
- Use this file (CLAUDE.md) as the authoritative reference for architecture

## Architecture Changes (Recent)

**Phase 9 (CSV Migration)**:
- Migrated from Excel to CSV (8.3x faster I/O)
- Replaced TF-IDF keywords with OpenAI group summaries
- Added CSV-based media directory with Sheets sync

**Phase 10 (LLM-Only)**:
- **Removed hybrid system**: Deleted `rule_engine.py`, `hybrid.py`, `rules.yaml`
- **LLM-only classification**: Uses OpenAI structured output
- **Configuration-driven**: All logic in `prompts.yaml` (no retraining needed)
- **Cleaner architecture**: Single classification path via `classify_llm.py`

**Phase 11 (Analysis Module Refactoring)**:
- **Removed legacy classify.py**: Deleted unused 3-stage classification system (473 lines)
- **Module separation**: Extracted `classification_stats.py` and `result_writer.py` from `classify_llm.py`
- **Single Responsibility**: Each module now has one clear purpose:
  - `classify_llm.py`: Parallel processing orchestration only
  - `llm_engine.py`: OpenAI API calls with structured output (schema caching optimized)
  - `classification_stats.py`: Statistics generation and reporting
  - `result_writer.py`: Thread-safe CSV/Sheets incremental saving
- **Code reduction**: 44% reduction in analysis module size (841 net lines removed)
- **Thread safety**: Lock-based CSV writing for multithread environments

**Phase 12 (Press Release LLM Classification)**:
- **Replaced preset system**: `preset_pr.py` deprecated → `classify_press_releases.py`
- **Cluster-based classification**: Representative article per cluster analyzed by LLM, results shared across cluster
- **New orchestration**: `llm_orchestrator.py` for chunked parallel processing
- **API model config**: `src/api_models.yaml` for per-task model selection
- **Date filtering**: Hard-coded 2026-02-01+ filter in main.py and sheets.py
- **Utility extraction**: `src/utils/` with `openai_client.py`, `sheets_helpers.py`, `text_cleaning.py`

**Phase 13 (Current - Logging System Redesign)**:
- **Fixed schema**: `RUN_HISTORY_SCHEMA` (51 columns) prevents column drift across runs
- **3-sheet structure**: `run_history` + `errors` + `events` (replaces single `logs` tab)
- **6-stage duration tracking**: collection, processing, pr_classification, general_classification, source_verification, sheets_sync
- **Metrics collection**: `media_classify` and `classify_press_releases` return `(df, stats)` tuples
- **Schema validation**: Auto-backup old CSV on schema mismatch, prevents blank Sheets on upload failure
- **Stage context**: All errors/events tagged with pipeline stage for debugging

**Phase 14 (Source Verification & Topic Grouping)**:
- **Step 3-3**: Post-LLM source verification with LLM cluster verification
- **Part A**: LLM cluster verification (1 API call/cluster) — 보도자료/유사주제 판단; rule-based fallback on failure
- **Part A-2**: Cross-query cluster merge — TF-IDF cosine + Jaccard (no date constraint), LLM borderline verification
- **Part B**: Discover topic groups via Jaccard similarity + LLM borderline verification (0.35~0.50)
- **New source labels**: 보도자료 / 유사주제 / 일반기사
- **Prompts**: `source_verifier_prompts.yaml` (external, cluster_verification + topic_similarity)
- **Model config**: `source_verification` key in `api_models.yaml` (default: gpt-4o-mini)
- **Module**: `source_verifier.py` in `src/modules/analysis/`
- **Engine**: `load_source_verifier_prompts()`, `render_prompt()`, `call_openai_structured()` in `llm_engine.py`
- **Logger**: 8 new columns in RUN_HISTORY_SCHEMA (55 total, +2 for cross-query merge)

**Google Sheets as Primary Store**:
- CSV files are backups for troubleshooting
- Sheets provides real-time collaboration
- Incremental sync prevents duplicate processing
