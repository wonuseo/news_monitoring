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
├─ Rules 2-6: Field-level empty check (brand_relevance, sentiment_stage, source, media_domain, date_only)
├─ Union all targets → clear classified_at → merge with new articles
└─ Module: reprocess_checker.py

STEP 2: Processing
├─ Normalize (HTML strip, ISO dates, article_id, article_no)
├─ Deduplicate by link
├─ TF-IDF similarity (detect press releases, cosine ≥ 0.8)
├─ OpenAI group summarization (press_release_group)
└─ Media classification (domain → name/group/type via OpenAI batch)

STEP 3: LLM Classification
├─ Press release preset (skip LLM for cost savings)
├─ LLM Analysis (OpenAI gpt-4o-mini, selective: 우리 브랜드 all + 경쟁사 all)
│   └─ Outputs: brand_relevance, sentiment_stage, danger_level,
│       issue_category, news_category, news_keyword_summary
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
│   ├── collect.py       # Naver API pagination
│   └── scrape.py        # Browser automation (optional)
├── processing/
│   ├── process.py       # Normalize, dedupe, CSV I/O
│   ├── press_release_detector.py # Press release detection & summarization
│   ├── media_classify.py # OpenAI media outlet classification
│   ├── reprocess_checker.py # Reprocess target detection (missing/incomplete fields)
│   ├── fulltext.py      # Full-text extraction (optional)
│   └── looker_prep.py   # Time-series columns (optional)
├── analysis/
│   ├── classify_llm.py  # LLM classification orchestrator (parallel processing)
│   ├── llm_engine.py    # OpenAI Structured Output engine (low-level API)
│   ├── classification_stats.py # Statistics generation & reporting
│   ├── result_writer.py # CSV/Sheets incremental saving (thread-safe)
│   ├── preset_pr.py     # Press release preset values
│   ├── keyword_extractor.py # Category-specific keyword extraction (kiwipiepy + Log-odds)
│   └── prompts.yaml     # LLM prompts & schemas
├── monitoring/
│   └── logger.py        # Run metrics logging (CSV + Google Sheets)
└── export/
    ├── report.py        # CSV + Word generation
    └── sheets.py        # Google Sheets sync
```

**Note**: System uses LLM-only classification. All legacy files (`classify.py`, `hybrid.py`, `rule_engine.py`, `rules.yaml`) have been removed.

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

# Browser scraping with date range
python main.py --scrape --start_date 2026-01-01 --end_date 2026-02-08

# Full-text extraction (high/medium risk only)
python main.py --fulltext --fulltext_risk_levels 상,중

# Custom output directory
python main.py --outdir reports

# Parallel processing (adjust workers)
python main.py --max_workers 10

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
- `source` ("보도자료" if similar, else "일반기사")
- `cluster_id` (press release cluster ID)
- `press_release_group` (OpenAI 3-word summary)
- `media_domain`, `media_name`, `media_group`, `media_type`

**LLM Classification (result.csv)**:
- `brand_relevance`: "관련" / "언급" / "무관" / "판단 필요"
- `brand_relevance_query_keywords`: [array of keywords]
- `sentiment_stage`: "긍정" / "중립" / "부정 후보" / "부정 확정"
- `danger_level`: "상" / "중" / "하" / null (only if brand_relevance + negative)
- `issue_category`: One of 11 Korean categories (안전/사고, 법무/규제, etc.) or null
- `news_category`: One of 9 Korean categories (사업/실적, 브랜드/마케팅, etc.)
- `news_keyword_summary`: 5-word Korean summary
- `classified_at`: ISO timestamp

**Google Sheets** (primary data store):
- `raw_data` tab: Raw collected articles
- `result` tab: Classified articles with all LLM columns
- `logs` tab: Run history with metrics (CSV backup: `data/logs/run_history.csv`)
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
- **Model**: OpenAI gpt-4o-mini (as configured in prompts.yaml)
- **Full analysis by default**: OUR_BRANDS (all) + COMPETITORS (all)
  - Use `--max_competitor_classify N` to limit competitor analysis
- **Press releases**: Preset values via `preset_pr.py`, skip LLM (save cost)
- **Chunking**: Default 100 articles/chunk (reduce for timeouts)
- **Parallel**: ThreadPoolExecutor with max_workers (default 10)
- **Incremental save**: Appends to result.csv after each successful chunk
- **Configuration**: All prompts and rules in `prompts.yaml` - no code changes needed

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

The system automatically tracks all pipeline metrics and saves them to CSV + Google Sheets.

**Tracked Metrics** (34 columns):
- **Basic**: run_id, timestamp, duration_total, cli_args
- **Collection**: articles_collected_total, articles_collected_per_query, existing_links_skipped, duration_collection
- **Processing**: duplicates_removed, articles_processed, press_releases_detected, press_release_groups, duration_processing
- **Media**: media_domains_total, media_domains_new, media_domains_cached
- **Classification**: articles_classified_llm, llm_api_calls, llm_cost_estimated, press_releases_skipped, classification_errors, duration_classification
- **Results**: our_brands_relevant, our_brands_negative, danger_high, danger_medium, competitor_articles
- **Sheets**: sheets_sync_enabled, sheets_rows_uploaded_raw, sheets_rows_uploaded_result, sheets_logs_uploaded, duration_sheets_sync

**Log Storage**:
- CSV: `data/logs/run_history.csv` (append mode, persistent)
- Google Sheets: `logs` tab (full replace on each sync)

**Usage**:
- Automatic: No configuration needed, logs saved after every run
- View logs: `cat data/logs/run_history.csv` or open Google Sheets
- Monitor costs: Check `llm_cost_estimated` column (USD)
- Track performance: Compare `duration_*` columns across runs

## Output Files

**Location**: `data/` directory (or `--outdir`)

- `raw.csv` - Raw API collection (UTF-8 BOM, troubleshooting backup)
- `result.csv` - LLM classified results (UTF-8 BOM, troubleshooting backup)
- `media_directory.csv` - Media outlet directory (persistent)
- `keywords/` - Category-specific keyword CSV files (automatic, troubleshooting backup)
- `logs/run_history.csv` - Run metrics history (persistent, append mode)

**Google Sheets** (primary data store, if configured):
- `raw_data` tab: Raw collected articles
- `result` tab: Classified articles with all LLM columns
- `logs` tab: Run history with metrics
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
| Playwright errors | Run `playwright install` |

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
- Rules 2-6: Checks field-level completeness (brand_relevance, sentiment_stage, source, media_domain, date_only)
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
- Tests directory (`tests/`) exists for unit tests (in development)

**Important Functions**:
- Collection: `collect_all_news()` in `collect.py`
- Processing: `normalize_df()`, `dedupe_df()`, `detect_similar_articles()` in `process.py`
- Classification:
  - `classify_llm()` in `classify_llm.py` - Main orchestrator with parallel processing
  - `analyze_article_llm()` in `llm_engine.py` - Single article LLM analysis
  - `save_result_to_csv_incremental()` in `result_writer.py` - Thread-safe CSV saving
  - `get_classification_stats()` in `classification_stats.py` - Statistics generation
- Reprocess: `check_reprocess_targets()`, `load_raw_data_from_sheets()`, `clear_classified_at_for_targets()` in `reprocess_checker.py`
- Reporting: `create_word_report()` in `report.py`
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

**Phase 11 (Current - Analysis Module Refactoring)**:
- **Removed legacy classify.py**: Deleted unused 3-stage classification system (473 lines)
- **Module separation**: Extracted `classification_stats.py` and `result_writer.py` from `classify_llm.py`
- **Single Responsibility**: Each module now has one clear purpose:
  - `classify_llm.py`: Parallel processing orchestration only
  - `llm_engine.py`: OpenAI API calls with structured output (schema caching optimized)
  - `classification_stats.py`: Statistics generation and reporting
  - `result_writer.py`: Thread-safe CSV/Sheets incremental saving
- **Code reduction**: 44% reduction in analysis module size (841 net lines removed)
- **Thread safety**: Lock-based CSV writing for multithread environments

**Google Sheets as Primary Store**:
- CSV files are backups for troubleshooting
- Sheets provides real-time collaboration
- Incremental sync prevents duplicate processing
