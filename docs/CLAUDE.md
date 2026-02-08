# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**News Monitoring System** - A fully automated pipeline that collects hotel brand news articles from Naver, analyzes them with a Hybrid Analysis System (Rule-Based + LLM), detects duplicate content via similarity detection, optionally extracts full-text articles, and generates CSV + Word reports with sentiment analysis, categorization, and risk assessment.

The system runs end-to-end in ~40 seconds and uses a hybrid approach combining regex patterns (instant) with selective LLM analysis (high-value articles only). **Recent additions**: Phase 5-6 (API pagination, Google Sheets sync, TF-IDF similarity), Phase 7 (media classification), Phase 8-9 (CSV migration, OpenAI group summarization), **Phase 10 (Hybrid Analysis System with 3-stage reconciliation)**.

## Architecture

The project follows a linear 5-step pipeline with optional enhancements:

```
STEP 1: Collection (6 brands × 900 articles/brand via pagination)
  ├─ Naver API pagination (9 pages, 90% quota safety)
  ├─ Optional: Playwright browser scraping (JavaScript rendering)
  └─ Optional: Filter by existing Google Sheets data (incremental)
    ↓
STEP 2: Processing (normalize, dedupe, similarity detection, media classification)
  ├─ Strip HTML, normalize dates to ISO 8601
  ├─ Deduplicate by link, keep newest
  ├─ TF-IDF similarity detection marks press release duplicates
  ├─ OpenAI summarization for press release groups
  └─ Media outlet classification (domain → name/group/type)
    ↓
STEP 3: Hybrid Analysis (Rule-Based + LLM with 3-stage reconciliation)
  ├─ [1/3] Rule-Based Analysis (전체 기사, instant, regex patterns)
  │   ├─ Brand Scope: BRAND_TARGETED / BRAND_MENTIONED / VENUE_ONLY
  │   ├─ Sentiment: POSITIVE / NEUTRAL / NEGATIVE_CANDIDATE / NEGATIVE_CONFIRMED
  │   ├─ Danger: D1 / D2 / D3 (score-based, conditional)
  │   ├─ Issue Category: 11 categories (Safety, Legal, Security, etc.)
  │   └─ Coverage Themes: 8 themes, max 2 (Business, Risk/Crisis, etc.)
  ├─ [2/3] LLM Analysis (selective: 우리 브랜드 전체 + 경쟁사 상위 N개)
  │   ├─ sentiment_llm → sentiment_final (RB vs LLM reconciliation)
  │   ├─ danger_llm → danger_final (conditional, RB vs LLM reconciliation)
  │   └─ category_llm → category_final (RB vs LLM reconciliation)
  └─ [3/3] Final Decision (per dimension)
      ├─ decision_rule: KEEP_RB / KEEP_LLM / RECALL_UPGRADE / PLAYBOOK_TIE_BREAK
      ├─ confidence: 0.0-1.0
      └─ evidence + rationale: transparency for audits
    ↓
STEP 3.5: Enhancement (optional full-text extraction, Looker prep)
    ↓
STEP 4: Reporting (CSV + Word document)
    ↓
STEP 5: Google Sheets (incremental upload to 2 tabs: raw_data, result)
```

### Module Organization

```
src/modules/
├── collection/
│   ├── collect.py         # Naver API pagination + brand search
│   └── scrape.py          # Playwright browser automation (optional)
├── processing/
│   ├── process.py         # Normalization, dedup, similarity detection, CSV I/O
│   ├── fulltext.py        # Article text extraction (optional)
│   ├── looker_prep.py     # Time-series columns (optional)
│   └── media_classify.py  # Media outlet classification (OpenAI batch)
├── analysis/
│   ├── classify.py        # Legacy AI classification (deprecated)
│   ├── hybrid.py          # Hybrid orchestrator (Rule-Based + LLM)
│   ├── rule_engine.py     # Regex pattern matching engine
│   ├── llm_engine.py      # OpenAI Structured Output engine
│   ├── rules.yaml         # Rule-Based patterns, scores, thresholds
│   └── prompts.yaml       # LLM prompts, policies, JSON schemas
└── export/
    ├── report.py          # CSV/Word generation
    └── sheets.py          # Google Sheets incremental sync
```

## Key Configuration

**Brands** (`src/modules/collection/collect.py` lines 13-14):
- `OUR_BRANDS = ["롯데호텔", "호텔롯데", "L7", "시그니엘"]` - Monitored hotel brands
- `COMPETITORS = ["신라호텔", "조선호텔"]` - Competitor brands

**Hybrid Analysis System** (`src/modules/analysis/`):
- **System**: Rule-Based (regex, instant) + LLM (OpenAI, selective)
- **3-stage process per dimension**:
  1. Rule-Based analysis (전체 기사, 0.1s/article)
  2. LLM independent judgment (우리 브랜드 + 경쟁사 상위 N개)
  3. Final reconciliation (RB vs LLM, decision_rule recorded)
- **Dimensions**:
  - Sentiment (4-state): POSITIVE / NEUTRAL / NEGATIVE_CANDIDATE / NEGATIVE_CONFIRMED
  - Danger (3-level): D1 (minor) / D2 (monitor) / D3 (statement needed)
  - Issue Category (11 options): Safety/Incident, Legal/Regulation, Security/Privacy/IT, Hygiene/Food, Customer Dispute, Service Quality, Pricing/Commercial, Labor/HR, Governance/Ethics, Reputation/PR, OTHER
  - Coverage Themes (max 2 of 8): Business/Performance, Brand/Marketing, Product/Offering, Customer Experience, Operations/Technology, People/Organization, Risk/Crisis, ESG/Social, OTHER
- **Configuration**:
  - `rules.yaml` - Regex patterns, scores, thresholds
  - `prompts.yaml` - LLM prompts, policies, JSON schemas
  - No retraining needed: Edit YAML files to update logic

**Environment** (`.env`):
- Required: `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET`, `OPENAI_API_KEY`
- Optional: `GOOGLE_SHEETS_CREDENTIALS_PATH`, `GOOGLE_SHEET_ID` (for Sheets sync)

## Commands

### Installation & Setup

```bash
# Install dependencies
pip install -r requirements.txt

# One-time browser setup for Playwright (only if using --scrape)
playwright install

# Set up environment (.env file with API keys)
cp .env.template .env
# Then edit .env with your credentials
```

### Basic Usage

```bash
# Default: API only, 100 articles/brand, fast run
python main.py

# With incremental Google Sheets (skips duplicates from previous runs)
python main.py --sheets

# API pagination (9 pages = ~900 articles/brand, recommended)
python main.py --max_api_pages 9

# Browser scraping with date range (in addition to API)
python main.py --scrape --start_date 2026-01-01 --end_date 2026-02-08

# Full-text extraction for high/medium risk articles
python main.py --fulltext --fulltext_risk_levels 상,중

# Add Looker Studio time-series columns
python main.py --looker_prep

# Combine everything
python main.py --max_api_pages 9 --sheets --fulltext --looker_prep
```

### Advanced Options

```bash
# Dry run (test pipeline without AI costs)
python main.py --dry_run

# Raw collection only (no classification)
python main.py --raw_only --sheets

# Small test with 1 page, 50 articles
python main.py --display 50 --max_api_pages 1 --chunk_size 50

# Larger chunks for speed (if no timeout issues)
python main.py --chunk_size 200

# Output to custom directory
python main.py --outdir reports

# Sort by relevance instead of date
python main.py --sort sim

# Reduce competitor article processing
python main.py --max_competitor_classify 10

# Use legacy classification (OpenAI only, no hybrid)
python main.py --legacy_classify
```

## Implementation Details

### API Pagination Strategy (Phase 5)
- Naver API limit: 1000 results/query (100 per page × 10 pages max)
- Implemented: 9 pages = 900 results/query (90% quota for safety margin)
- Rate limiting: 0.2s delay between pages (prevents throttling)
- 6 brands × 900 articles = ~5400 articles per run

### Incremental Google Sheets Collection (Phase 5)
- Reads existing links from Sheets before collection
- Collects new articles only (skips duplicates)
- Saves API quota on repeated runs
- Double safety: auto-deduplication on Sheets upload

### Similarity Detection (Phase 6)
- TF-IDF vectorization with character-level 2-4 n-grams (Korean-optimized)
- Cosine similarity ≥ 0.8 marks articles as "보도자료" (press releases)
- Preserves all articles (non-destructive labeling)
- Performance: 2-15s for 1000-5000 articles

### Batching Strategy (Cost Optimization)
- Articles grouped into chunks (default 100) for OpenAI
- 5400 articles → 54 API calls for sentiment (vs 5400 individual)
- Negative articles batched separately for risk (stage 3 only)
- ~97% cost savings vs per-article processing

### Error Handling
- Naver API: 0.2s delay between requests, graceful 401 handling
- OpenAI API: Exponential backoff for rate limits (429), 5s default wait
- Sheets: Continues if credentials missing, Excel still generated
- Scraping: Continues with API data only if browser fails
- Full-text: Graceful timeout/paywall handling with status classification
- All failures non-blocking: pipeline continues to completion

## Data Columns

### From Naver API
- `title`, `description`, `link`, `originallink`, `pubDate`

### Added in Processing
- `query` - Search term (brand name)
- `group` - "OUR" or "COMPETITOR"
- `pub_datetime` - ISO 8601 date
- `source` - "보도자료" if similar to other articles, else ""
- `group_id` - Press release group ID (e.g., "group_0")
- `press_release_group` - OpenAI-generated 3-word summary (e.g., "신라호텔 개장")
- `media_domain`, `media_name`, `media_group`, `media_type` - Media outlet info

### Added in Hybrid Classification

**Rule-Based (RB) - Always present:**
- `brand_mentions` - {"our": [...], "competitors": [...]}
- `brand_scope_rb` - BRAND_TARGETED / BRAND_MENTIONED / VENUE_ONLY
- `sentiment_rb` - POSITIVE / NEUTRAL / NEGATIVE_CANDIDATE / NEGATIVE_CONFIRMED
- `danger_rb` - D1 / D2 / D3 (if applicable)
- `risk_score_rb` - Numeric score (0-100)
- `issue_category_rb` - One of 11 categories
- `coverage_themes_rb` - Array of up to 2 themes
- `reason_codes_rb` - Array (e.g., ["DATA_BREACH", "INVESTIGATION"])
- `matched_rules_rb` - Array of matched regex patterns
- `score_breakdown_rb` - JSON breakdown of risk score components

**LLM Stage - Present if LLM analyzed:**
- `sentiment_llm`, `sentiment_llm_confidence`, `sentiment_llm_evidence`, `sentiment_llm_rationale`
- `danger_llm`, `danger_llm_confidence`, `danger_llm_evidence`, `danger_llm_rationale`
- `issue_category_llm`, `coverage_themes_llm`, `category_llm_confidence`, `category_llm_evidence`, `category_llm_rationale`

**Final Stage - Present if LLM analyzed:**
- `sentiment_final`, `sentiment_final_confidence`, `sentiment_final_decision_rule`, `sentiment_final_evidence`, `sentiment_final_rationale`
- `danger_final`, `danger_final_confidence`, `danger_final_decision_rule`, `danger_final_evidence`, `danger_final_rationale`
- `issue_category_final`, `coverage_themes_final`, `category_final_confidence`, `category_final_decision_rule`, `category_final_evidence`, `category_final_rationale`
- `classified_at` - ISO 8601 timestamp

### Optional Enhancements
- `fulltext` - Full article text (if `--fulltext` enabled)
- `fulltext_status` - success, paywall, 404, timeout, etc.
- `date_only`, `week_number`, `month`, `article_count` (if `--looker_prep`)

## Output Files

Located in `data/` directory (or `--outdir`):

- `raw.csv` - Unprocessed API response (UTF-8 BOM)
- `result.csv` - Final results with hybrid analysis columns (UTF-8 BOM)
- `media_directory.csv` - Media outlet directory (persistent, auto-updated)
- `report.docx` - Formatted Word document by risk level

**Google Sheets** (if `--sheets` enabled):
- `raw_data` tab - Raw articles
- `result` tab - Classified articles with all hybrid analysis columns

## Development

### Quick Testing

```bash
# Test collection only
python main.py --raw_only --display 10

# Test processing without AI (--dry_run)
python main.py --dry_run --display 50

# Test with custom chunk size (debug timeout issues)
python main.py --chunk_size 10

# Test Sheets integration
python main.py --sheets --display 10 --dry_run
```

### Debugging

Enable debug output in individual modules by adding prints (no centralized logger currently).

Common issues:
- **Missing .env**: Copy template and add API keys
- **Playwright missing**: Run `playwright install` after pip install
- **Google Sheets credentials**: Ensure service account JSON exists at path in .env
- **Rate limits**: Add `--chunk_size 50` to reduce API call size
- **Import errors**: Verify `src/modules/` has `__init__.py` files

### Code Style

- No strict linting enforced
- Use f-strings for formatting
- Keep functions under 50 lines for readability
- DataFrames named: `df_raw`, `df_normalized`, `df_processed`, `df_classified`, `df_result`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **401 Auth Error** | Verify `.env` has correct Naver/OpenAI API keys |
| **No articles found** | Check Korean spelling in brand/competitor lists; try `--display 200` |
| **Timeout during classification** | Reduce `--chunk_size` (e.g., 50 or 30) |
| **Rate limiting (429)** | Already has retry logic; try `--chunk_size 50` |
| **No Sheets sync** | Verify credentials JSON path in `.env`; check Sheet ID is accessible |
| **Playwright errors** | Run `playwright install`; check Chromium is installed |
| **Full-text extraction slow** | Reduce articles with `--fulltext_max_articles 20` or use fewer risk levels |

## File Organization Principles

- **Single responsibility**: Each module handles one step
- **DataFrame-centric**: All data flows as pandas DataFrames
- **Configuration in files**: Brands/categories at module top for easy editing
- **Environment variables**: Credentials and paths in `.env` only
- **Graceful degradation**: Optional features don't block core pipeline
