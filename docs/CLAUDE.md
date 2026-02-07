# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**News Monitoring System** - A fully automated pipeline that collects hotel brand news articles from Naver, analyzes them with OpenAI's GPT models, and generates Excel + Word reports with sentiment analysis, categorization, and risk assessment.

The system runs end-to-end in ~40 seconds and uses batched AI processing to achieve 97% cost savings compared to per-article processing.

## Architecture

The project follows a linear 4-step pipeline architecture:

1. **Collection** (`collect.py`) - Fetch articles from Naver News API
   - Searches for OUR_BRANDS and COMPETITORS keywords
   - Returns raw DataFrame with API response fields (title, description, link, pubDate, etc.)

2. **Processing** (`process.py`) - Normalize and deduplicate data
   - Strip HTML tags and unescape entities
   - Parse pub dates to ISO format
   - Remove duplicate articles by link, keeping newest

3. **Classification** (`classify.py`) - AI-powered analysis using OpenAI API
   - **Stage 1**: Sentiment analysis (긍정/중립/부정) - batch all articles
   - **Stage 2**: Category classification - batch all articles into 7 categories
   - **Stage 3**: Risk assessment (상/중/하) - batch only negative articles to save costs
   - Uses chunking strategy to batch process articles (default chunk_size=100)
   - Retry logic with exponential backoff for rate limits and timeouts

4. **Reporting** (`report.py`) - Generate output files
   - Console summary (top 10 negative, positive, competitor articles)
   - Excel workbook with 4 sheets: 전체데이터, 우리_부정, 우리_긍정, 경쟁사
   - Word document with structured sections by risk level

The `main.py` orchestrates all steps and handles environment setup.

## Key Configuration

**Brands** (`collect.py` lines 13-14):
- `OUR_BRANDS = ["롯데호텔", "호텔롯데", "L7", "시그니엘"]` - Monitored hotel brands
- `COMPETITORS = ["신라호텔", "조선호텔"]` - Competitor brands

**Categories** (`classify.py` lines 21-29):
- 7 Korean categories: 법률/규제, 보안/데이터, 안전/사고, 재무/실적, 제품/서비스, 평판/SNS, 운영/기타

**Environment** (`.env`):
- `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET` - Naver News API credentials
- `OPENAI_API_KEY` - OpenAI API key

## Commands

**Installation**:
```bash
pip install -r requirements.txt
```

**Basic Run**:
```bash
python main.py
```

**Run with Options**:
```bash
# Collect 200 articles per brand
python main.py --display 200

# Change output directory
python main.py --outdir reports

# Adjust batch size for timeout issues (smaller = slower but more stable)
python main.py --chunk_size 50

# Dry run (skip AI classification, test data pipeline)
python main.py --dry_run

# All options
python main.py --display 100 --sort date --outdir reports --max_competitor_classify 20 --chunk_size 100
```

## Data Flow

### Input DataFrame columns (from Naver API):
- `title` - Article headline
- `description` - Summary text
- `link` - Article URL
- `originallink` - Original source URL
- `pubDate` - Publication date (RFC 2822)

### Added columns through pipeline:
- `query` - Search term used (brand name)
- `group` - "OUR" or "COMPETITOR"
- `pub_datetime` - ISO 8601 date format
- `sentiment` - "긍정", "중립", or "부정"
- `category` - One of 7 categories
- `risk_level` - "상", "중", "하" (only for negative articles)
- `reason` - AI explanation for classification

## Important Implementation Details

### Batching Strategy (Cost Optimization)
- Articles are grouped into chunks (default 100) before sending to OpenAI
- Each chunk is sent as a single API call with 100 articles
- 365 articles → 4 chunks → 12 API calls total (vs 365 individual calls)
- Negative articles are batched separately for risk assessment (only 3rd stage)

### Error Handling
- Naver API: Built-in 0.1s delay between requests, 401 auth error handling
- OpenAI API: Exponential backoff with retry for rate limits (429), 5s default wait
- Timeout: Configurable via `--chunk_size` (reduce if timeout occurs)

### DataFrame Deduplication
- Uses `originallink` as primary key, falls back to `link`
- Keeps most recent article when duplicates found
- Sorts by pubDate before deduplication to ensure newest is retained

## Output Files

Located in `data/` directory (or specified `--outdir`):

- `raw.xlsx` - Original Naver API response, unprocessed
- `processed.xlsx` - After HTML stripping, date normalization, deduplication
- `result.xlsx` - Final results with 4 sheets (all data, our negative, our positive, competitors)
- `report.docx` - Formatted Word document with 5 sections (3 risk levels, positive, competitors)

## Testing

**Unit Testing**: Not currently implemented. Manual testing via `--dry_run` flag.

**Common Test Cases**:
- `python main.py --dry_run` - Test data pipeline without AI costs
- `python main.py --display 10` - Small run with limited articles
- `python main.py --chunk_size 10` - Test chunking with tiny chunks

## Troubleshooting

- **API Auth Errors (401)**: Verify `.env` file has correct NAVER_CLIENT_ID, NAVER_CLIENT_SECRET, OPENAI_API_KEY
- **Timeout Errors**: Reduce `--chunk_size` (e.g., 50 or 30 instead of 100)
- **Rate Limiting (429)**: Already has retry logic; if still occurs, reduce chunk_size or add delays
- **No Results**: Check search terms in `collect.py` are correct Korean spelling; try `--display 200` for more articles
