# Quick Start Guide

## Installation

```bash
# Install new dependencies
pip install -r requirements.txt
```

## Common Commands

### 1. Traditional API-only (existing behavior)
```bash
python main.py --display 100
```

### 2. Scrape historical data (2026-01-01 to 2026-02-07)
```bash
python main.py --scrape
```

### 3. Add Looker Studio columns
```bash
python main.py --looker_prep
```

### 4. Extract full-text for high-risk articles only
```bash
python main.py --fulltext --fulltext_risk_levels 상
```

### 5. Sync to Google Sheets
```bash
python main.py --sheets
```

### 6. Full pipeline with all features
```bash
python main.py --scrape --fulltext --looker_prep --sheets --display 100
```

### 7. Test without AI costs (dry-run)
```bash
python main.py --scrape --dry_run
```

## New Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--scrape` | Enable Naver News scraping by date range | Disabled |
| `--start_date` | Scraping start date (YYYY-MM-DD) | 2026-01-01 |
| `--end_date` | Scraping end date (YYYY-MM-DD) | 2026-02-07 |
| `--max_scrape_pages` | Max pages per brand query | 10 |
| `--fulltext` | Extract full article text | Disabled |
| `--fulltext_risk_levels` | Risk levels to scrape (comma-separated) | 상,중 |
| `--fulltext_max_articles` | Max articles to process | Unlimited |
| `--looker_prep` | Add Looker Studio time-series columns | Disabled |
| `--sheets` | Upload to Google Sheets | Disabled |
| `--sheets_id` | Google Sheet ID (overrides .env) | From .env |

## Output Files

All output in `data/` directory (or `--outdir` if specified):

| File | Description |
|------|-------------|
| `raw.xlsx` | Raw API/scraping data (1 sheet) |
| `processed.xlsx` | After normalization & dedup (1 sheet) |
| `result.xlsx` | Final classified results (4 sheets) |
| `report.docx` | Word report with risk analysis |
| Google Sheets | If `--sheets` flag used (4 tabs) |

## Google Sheets Tabs

When using `--sheets`:
1. **전체데이터** - All articles with AI classification
2. **우리_부정** - Our brands, negative sentiment
3. **우리_긍정** - Our brands, positive sentiment
4. **경쟁사** - Competitor articles

## Pipeline Steps

```
STEP 1: Collection (API ± Scraping)
STEP 2: Processing (normalize, deduplicate)
STEP 3: Classification (AI sentiment/category/risk)
STEP 3.5: Full-Text Scraping (optional, risk-filtered)
STEP 3.7: Looker Prep (optional, time-series columns)
STEP 4: Reporting (Excel + Word)
STEP 5: Google Sheets Upload (optional)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| ImportError for beautifulsoup4 | Run `pip install -r requirements.txt` |
| Google Sheets auth error | Check `.env` has GOOGLE_SHEETS_CREDENTIALS_PATH |
| Scraping timeout | Reduce `--max_scrape_pages` (e.g., 5) |
| Full-text too slow | Use `--fulltext_max_articles 50` to limit |
| No results from scraping | Try different `--start_date`/`--end_date` |

## Configuration Files

- `.env` - API keys & Google Sheets credentials
- `.gitignore` - Protects sensitive files
- `requirements.txt` - All dependencies including new ones
- `IMPLEMENTATION_SUMMARY.md` - Full technical documentation
- `CLAUDE.md` - Original project architecture

## New Modules

| Module | Purpose | Key Functions |
|--------|---------|---|
| `scrape.py` | Naver scraping by date | `scrape_naver_news_by_date()`, `merge_api_and_scrape()` |
| `fulltext.py` | Article text extraction | `fetch_full_text()`, `batch_fetch_full_text()` |
| `looker_prep.py` | Looker Studio prep | `add_time_series_columns()` |
| `sheets.py` | Google Sheets sync | `connect_sheets()`, `sync_all_sheets()` |

## Example Workflows

### Workflow 1: Daily monitoring (API only)
```bash
python main.py --display 100
```
- Fast, uses existing API
- No new feature overhead

### Workflow 2: Historical analysis + Looker prep
```bash
python main.py --scrape --looker_prep --sheets
```
- Collects 2026-01-01 to 2026-02-07 data
- Adds time-series columns
- Syncs to Google Sheets for Looker Studio

### Workflow 3: Risk analysis with full-text
```bash
python main.py --fulltext --fulltext_risk_levels 상 --sheets
```
- Extracts full text only for HIGH risk articles
- Syncs results with full text to Sheets
- Reduces costs by filtering

### Workflow 4: Complete pipeline
```bash
python main.py \
  --scrape \
  --fulltext --fulltext_risk_levels 상,중 \
  --looker_prep \
  --sheets \
  --display 100 \
  --chunk_size 50
```
- Collects historical + recent articles
- Extracts full text for medium/high risk
- Adds Looker columns
- Syncs everything to Sheets
- Use smaller chunk_size if AI API rate limiting

## Performance Notes

- **Scraping:** ~50ms per article + 0.5s delay = ~1 min per 100 articles
- **Full-text:** ~100ms per article + 1s delay = ~2 min per 100 articles
- **Looker prep:** <100ms for all articles
- **Sheets sync:** ~1s per 100 rows
- **AI classification:** ~10-30 seconds per 100 articles (existing)

**Typical full pipeline:** ~3-5 minutes for 100 articles with all features

## Backward Compatibility

All new features are **optional flags**. Existing commands work unchanged:

```bash
# These still work exactly as before:
python main.py
python main.py --display 50 --dry_run
python main.py --chunk_size 30 --outdir reports
```

No existing functionality modified - only additions.

---

For detailed technical documentation, see: `IMPLEMENTATION_SUMMARY.md`
For project architecture, see: `CLAUDE.md`
