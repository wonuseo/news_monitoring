"""
Run Logger - Tracks pipeline execution metrics and saves to CSV + Google Sheets
"""

import time
import uuid
import json
import os
import csv
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd


class RunLogger:
    """
    Tracks metrics across pipeline stages and saves to CSV + Google Sheets.

    Usage:
        logger = RunLogger()
        logger.log("articles_collected", 100)
        logger.log_stage("collection", {"api_calls": 9, "articles": 900})
        logger.save_csv("data/logs/run_history.csv")
    """

    def __init__(self):
        self.run_id = str(uuid.uuid4())[:8]  # Short UUID (e.g., "a3f2c1d4")
        self.start_time = time.time()
        self.metrics = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
        }
        self._logs = []
        self._logs_flushed = 0
        self.stage_start_times = {}

    def log(self, key: str, value: Any):
        """Log a single metric"""
        self.metrics[key] = value

    def log_dict(self, metrics_dict: Dict[str, Any]):
        """Log multiple metrics at once"""
        self.metrics.update(metrics_dict)

    def _append_log(self, level: str, category: str, message: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Internal: append a log entry to the unified buffer.
        """
        payload = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "category": category,
            "message": message,
            "data_json": json.dumps(data, ensure_ascii=False) if data else ""
        }
        self._logs.append(payload)
        return payload

    def log_event(self, message: str, data: Optional[Dict[str, Any]] = None, category: str = "pipeline") -> Dict[str, Any]:
        """
        Log an INFO-level event.
        """
        return self._append_log(level="INFO", category=category, message=message, data=data)

    def log_error(self, message: str, data: Optional[Dict[str, Any]] = None, category: str = "system") -> Dict[str, Any]:
        """
        Log an ERROR-level event (also print to terminal).
        """
        print(f"  ERROR: {message}")
        self.metrics["errors_total"] = self.metrics.get("errors_total", 0) + 1
        return self._append_log(level="ERROR", category=category, message=message, data=data)

    def start_stage(self, stage_name: str):
        """Mark the start of a pipeline stage (for duration tracking)"""
        self.stage_start_times[stage_name] = time.time()

    def end_stage(self, stage_name: str):
        """Mark the end of a pipeline stage and calculate duration"""
        if stage_name in self.stage_start_times:
            duration = time.time() - self.stage_start_times[stage_name]
            self.metrics[f"duration_{stage_name}"] = round(duration, 2)

    def finalize(self):
        """Finalize metrics (calculate total duration)"""
        self.metrics["duration_total"] = round(time.time() - self.start_time, 2)

    def to_dict(self) -> Dict[str, Any]:
        """Return all metrics as dictionary"""
        return self.metrics.copy()

    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to single-row DataFrame"""
        # Ensure all values are serializable
        clean_metrics = {}
        for key, value in self.metrics.items():
            if isinstance(value, (dict, list)):
                clean_metrics[key] = json.dumps(value, ensure_ascii=False)
            else:
                clean_metrics[key] = value

        return pd.DataFrame([clean_metrics])

    def save_csv(self, csv_path: str):
        """
        Save metrics to CSV file (append mode).
        Creates file if it doesn't exist.
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)

            df = self.to_dataframe()

            # Append to existing file or create new one
            if os.path.exists(csv_path):
                df.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
            else:
                df.to_csv(csv_path, mode='w', header=True, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)

            print(f"\n  Log saved: {csv_path}")
            return True

        except Exception as e:
            print(f"\n  Log CSV save failed: {e}")
            return False

    def print_summary(self):
        """Print a formatted summary of key metrics"""
        print("\n" + "="*60)
        print("  Run Summary")
        print("="*60)

        # Basic info
        print(f"Run ID: {self.metrics.get('run_id', 'N/A')}")
        print(f"Timestamp: {self.metrics.get('timestamp', 'N/A')}")
        print(f"Total Duration: {self.metrics.get('duration_total', 0):.2f}s")
        print()

        # Collection
        print("  Collection")
        print(f"  - Total collected: {self.metrics.get('articles_collected_total', 0)}")
        if 'articles_collected_per_query' in self.metrics:
            per_query = self.metrics['articles_collected_per_query']
            if isinstance(per_query, str):
                per_query = json.loads(per_query)
            for brand, count in per_query.items():
                print(f"    - {brand}: {count}")
        print(f"  - Existing links skipped: {self.metrics.get('existing_links_skipped', 0)}")
        print(f"  - Duration: {self.metrics.get('duration_collection', 0):.2f}s")
        print()

        # Processing
        print("  Processing")
        print(f"  - Duplicates removed: {self.metrics.get('duplicates_removed', 0)}")
        print(f"  - Processed: {self.metrics.get('articles_processed', 0)}")
        print(f"  - Press releases detected: {self.metrics.get('press_releases_detected', 0)}")
        print(f"  - Press release groups: {self.metrics.get('press_release_groups', 0)}")
        print(f"  - Duration: {self.metrics.get('duration_processing', 0):.2f}s")
        print()

        # Media classification
        if self.metrics.get('media_domains_total', 0) > 0:
            print("  Media Classification")
            print(f"  - Total domains: {self.metrics.get('media_domains_total', 0)}")
            print(f"  - New classified: {self.metrics.get('media_domains_new', 0)}")
            print(f"  - Cached: {self.metrics.get('media_domains_cached', 0)}")
            print()

        # Classification
        print("  LLM Classification")
        print(f"  - Classified: {self.metrics.get('articles_classified_llm', 0)}")
        print(f"  - API calls: {self.metrics.get('llm_api_calls', 0)}")
        print(f"  - Estimated cost: ${self.metrics.get('llm_cost_estimated', 0):.4f}")
        print(f"  - Press releases skipped: {self.metrics.get('press_releases_skipped', 0)}")
        print(f"  - Classification errors: {self.metrics.get('classification_errors', 0)}")
        print(f"  - Duration: {self.metrics.get('duration_classification', 0):.2f}s")
        print()

        # Results
        print("  Classification Results")
        print(f"  - Our brands relevant: {self.metrics.get('our_brands_relevant', 0)}")
        print(f"  - Our brands negative: {self.metrics.get('our_brands_negative', 0)}")
        print(f"  - Danger high: {self.metrics.get('danger_high', 0)}")
        print(f"  - Danger medium: {self.metrics.get('danger_medium', 0)}")
        print(f"  - Competitor articles: {self.metrics.get('competitor_articles', 0)}")
        print()

        # Sheets sync
        if self.metrics.get('sheets_sync_enabled'):
            print("  Google Sheets Sync")
            print(f"  - raw_data uploaded: {self.metrics.get('sheets_rows_uploaded_raw', 0)} rows")
            print(f"  - result uploaded: {self.metrics.get('sheets_rows_uploaded_result', 0)} rows")
            print(f"  - run_history uploaded: {self.metrics.get('sheets_run_history_uploaded', 0)} rows")
            print(f"  - logs uploaded: {self.metrics.get('sheets_logs_uploaded', 0)} rows")
            print(f"  - Duration: {self.metrics.get('duration_sheets_sync', 0):.2f}s")
            print()

        # Errors
        if self.metrics.get('errors_total', 0) > 0:
            print("  Errors")
            print(f"  - Total errors: {self.metrics.get('errors_total', 0)}")
            print()

        print("="*60)

    def _get_or_create_headers(self, worksheet, default_headers):
        existing = worksheet.row_values(1)
        if not existing:
            worksheet.append_row(default_headers)
            return default_headers
        return existing

    def flush_logs_to_sheets(self, spreadsheet, sheet_name: str = "logs") -> bool:
        """
        Append all buffered logs (INFO + ERROR) to a single Google Sheets tab.
        Schema: run_id, timestamp, level, category, message, data_json
        """
        if self._logs_flushed >= len(self._logs):
            return True

        try:
            try:
                worksheet = spreadsheet.worksheet(sheet_name)
            except:
                worksheet = spreadsheet.add_worksheet(title=sheet_name, rows=1000, cols=10)

            headers = self._get_or_create_headers(
                worksheet, ["run_id", "timestamp", "level", "category", "message", "data_json"]
            )

            pending = self._logs[self._logs_flushed:]
            rows = []
            for entry in pending:
                row_map = {
                    "run_id": entry["run_id"],
                    "timestamp": entry["timestamp"],
                    "level": entry["level"],
                    "category": entry["category"],
                    "message": entry["message"],
                    "data_json": entry["data_json"]
                }
                rows.append([row_map.get(h, "") for h in headers])
            worksheet.append_rows(rows)
            self._logs_flushed = len(self._logs)
            return True
        except Exception as e:
            print(f"  logs upload failed: {e}")
            return False


def sync_logs_to_sheets(csv_path: str, spreadsheet, sheet_name: str = "run_history"):
    """
    Sync run history CSV to Google Sheets.

    Args:
        csv_path: Path to run_history.csv
        spreadsheet: gspread spreadsheet object
        sheet_name: Sheet name (default: "run_history")
    """
    try:
        if not os.path.exists(csv_path):
            print(f"  Log file not found: {csv_path}")
            return False

        # Load CSV (with proper quoting for JSON fields)
        df = pd.read_csv(csv_path, encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)

        if df.empty:
            print("  No log data")
            return False

        # Get or create worksheet
        try:
            worksheet = spreadsheet.worksheet(sheet_name)
        except:
            worksheet = spreadsheet.add_worksheet(title=sheet_name, rows=1000, cols=50)

        # Clear and upload (full replace strategy for logs)
        worksheet.clear()

        # Convert DataFrame to list of lists (with header)
        data = [df.columns.tolist()] + df.values.tolist()

        # Upload
        worksheet.update('A1', data, value_input_option='RAW')

        print(f"  Log sync complete: {len(df)} run records")
        return True

    except Exception as e:
        print(f"  Log Sheets sync failed: {e}")
        return False
