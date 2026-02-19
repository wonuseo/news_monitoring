"""
Run Logger - Tracks pipeline execution metrics and saves to CSV + Google Sheets

Fixed schema with 3-sheet structure:
- run_history: Fixed-column metrics per run
- errors: ERROR-level logs only
- events: INFO-level logs only
"""

import time
import uuid
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd

from src.utils.sheets_helpers import get_or_create_worksheet


class _NumpyEncoder(json.JSONEncoder):
    """numpy int64/float64 등을 Python 기본형으로 변환하는 JSON 인코더."""
    def default(self, o):
        # numpy 설치 여부와 무관하게 동작하도록 타입명으로 판단
        type_name = type(o).__name__
        if type_name in ("int64", "int32", "int16", "int8", "uint64", "uint32", "uint16", "uint8"):
            return int(o)
        if type_name in ("float64", "float32", "float16"):
            return float(o)
        if type_name == "ndarray":
            return o.tolist()
        if type_name == "bool_":
            return bool(o)
        return super().default(o)


def _dumps(obj, **kwargs) -> str:
    """numpy 타입 안전 json.dumps 래퍼."""
    return json.dumps(obj, cls=_NumpyEncoder, ensure_ascii=False, **kwargs)


# ─── Fixed Schemas ───────────────────────────────────────────────────────────

RUN_HISTORY_SCHEMA = [
    # Basic
    "run_id", "timestamp", "run_mode", "cli_args",
    # Collection
    "articles_collected_total", "articles_collected_per_query", "existing_links_skipped",
    "duration_collection",
    # Reprocess
    "reprocess_targets_total", "reprocess_missing_from_result", "reprocess_field_missing",
    # Processing
    "articles_processed", "duplicates_removed", "articles_filtered_by_date",
    "press_releases_detected", "press_release_groups", "press_release_avg_cluster_size",
    "media_domains_total", "media_domains_new", "media_domains_cached",
    "duration_processing",
    # PR Classification
    "pr_clusters_analyzed", "pr_articles_propagated",
    "pr_llm_success", "pr_llm_failed", "pr_cost_estimated",
    "duration_pr_classification",
    # General Classification
    "articles_classified_llm", "llm_api_calls",
    "classification_errors", "press_releases_skipped", "llm_cost_estimated",
    "duration_general_classification",
    # Source Verification
    "sv_clusters_verified", "sv_kept_press_release",
    "sv_reclassified_similar_topic",
    "sv_cross_merged_groups", "sv_cross_merged_articles",
    "sv_new_topic_groups", "sv_new_topic_articles",
    "sv_llm_verified", "sv_llm_rejected",
    "duration_source_verification",
    # Results
    "our_brands_relevant", "our_brands_negative",
    "danger_high", "danger_medium",
    "competitor_articles", "total_result_count",
    # Sheets Sync
    "sheets_sync_enabled", "sheets_rows_uploaded_raw", "sheets_rows_uploaded_result",
    "duration_sheets_sync",
    # Errors
    "errors_total",
    # Total
    "duration_total",
]

ERRORS_SCHEMA = ["run_id", "timestamp", "category", "stage", "message", "data_json"]
EVENTS_SCHEMA = ["run_id", "timestamp", "category", "stage", "message", "data_json"]


class RunLogger:
    """
    Tracks metrics across pipeline stages and saves to CSV + Google Sheets.

    Usage:
        logger = RunLogger()
        logger.log("articles_collected", 100)
        logger.log_dict({"pr_clusters_analyzed": 5, "pr_llm_success": 5})
        logger.save_to_sheets(spreadsheet)
    """

    def __init__(self):
        self.run_id = str(uuid.uuid4())[:8]
        self.start_time = time.time()
        self.metrics = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
        }
        self._logs: List[Dict[str, Any]] = []
        self._errors_flushed = 0
        self._events_flushed = 0
        self.stage_start_times = {}

    def log(self, key: str, value: Any):
        """Log a single metric"""
        self.metrics[key] = value

    def log_dict(self, metrics_dict: Dict[str, Any]):
        """Log multiple metrics at once"""
        self.metrics.update(metrics_dict)

    def _append_log(self, level: str, category: str, message: str,
                    stage: str = "", data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Internal: append a log entry to the unified buffer."""
        payload = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "category": category,
            "stage": stage,
            "message": message,
            "data_json": _dumps(data) if data else ""
        }
        self._logs.append(payload)
        return payload

    def log_event(self, message: str, data: Optional[Dict[str, Any]] = None,
                  category: str = "pipeline", stage: str = "") -> Dict[str, Any]:
        """Log an INFO-level event."""
        return self._append_log(level="INFO", category=category, message=message,
                                stage=stage, data=data)

    def log_error(self, message: str, data: Optional[Dict[str, Any]] = None,
                  category: str = "system", stage: str = "") -> Dict[str, Any]:
        """Log an ERROR-level event (also print to terminal)."""
        print(f"  ERROR: {message}")
        self.metrics["errors_total"] = self.metrics.get("errors_total", 0) + 1
        return self._append_log(level="ERROR", category=category, message=message,
                                stage=stage, data=data)

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
        """Convert metrics to single-row DataFrame with fixed schema columns."""
        clean_metrics = {}
        for key in RUN_HISTORY_SCHEMA:
            value = self.metrics.get(key, "")
            if value is None:
                clean_metrics[key] = ""
            elif isinstance(value, bool):
                clean_metrics[key] = str(value)
            elif isinstance(value, (dict, list)):
                clean_metrics[key] = _dumps(value)
            else:
                clean_metrics[key] = value

        return pd.DataFrame([clean_metrics], columns=RUN_HISTORY_SCHEMA)

    def save_to_sheets(self, spreadsheet, sheet_name: str = "run_history") -> bool:
        """
        Append current run metrics to run_history sheet in Google Sheets.
        """
        try:
            df = self.to_dataframe()
            worksheet = get_or_create_worksheet(spreadsheet, sheet_name, rows=1000, cols=60)
            headers = self._get_or_create_headers(worksheet, RUN_HISTORY_SCHEMA)

            row_values = [
                _clean_value_for_sheets(df[col].iloc[0]) if col in df.columns else ""
                for col in headers
            ]
            worksheet.append_row(row_values)
            print(f"\n  Run history saved to Sheets: {sheet_name}")
            return True

        except Exception as e:
            print(f"\n  Run history Sheets save failed: {e}")
            return False

    def print_summary(self):
        """Print a formatted summary of key metrics (5-stage structure)"""
        print("\n" + "=" * 60)
        print("  Run Summary")
        print("=" * 60)

        # Basic info
        print(f"Run ID: {self.metrics.get('run_id', 'N/A')}")
        print(f"Timestamp: {self.metrics.get('timestamp', 'N/A')}")
        print(f"Run Mode: {self.metrics.get('run_mode', 'N/A')}")
        print(f"Total Duration: {self.metrics.get('duration_total', 0):.2f}s")
        print()

        # Collection
        print("  Collection")
        print(f"  - Total collected: {self.metrics.get('articles_collected_total', 0)}")
        if 'articles_collected_per_query' in self.metrics:
            per_query = self.metrics['articles_collected_per_query']
            if isinstance(per_query, str):
                try:
                    per_query = json.loads(per_query)
                except (json.JSONDecodeError, TypeError):
                    per_query = {}
            if isinstance(per_query, dict):
                for brand, count in per_query.items():
                    print(f"    - {brand}: {count}")
        print(f"  - Existing links skipped: {self.metrics.get('existing_links_skipped', 0)}")
        print(f"  - Duration: {self.metrics.get('duration_collection', 0):.2f}s")
        print()

        # Reprocess
        reprocess_total = self.metrics.get('reprocess_targets_total', 0)
        if reprocess_total:
            print("  Reprocess Check")
            print(f"  - Total targets: {reprocess_total}")
            print(f"  - Missing from result: {self.metrics.get('reprocess_missing_from_result', 0)}")
            field_missing = self.metrics.get('reprocess_field_missing', '')
            if field_missing:
                if isinstance(field_missing, str):
                    try:
                        field_missing = json.loads(field_missing)
                    except (json.JSONDecodeError, TypeError):
                        field_missing = {}
                if isinstance(field_missing, dict):
                    for field, count in field_missing.items():
                        print(f"    - {field}: {count}")
            print()

        # Processing
        print("  Processing")
        print(f"  - Duplicates removed: {self.metrics.get('duplicates_removed', 0)}")
        print(f"  - Processed: {self.metrics.get('articles_processed', 0)}")
        print(f"  - Date filtered: {self.metrics.get('articles_filtered_by_date', 0)}")
        print(f"  - Press releases detected: {self.metrics.get('press_releases_detected', 0)}")
        print(f"  - Press release groups: {self.metrics.get('press_release_groups', 0)}")
        avg_size = self.metrics.get('press_release_avg_cluster_size', '')
        if avg_size:
            print(f"  - Avg cluster size: {avg_size}")
        print(f"  - Duration: {self.metrics.get('duration_processing', 0):.2f}s")
        print()

        # Media classification
        if self.metrics.get('media_domains_total', 0):
            print("  Media Classification")
            print(f"  - Total domains: {self.metrics.get('media_domains_total', 0)}")
            print(f"  - New classified: {self.metrics.get('media_domains_new', 0)}")
            print(f"  - Cached: {self.metrics.get('media_domains_cached', 0)}")
            print()

        # PR Classification
        pr_clusters = self.metrics.get('pr_clusters_analyzed', 0)
        if pr_clusters:
            print("  PR Classification")
            print(f"  - Clusters analyzed: {pr_clusters}")
            print(f"  - Articles propagated: {self.metrics.get('pr_articles_propagated', 0)}")
            print(f"  - LLM success: {self.metrics.get('pr_llm_success', 0)}")
            print(f"  - LLM failed: {self.metrics.get('pr_llm_failed', 0)}")
            pr_cost = self.metrics.get('pr_cost_estimated', 0)
            if pr_cost:
                print(f"  - Estimated cost: ${pr_cost:.4f}")
            print(f"  - Duration: {self.metrics.get('duration_pr_classification', 0):.2f}s")
            print()

        # General Classification
        print("  General Classification")
        print(f"  - Classified: {self.metrics.get('articles_classified_llm', 0)}")
        print(f"  - API calls: {self.metrics.get('llm_api_calls', 0)}")
        llm_cost = self.metrics.get('llm_cost_estimated', 0)
        if llm_cost:
            print(f"  - Estimated cost: ${llm_cost:.4f}")
        print(f"  - Press releases skipped: {self.metrics.get('press_releases_skipped', 0)}")
        print(f"  - Classification errors: {self.metrics.get('classification_errors', 0)}")
        print(f"  - Duration: {self.metrics.get('duration_general_classification', 0):.2f}s")
        print()

        # Source Verification
        sv_verified = self.metrics.get('sv_clusters_verified', 0)
        sv_cross_groups = self.metrics.get('sv_cross_merged_groups', 0)
        sv_new_groups = self.metrics.get('sv_new_topic_groups', 0)
        if sv_verified or sv_cross_groups or sv_new_groups:
            print("  Source Verification")
            if sv_verified:
                print(f"  - Clusters verified: {sv_verified}")
                print(f"  - Kept as press release: {self.metrics.get('sv_kept_press_release', 0)}")
                print(f"  - Reclassified similar topic: {self.metrics.get('sv_reclassified_similar_topic', 0)}")
            if sv_cross_groups:
                print(f"  - Cross-query merged groups: {sv_cross_groups}")
                print(f"  - Cross-query merged articles: {self.metrics.get('sv_cross_merged_articles', 0)}")
            if sv_new_groups:
                print(f"  - New topic groups: {sv_new_groups}")
                print(f"  - New topic articles: {self.metrics.get('sv_new_topic_articles', 0)}")
            print(f"  - Duration: {self.metrics.get('duration_source_verification', 0):.2f}s")
            print()

        # Results
        print("  Results")
        print(f"  - Our brands relevant: {self.metrics.get('our_brands_relevant', 0)}")
        print(f"  - Our brands negative: {self.metrics.get('our_brands_negative', 0)}")
        print(f"  - Danger high: {self.metrics.get('danger_high', 0)}")
        print(f"  - Danger medium: {self.metrics.get('danger_medium', 0)}")
        print(f"  - Competitor articles: {self.metrics.get('competitor_articles', 0)}")
        total_result = self.metrics.get('total_result_count', '')
        if total_result:
            print(f"  - Total result count: {total_result}")
        print()

        # Sheets sync
        if self.metrics.get('sheets_sync_enabled'):
            print("  Google Sheets Sync")
            print(f"  - raw_data uploaded: {self.metrics.get('sheets_rows_uploaded_raw', 0)} rows")
            print(f"  - result uploaded: {self.metrics.get('sheets_rows_uploaded_result', 0)} rows")
            print(f"  - Duration: {self.metrics.get('duration_sheets_sync', 0):.2f}s")
            print()

        # Errors
        if self.metrics.get('errors_total', 0) > 0:
            print("  Errors")
            print(f"  - Total errors: {self.metrics.get('errors_total', 0)}")
            print()

        print("=" * 60)

    # ─── Sheets Flush (3-sheet split) ────────────────────────────────────────

    def _get_or_create_headers(self, worksheet, default_headers):
        existing = worksheet.row_values(1)
        if not existing:
            worksheet.append_row(default_headers)
            return default_headers
        return existing

    def _flush_filtered_logs(self, spreadsheet, sheet_name: str,
                             level_filter: str, schema: List[str],
                             flushed_attr: str) -> bool:
        """
        Flush logs matching level_filter to a specific sheet.
        Returns True on success.
        """
        flushed_count = getattr(self, flushed_attr)

        # Filter pending logs by level
        pending = [log for log in self._logs[flushed_count:]
                   if log["level"] == level_filter]

        if not pending:
            # Update flushed pointer even if no matching logs
            setattr(self, flushed_attr, len(self._logs))
            return True

        try:
            worksheet = get_or_create_worksheet(spreadsheet, sheet_name, rows=1000, cols=10)
            headers = self._get_or_create_headers(worksheet, schema)

            rows = []
            for entry in pending:
                rows.append([entry.get(h, "") for h in headers])

            worksheet.append_rows(rows)
            setattr(self, flushed_attr, len(self._logs))
            return True
        except Exception as e:
            print(f"  {sheet_name} upload failed: {e}")
            return False

    def flush_errors_to_sheets(self, spreadsheet) -> bool:
        """Flush ERROR-level logs to 'errors' sheet."""
        return self._flush_filtered_logs(
            spreadsheet, "errors", "ERROR", ERRORS_SCHEMA, "_errors_flushed"
        )

    def flush_events_to_sheets(self, spreadsheet) -> bool:
        """Flush INFO-level logs to 'events' sheet."""
        return self._flush_filtered_logs(
            spreadsheet, "events", "INFO", EVENTS_SCHEMA, "_events_flushed"
        )

    def flush_all_to_sheets(self, spreadsheet) -> bool:
        """Flush both errors and events to their respective sheets."""
        err_ok = self.flush_errors_to_sheets(spreadsheet)
        evt_ok = self.flush_events_to_sheets(spreadsheet)
        return err_ok and evt_ok

    # Keep old method name for backward compatibility during transition
    def flush_logs_to_sheets(self, spreadsheet, sheet_name: str = "logs") -> bool:
        """Deprecated: use flush_all_to_sheets() instead."""
        return self.flush_all_to_sheets(spreadsheet)


def _clean_value_for_sheets(v):
    """NaN/inf를 빈 문자열로 변환 (Sheets API JSON 직렬화 안전)"""
    import math
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return ""
    return v


# Kept for backward compatibility — prefer RunLogger.save_to_sheets() for new code
