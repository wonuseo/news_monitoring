"""Pipeline context - shared state passed between steps."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd


@dataclass
class PipelineContext:
    """Pipeline context holding shared state across all steps."""

    args: Any                              # argparse.Namespace
    run_mode: str                          # "full" | "dry_run" | "raw_only" | "preprocess_only" | "recheck_only"
    env: Dict[str, str]                    # {"naver_id", "naver_secret", "openai_key"}
    outdir: Path                           # Output directory (used for emergency raw.csv only)
    logger: Any                            # RunLogger instance
    spreadsheet: Optional[Any] = None      # gspread Spreadsheet or None
    sheets_required: bool = True           # False when Sheets not connected (emergency raw-only mode)
    current_stage: str = "init"            # Current pipeline stage
    reasoning_collector: Optional[Any] = None  # ReasoningCollector instance

    # DataFrames that flow between steps
    df_raw: Optional[pd.DataFrame] = None
    df_raw_new: Optional[pd.DataFrame] = None
    df_to_process: Optional[pd.DataFrame] = None
    df_processed: Optional[pd.DataFrame] = None
    df_result: Optional[pd.DataFrame] = None

    def record_error(self, message, data=None, category="system"):
        """Record error and flush to Sheets if connected."""
        self.logger.log_error(message, data, category=category, stage=self.current_stage)
        if self.spreadsheet:
            self.logger.flush_all_to_sheets(self.spreadsheet)
