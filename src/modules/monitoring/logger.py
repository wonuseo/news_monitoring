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
        self.stage_start_times = {}

    def log(self, key: str, value: Any):
        """Log a single metric"""
        self.metrics[key] = value

    def log_dict(self, metrics_dict: Dict[str, Any]):
        """Log multiple metrics at once"""
        self.metrics.update(metrics_dict)

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
                df.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)
            else:
                df.to_csv(csv_path, mode='w', header=True, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)

            print(f"\n📊 로그 저장: {csv_path}")
            return True

        except Exception as e:
            print(f"\n⚠️ 로그 CSV 저장 실패: {e}")
            return False

    def print_summary(self):
        """Print a formatted summary of key metrics"""
        print("\n" + "="*60)
        print("📊 실행 요약 (Run Summary)")
        print("="*60)

        # Basic info
        print(f"Run ID: {self.metrics.get('run_id', 'N/A')}")
        print(f"Timestamp: {self.metrics.get('timestamp', 'N/A')}")
        print(f"Total Duration: {self.metrics.get('duration_total', 0):.2f}초")
        print()

        # Collection
        print("📥 수집 (Collection)")
        print(f"  - 전체 수집: {self.metrics.get('articles_collected_total', 0)}개")
        if 'articles_collected_per_query' in self.metrics:
            per_query = self.metrics['articles_collected_per_query']
            if isinstance(per_query, str):
                per_query = json.loads(per_query)
            for brand, count in per_query.items():
                print(f"    • {brand}: {count}개")
        print(f"  - 기존 링크 스킵: {self.metrics.get('existing_links_skipped', 0)}개")
        print(f"  - 소요 시간: {self.metrics.get('duration_collection', 0):.2f}초")
        print()

        # Processing
        print("⚙️ 전처리 (Processing)")
        print(f"  - 중복 제거: {self.metrics.get('duplicates_removed', 0)}개")
        print(f"  - 전처리 완료: {self.metrics.get('articles_processed', 0)}개")
        print(f"  - 보도자료 탐지: {self.metrics.get('press_releases_detected', 0)}개")
        print(f"  - 보도자료 그룹: {self.metrics.get('press_release_groups', 0)}개")
        print(f"  - 소요 시간: {self.metrics.get('duration_processing', 0):.2f}초")
        print()

        # Media classification
        if self.metrics.get('media_domains_total', 0) > 0:
            print("🏢 미디어 분류 (Media Classification)")
            print(f"  - 전체 도메인: {self.metrics.get('media_domains_total', 0)}개")
            print(f"  - 신규 분류: {self.metrics.get('media_domains_new', 0)}개")
            print(f"  - 캐시 사용: {self.metrics.get('media_domains_cached', 0)}개")
            print()

        # Classification
        print("🤖 LLM 분석 (Classification)")
        print(f"  - 분석 완료: {self.metrics.get('articles_classified_llm', 0)}개")
        print(f"  - API 호출: {self.metrics.get('llm_api_calls', 0)}회")
        print(f"  - 추정 비용: ${self.metrics.get('llm_cost_estimated', 0):.4f}")
        print(f"  - 보도자료 스킵: {self.metrics.get('press_releases_skipped', 0)}개")
        print(f"  - 분류 실패: {self.metrics.get('classification_errors', 0)}개")
        print(f"  - 소요 시간: {self.metrics.get('duration_classification', 0):.2f}초")
        print()

        # Results
        print("📈 분류 결과 (Results)")
        print(f"  - 우리 브랜드 관련: {self.metrics.get('our_brands_relevant', 0)}개")
        print(f"  - 우리 브랜드 부정: {self.metrics.get('our_brands_negative', 0)}개")
        print(f"  - 위험도 상: {self.metrics.get('danger_high', 0)}개")
        print(f"  - 위험도 중: {self.metrics.get('danger_medium', 0)}개")
        print(f"  - 경쟁사 기사: {self.metrics.get('competitor_articles', 0)}개")
        print()

        # Sheets sync
        if self.metrics.get('sheets_sync_enabled'):
            print("☁️ Google Sheets 동기화")
            print(f"  - raw_data 업로드: {self.metrics.get('sheets_rows_uploaded_raw', 0)}행")
            print(f"  - result 업로드: {self.metrics.get('sheets_rows_uploaded_result', 0)}행")
            print(f"  - logs 업로드: {self.metrics.get('sheets_logs_uploaded', 0)}행")
            print(f"  - 소요 시간: {self.metrics.get('duration_sheets_sync', 0):.2f}초")
            print()

        # Errors
        if self.metrics.get('errors_total', 0) > 0:
            print("⚠️ 에러")
            print(f"  - 전체 에러: {self.metrics.get('errors_total', 0)}개")
            print(f"  - 경고: {self.metrics.get('warnings_total', 0)}개")
            print()

        print("="*60)


def sync_logs_to_sheets(csv_path: str, spreadsheet, sheet_name: str = "logs"):
    """
    Sync run history CSV to Google Sheets.

    Args:
        csv_path: Path to run_history.csv
        spreadsheet: gspread spreadsheet object
        sheet_name: Sheet name (default: "logs")
    """
    try:
        if not os.path.exists(csv_path):
            print(f"⚠️ 로그 파일 없음: {csv_path}")
            return False

        # Load CSV (with proper quoting for JSON fields)
        df = pd.read_csv(csv_path, encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)

        if df.empty:
            print("⚠️ 로그 데이터 없음")
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

        print(f"✅ 로그 동기화 완료: {len(df)}개 실행 기록")
        return True

    except Exception as e:
        print(f"⚠️ 로그 Sheets 동기화 실패: {e}")
        return False
