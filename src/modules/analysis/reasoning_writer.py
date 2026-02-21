"""
reasoning_writer.py - Thread-safe collector for LLM reasoning entries.
분류 reasoning을 수집하여 Google Sheets "reasoning" 탭에 batch append.
"""

import threading
from typing import Dict, List, Optional

from src.utils.sheets_helpers import get_or_create_worksheet


# reasoning 탭 컬럼 순서 (11컬럼)
REASONING_COLUMNS = [
    "article_id",
    "classified_at",
    "article_subject",
    "brand_role",
    "subject_test",
    "sentiment_rationale",
    "news_category_rationale",
    "severity_analysis",
    "attribution_analysis",
    "spread_signals",
    "final_judgment",
]


class ReasoningCollector:
    """Thread-safe collector for LLM reasoning entries."""

    def __init__(self):
        self._entries: Dict[str, Dict] = {}  # article_id -> entry
        self._lock = threading.Lock()

    def add_first_pass(self, article_id: str, classified_at: str, reasoning: Optional[Dict]):
        """1차 reasoning 수집 (brand_relevance / sentiment / news_category)."""
        if not reasoning or not article_id:
            return

        entry = {
            "article_id": article_id,
            "classified_at": classified_at,
            "article_subject": reasoning.get("article_subject", ""),
            "brand_role": reasoning.get("brand_role", ""),
            "subject_test": reasoning.get("subject_test", ""),
            "sentiment_rationale": reasoning.get("sentiment_rationale", ""),
            "news_category_rationale": reasoning.get("news_category_rationale", ""),
            # 2차 reasoning (부정 기사만 채워짐)
            "severity_analysis": "",
            "attribution_analysis": "",
            "spread_signals": "",
            "final_judgment": "",
        }
        with self._lock:
            self._entries[article_id] = entry

    def update_second_pass(self, article_id: str, reasoning: Optional[Dict]):
        """2차 reasoning 병합 (부정 기사 — danger_level / issue_category 분석)."""
        if not reasoning or not article_id:
            return

        with self._lock:
            if article_id not in self._entries:
                return
            self._entries[article_id].update({
                "severity_analysis": reasoning.get("severity_analysis", ""),
                "attribution_analysis": reasoning.get("attribution_analysis", ""),
                "spread_signals": reasoning.get("spread_signals", ""),
                "final_judgment": reasoning.get("final_judgment", ""),
            })

    def flush_to_sheets(self, spreadsheet) -> bool:
        """"reasoning" 탭에 수집된 항목을 batch append."""
        if spreadsheet is None:
            return False

        with self._lock:
            entries = list(self._entries.values())

        if not entries:
            print("ℹ️  reasoning 수집 항목 없음, Sheets 업로드 스킵")
            return True

        try:
            worksheet = get_or_create_worksheet(
                spreadsheet, "reasoning", rows=5000, cols=len(REASONING_COLUMNS)
            )

            # 헤더가 없으면 삽입
            existing_data = worksheet.get_all_values()
            if not existing_data:
                worksheet.append_row(REASONING_COLUMNS)

            # 데이터 행 구성
            rows: List[List[str]] = []
            for entry in entries:
                row = [str(entry.get(col, "")) for col in REASONING_COLUMNS]
                rows.append(row)

            worksheet.append_rows(rows, value_input_option="USER_ENTERED")
            print(f"✅ reasoning 탭 업로드 완료: {len(rows)}개 항목")
            return True

        except Exception as e:
            print(f"⚠️  reasoning 탭 업로드 실패: {e}")
            return False
