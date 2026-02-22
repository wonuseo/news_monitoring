#!/usr/bin/env python3
"""Ensure reasoning sheet has required headers."""

import argparse
import os
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modules.analysis.reasoning_writer import REASONING_COLUMNS
from src.modules.export.sheets import connect_sheets
from src.utils.sheets_helpers import get_or_create_worksheet


def _normalize(value: str) -> str:
    return str(value or "").strip().lower()


def _col_to_letter(col_num: int) -> str:
    result = []
    while col_num > 0:
        col_num, rem = divmod(col_num - 1, 26)
        result.append(chr(65 + rem))
    return "".join(reversed(result))


def _build_headers(existing: List[str]) -> List[str]:
    if not existing or all(not str(h).strip() for h in existing):
        return list(REASONING_COLUMNS)

    final_headers = [str(h).strip() for h in existing]
    existing_norm = {_normalize(h) for h in final_headers}
    for required in REASONING_COLUMNS:
        if _normalize(required) not in existing_norm:
            final_headers.append(required)
            existing_norm.add(_normalize(required))
    return final_headers


def main() -> int:
    parser = argparse.ArgumentParser(description="Add missing headers to reasoning sheet.")
    parser.add_argument(
        "--credentials",
        help="Service account JSON path (defaults to GOOGLE_SHEETS_CREDENTIALS_PATH).",
    )
    parser.add_argument(
        "--sheet-id",
        help="Google Sheet ID (defaults to GOOGLE_SHEET_ID).",
    )
    parser.add_argument(
        "--sheet-name",
        default="reasoning",
        help="Worksheet name (default: reasoning).",
    )
    args = parser.parse_args()

    load_dotenv()
    credentials_path = args.credentials or os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH")
    sheet_id = args.sheet_id or os.getenv("GOOGLE_SHEET_ID")

    if not credentials_path or not os.path.exists(credentials_path):
        print("❌ credentials 파일이 없거나 경로가 잘못되었습니다.")
        print("   --credentials 또는 GOOGLE_SHEETS_CREDENTIALS_PATH를 확인하세요.")
        return 1
    if not sheet_id:
        print("❌ Google Sheet ID가 없습니다.")
        print("   --sheet-id 또는 GOOGLE_SHEET_ID를 설정하세요.")
        return 1

    spreadsheet = connect_sheets(credentials_path, sheet_id)
    if not spreadsheet:
        print("❌ Google Sheets 연결 실패")
        return 1

    worksheet = get_or_create_worksheet(
        spreadsheet,
        args.sheet_name,
        rows=5000,
        cols=max(len(REASONING_COLUMNS), 11),
    )

    existing_headers = worksheet.row_values(1)
    final_headers = _build_headers(existing_headers)

    if worksheet.col_count < len(final_headers):
        worksheet.add_cols(len(final_headers) - worksheet.col_count)

    end_col = _col_to_letter(len(final_headers))
    worksheet.update(f"A1:{end_col}1", [final_headers], value_input_option="RAW")

    added_count = max(0, len(final_headers) - len(existing_headers))
    print(f"✅ '{args.sheet_name}' 헤더 보정 완료")
    print(f"   - 기존 헤더 수: {len(existing_headers)}")
    print(f"   - 최종 헤더 수: {len(final_headers)}")
    print(f"   - 추가된 헤더 수: {added_count}")
    print(f"   - 필수 헤더: {', '.join(REASONING_COLUMNS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
