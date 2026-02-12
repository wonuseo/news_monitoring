#!/usr/bin/env python3
"""Google Sheets cleanup: delete rows whose pubDate contains "Jan 2026"."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import os
import time
from typing import Iterable, List, Optional, Tuple

from dotenv import load_dotenv

from src.modules.export.sheets import connect_sheets

PUBDATE_SUBSTRING = "jan 2026"
DATE_COLUMN_NAME = "pubDate"
MAX_ERROR_MESSAGES = 3


def normalize_headers(headers: Iterable[str]) -> List[str]:
    """Return headers stripped and kept in the original order."""
    normalized = []
    for header in headers:
        if isinstance(header, str):
            normalized.append(header.strip())
        else:
            normalized.append(header)
    return normalized


def find_pubdate_column(headers: Iterable[str]) -> Optional[str]:
    """Return the actual header name that matches `pubDate` case-insensitively."""
    normalized = normalize_headers(headers)
    for header in normalized:
        if isinstance(header, str) and header.lower() == DATE_COLUMN_NAME.lower():
            return header
    return None


def rows_with_jan_2026(records: List[dict], column: str) -> List[int]:
    """Return row indices (1-based) that should be deleted."""
    matches = []
    for row_idx, record in enumerate(records, start=2):
        cell_value = record.get(column, "")
        if cell_value is None:
            continue
        if PUBDATE_SUBSTRING in str(cell_value).lower():
            matches.append(row_idx)
    return matches


def make_ranges(row_indices: List[int]) -> List[Tuple[int, int]]:
    """Return descending-order contiguous row ranges (inclusive)."""
    if not row_indices:
        return []

    ranges = []
    sorted_rows = sorted(row_indices)
    current_start = sorted_rows[0]
    current_end = current_start

    for row in sorted_rows[1:]:
        if row == current_end + 1:
            current_end = row
        else:
            ranges.append((current_start, current_end))
            current_start = row
            current_end = row

    ranges.append((current_start, current_end))
    return ranges


def remove_rows(worksheet, row_indices: List[int], dry_run: bool) -> Tuple[int, bool]:
    """Delete rows via contiguous range requests and return how many were deleted."""
    if not row_indices:
        print("  ✅ 삭제 대상 없음 (Jan 2026 미포함).")
        return 0, False

    print(f"  🔬 {len(row_indices)}개 행에서 '{PUBDATE_SUBSTRING}' 발견")
    if dry_run:
        print("  🧪 dry run 모드, 실제 삭제는 수행하지 않습니다.")
        return len(row_indices), False

    ranges = make_ranges(row_indices)
    deleted = 0
    error_count = 0

    sheet_id = getattr(worksheet, "_properties", {}).get("sheetId") or getattr(
        worksheet, "id", None
    )
    if sheet_id is None:
        print("    ⚠️ worksheet.sheetId를 찾을 수 없어 단일 삭제로 fallback합니다.")
        for row_idx in reversed(row_indices):
            try:
                worksheet.delete_rows(row_idx)
                deleted += 1
                time.sleep(0.2)
            except Exception as exc:
                error_count += 1
                print(f"    ⚠️ 삭제 실패: row {row_idx} -> {exc}")
                if error_count >= MAX_ERROR_MESSAGES:
                    print("    ⚠️ 오류가 3회 발생하여 작업을 중단합니다.")
                    return deleted, True
        return deleted, False

    for start, end in reversed(ranges):
        if error_count >= MAX_ERROR_MESSAGES:
            print("    ⚠️ 오류가 3회 발생하여 작업을 중단합니다.")
            break

        row_count = end - start + 1
        request = {
            "requests": [
                {
                    "deleteDimension": {
                        "range": {
                            "sheetId": sheet_id,
                            "dimension": "ROWS",
                            "startIndex": start - 1,
                            "endIndex": end,
                        }
                    }
                }
            ]
        }

        try:
            spreadsheet = getattr(worksheet, "parent", None) or getattr(worksheet, "spreadsheet", None)
            if not spreadsheet:
                raise RuntimeError("Spreadsheet 객체를 불러오지 못했습니다.")
            spreadsheet.batch_update(request)
            deleted += row_count
            print(f"    🗑️ {start}-{end} 삭제 완료 ({row_count}행)")
            time.sleep(0.2)
        except Exception as exc:
            error_count += 1
            print(f"    ⚠️ {start}-{end} 삭제 실패: {exc}")
            if error_count >= MAX_ERROR_MESSAGES:
                print("    ⚠️ 오류가 3회 발생하여 작업을 중단합니다.")
                return deleted, True

    return deleted, error_count >= MAX_ERROR_MESSAGES


def cleanup_worksheet(worksheet, dry_run: bool) -> Tuple[int, bool]:
    """Delete rows whose pubDate column includes the target substring."""
    headers = worksheet.row_values(1)
    if not headers:
        print(f"  ⚠️  '{worksheet.title}'에 헤더가 없어 건너뜀.")
        return 0

    date_column = find_pubdate_column(headers)
    if not date_column:
        print(f"  ⚠️  '{worksheet.title}'에 '{DATE_COLUMN_NAME}' 컬럼이 없습니다.")
        return 0

    try:
        records = worksheet.get_all_records()
    except Exception as exc:
        print(f"  ⚠️  '{worksheet.title}'에서 데이터를 읽는 중 오류: {exc}")
        return 0

    if not records:
        print(f"  ℹ️  '{worksheet.title}'에 데이터가 없습니다.")
        return 0

    row_indices = rows_with_jan_2026(records, date_column)
    deleted, aborted = remove_rows(worksheet, row_indices, dry_run)
    if deleted:
        print(f"  ✅ '{worksheet.title}'에서 {deleted}개 행 삭제 완료")

    return deleted, aborted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="'Jan 2026'이 pubDate에 포함된 행을 Google Sheets에서 삭제합니다."
    )
    parser.add_argument(
        "--worksheets",
        default="total_result",
        help="쉼표로 구분된 워크시트 목록 (기본: total_result).",
    )
    parser.add_argument(
        "--credentials",
        default=None,
        help="서비스 계정 JSON 경로 (.env의 GOOGLE_SHEETS_CREDENTIALS_PATH 우선).",
    )
    parser.add_argument(
        "--sheet-id",
        default=None,
        help="Google Sheets ID (.env의 GOOGLE_SHEET_ID 대신 사용).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="삭제 없이 분석만 수행합니다.",
    )

    args = parser.parse_args()
    load_dotenv()

    creds_path = (
        args.credentials
        or os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH")
        or "service-account.json"
    )
    sheet_id = args.sheet_id or os.getenv("GOOGLE_SHEET_ID")
    if not sheet_id:
        parser.error("Google Sheets ID (.env 또는 --sheet-id) 값이 필요합니다.")

    worksheet_names = [
        name.strip() for name in args.worksheets.split(",") if name.strip()
    ]

    separator = ", "
    print("Jan 2026 Rows 제거 스크립트 시작")
    print(f"  - 대상 시트: {separator.join(worksheet_names)}")
    print(f"  - pubDate 필터: '{PUBDATE_SUBSTRING}'")
    print(f"  - dry run: {'예' if args.dry_run else '아니오'}")

    spreadsheet = connect_sheets(creds_path, sheet_id)
    if not spreadsheet:
        raise SystemExit(1)

    total_deleted = 0
    aborted = False

    for sheet_name in worksheet_names:
        try:
            worksheet = spreadsheet.worksheet(sheet_name)
        except Exception as exc:
            print(f"  ⚠️  워크시트 '{sheet_name}'을(를) 열 수 없음: {exc}")
            continue

        deleted, sheet_aborted = cleanup_worksheet(worksheet, args.dry_run)
        total_deleted += deleted
        if sheet_aborted:
            aborted = True
            break

    if args.dry_run:
        print(f"전체 예상 삭제: {total_deleted}개")
    else:
        print(f"전체 삭제 완료: {total_deleted}개")
    if aborted:
        print("⚠️ 오류 횟수 한도 도달으로 추가 작업을 중단했습니다.")


if __name__ == "__main__":
    main()
