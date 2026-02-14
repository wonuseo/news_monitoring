#!/usr/bin/env python3
"""Delete duplicate rows in Google Sheets based on originallink column."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import os
import time
from typing import Iterable, List, Optional, Tuple

from dotenv import load_dotenv

from src.modules.export.sheets import connect_sheets

ORIGINALLINK_COLUMN_PRIORITY = [
    "originallink",
    "original_link",
    "original link",
]
BRAND_RELEVANCE_COLUMN_PRIORITY = [
    "brand_relevance",
    "brand relevance",
]
MAX_ERROR_MESSAGES = 3


def find_column_index(headers: Iterable[str], candidates: List[str]) -> Optional[int]:
    normalized_headers = [
        header.strip().lower() if isinstance(header, str) else "" for header in headers
    ]
    for candidate in candidates:
        candidate_lower = candidate.lower()
        for idx, header in enumerate(normalized_headers):
            if header == candidate_lower:
                return idx
    return None


def normalize_link(value: str) -> str:
    return value.strip()


def has_non_empty_cell(row: List[str], col_idx: Optional[int]) -> bool:
    if col_idx is None or len(row) <= col_idx:
        return False
    return bool(str(row[col_idx]).strip())


def pick_keeper(group_rows: List[Tuple[int, List[str]]], brand_col_idx: Optional[int], keep: str) -> int:
    brand_rows = [
        row_idx for row_idx, row in group_rows if has_non_empty_cell(row, brand_col_idx)
    ]
    if brand_rows:
        return brand_rows[0] if keep == "first" else brand_rows[-1]
    return group_rows[0][0] if keep == "first" else group_rows[-1][0]


def find_duplicate_rows(
    data_rows: List[List[str]], link_col_idx: int, brand_col_idx: Optional[int], keep: str
) -> Tuple[List[int], int, int]:
    grouped: dict[str, List[Tuple[int, List[str]]]] = {}
    for row_idx, row in enumerate(data_rows, start=2):
        link = normalize_link(row[link_col_idx]) if len(row) > link_col_idx else ""
        if not link:
            continue
        grouped.setdefault(link, []).append((row_idx, row))

    rows_to_delete: List[int] = []
    duplicate_count = 0
    kept_by_brand = 0

    for group_rows in grouped.values():
        if len(group_rows) <= 1:
            continue

        keeper = pick_keeper(group_rows, brand_col_idx, keep)
        if any(has_non_empty_cell(row, brand_col_idx) for _, row in group_rows):
            kept_by_brand += 1

        for row_idx, _ in group_rows:
            if row_idx != keeper:
                rows_to_delete.append(row_idx)
                duplicate_count += 1

    return rows_to_delete, duplicate_count, kept_by_brand


def make_ranges(row_indices: List[int]) -> List[Tuple[int, int]]:
    if not row_indices:
        return []

    sorted_rows = sorted(row_indices)
    ranges: List[Tuple[int, int]] = []
    start = sorted_rows[0]
    end = start

    for row in sorted_rows[1:]:
        if row == end + 1:
            end = row
        else:
            ranges.append((start, end))
            start = row
            end = row

    ranges.append((start, end))
    return ranges


def delete_rows_with_ranges(worksheet, row_indices: List[int], dry_run: bool) -> Tuple[int, bool]:
    if not row_indices:
        print("  ✅ 중복 행이 없습니다.")
        return 0, False

    print(f"  🔎 중복 행 {len(row_indices)}개 발견")
    if dry_run:
        print("  🧪 dry run 모드, 실제 삭제는 수행하지 않습니다.")
        return len(row_indices), False

    sheet_id = getattr(worksheet, "_properties", {}).get("sheetId") or getattr(
        worksheet, "id", None
    )
    spreadsheet = getattr(worksheet, "parent", None) or getattr(
        worksheet, "spreadsheet", None
    )

    deleted = 0
    error_count = 0

    if sheet_id is None or spreadsheet is None:
        print("  ⚠️ batch delete를 사용할 수 없어 단일 삭제로 진행합니다.")
        for row_idx in sorted(row_indices, reverse=True):
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

    for start, end in reversed(make_ranges(row_indices)):
        if error_count >= MAX_ERROR_MESSAGES:
            print("    ⚠️ 오류가 3회 발생하여 작업을 중단합니다.")
            break

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
        row_count = end - start + 1
        try:
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


def cleanup_duplicates(worksheet, keep: str, dry_run: bool) -> Tuple[int, bool]:
    all_rows = worksheet.get_all_values()
    if not all_rows:
        print(f"  ℹ️  '{worksheet.title}' 시트가 비어 있습니다.")
        return 0, False

    headers = all_rows[0]
    link_col_idx = find_column_index(headers, ORIGINALLINK_COLUMN_PRIORITY)
    if link_col_idx is None:
        print(
            "  ⚠️  originallink 컬럼을 찾을 수 없습니다. "
            f"(시도: {', '.join(ORIGINALLINK_COLUMN_PRIORITY)})"
        )
        return 0, False

    brand_col_idx = find_column_index(headers, BRAND_RELEVANCE_COLUMN_PRIORITY)
    if brand_col_idx is None:
        print(
            "  ⚠️  brand_relevance 컬럼을 찾지 못했습니다. "
            "중복 내 보존 우선순위는 행 순서로만 결정됩니다."
        )

    data_rows = all_rows[1:]
    if not data_rows:
        print(f"  ℹ️  '{worksheet.title}'에 데이터 행이 없습니다.")
        return 0, False

    row_indices, duplicate_count, kept_by_brand = find_duplicate_rows(
        data_rows, link_col_idx, brand_col_idx, keep
    )
    if duplicate_count == 0:
        print("  ✅ originallink 기준 중복이 없습니다.")
        return 0, False

    print(
        f"  - 기준 컬럼: '{headers[link_col_idx]}' / keep='{keep}' / duplicates={duplicate_count}"
    )
    if kept_by_brand:
        print(f"  - brand_relevance 보존 적용 그룹: {kept_by_brand}개")
    deleted, aborted = delete_rows_with_ranges(worksheet, row_indices, dry_run)
    if deleted:
        print(f"  ✅ '{worksheet.title}'에서 {deleted}개 행 처리 완료")
    return deleted, aborted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Google Sheets에서 originallink 기준 중복 행을 삭제합니다."
    )
    parser.add_argument(
        "--worksheets",
        default="total_result",
        help="쉼표로 구분된 워크시트 목록 (기본: total_result).",
    )
    parser.add_argument(
        "--keep",
        choices=["first", "last"],
        default="first",
        help="brand_relevance 우선 후 동률일 때 어떤 행을 남길지 선택 (기본: first).",
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

    worksheet_names = [name.strip() for name in args.worksheets.split(",") if name.strip()]

    print("originallink 중복 제거 스크립트 시작")
    print(f"  - 대상 워크시트: {', '.join(worksheet_names)}")
    print(f"  - keep 정책: brand_relevance 우선 + {args.keep} tie-break")
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
            print(f"  ⚠️ 워크시트 '{sheet_name}'을(를) 열 수 없음: {exc}")
            continue

        deleted, sheet_aborted = cleanup_duplicates(worksheet, args.keep, args.dry_run)
        total_deleted += deleted
        if sheet_aborted:
            aborted = True
            break

    if args.dry_run:
        print(f"전체 예상 삭제: {total_deleted}개")
    else:
        print(f"전체 삭제 완료: {total_deleted}개")
    if aborted:
        print("⚠️ 오류 횟수 한도 도달로 작업을 중단했습니다.")


if __name__ == "__main__":
    main()
