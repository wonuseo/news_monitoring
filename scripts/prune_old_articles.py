#!/usr/bin/env python3
"""One-time Google Sheets cleanup: drop pre-2026-02-01 articles."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import os
import time
from datetime import datetime, timezone
from typing import Iterable, List, Optional

from dateutil import parser as date_parser
from dotenv import load_dotenv

from src.modules.export.sheets import connect_sheets


DATE_COLUMN_PRIORITY = [
    "pub_datetime",
    "date_only",
    "pubDate",
    "pub_date",
    "published_at",
    "date",
]


def parse_cutoff(value: str) -> datetime:
    """Convert user cutoff (e.g., 2026-02-01) to timezone-aware UTC."""
    parsed = date_parser.parse(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_sheet_date(value) -> Optional[datetime]:
    """Try to parse a date string from the sheet and return UTC."""
    if value is None:
        return None

    if isinstance(value, (int, float)):
        value = str(value)

    try:
        parsed = date_parser.parse(str(value))
    except (ValueError, OverflowError):
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def find_date_column(headers: Iterable[str]) -> Optional[str]:
    """Return the header name that best matches a publication column."""
    normalized = {header.lower(): header for header in headers if isinstance(header, str)}
    for candidate in DATE_COLUMN_PRIORITY:
        candidate_lower = candidate.lower()
        if candidate_lower in normalized:
            return normalized[candidate_lower]
    return None


def prune_worksheet(worksheet, cutoff: datetime, dry_run: bool) -> int:
    """Remove rows older than cutoff; returns number of deleted rows."""
    headers = worksheet.row_values(1)
    if not headers:
        print(f"  ⚠️  '{worksheet.title}'에 헤더가 없어 건너뜀.")
        return 0

    date_column = find_date_column(headers)
    if not date_column:
        header_names = ", ".join(headers[:5])
        print(
            f"  ⚠️  '{worksheet.title}'에서 날짜 컬럼을 찾을 수 없음 (예시: {header_names})."
        )
        return 0

    records = worksheet.get_all_records()
    if not records:
        print(f"  ℹ️  '{worksheet.title}'에 데이터가 없습니다.")
        return 0

    rows_to_delete: List[int] = []
    skipped = 0
    for row_idx, record in enumerate(records, start=2):
        published_value = record.get(date_column, "")
        published_dt = parse_sheet_date(published_value)
        if not published_dt:
            skipped += 1
            continue
        if published_dt < cutoff:
            rows_to_delete.append(row_idx)

    if not rows_to_delete:
        print(f"  ✅ '{worksheet.title}'에 삭제 대상이 없습니다 (해당 날짜 이전 없음).")
        return 0

    deleted_count = 0
    if dry_run:
        print(
            f"  🧪 '{worksheet.title}'에서 삭제 예정 행: {len(rows_to_delete)} (절삭 날짜: {cutoff.date()})"
        )
        return len(rows_to_delete)

    print(
        f"  🧹 '{worksheet.title}'에서 {len(rows_to_delete)}개 행 삭제 (기준: {cutoff.date()})"
    )
    for row_idx in reversed(rows_to_delete):
        try:
            worksheet.delete_rows(row_idx)
            deleted_count += 1
            time.sleep(0.2)
        except Exception as exc:
            print(f"    ⚠️ 삭제 실패: {worksheet.title} row {row_idx} -> {exc}")
    print(f"  ✅ '{worksheet.title}': 총 {deleted_count}개 행 삭제")
    if skipped:
        print(f"    ℹ️  {skipped}개 행의 날짜 파싱 실패 (무시됨)")
    return deleted_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Google Sheets에서 2026-02-01 이전 기사를 정리합니다."
    )
    parser.add_argument(
        "--cutoff",
        default="2026-02-01",
        help="삭제 기준 날짜 (ISO, 기본: 2026-02-01).",
    )
    parser.add_argument(
        "--worksheets",
        default="total_result",
        help="쉼표로 구분된 워크시트 목록 (기본: total_result).",
    )
    parser.add_argument(
        "--credentials",
        default=None,
        help="서비스 계정 JSON 경로 (.env의 GOOGLE_SHEETS_CREDENTIALS_PATH를 건너뜀)",
    )
    parser.add_argument(
        "--sheet-id",
        default=None,
        help="Google Sheets ID (.env의 GOOGLE_SHEET_ID 대신 사용)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="삭제 없이 미리보기만 실행합니다.",
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

    cutoff = parse_cutoff(args.cutoff)
    worksheet_names = [name.strip() for name in args.worksheets.split(",") if name.strip()]

    print("Google Sheets 정리 스크립트 실행")
    print(f"  - 기준 날짜: {cutoff.isoformat()}")
    print(f"  - 워크시트: {', '.join(worksheet_names)}")
    print("  - dry run:" + (" 예" if args.dry_run else " 아니오"))

    spreadsheet = connect_sheets(creds_path, sheet_id)
    if not spreadsheet:
        raise SystemExit(1)

    total_deleted = 0
    for sheet_name in worksheet_names:
        try:
            worksheet = spreadsheet.worksheet(sheet_name)
        except Exception as exc:
            print(f"  ⚠️  워크시트 '{sheet_name}'을(를) 열 수 없음: {exc}")
            continue
        deleted = prune_worksheet(worksheet, cutoff, args.dry_run)
        total_deleted += deleted

    if args.dry_run:
        print(f"전체 예상 삭제: {total_deleted}개")
    else:
        print(f"전체 삭제 완료: {total_deleted}개")


if __name__ == "__main__":
    main()
