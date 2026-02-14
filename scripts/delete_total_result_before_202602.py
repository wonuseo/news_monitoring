#!/usr/bin/env python3
"""Delete rows in total_result where article date is before 2026-02-01."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import os
import time
from datetime import datetime, timezone, date
from email.utils import parsedate_to_datetime
from typing import Iterable, List, Optional
from zoneinfo import ZoneInfo

from dateutil import parser as date_parser
from dotenv import load_dotenv

from src.modules.export.sheets import connect_sheets


DATE_COLUMN_PRIORITY = [
    "pubDate",
    "pub_datetime",
    "date_only",
    "pub_date",
    "published_at",
    "date",
]
DEFAULT_TIMEZONE = "Asia/Seoul"
DEFAULT_CUTOFF_DATE = "2026-01-31"


def make_ranges(row_indices: List[int]) -> List[tuple[int, int]]:
    if not row_indices:
        return []

    sorted_rows = sorted(row_indices)
    ranges: List[tuple[int, int]] = []
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


def parse_cutoff_date(value: str) -> date:
    return date_parser.parse(value).date()


def parse_sheet_date(value) -> Optional[datetime]:
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


def parse_pubdate(value) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return parse_sheet_date(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def find_date_column_index(headers: Iterable[str]) -> Optional[int]:
    normalized = [
        header.strip().lower() if isinstance(header, str) else "" for header in headers
    ]
    for candidate in DATE_COLUMN_PRIORITY:
        candidate_lower = candidate.lower()
        for idx, header in enumerate(normalized):
            if header == candidate_lower:
                return idx
    return None


def delete_before_cutoff(
    worksheet,
    cutoff_local_date: date,
    tz_name: str,
    dry_run: bool,
) -> int:
    all_rows = worksheet.get_all_values()
    if not all_rows:
        print("No rows found in worksheet. Nothing to do.")
        return 0

    headers = all_rows[0]
    if not headers:
        print("No header row found. Nothing to do.")
        return 0

    date_column_index = find_date_column_index(headers)
    if date_column_index is None:
        print(f"Date column not found. Tried: {', '.join(DATE_COLUMN_PRIORITY)}")
        return 0
    date_column_name = headers[date_column_index] if len(headers) > date_column_index else ""
    target_tz = ZoneInfo(tz_name)
    print(f"Using date column: '{date_column_name}' (timezone: {tz_name})")

    data_rows = all_rows[1:]
    if not data_rows:
        print("No data rows found. Nothing to delete.")
        return 0

    rows_to_delete: List[int] = []
    skipped = 0
    for row_idx, row in enumerate(data_rows, start=2):
        date_value = row[date_column_index] if len(row) > date_column_index else ""
        if date_column_name.strip().lower() == "pubdate":
            dt = parse_pubdate(date_value)
        else:
            dt = parse_sheet_date(date_value)
        if not dt:
            skipped += 1
            continue
        local_date = dt.astimezone(target_tz).date()
        if local_date <= cutoff_local_date:
            rows_to_delete.append(row_idx)

    if not rows_to_delete:
        print("No rows matched the deletion condition.")
        if skipped:
            print(f"Skipped rows due to parse failure: {skipped}")
        return 0

    print(
        f"Rows matched: {len(rows_to_delete)} (local date <= {cutoff_local_date}) on worksheet '{worksheet.title}'"
    )
    if dry_run:
        print("Dry run enabled. No rows were deleted.")
        return len(rows_to_delete)

    sheet_id = getattr(worksheet, "_properties", {}).get("sheetId") or getattr(
        worksheet, "id", None
    )
    spreadsheet = getattr(worksheet, "parent", None) or getattr(
        worksheet, "spreadsheet", None
    )

    deleted = 0
    if sheet_id is None or spreadsheet is None:
        print("Batch delete unavailable, falling back to row-by-row deletion.")
        for row_idx in reversed(rows_to_delete):
            try:
                worksheet.delete_rows(row_idx)
                deleted += 1
                time.sleep(0.2)
            except Exception as exc:
                print(f"Failed deleting row {row_idx}: {exc}")
    else:
        ranges = make_ranges(rows_to_delete)
        print(f"Deleting in {len(ranges)} range chunk(s).")
        for start, end in reversed(ranges):
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
                spreadsheet.batch_update(request)
                deleted += row_count
                print(f"Deleted rows {start}-{end} ({row_count})")
                time.sleep(0.2)
            except Exception as exc:
                print(f"Failed deleting rows {start}-{end}: {exc}")

    print(f"Deleted rows: {deleted}")
    if skipped:
        print(f"Skipped rows due to parse failure: {skipped}")
    return deleted


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Delete rows from total_result if article date is on/before cutoff date "
            "(default: 2026-01-31, timezone-aware)."
        )
    )
    parser.add_argument(
        "--cutoff-date",
        default=DEFAULT_CUTOFF_DATE,
        help="Cutoff local date in ISO format. Rows on/before this date are deleted.",
    )
    parser.add_argument(
        "--timezone",
        default=DEFAULT_TIMEZONE,
        help="Timezone used for date comparison (default: Asia/Seoul).",
    )
    parser.add_argument(
        "--worksheet",
        default="total_result",
        help="Worksheet name. Default is total_result.",
    )
    parser.add_argument(
        "--credentials",
        default=None,
        help="Service account JSON path (defaults to env GOOGLE_SHEETS_CREDENTIALS_PATH).",
    )
    parser.add_argument(
        "--sheet-id",
        default=None,
        help="Google Sheet ID (defaults to env GOOGLE_SHEET_ID).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview only. Do not delete rows.",
    )
    args = parser.parse_args()

    load_dotenv()

    credentials_path = (
        args.credentials
        or os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH")
        or "service-account.json"
    )
    sheet_id = args.sheet_id or os.getenv("GOOGLE_SHEET_ID")
    if not sheet_id:
        parser.error("Google Sheet ID is required. Set GOOGLE_SHEET_ID or use --sheet-id.")

    cutoff_local_date = parse_cutoff_date(args.cutoff_date)
    print(f"Connecting to Google Sheet: {sheet_id}")
    print(f"Worksheet: {args.worksheet}")
    print(
        f"Delete condition: local date <= {cutoff_local_date} "
        f"(timezone: {args.timezone})"
    )
    print(f"Dry run: {'yes' if args.dry_run else 'no'}")

    spreadsheet = connect_sheets(credentials_path, sheet_id)
    if not spreadsheet:
        raise SystemExit(1)

    try:
        worksheet = spreadsheet.worksheet(args.worksheet)
    except Exception as exc:
        print(f"Failed to open worksheet '{args.worksheet}': {exc}")
        raise SystemExit(1)

    deleted = delete_before_cutoff(
        worksheet=worksheet,
        cutoff_local_date=cutoff_local_date,
        tz_name=args.timezone,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(f"Dry run complete. Rows to delete: {deleted}")
    else:
        print(f"Cleanup complete. Rows deleted: {deleted}")


if __name__ == "__main__":
    main()
