#!/usr/bin/env python3
"""Helper: print the value of a single cell in Google Sheets."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import os

from dotenv import load_dotenv

from src.modules.export.sheets import connect_sheets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Google Sheets total_result 시트에서 특정 셀 값을 출력합니다."
    )
    parser.add_argument(
        "--cell",
        default="U2",
        help="조회할 셀 주소 (예: U2).",
    )
    parser.add_argument(
        "--worksheet",
        default="total_result",
        help="워크시트 이름 (기본: total_result).",
    )
    parser.add_argument(
        "--credentials",
        default=None,
        help="서비스 계정 JSON 경로 (.env의 GOOGLE_SHEETS_CREDENTIALS_PATH 우선).",
    )
    parser.add_argument(
        "--sheet-id",
        default=None,
        help="Google Sheets ID (.env의 GOOGLE_SHEET_ID 우선).",
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
        parser.error("Google Sheets ID (.env 또는 --sheet-id) 값이 필요합니다.")

    print(f"Google Sheets ({sheet_id}) 연결 시도")
    spreadsheet = connect_sheets(credentials_path, sheet_id)
    if not spreadsheet:
        raise SystemExit(1)

    try:
        worksheet = spreadsheet.worksheet(args.worksheet)
    except Exception as exc:
        print(f"⚠️  워크시트 '{args.worksheet}' 열기 실패: {exc}")
        raise SystemExit(1)

    try:
        value = worksheet.acell(args.cell).value
    except Exception as exc:
        print(f"⚠️  셀 조회 실패 ({args.cell}): {exc}")
        raise SystemExit(1)

    print(f"워크시트 '{args.worksheet}' {args.cell} = {value}")


if __name__ == "__main__":
    main()
