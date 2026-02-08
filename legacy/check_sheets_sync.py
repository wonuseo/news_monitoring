#!/usr/bin/env python3
"""
check_sheets_sync.py - Google Sheets 동기화 상태 확인
"""

import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os
from src.modules.export.sheets import connect_sheets

# .env 로드
load_dotenv()

def check_sync_status():
    """CSV와 Google Sheets의 동기화 상태 확인"""

    # CSV 파일 읽기
    raw_csv = Path("data/raw.csv")
    result_csv = Path("data/result.csv")

    print("="*80)
    print("📊 동기화 상태 확인")
    print("="*80)

    # CSV 파일 행 수
    if raw_csv.exists():
        df_raw = pd.read_csv(raw_csv, encoding='utf-8-sig')
        print(f"\n📂 로컬 CSV:")
        print(f"  - raw.csv: {len(df_raw)}개 기사")
    else:
        print("\n⚠️  raw.csv 파일이 없습니다.")
        return

    if result_csv.exists():
        df_result = pd.read_csv(result_csv, encoding='utf-8-sig')
        print(f"  - result.csv: {len(df_result)}개 기사")
    else:
        print("  - result.csv: 파일 없음")
        df_result = None

    # Google Sheets 연결
    credentials_path = os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH")
    sheet_id = os.getenv("GOOGLE_SHEET_ID")

    if not credentials_path or not sheet_id:
        print("\n⚠️  .env 파일에 Google Sheets 설정이 없습니다.")
        return

    spreadsheet = connect_sheets(credentials_path, sheet_id)
    if not spreadsheet:
        print("\n❌ Google Sheets 연결 실패")
        return

    print(f"\n☁️  Google Sheets:")

    # raw_data 시트 확인
    try:
        ws_raw = spreadsheet.worksheet("raw_data")
        raw_sheets_count = len(ws_raw.get_all_records())
        print(f"  - raw_data 시트: {raw_sheets_count}개 기사")

        raw_diff = len(df_raw) - raw_sheets_count
        if raw_diff > 0:
            print(f"    ⚠️  {raw_diff}개 기사 누락!")
        elif raw_diff < 0:
            print(f"    ⚠️  Sheets에 {abs(raw_diff)}개 더 많음 (이상)")
        else:
            print(f"    ✅ 동기화 완료")
    except:
        print(f"  - raw_data 시트: 없음")
        raw_sheets_count = 0
        raw_diff = len(df_raw)

    # result 시트 확인
    if df_result is not None:
        try:
            ws_result = spreadsheet.worksheet("result")
            result_sheets_count = len(ws_result.get_all_records())
            print(f"  - result 시트: {result_sheets_count}개 기사")

            result_diff = len(df_result) - result_sheets_count
            if result_diff > 0:
                print(f"    ⚠️  {result_diff}개 기사 누락!")
            elif result_diff < 0:
                print(f"    ⚠️  Sheets에 {abs(result_diff)}개 더 많음 (이상)")
            else:
                print(f"    ✅ 동기화 완료")
        except:
            print(f"  - result 시트: 없음")
            result_sheets_count = 0
            result_diff = len(df_result)

    print("\n" + "="*80)
    print("📋 요약")
    print("="*80)

    total_missing = 0
    if raw_diff > 0:
        print(f"⚠️  raw_data: {raw_diff}개 누락")
        total_missing += raw_diff

    if df_result is not None and result_diff > 0:
        print(f"⚠️  result: {result_diff}개 누락")
        total_missing += result_diff

    if total_missing > 0:
        print(f"\n⚠️  총 {total_missing}개 기사가 Google Sheets에 동기화되지 않았습니다.")
        print("\n해결 방법:")
        print("  1. python upload_all_to_sheets.py  (전체 재동기화)")
        print("  2. python main.py --sheets  (다음 실행 시 자동 동기화)")
    else:
        print("\n✅ 모든 데이터가 Google Sheets에 동기화되었습니다!")

if __name__ == "__main__":
    check_sync_status()
