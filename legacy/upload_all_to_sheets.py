#!/usr/bin/env python3
"""
upload_all_to_sheets.py - 전체 데이터를 Google Sheets에 재동기화
"""

import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os
from src.modules.export.sheets import connect_sheets

# .env 로드
load_dotenv()

def clear_and_upload():
    """Google Sheets를 비우고 전체 데이터를 재업로드"""

    # CSV 파일 읽기
    raw_csv = Path("data/raw.csv")
    result_csv = Path("data/result.csv")

    print("="*80)
    print("🔄 전체 데이터 재동기화")
    print("="*80)

    if not raw_csv.exists():
        print("\n❌ raw.csv 파일이 없습니다.")
        return

    df_raw = pd.read_csv(raw_csv, encoding='utf-8-sig')
    print(f"\n📂 로컬 CSV:")
    print(f"  - raw.csv: {len(df_raw)}개 기사")

    if result_csv.exists():
        df_result = pd.read_csv(result_csv, encoding='utf-8-sig')
        print(f"  - result.csv: {len(df_result)}개 기사")
    else:
        print("  - result.csv: 없음 (스킵)")
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

    # 사용자 확인
    print("\n⚠️  경고: 기존 Google Sheets 데이터를 모두 삭제하고 재업로드합니다.")
    response = input("계속하시겠습니까? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("\n❌ 취소됨")
        return

    print("\n🔄 재동기화 시작...")

    # 1. raw_data 시트 재생성
    print("\n[1/2] raw_data 시트 재생성 중...")
    try:
        # 기존 시트 삭제
        try:
            ws_raw = spreadsheet.worksheet("raw_data")
            spreadsheet.del_worksheet(ws_raw)
            print("  - 기존 시트 삭제 완료")
        except:
            print("  - 기존 시트 없음 (새로 생성)")

        # 새 시트 생성
        ws_raw = spreadsheet.add_worksheet(title="raw_data", rows=len(df_raw)+10, cols=len(df_raw.columns))

        # 헤더 추가
        ws_raw.append_row(df_raw.columns.tolist())

        # 데이터 추가 (배치 처리)
        batch_size = 1000
        total_rows = len(df_raw)

        for i in range(0, total_rows, batch_size):
            batch = df_raw.iloc[i:i+batch_size]
            values = batch.fillna("").astype(str).values.tolist()
            ws_raw.append_rows(values)
            print(f"  - {min(i+batch_size, total_rows)}/{total_rows} 행 업로드 완료")

        print(f"✅ raw_data: {len(df_raw)}개 기사 업로드 완료")

    except Exception as e:
        print(f"❌ raw_data 업로드 실패: {e}")
        return

    # 2. result 시트 재생성
    if df_result is not None:
        print("\n[2/2] result 시트 재생성 중...")
        try:
            # 기존 시트 삭제
            try:
                ws_result = spreadsheet.worksheet("result")
                spreadsheet.del_worksheet(ws_result)
                print("  - 기존 시트 삭제 완료")
            except:
                print("  - 기존 시트 없음 (새로 생성)")

            # 새 시트 생성
            ws_result = spreadsheet.add_worksheet(title="result", rows=len(df_result)+10, cols=len(df_result.columns))

            # 헤더 추가
            ws_result.append_row(df_result.columns.tolist())

            # 데이터 추가 (배치 처리)
            batch_size = 1000
            total_rows = len(df_result)

            for i in range(0, total_rows, batch_size):
                batch = df_result.iloc[i:i+batch_size]
                values = batch.fillna("").astype(str).values.tolist()
                ws_result.append_rows(values)
                print(f"  - {min(i+batch_size, total_rows)}/{total_rows} 행 업로드 완료")

            print(f"✅ result: {len(df_result)}개 기사 업로드 완료")

        except Exception as e:
            print(f"❌ result 업로드 실패: {e}")
            return

    print("\n" + "="*80)
    print("✅ 전체 데이터 재동기화 완료!")
    print("="*80)
    print(f"\n📊 업로드된 데이터:")
    print(f"  - raw_data: {len(df_raw)}개 기사")
    if df_result is not None:
        print(f"  - result: {len(df_result)}개 기사")

if __name__ == "__main__":
    clear_and_upload()
