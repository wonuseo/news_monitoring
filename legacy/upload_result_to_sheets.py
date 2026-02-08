#!/usr/bin/env python3
"""
result.csv를 Google Sheets에 업로드하는 스크립트
Rate limiting 테스트용
"""

import pandas as pd
import os
from dotenv import load_dotenv
from src.modules.export.sheets import connect_sheets, sync_to_sheets

# 환경 변수 로드
load_dotenv()

def main():
    # 파일 경로
    result_csv_path = "data/result.csv"

    # CSV 파일 확인
    if not os.path.exists(result_csv_path):
        print(f"❌ {result_csv_path} 파일이 없습니다.")
        return

    # CSV 읽기
    print(f"📂 {result_csv_path} 읽는 중...")
    df = pd.read_csv(result_csv_path, encoding="utf-8-sig")
    print(f"✅ {len(df)}개 행 로드됨")

    # Google Sheets 연결
    credentials_path = os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH")
    sheet_id = os.getenv("GOOGLE_SHEET_ID")

    if not credentials_path or not sheet_id:
        print("❌ .env 파일에 Google Sheets 설정이 없습니다.")
        print("  GOOGLE_SHEETS_CREDENTIALS_PATH")
        print("  GOOGLE_SHEET_ID")
        return

    print("\n📊 Google Sheets 연결 중...")
    spreadsheet = connect_sheets(credentials_path, sheet_id)

    if not spreadsheet:
        print("❌ Google Sheets 연결 실패")
        return

    # result 시트에 업로드
    print(f"\n📤 'result' 시트에 {len(df)}개 행 업로드 중...")
    print("  (Rate limiting 테스트: 1000행/배치, 1초 delay, 재시도 로직)")

    result = sync_to_sheets(
        df=df,
        spreadsheet=spreadsheet,
        sheet_name="result",
        key_column="link"
    )

    # 결과 출력
    print("\n" + "="*80)
    print("✅ 업로드 완료!")
    print(f"  - 추가됨: {result['added']}개")
    print(f"  - 건너뜀: {result['skipped']}개 (이미 존재)")
    print(f"  - 오류: {result['errors']}개")
    print("="*80)

if __name__ == "__main__":
    main()
