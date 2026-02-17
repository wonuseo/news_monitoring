"""
Sync sorted result.csv to Google Sheets

재정렬된 result.csv를 Google Sheets total_result 탭에 동기화합니다.
전체 데이터를 덮어씁니다 (기존 데이터 대체).

실행: python scripts/sync_sorted_to_sheets.py
"""

import os
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.modules.export.sheets import get_spreadsheet, sync_raw_and_processed
from dotenv import load_dotenv

def main():
    print("=" * 80)
    print("Google Sheets 동기화 스크립트")
    print("=" * 80)
    print()

    # Load environment
    load_dotenv()

    # Check credentials
    creds_path = os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH")
    sheet_id = os.getenv("GOOGLE_SHEET_ID")

    if not creds_path or not sheet_id:
        print("❌ Google Sheets 설정을 찾을 수 없습니다.")
        print("   .env 파일에 다음 설정이 필요합니다:")
        print("   - GOOGLE_SHEETS_CREDENTIALS_PATH")
        print("   - GOOGLE_SHEET_ID")
        return

    # Load result.csv
    result_path = Path("data/result.csv")
    if not result_path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {result_path}")
        return

    print(f"📂 로딩: {result_path}")
    df_result = pd.read_csv(result_path, encoding='utf-8-sig')
    print(f"   총 {len(df_result)}개 기사")
    print(f"   article_no 범위: {df_result['article_no'].min()} ~ {df_result['article_no'].max()}")
    print()

    # Connect to Sheets
    print("📊 Google Sheets 연결 중...")
    try:
        spreadsheet = get_spreadsheet(creds_path, sheet_id)
        print(f"✅ 연결 성공: {spreadsheet.title}")
    except Exception as e:
        print(f"❌ Sheets 연결 실패: {e}")
        return

    print()

    # Sync to total_result tab
    print("🔄 total_result 탭 동기화 중...")
    print("   ⚠️  전체 데이터를 덮어씁니다 (기존 데이터 대체)")

    try:
        # Get or create total_result worksheet
        try:
            worksheet = spreadsheet.worksheet("total_result")
            print(f"   📋 기존 시트 발견: total_result")
        except:
            worksheet = spreadsheet.add_worksheet(title="total_result", rows=10000, cols=30)
            print(f"   📋 새 시트 생성: total_result")

        # Clear existing data
        print("   🧹 기존 데이터 삭제 중...")
        worksheet.clear()

        # Prepare data for upload
        # Convert DataFrame to list of lists (including header)
        values = [df_result.columns.tolist()] + df_result.fillna("").astype(str).values.tolist()

        print(f"   ⬆️  업로드 중: {len(values)-1}개 행 (헤더 포함 {len(values)}개)...")

        # Upload in one batch
        worksheet.update(values, value_input_option='RAW')

        print(f"   ✅ 업로드 완료: {len(df_result)}개 행")

        # Format header row
        print("   🎨 헤더 서식 적용 중...")
        worksheet.format("1:1", {
            "textFormat": {"bold": True},
            "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9}
        })

        print("   ✅ total_result 동기화 완료")

    except Exception as e:
        print(f"   ❌ 동기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    print()
    print("=" * 80)
    print("✅ 완료!")
    print(f"   Sheets: {spreadsheet.url}")
    print("=" * 80)

if __name__ == "__main__":
    main()
