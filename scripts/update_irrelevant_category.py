"""
update_irrelevant_category.py - brand_relevance="무관" + news_category="기타" → "비관련" 일괄 변경

Google Sheets total_result 탭에서 해당하는 행들을 찾아 news_category를 "비관련"으로 업데이트합니다.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import gspread
from google.oauth2.service_account import Credentials

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def update_irrelevant_categories():
    """
    brand_relevance="무관" + news_category="기타" → news_category="비관련" 업데이트
    """
    # 환경 변수 로드
    load_dotenv()

    credentials_path = os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH")
    sheet_id = os.getenv("GOOGLE_SHEET_ID")

    if not credentials_path or not sheet_id:
        print("❌ GOOGLE_SHEETS_CREDENTIALS_PATH 또는 GOOGLE_SHEET_ID가 .env에 설정되지 않았습니다.")
        return

    # Google Sheets 인증
    print("🔐 Google Sheets 인증 중...")
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = Credentials.from_service_account_file(credentials_path, scopes=scopes)
    client = gspread.authorize(creds)

    # Spreadsheet 열기
    print(f"📊 Spreadsheet 열기: {sheet_id}")
    spreadsheet = client.open_by_key(sheet_id)

    # total_result 워크시트 가져오기
    try:
        worksheet = spreadsheet.worksheet("total_result")
    except gspread.WorksheetNotFound:
        print("❌ total_result 워크시트를 찾을 수 없습니다.")
        return

    # 전체 데이터 가져오기
    print("📥 데이터 로드 중...")
    all_values = worksheet.get_all_values()

    if not all_values:
        print("⚠️  워크시트가 비어있습니다.")
        return

    # 헤더와 데이터 분리
    headers = all_values[0]
    data_rows = all_values[1:]

    # 필요한 컬럼 인덱스 찾기
    try:
        brand_rel_idx = headers.index("brand_relevance")
        news_cat_idx = headers.index("news_category")
    except ValueError as e:
        print(f"❌ 필수 컬럼을 찾을 수 없습니다: {e}")
        print(f"   사용 가능한 컬럼: {headers}")
        return

    # 조건에 맞는 행 찾기 (1-based row number for Sheets API)
    target_rows = []
    for i, row in enumerate(data_rows, start=2):  # 2부터 시작 (헤더는 1행)
        if len(row) > max(brand_rel_idx, news_cat_idx):
            brand_rel = row[brand_rel_idx]
            news_cat = row[news_cat_idx]

            if brand_rel == "무관" and news_cat == "기타":
                target_rows.append(i)

    if not target_rows:
        print("✅ 업데이트할 행이 없습니다. (brand_relevance='무관' + news_category='기타' 조건에 맞는 행 없음)")
        return

    print(f"🎯 업데이트 대상: {len(target_rows)}개 행")
    print(f"   행 번호 (처음 10개): {target_rows[:10]}")

    # 확인 메시지
    response = input(f"\n⚠️  {len(target_rows)}개 행의 news_category를 '비관련'으로 변경하시겠습니까? (y/N): ")
    if response.lower() != 'y':
        print("❌ 취소되었습니다.")
        return

    # 연속된 행들을 범위로 묶기
    print("✏️  업데이트 범위 생성 중...")
    ranges = []
    if target_rows:
        current_start = target_rows[0]
        current_end = target_rows[0]

        for row in target_rows[1:]:
            if row == current_end + 1:
                current_end = row
            else:
                ranges.append((current_start, current_end))
                current_start = row
                current_end = row
        ranges.append((current_start, current_end))

    print(f"   {len(ranges)}개 범위로 묶음 (총 {len(target_rows)}개 행)")

    # 배치 업데이트 준비
    print("✏️  업데이트 중...")
    updates = []
    news_cat_col_letter = chr(65 + news_cat_idx)  # A=65

    for start_row, end_row in ranges:
        num_rows = end_row - start_row + 1
        cell_range = f"{news_cat_col_letter}{start_row}:{news_cat_col_letter}{end_row}"
        updates.append({
            'range': cell_range,
            'values': [['비관련']] * num_rows
        })

    # 배치 업데이트 실행 (100개 범위씩 나눠서)
    batch_size = 100
    total_rows_updated = 0
    for i in range(0, len(updates), batch_size):
        batch = updates[i:i+batch_size]
        worksheet.batch_update(batch)
        # 이 배치에서 업데이트한 행 수 계산
        batch_row_count = sum(len(update['values']) for update in batch)
        total_rows_updated += batch_row_count
        print(f"   {total_rows_updated}/{len(target_rows)} 행 완료...")

    print(f"✅ 완료! {len(target_rows)}개 행의 news_category를 '비관련'으로 변경했습니다.")


if __name__ == "__main__":
    update_irrelevant_categories()
