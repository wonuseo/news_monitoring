#!/usr/bin/env python3
"""
기존 데이터의 group_id를 1부터 시작하도록 재번호 부여
CSV와 Google Sheets 업데이트
"""
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# .env 로드
load_dotenv()

def renumber_group_ids(df):
    """
    group_id를 1부터 시작하는 순차적 번호로 재할당

    Args:
        df: group_id 컬럼이 있는 DataFrame

    Returns:
        재번호가 부여된 DataFrame
    """
    if 'group_id' not in df.columns:
        print("⚠️  group_id 컬럼이 없습니다.")
        return df

    df = df.copy()

    # 기존 group_id 중 비어있지 않은 것들만 추출
    existing_groups = df[df['group_id'] != '']['group_id'].unique()

    if len(existing_groups) == 0:
        print("  ℹ️  재번호 부여할 그룹이 없습니다.")
        return df

    # 기존 그룹 ID를 정렬 (숫자 부분 추출)
    def extract_number(group_id):
        try:
            return int(group_id.replace('group_', ''))
        except:
            return 0

    sorted_groups = sorted(existing_groups, key=extract_number)

    # 매핑 테이블 생성 (old_id → new_id)
    group_mapping = {}
    for new_id, old_id in enumerate(sorted_groups, start=1):
        group_mapping[old_id] = f"group_{new_id}"

    # group_id 재할당
    df['group_id'] = df['group_id'].apply(lambda x: group_mapping.get(x, x))

    print(f"  ✅ {len(group_mapping)}개 그룹 재번호 완료")
    print(f"  - 예시: {list(group_mapping.items())[:3]}")

    return df


def main():
    """메인 실행 함수"""
    data_dir = Path("data")
    result_csv = data_dir / "result.csv"

    if not result_csv.exists():
        print(f"❌ {result_csv} 파일이 없습니다.")
        return

    # 1. CSV 읽기
    print(f"📖 {result_csv} 읽는 중...")
    df = pd.read_csv(result_csv, encoding='utf-8-sig')
    print(f"  - 전체 기사: {len(df)}개")
    print(f"  - 보도자료: {(df['source'] == '보도자료').sum()}개")

    # 2. group_id 재번호 부여
    print("\n🔢 group_id 재번호 부여 중...")
    df = renumber_group_ids(df)

    # 3. CSV 저장
    print(f"\n💾 {result_csv} 저장 중...")
    df.to_csv(result_csv, index=False, encoding='utf-8-sig')
    print(f"  ✅ CSV 업데이트 완료")

    # 4. Google Sheets 업데이트 (선택사항)
    sheets_id = os.getenv('GOOGLE_SHEET_ID')
    credentials_path = os.getenv('GOOGLE_SHEETS_CREDENTIALS_PATH')

    if sheets_id and credentials_path and Path(credentials_path).exists():
        print("\n📊 Google Sheets 업데이트 중...")
        try:
            from src.modules.export.sheets import connect_sheets
            import gspread

            spreadsheet = connect_sheets(credentials_path, sheets_id)

            # result 시트 업데이트
            try:
                worksheet = spreadsheet.worksheet('result')

                # 기존 데이터 삭제
                worksheet.clear()

                # 새 데이터 업로드
                # 헤더 + 데이터 변환
                header = df.columns.tolist()
                values = [header] + df.fillna('').astype(str).values.tolist()

                worksheet.update(values, value_input_option='RAW')
                print(f"  ✅ Google Sheets 'result' 시트 업데이트 완료 ({len(df)}개 기사)")

            except gspread.exceptions.WorksheetNotFound:
                print("  ⚠️  'result' 시트를 찾을 수 없습니다.")

        except ImportError:
            print("  ⚠️  Google Sheets 모듈을 로드할 수 없습니다.")
        except Exception as e:
            print(f"  ⚠️  Google Sheets 업데이트 실패: {e}")
    else:
        print("\n⚠️  Google Sheets 설정이 없습니다. CSV만 업데이트됩니다.")

    print("\n🎉 완료!")


if __name__ == '__main__':
    main()
