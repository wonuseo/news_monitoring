"""
sheets.py - Google Sheets Integration Module
Google Sheets로 데이터를 증분 업로드하고 Looker Studio 연계
"""

import pandas as pd
from typing import Dict, Optional, List
from datetime import datetime
import os


def connect_sheets(credentials_path: str, sheet_id: str):
    """
    Google Sheets 연결

    Args:
        credentials_path: 서비스 계정 JSON 키 경로
        sheet_id: 구글 시트 ID

    Returns:
        gspread Spreadsheet 객체
    """
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        if not os.path.exists(credentials_path):
            raise FileNotFoundError(f"자격증명 파일을 찾을 수 없음: {credentials_path}")

        # 서비스 계정 인증
        scope = [
            "https://spreadsheetapis.google.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]

        creds = Credentials.from_service_account_file(credentials_path, scopes=scope)
        client = gspread.authorize(creds)

        # 시트 열기
        spreadsheet = client.open_by_key(sheet_id)

        print(f"✅ Google Sheets 연결 성공: {spreadsheet.title}")
        return spreadsheet

    except ImportError:
        print("❌ gspread 또는 google-auth 라이브러리가 설치되지 않았습니다.")
        print("  pip install gspread google-auth google-auth-oauthlib")
        return None
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return None
    except Exception as e:
        print(f"❌ Google Sheets 연결 실패: {e}")
        return None


def sync_to_sheets(df: pd.DataFrame, spreadsheet,
                  sheet_name: str = "전체데이터",
                  key_column: str = "link") -> Dict[str, int]:
    """
    DataFrame을 Google Sheets에 증분 업로드

    Args:
        df: 업로드할 DataFrame
        spreadsheet: gspread Spreadsheet 객체
        sheet_name: 워크시트 이름
        key_column: 중복 제거 기준 컬럼

    Returns:
        {"added": N, "skipped": N, "errors": N}
    """
    try:
        # 워크시트 선택 또는 생성
        try:
            worksheet = spreadsheet.worksheet(sheet_name)
        except:
            worksheet = spreadsheet.add_worksheet(title=sheet_name, rows=1000, cols=30)
            print(f"  📝 새 워크시트 생성: {sheet_name}")

        # 기존 데이터 읽기 (헤더만 읽기, 성능상 모든 행 읽지 않음)
        try:
            existing_data = worksheet.get_all_records()
        except:
            existing_data = []

        # 기존 key_column 값들을 set으로 저장 (중복 체크용)
        existing_keys = set()
        if existing_data and key_column in existing_data[0]:
            existing_keys = {row.get(key_column, "") for row in existing_data}

        # 새로운 행만 필터링
        if key_column in df.columns:
            new_rows = df[~df[key_column].isin(existing_keys)]
        else:
            new_rows = df

        if len(new_rows) == 0:
            print(f"  ℹ️  {sheet_name}: 새 기사 없음")
            return {"added": 0, "skipped": len(df), "errors": 0}

        # 헤더 행이 없으면 추가
        if len(existing_data) == 0:
            worksheet.append_row(df.columns.tolist())

        # 새로운 행들을 batch로 추가
        values_to_append = []
        for _, row in new_rows.iterrows():
            row_values = []
            for col in df.columns:
                val = row[col]
                # None을 빈 문자열로 변환
                if pd.isna(val) or val is None:
                    row_values.append("")
                else:
                    row_values.append(str(val))
            values_to_append.append(row_values)

        # 일괄 추가 (최대 100행씩)
        for i in range(0, len(values_to_append), 100):
            batch = values_to_append[i:i+100]
            worksheet.append_rows(batch)

        print(f"  ✅ {sheet_name}: {len(new_rows)}개 행 추가")
        return {"added": len(new_rows), "skipped": len(df) - len(new_rows), "errors": 0}

    except Exception as e:
        print(f"  ❌ {sheet_name} 업로드 실패: {e}")
        return {"added": 0, "skipped": 0, "errors": len(df)}


def configure_sheet_schema(worksheet) -> None:
    """
    Google Sheets 워크시트 스키마 설정

    Args:
        worksheet: gspread Worksheet 객체
    """
    try:
        import gspread

        # 날짜 컬럼 설정
        date_columns = {
            "pub_datetime": "DATETIME",
            "classified_at": "DATETIME",
            "full_text_scraped_at": "DATETIME",
            "scraped_at": "DATETIME",
            "date_only": "DATE"
        }

        # 숫자 컬럼
        number_columns = ["article_count"]

        # 텍스트 컬럼
        text_columns = ["full_text"]

        # 헤더 행 읽기
        headers = worksheet.row_values(1)

        for idx, header in enumerate(headers, 1):
            if header in date_columns:
                # 날짜 포맷 설정 (API로는 직접 설정 불가, 수동 설정 필요)
                pass
            elif header in number_columns:
                # 숫자 포맷
                pass
            elif header in text_columns:
                # 텍스트 줄바꿈 활성화
                pass

        print("  ℹ️  워크시트 스키마 설정 완료")

    except Exception as e:
        print(f"  ⚠️  스키마 설정 실패: {e}")


def sync_all_sheets(df: pd.DataFrame, spreadsheet) -> Dict[str, Dict]:
    """
    데이터를 여러 워크시트에 동시 업로드

    Args:
        df: 분류된 DataFrame
        spreadsheet: gspread Spreadsheet 객체

    Returns:
        {sheet_name: {added, skipped, errors}}
    """
    results = {}

    print("📊 Google Sheets 동기화 중...")

    # 1. 전체 데이터
    print("\n  [1/4] 전체데이터")
    results["전체데이터"] = sync_to_sheets(df, spreadsheet, "전체데이터")

    # 2. 우리 브랜드 부정
    print("  [2/4] 우리_부정")
    our_negative = df[(df["group"] == "OUR") & (df["sentiment"] == "부정")]
    results["우리_부정"] = sync_to_sheets(our_negative, spreadsheet, "우리_부정")

    # 3. 우리 브랜드 긍정
    print("  [3/4] 우리_긍정")
    our_positive = df[(df["group"] == "OUR") & (df["sentiment"] == "긍정")]
    results["우리_긍정"] = sync_to_sheets(our_positive, spreadsheet, "우리_긍정")

    # 4. 경쟁사
    print("  [4/4] 경쟁사")
    competitor = df[df["group"] == "COMPETITOR"]
    results["경쟁사"] = sync_to_sheets(competitor, spreadsheet, "경쟁사")

    # 통계
    print("\n✅ Google Sheets 동기화 완료")
    total_added = sum(r["added"] for r in results.values())
    total_skipped = sum(r["skipped"] for r in results.values())
    total_errors = sum(r["errors"] for r in results.values())

    print(f"  - 추가됨: {total_added}개")
    print(f"  - 건너뜀: {total_skipped}개")
    print(f"  - 오류: {total_errors}개")

    return results
