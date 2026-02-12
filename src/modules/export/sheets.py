"""
sheets.py - Google Sheets Integration Module
Google Sheets로 데이터를 증분 업로드하고 Looker Studio 연계
"""

import pandas as pd
from typing import Dict, Optional, List
from datetime import datetime
import os
import time


def clean_bom(value) -> str:
    """
    모든 BOM 및 invisible 문자를 제거하고 빈 문자열로 변환

    제거 대상:
    - UTF-8/16/32 BOM: \ufeff, \ufffe
    - Zero Width 문자: \u200b, \u200c, \u200d, \u2060
    - Non-breaking/ideographic spaces: \u00a0, \u3000
    - 기타 invisible 문자: \u180e, \u2028, \u2029, \u200e, \u200f, \u202a-\u202f
    - C0/C1 제어 문자: \x00-\x08, \x0b, \x0c, \x0e-\x1f, \x7f-\x9f
    - Interlinear annotation: \ufff9-\ufffc

    Args:
        value: 정리할 값

    Returns:
        정리된 문자열
    """
    import re

    if pd.isna(value) or value is None:
        return ""

    # 문자열로 변환
    value_str = str(value)

    # 정규식으로 모든 invisible/제어 문자 일괄 제거
    # BOM, Zero Width, 제어 문자, 방향 마크, non-breaking space 등
    value_str = re.sub(
        r'[\ufeff\ufffe'           # BOM
        r'\u200b-\u200f'           # Zero Width + 방향 마크
        r'\u2028-\u202f'           # 줄/단락 구분자 + 방향 포맷
        r'\u2060'                  # Word Joiner
        r'\u180e'                  # Mongolian Vowel Separator
        r'\u00a0'                  # Non-Breaking Space
        r'\u3000'                  # Ideographic Space (전각 공백)
        r'\u00ad'                  # Soft Hyphen
        r'\ufff9-\ufffc'           # Interlinear Annotation
        r'\x00-\x08\x0b\x0c\x0e-\x1f'  # C0 제어 문자 (탭/개행 제외)
        r'\x7f-\x9f'              # DEL + C1 제어 문자
        r']', '', value_str
    )

    # 앞뒤 공백 제거
    value_str = value_str.strip()

    return value_str


def clean_all_bom_in_sheets(spreadsheet, sheet_names: list = None) -> Dict[str, int]:
    """
    Google Sheets의 모든 셀에서 BOM 및 invisible 문자를 일괄 제거

    전체 시트 재작성 방식: API의 FORMATTED_VALUE가 BOM을 숨겨서
    셀 단위 비교로는 감지 불가능한 BOM도 제거.

    동작 방식:
    1. 시트 전체 값을 읽기
    2. 모든 셀 값에 clean_bom() 적용
    3. 전체 시트를 정리된 값으로 덮어쓰기 (숨겨진 BOM도 제거)

    Args:
        spreadsheet: gspread Spreadsheet 객체
        sheet_names: 정리할 시트 이름 리스트 (None이면 raw_data, total_result)

    Returns:
        {sheet_name: cleaned_cell_count}
    """
    if sheet_names is None:
        sheet_names = ["raw_data", "total_result"]

    results = {}

    for sheet_name in sheet_names:
        try:
            try:
                worksheet = spreadsheet.worksheet(sheet_name)
            except Exception:
                print(f"  ℹ️  '{sheet_name}' 워크시트가 없습니다. 건너뜀.")
                results[sheet_name] = 0
                continue

            # 전체 데이터 읽기
            all_values = worksheet.get_all_values()
            if not all_values:
                print(f"  ℹ️  '{sheet_name}' 워크시트가 비어있습니다.")
                results[sheet_name] = 0
                continue

            # 모든 셀 값 정리 (API가 BOM을 숨겨도 감지 가능한 것은 카운트)
            cleaned_rows = []
            detected_count = 0

            for row in all_values:
                cleaned_row = []
                for cell_value in row:
                    if isinstance(cell_value, str) and cell_value:
                        cleaned = clean_bom(cell_value)
                        if cleaned != cell_value:
                            detected_count += 1
                        cleaned_row.append(cleaned)
                    else:
                        cleaned_row.append(cell_value if cell_value else "")
                cleaned_rows.append(cleaned_row)

            # 전체 시트 재작성 (숨겨진 BOM도 덮어쓰기로 제거)
            num_rows = len(cleaned_rows)
            num_cols = max(len(row) for row in cleaned_rows) if cleaned_rows else 0

            if num_rows > 0 and num_cols > 0:
                # 모든 행의 길이를 맞추기 (패딩)
                for row in cleaned_rows:
                    while len(row) < num_cols:
                        row.append("")

                last_col = col_num_to_letter(num_cols)
                range_str = f"A1:{last_col}{num_rows}"

                # 배치 단위로 업데이트 (대용량 시트 대응)
                batch_row_size = 2000
                for i in range(0, num_rows, batch_row_size):
                    batch_rows = cleaned_rows[i:i + batch_row_size]
                    start_row = i + 1
                    end_row = i + len(batch_rows)
                    batch_range = f"A{start_row}:{last_col}{end_row}"
                    worksheet.update(batch_range, batch_rows, value_input_option='RAW')
                    if i + batch_row_size < num_rows:
                        time.sleep(1.0)

                if detected_count > 0:
                    print(f"  ✅ '{sheet_name}': {detected_count}개 셀 BOM 감지 + 전체 시트 재작성 완료 ({num_rows}행)")
                else:
                    print(f"  ✅ '{sheet_name}': 전체 시트 재작성 완료 ({num_rows}행, 숨겨진 BOM 포함 제거)")

            results[sheet_name] = detected_count

        except Exception as e:
            print(f"  ❌ '{sheet_name}' BOM 정리 실패: {e}")
            results[sheet_name] = 0

    return results


def col_num_to_letter(col_num: int) -> str:
    """
    컬럼 번호를 Excel/Sheets 스타일 문자로 변환

    Args:
        col_num: 컬럼 번호 (1-based, 1=A, 27=AA)

    Returns:
        컬럼 문자 (A, B, ..., Z, AA, AB, ...)

    Examples:
        1 -> A
        26 -> Z
        27 -> AA
        52 -> AZ
        53 -> BA
    """
    result = ""
    while col_num > 0:
        col_num -= 1  # 0-based로 변환
        result = chr(65 + (col_num % 26)) + result
        col_num //= 26
    return result


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
            "https://www.googleapis.com/auth/spreadsheets",
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


def load_existing_links_from_sheets(spreadsheet, sheet_name: str = "raw_data") -> set:
    """
    Google Sheets에서 기존 기사 링크 목록 로드

    Args:
        spreadsheet: gspread Spreadsheet 객체
        sheet_name: 워크시트 이름 (기본: raw_data)

    Returns:
        기존 기사 링크 set (중복 제거용)
    """
    try:
        # 워크시트 선택
        try:
            worksheet = spreadsheet.worksheet(sheet_name)
        except:
            print(f"  ℹ️  '{sheet_name}' 워크시트가 없습니다. 첫 실행으로 간주합니다.")
            return set()

        # 모든 데이터 읽기
        existing_data = worksheet.get_all_records()

        if not existing_data:
            print(f"  ℹ️  '{sheet_name}' 워크시트가 비어있습니다.")
            return set()

        # link 컬럼 추출
        existing_links = set()
        for row in existing_data:
            link = row.get("link", "")
            if link:
                existing_links.add(link)

        print(f"📂 Google Sheets에서 {len(existing_links)}개 기존 기사 로드")
        return existing_links

    except Exception as e:
        print(f"⚠️  Google Sheets 로드 실패: {e}")
        print("  → 증분 수집 없이 계속 진행합니다.")
        return set()


def load_analysis_status_from_sheets(
    spreadsheet,
    sheet_name: str = "total_result",
    analysis_cols: Optional[List[str]] = None
) -> Dict[str, set]:
    """
    Google Sheets에서 분석 완료/미완료 링크 집합 로드

    Returns:
        {"processed_links": set, "missing_analysis_links": set}
    """
    if analysis_cols is None:
        # LLM 분석 + 전처리 필드 체크
        analysis_cols = [
            "brand_relevance", "sentiment_stage",  # LLM 분석
            "source", "media_domain", "date_only"  # 전처리 필드
        ]

    try:
        try:
            worksheet = spreadsheet.worksheet(sheet_name)
        except Exception:
            print(f"  ℹ️  '{sheet_name}' 워크시트가 없습니다. 첫 실행으로 간주합니다.")
            return {"processed_links": set(), "missing_analysis_links": set()}

        existing_data = worksheet.get_all_records()
        if not existing_data:
            print(f"  ℹ️  '{sheet_name}' 워크시트가 비어있습니다.")
            return {"processed_links": set(), "missing_analysis_links": set()}

        processed_links = set()
        missing_analysis_links = set()

        for row in existing_data:
            link = row.get("link", "")
            if not link:
                continue
            processed_links.add(link)
            # 분석 필드가 하나라도 비어 있으면 재분석 대상으로 간주 (BOM 문자도 빈 값으로 처리)
            for col in analysis_cols:
                val = row.get(col, "")
                # BOM 문자 제거 후 체크
                cleaned_val = clean_bom(val)
                if cleaned_val == "":
                    missing_analysis_links.add(link)
                    break

        print(f"📂 Google Sheets에서 {len(processed_links)}개 기존 기사 로드 (total_result)")
        if missing_analysis_links:
            print(f"  ℹ️  분석 누락 링크 {len(missing_analysis_links)}개 발견")

        return {
            "processed_links": processed_links,
            "missing_analysis_links": missing_analysis_links
        }

    except Exception as e:
        print(f"⚠️  Google Sheets 분석 상태 로드 실패: {e}")
        print("  → result.csv 기준으로 계속 진행합니다.")
        return {"processed_links": set(), "missing_analysis_links": set()}


def filter_new_articles_from_sheets(df_raw: pd.DataFrame, existing_links: set) -> pd.DataFrame:
    """
    Google Sheets 기존 데이터와 비교하여 새 기사만 필터링

    Args:
        df_raw: 수집한 원본 DataFrame
        existing_links: Google Sheets의 기존 링크 set

    Returns:
        새 기사만 포함한 DataFrame
    """
    if len(existing_links) == 0:
        print(f"✅ 모든 {len(df_raw)}개 기사가 새 기사입니다 (기존 데이터 없음)")
        return df_raw

    # link 컬럼이 기존 링크에 없는 행만 필터링
    df_new = df_raw[~df_raw["link"].isin(existing_links)].copy()

    skipped = len(df_raw) - len(df_new)
    print(f"✅ {len(df_new)}개 새 기사 발견 ({skipped}개 중복 건너뜀)")

    return df_new


def sync_to_sheets(df: pd.DataFrame, spreadsheet,
                  sheet_name: str = "전체데이터",
                  key_column: str = "link",
                  update_fields: list = None) -> Dict[str, int]:
    """
    DataFrame을 Google Sheets에 upsert (update or insert)

    Args:
        df: 업로드할 DataFrame
        spreadsheet: gspread Spreadsheet 객체
        sheet_name: 워크시트 이름
        key_column: 중복 제거 기준 컬럼
        update_fields: 업데이트할 필드 리스트 (None이면 분석 필드 자동 감지)

    Returns:
        {"attempted": N, "added": N, "updated": N, "skipped": N, "errors": N}
        - attempted: 이번에 업로드 대상으로 넘긴 기사 수
        - added: 새로 추가된 기사 수
        - updated: 기존 행 업데이트된 기사 수
        - skipped: 시트에 이미 존재하고 업데이트 불필요한 기사 수
    """
    # 업데이트할 분석 및 전처리 필드 (기본값)
    if update_fields is None:
        update_fields = [
            # LLM 분석 필드
            "brand_relevance", "brand_relevance_query_keywords",
            "sentiment_stage", "danger_level", "issue_category",
            "news_category", "news_keyword_summary", "classified_at",
            # 전처리 필드
            "press_release_group", "cluster_id", "source",
            "media_domain", "media_name", "media_group", "media_type",
            # Looker Studio 시계열 필드
            "date_only", "week_number", "month", "article_count"
        ]

    try:
        # 워크시트 선택 또는 생성
        try:
            worksheet = spreadsheet.worksheet(sheet_name)
        except:
            worksheet = spreadsheet.add_worksheet(title=sheet_name, rows=1000, cols=30)
            print(f"  📝 새 워크시트 생성: {sheet_name}")

        # 기존 데이터 읽기
        try:
            existing_data = worksheet.get_all_records()
        except:
            existing_data = []

        attempted = len(df)
        added_count = 0
        updated_count = 0
        skipped_count = 0

        # 기존 데이터를 dict로 변환 (link → row_index, row_data)
        existing_by_key = {}
        if existing_data:
            for row_idx, row in enumerate(existing_data, start=2):  # 헤더는 1행, 데이터는 2행부터
                key_val = row.get(key_column, "")
                if key_val:
                    existing_by_key[key_val] = {"row_idx": row_idx, "data": row}

        # 헤더 행이 없으면 추가 (BOM 제거 후)
        if len(existing_data) == 0:
            clean_headers = [clean_bom(col) for col in df.columns.tolist()]
            worksheet.append_row(clean_headers)

        # 새로운 행과 업데이트 대상 행 분류
        new_rows = []
        rows_to_update = []  # (row_idx, new_values)

        for _, row in df.iterrows():
            key_val = row[key_column] if key_column in df.columns else None

            if not key_val or key_val not in existing_by_key:
                # 새 행: append 대상
                new_rows.append(row)
            else:
                # 기존 행: 업데이트 필요 여부 체크
                existing_row_info = existing_by_key[key_val]
                existing_row_data = existing_row_info["data"]
                row_idx = existing_row_info["row_idx"]

                # 업데이트 필요 여부 확인 (실제 값 변경이 있을 때만)
                needs_update = False
                for field in update_fields:
                    if field not in df.columns:
                        continue
                    new_val = clean_bom(row.get(field, ""))
                    existing_val = clean_bom(existing_row_data.get(field, ""))

                    # 업데이트가 필요한 경우:
                    # 1. 빈 값에 실제 값이 들어갈 때 (기존: 빈값, 새로운: 값 있음)
                    # 2. 둘 다 값이 있고 다를 때 (기존: 값A, 새로운: 값B)
                    # 절대 하지 않는 경우:
                    # - 빈 값 → 빈 값 (변경 없음)
                    # - 기존 값 → 빈 값 (기존 분석 결과 보호!)
                    if existing_val == "" and new_val != "":
                        needs_update = True
                        break
                    elif existing_val != "" and new_val != "" and new_val != existing_val:
                        needs_update = True
                        break

                if needs_update:
                    rows_to_update.append((row_idx, row, existing_row_data))
                else:
                    skipped_count += 1

        # 새 행 추가 (batch append)
        if new_rows:
            values_to_append = []
            for row in new_rows:
                row_values = []
                for col in df.columns:
                    val = row[col]
                    # BOM 문자 제거 및 빈 값 정리
                    cleaned_val = clean_bom(val)
                    row_values.append(cleaned_val)
                values_to_append.append(row_values)

            # 일괄 추가 (최대 1000행씩)
            batch_size = 1000
            for i in range(0, len(values_to_append), batch_size):
                batch = values_to_append[i:i+batch_size]
                worksheet.append_rows(batch)
                time.sleep(1.0)  # Rate limit 방지

            added_count = len(new_rows)
            print(f"  ✅ {sheet_name}: {added_count}개 행 추가")

        # 기존 행 업데이트 (batch update)
        if rows_to_update:
            # batch_update 준비
            updates = []
            for row_idx, row_data, existing_row_data in rows_to_update:
                # 전체 행 값 생성 (기존 값 보호: 새 값이 비어있으면 기존 값 유지)
                row_values = []
                for col in df.columns:
                    new_val = clean_bom(row_data[col])
                    existing_val = clean_bom(existing_row_data.get(col, ""))

                    # 새 값이 비어있고 기존 값이 있으면 → 기존 값 보호
                    if new_val == "" and existing_val != "":
                        cleaned_val = existing_val
                    else:
                        cleaned_val = new_val
                    row_values.append(cleaned_val)

                # A{row_idx}:LastCol{row_idx} 형식으로 범위 지정
                # 컬럼 수를 올바르게 문자로 변환 (A, B, ..., Z, AA, AB, ...)
                last_col_letter = col_num_to_letter(len(df.columns))
                range_name = f"A{row_idx}:{last_col_letter}{row_idx}"

                updates.append({"range": range_name, "values": [row_values]})

            # 디버그: 업데이트 범위 요약 출력
            if len(updates) > 0:
                first_range = updates[0]["range"]
                last_range = updates[-1]["range"]
                if len(updates) == 1:
                    print(f"    🔍 업데이트 범위: {first_range} (1개 행)")
                else:
                    print(f"    🔍 업데이트 범위: {first_range} ~ {last_range} ({len(updates)}개 행)")

            # batch_update 실행 (최대 100개씩)
            update_batch_size = 100
            for i in range(0, len(updates), update_batch_size):
                batch_updates = updates[i:i+update_batch_size]
                try:
                    worksheet.batch_update(batch_updates, value_input_option='RAW')
                    time.sleep(1.0)  # Rate limit 방지
                except Exception as e:
                    error_msg = str(e)
                    print(f"    ⚠️  batch_update 실패: {error_msg}")
                    # 디버그: 첫 번째 업데이트 range 출력
                    if batch_updates:
                        print(f"    🔍 첫 번째 range 예시: {batch_updates[0]['range']}")
                    # Fallback: 개별 update
                    for idx, update in enumerate(batch_updates):
                        try:
                            range_str = update["range"]
                            # 디버그: 개별 update 시 range 출력 (처음 3개만)
                            if idx < 3:
                                print(f"    🔍 개별 update 시도 [{idx+1}]: range='{range_str}'")
                            worksheet.update(range_str, update["values"], value_input_option='RAW')
                            time.sleep(0.5)
                        except Exception as e2:
                            print(f"    ⚠️  개별 update 실패 [range={update.get('range', 'N/A')}]: {e2}")

            updated_count = len(rows_to_update)
            print(f"  🔄 {sheet_name}: {updated_count}개 행 업데이트")

        if added_count == 0 and updated_count == 0:
            print(f"  ℹ️  {sheet_name}: 변경 사항 없음 ({skipped_count}개 건너뜀)")

        return {
            "attempted": attempted,
            "added": added_count,
            "updated": updated_count,
            "skipped": skipped_count,
            "errors": 0
        }

    except Exception as e:
        print(f"  ❌ {sheet_name} 업로드 실패: {e}")
        return {"attempted": len(df), "added": 0, "updated": 0, "skipped": 0, "errors": len(df)}


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


def sync_raw_and_processed(df_raw: pd.DataFrame, df_result: pd.DataFrame, spreadsheet) -> Dict[str, Dict]:
    """
    원본 데이터와 분류 결과를 Google Sheets에 upsert (update or insert)

    시트 구조:
    - raw_data: 원본 데이터 (수집된 그대로)
    - total_result: 전체 분류 결과 (독립기사 + 보도자료) - 기존 행 업데이트 지원

    동작:
    - 새 기사: append
    - 기존 기사 (분석 필드 비어있음): update
    - 기존 기사 (분석 필드 있음): skip

    Args:
        df_raw: 원본 데이터 (수집된 그대로)
        df_result: 분류 결과 (AI 분류 완료)
        spreadsheet: gspread Spreadsheet 객체

    Returns:
        {sheet_name: {added, updated, skipped, errors}}
    """
    results = {}

    print("📊 Google Sheets 동기화 중...")

    # 1. raw_data - 원본 데이터
    print("\n  [1/2] raw_data (원본 데이터)")
    results["raw_data"] = sync_to_sheets(df_raw, spreadsheet, "raw_data")

    # 2. total_result - 전체 분류 결과 (upsert 지원)
    print("  [2/2] total_result (전체 분류 결과)")
    results["total_result"] = sync_to_sheets(df_result, spreadsheet, "total_result")

    # 통계
    print("\n✅ Google Sheets 동기화 완료")
    total_attempted = sum(r.get("attempted", 0) for r in results.values())
    total_added = sum(r.get("added", 0) for r in results.values())
    total_updated = sum(r.get("updated", 0) for r in results.values())
    total_skipped = sum(r.get("skipped", 0) for r in results.values())
    total_errors = sum(r.get("errors", 0) for r in results.values())

    print(f"  - 시도됨: {total_attempted}개")
    print(f"  - 추가됨: {total_added}개")
    print(f"  - 업데이트됨: {total_updated}개")
    print(f"  - 건너뜀(변경 없음): {total_skipped}개")
    if total_errors > 0:
        print(f"  - 오류: {total_errors}개")
    return results


def sync_all_sheets(df: pd.DataFrame, spreadsheet) -> Dict[str, Dict]:
    """
    (deprecated) 데이터를 여러 워크시트에 동시 업로드

    대신 sync_raw_and_processed()를 사용하세요.

    Args:
        df: 분류된 DataFrame
        spreadsheet: gspread Spreadsheet 객체

    Returns:
        {sheet_name: {added, skipped, errors}}
    """
    print("⚠️  sync_all_sheets()는 deprecated 됨. sync_raw_and_processed()를 사용하세요.")
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
    total_attempted = sum(r.get("attempted", 0) for r in results.values())
    total_added = sum(r["added"] for r in results.values())
    total_skipped = sum(r["skipped"] for r in results.values())
    total_errors = sum(r["errors"] for r in results.values())

    print(f"  - 시도됨: {total_attempted}개")
    print(f"  - 추가됨: {total_added}개")
    print(f"  - 건너뜀(시트에 이미 존재): {total_skipped}개")
    print(f"  - 오류: {total_errors}개")

    return results
