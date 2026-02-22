"""
sheets.py - Google Sheets Integration Module
Google Sheets로 데이터를 증분 업로드하고 Looker Studio 연계
"""

import pandas as pd
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import os
import time

from src.utils.text_cleaning import clean_bom
from src.utils.sheets_helpers import get_or_create_worksheet
from src.utils.group_labels import is_competitor_group, is_our_group


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


def _normalize_header(value) -> str:
    """Normalize headers/cells for robust column matching."""
    return clean_bom(value)


def _read_sheet_values(worksheet) -> Tuple[List[str], List[List[str]]]:
    """
    Read worksheet values and return normalized headers + data rows.

    Returns:
        (headers, rows)
    """
    all_values = worksheet.get_all_values()
    if not all_values:
        return [], []
    headers = [_normalize_header(h) for h in all_values[0]]
    return headers, all_values[1:]


def _find_header_index(headers: List[str], column_name: str) -> Optional[int]:
    """Find header index (case-insensitive)."""
    target = _normalize_header(column_name).lower()
    for idx, header in enumerate(headers):
        if _normalize_header(header).lower() == target:
            return idx
    return None


def _ensure_sheet_headers(
    worksheet,
    existing_headers: List[str],
    df_headers: List[str],
) -> List[str]:
    """
    Ensure worksheet has clean headers and includes all DataFrame columns.
    Returns the final header order used for read/write.
    """
    final_headers = [_normalize_header(h) for h in existing_headers]

    # 빈 시트(또는 헤더만 손상)면 DataFrame 헤더로 초기화
    if not final_headers or all(h == "" for h in final_headers):
        final_headers = [_normalize_header(col) for col in df_headers]
    else:
        # 시트에 없는 신규 컬럼은 뒤에 추가
        for col in df_headers:
            normalized = _normalize_header(col)
            if normalized and _find_header_index(final_headers, normalized) is None:
                final_headers.append(normalized)

    if not final_headers:
        return final_headers

    # 컬럼 수 부족 시 확장
    if getattr(worksheet, "col_count", len(final_headers)) < len(final_headers):
        worksheet.add_cols(len(final_headers) - worksheet.col_count)

    # 헤더를 명시적으로 재작성해 BOM/공백 문제를 고정
    last_col = col_num_to_letter(len(final_headers))
    worksheet.update(f"A1:{last_col}1", [final_headers], value_input_option='RAW')
    return final_headers


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
        except Exception:
            print(f"  ℹ️  '{sheet_name}' 워크시트가 없습니다. 첫 실행으로 간주합니다.")
            return set()

        headers, rows = _read_sheet_values(worksheet)
        if not headers or not rows:
            print(f"  ℹ️  '{sheet_name}' 워크시트가 비어있습니다.")
            return set()

        link_idx = _find_header_index(headers, "link")
        if link_idx is None:
            print(f"  ⚠️  '{sheet_name}'에서 link 컬럼을 찾지 못했습니다.")
            return set()

        existing_links = set()
        for row in rows:
            link = clean_bom(row[link_idx]) if len(row) > link_idx else ""
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
    [DEPRECATED] reprocess_checker.py의 check_reprocess_targets()로 대체됨.
    호환성을 위해 유지하지만, main.py에서는 더 이상 호출하지 않음.

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

        headers, rows = _read_sheet_values(worksheet)
        if not headers or not rows:
            print(f"  ℹ️  '{sheet_name}' 워크시트가 비어있습니다.")
            return {"processed_links": set(), "missing_analysis_links": set()}

        link_idx = _find_header_index(headers, "link")
        if link_idx is None:
            print(f"  ⚠️  '{sheet_name}'에서 link 컬럼을 찾지 못했습니다.")
            return {"processed_links": set(), "missing_analysis_links": set()}

        analysis_indexes = {
            col: _find_header_index(headers, col)
            for col in analysis_cols
        }

        processed_links = set()
        missing_analysis_links = set()

        for row in rows:
            link = clean_bom(row[link_idx]) if len(row) > link_idx else ""
            if not link:
                continue
            processed_links.add(link)
            # 분석 필드가 하나라도 비어 있으면 재분석 대상으로 간주 (BOM 문자도 빈 값으로 처리)
            for col in analysis_cols:
                col_idx = analysis_indexes.get(col)
                val = row[col_idx] if col_idx is not None and len(row) > col_idx else ""
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
                  update_fields: list = None,
                  force_update_existing: bool = False) -> Dict[str, int]:
    """
    DataFrame을 Google Sheets에 upsert (update or insert)

    Args:
        df: 업로드할 DataFrame
        spreadsheet: gspread Spreadsheet 객체
        sheet_name: 워크시트 이름
        key_column: 중복 제거 기준 컬럼
        update_fields: 업데이트할 필드 리스트 (None이면 분석 필드 자동 감지)
        force_update_existing: True면 기존 키 행도 강제 업데이트

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
            "cluster_id", "cluster_summary", "source",
            "media_domain", "media_name", "media_group", "media_type",
            # Looker Studio 시계열 필드
            "date_only", "week_number", "month", "article_count"
        ]

    try:
        if df is None or len(df) == 0:
            return {"attempted": 0, "added": 0, "updated": 0, "skipped": 0, "errors": 0}

        # DataFrame 컬럼 정규화 매핑 (소문자 키 기준)
        df_column_lookup = {}
        for col in df.columns:
            normalized = _normalize_header(col).lower()
            if normalized and normalized not in df_column_lookup:
                df_column_lookup[normalized] = col

        key_col_norm = _normalize_header(key_column).lower()
        if key_col_norm not in df_column_lookup:
            raise ValueError(f"'{key_column}' 컬럼이 DataFrame에 없습니다.")

        # 워크시트 선택 또는 생성
        worksheet = get_or_create_worksheet(spreadsheet, sheet_name, rows=1000, cols=30)

        # 기존 시트 읽기 + 헤더 보정
        sheet_headers, raw_rows = _read_sheet_values(worksheet)
        key_idx_before_fix = _find_header_index(sheet_headers, key_column)
        sheet_headers = _ensure_sheet_headers(worksheet, sheet_headers, list(df.columns))
        if not sheet_headers:
            raise ValueError("시트 헤더를 구성할 수 없습니다.")

        # 데이터 행이 이미 있는데 key 헤더가 없으면 안전하게 중단
        if raw_rows and key_idx_before_fix is None:
            raise ValueError(
                f"'{sheet_name}' 시트에 데이터가 있지만 key 헤더 '{key_column}'를 찾지 못했습니다. "
                "헤더를 확인/복구 후 다시 실행하세요."
            )

        key_idx = _find_header_index(sheet_headers, key_column)
        if key_idx is None:
            raise ValueError(f"'{sheet_name}' 시트 헤더에 key 컬럼 '{key_column}'이 없습니다.")

        attempted = len(df)
        added_count = 0
        updated_count = 0
        skipped_count = 0

        # 기존 데이터를 dict로 변환 (key → row_index, row_data)
        existing_by_key = {}
        for row_idx, row in enumerate(raw_rows, start=2):  # 헤더는 1행, 데이터는 2행부터
            row_values = [
                clean_bom(row[col_idx]) if col_idx < len(row) else ""
                for col_idx in range(len(sheet_headers))
            ]
            key_val = row_values[key_idx] if key_idx < len(row_values) else ""
            if not key_val:
                continue

            row_dict = {}
            for col_idx, header in enumerate(sheet_headers):
                header_norm = _normalize_header(header).lower()
                # 중복 헤더가 있어도 첫 번째 헤더를 기준으로 고정
                if header_norm and header_norm not in row_dict:
                    row_dict[header_norm] = row_values[col_idx]
            existing_by_key[key_val] = {"row_idx": row_idx, "data": row_dict}

        update_field_norms = {_normalize_header(field).lower() for field in update_fields}

        # 새로운 행과 업데이트 대상 행 분류
        new_rows = []
        rows_to_update = []  # (row_idx, row_dict, existing_row_dict)
        for row in df.to_dict("records"):
            key_val = clean_bom(row.get(df_column_lookup[key_col_norm], ""))

            if not key_val or key_val not in existing_by_key:
                new_rows.append(row)
                continue

            existing_row_info = existing_by_key[key_val]
            existing_row_data = existing_row_info["data"]
            row_idx = existing_row_info["row_idx"]

            if force_update_existing:
                needs_update = True
            else:
                needs_update = False
                for field_norm in update_field_norms:
                    if field_norm not in df_column_lookup:
                        continue
                    field_col = df_column_lookup[field_norm]
                    new_val = clean_bom(row.get(field_col, ""))
                    existing_val = clean_bom(existing_row_data.get(field_norm, ""))

                    # 업데이트가 필요한 경우:
                    # 1. 빈 값에 실제 값이 들어갈 때 (기존: 빈값, 새로운: 값 있음)
                    # 2. 둘 다 값이 있고 다를 때 (기존: 값A, 새로운: 값B)
                    # 절대 하지 않는 경우:
                    # - 빈 값 → 빈 값 (변경 없음)
                    # - 기존 값 → 빈 값 (기존 분석 결과 보호!)
                    if existing_val == "" and new_val != "":
                        needs_update = True
                        break
                    if existing_val != "" and new_val != "" and new_val != existing_val:
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
                for header in sheet_headers:
                    header_norm = _normalize_header(header).lower()
                    if header_norm in df_column_lookup:
                        col_name = df_column_lookup[header_norm]
                        row_values.append(clean_bom(row.get(col_name, "")))
                    else:
                        row_values.append("")
                values_to_append.append(row_values)

            # 일괄 추가 (최대 1000행씩)
            batch_size = 1000
            for i in range(0, len(values_to_append), batch_size):
                batch = values_to_append[i:i + batch_size]
                worksheet.append_rows(batch)
                time.sleep(1.0)  # Rate limit 방지

            added_count = len(new_rows)
            print(f"  ✅ {sheet_name}: {added_count}개 행 추가")

        # 기존 행 업데이트 (batch update)
        if rows_to_update:
            updates = []
            for row_idx, row_data, existing_row_data in rows_to_update:
                row_values = []
                for header in sheet_headers:
                    header_norm = _normalize_header(header).lower()
                    existing_val = clean_bom(existing_row_data.get(header_norm, ""))

                    if header_norm in df_column_lookup:
                        col_name = df_column_lookup[header_norm]
                        new_val = clean_bom(row_data.get(col_name, ""))
                    else:
                        new_val = ""

                    # 새 값이 비어있고 기존 값이 있으면 → 기존 값 보호
                    if (header_norm not in df_column_lookup) or (new_val == "" and existing_val != ""):
                        cleaned_val = existing_val
                    else:
                        cleaned_val = new_val
                    row_values.append(cleaned_val)

                last_col_letter = col_num_to_letter(len(sheet_headers))
                range_name = f"A{row_idx}:{last_col_letter}{row_idx}"
                updates.append({"range": range_name, "values": [row_values]})

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
                batch_updates = updates[i:i + update_batch_size]
                try:
                    worksheet.batch_update(batch_updates, value_input_option='RAW')
                    time.sleep(1.0)  # Rate limit 방지
                except Exception as e:
                    error_msg = str(e)
                    print(f"    ⚠️  batch_update 실패: {error_msg}")
                    if batch_updates:
                        print(f"    🔍 첫 번째 range 예시: {batch_updates[0]['range']}")
                    # Fallback: 개별 update
                    for idx, update in enumerate(batch_updates):
                        try:
                            range_str = update["range"]
                            if idx < 3:
                                print(f"    🔍 개별 update 시도 [{idx + 1}]: range='{range_str}'")
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


TOTAL_RESULT_MIN_DATE = "2026-02-01"
TOTAL_RESULT_DATE_COLUMNS = [
    "pub_datetime",
    "date_only",
    "pubDate",
    "pub_date",
    "published_at",
    "date",
]


def filter_total_result_by_date(
    df_result: pd.DataFrame,
    min_date: str = TOTAL_RESULT_MIN_DATE,
) -> pd.DataFrame:
    """
    Keep only rows on/after min_date for total_result upload.
    raw_data는 영향을 받지 않는다.
    """
    if df_result.empty:
        return df_result

    candidate_cols = [col for col in TOTAL_RESULT_DATE_COLUMNS if col in df_result.columns]
    if not candidate_cols:
        print("  ⚠️  total_result 날짜 컬럼이 없어 날짜 필터를 건너뜁니다.")
        return df_result

    cutoff = pd.Timestamp(min_date, tz="UTC")

    # 컬럼별 파싱 성공률/유지 건수를 비교해 가장 신뢰도 높은 날짜 컬럼 선택
    best_col = None
    best_parsed = None
    best_score = (-1, -1)  # (kept_count, valid_count)
    for col in candidate_cols:
        parsed = pd.to_datetime(df_result[col], errors="coerce", utc=True)
        valid_count = int(parsed.notna().sum())
        kept_count = int((parsed >= cutoff).sum())
        score = (kept_count, valid_count)
        if score > best_score:
            best_col = col
            best_parsed = parsed
            best_score = score

    if best_col is None or best_parsed is None:
        print("  ⚠️  total_result 날짜 파싱 실패로 날짜 필터를 건너뜁니다.")
        return df_result

    keep_mask = best_parsed >= cutoff

    before_count = len(df_result)
    filtered = df_result[keep_mask].copy()
    removed_count = before_count - len(filtered)
    print(
        f"  🔎 total_result 날짜 필터({best_col}): "
        f"{removed_count}개 제외 (< {min_date}), {len(filtered)}개 유지"
    )

    return filtered


def deduplicate_sheet(
    spreadsheet,
    sheet_name: str,
    key_column: str = "link",
) -> Dict[str, int]:
    """
    Sheets 탭 내 중복 행 제거 (key_column 기준 첫 번째 행 유지).

    Args:
        spreadsheet: gspread Spreadsheet 객체
        sheet_name: 대상 워크시트 이름
        key_column: 중복 기준 컬럼 (기본: link)

    Returns:
        {"before": N, "after": N, "removed": N}
    """
    try:
        worksheet = spreadsheet.worksheet(sheet_name)
    except Exception:
        return {"before": 0, "after": 0, "removed": 0}

    all_values = worksheet.get_all_values()
    if not all_values or len(all_values) <= 1:
        return {"before": 0, "after": 0, "removed": 0}

    headers = [_normalize_header(h) for h in all_values[0]]
    rows = all_values[1:]
    before = len(rows)

    key_idx = _find_header_index(headers, key_column)
    if key_idx is None:
        return {"before": before, "after": before, "removed": 0}

    seen: set = set()
    deduped = []
    for row in rows:
        key = clean_bom(row[key_idx]) if key_idx < len(row) else ""
        if not key or key not in seen:
            deduped.append(row)
            if key:
                seen.add(key)

    removed = before - len(deduped)
    if removed == 0:
        return {"before": before, "after": before, "removed": 0}

    # 헤더 길이에 맞게 패딩 후 전체 시트 재작성
    n_cols = len(headers)
    padded = [
        (row + [""] * n_cols)[:n_cols]
        for row in deduped
    ]
    all_data = [headers] + padded

    worksheet.clear()
    batch_size = 2000
    last_col = col_num_to_letter(n_cols)
    for i in range(0, len(all_data), batch_size):
        batch = all_data[i:i + batch_size]
        start_row = i + 1
        end_row = i + len(batch)
        worksheet.update(f"A{start_row}:{last_col}{end_row}", batch, value_input_option="RAW")
        if i + batch_size < len(all_data):
            time.sleep(1.0)

    return {"before": before, "after": len(deduped), "removed": removed}


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
    df_result_for_total = filter_total_result_by_date(df_result, TOTAL_RESULT_MIN_DATE)
    print("  [2/2] total_result (전체 분류 결과)")
    results["total_result"] = sync_to_sheets(
        df_result_for_total,
        spreadsheet,
        "total_result",
        force_update_existing=True
    )

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
    our_negative = df[(df["group"].map(is_our_group)) & (df["sentiment"] == "부정")]
    results["우리_부정"] = sync_to_sheets(our_negative, spreadsheet, "우리_부정")

    # 3. 우리 브랜드 긍정
    print("  [3/4] 우리_긍정")
    our_positive = df[(df["group"].map(is_our_group)) & (df["sentiment"] == "긍정")]
    results["우리_긍정"] = sync_to_sheets(our_positive, spreadsheet, "우리_긍정")

    # 4. 경쟁사
    print("  [4/4] 경쟁사")
    competitor = df[df["group"].map(is_competitor_group)]
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


def sync_result_to_sheets(
    df_result,
    raw_df,
    spreadsheet,
    verbose: bool = True,
):
    """
    분류 결과 DataFrame을 Google Sheets에 동기화 (upsert 방식).

    classify_llm, classify_press_releases의 청크별 콜백에서 사용.

    Args:
        df_result: 분류 결과 DataFrame
        raw_df: 원본 raw DataFrame
        spreadsheet: gspread Spreadsheet 객체
        verbose: 진행 메시지 출력 여부

    Returns:
        동기화 결과 딕셔너리 또는 None (실패 시)
    """
    if not spreadsheet or df_result is None or len(df_result) == 0:
        return None

    try:
        sync_results = sync_raw_and_processed(raw_df, df_result, spreadsheet)

        added_count = sum(r.get("added", 0) for r in sync_results.values())
        updated_count = sum(r.get("updated", 0) for r in sync_results.values())

        if verbose and (added_count > 0 or updated_count > 0):
            msg_parts = []
            if added_count > 0:
                msg_parts.append(f"{added_count}개 추가")
            if updated_count > 0:
                msg_parts.append(f"{updated_count}개 업데이트")
            print(f"    ☁️  Sheets 동기화: {', '.join(msg_parts)}")

        return sync_results

    except Exception as e:
        if verbose:
            print(f"    ⚠️  Sheets 동기화 실패: {e}")
        return None
