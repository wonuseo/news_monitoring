"""
sheets_helpers.py - Shared Google Sheets Utilities
워크시트 get-or-create, 중간 동기화 패턴
"""

import pandas as pd


def get_or_create_worksheet(spreadsheet, sheet_name: str, rows: int = 1000, cols: int = 10):
    """
    워크시트를 가져오거나 없으면 새로 생성.

    Args:
        spreadsheet: gspread Spreadsheet 객체
        sheet_name: 워크시트 이름
        rows: 새 시트 생성 시 행 수
        cols: 새 시트 생성 시 열 수

    Returns:
        gspread Worksheet 객체
    """
    try:
        return spreadsheet.worksheet(sheet_name)
    except:
        worksheet = spreadsheet.add_worksheet(title=sheet_name, rows=rows, cols=cols)
        print(f"  📝 새 워크시트 생성: {sheet_name}")
        return worksheet


def get_max_values_from_sheets(spreadsheet, sheet_name: str = "total_result"):
    """
    Google Sheets에서 기존 최대값 가져오기 (cumulative numbering용)

    Args:
        spreadsheet: gspread Spreadsheet 객체
        sheet_name: 워크시트 이름 (기본값: "total_result")

    Returns:
        dict: {
            "max_article_no": int,  # 최대 article_no (0이면 데이터 없음)
            "max_cluster_num": int  # 최대 cluster_id 숫자 (0이면 데이터 없음)
        }
    """
    try:
        worksheet = spreadsheet.worksheet(sheet_name)
        data = worksheet.get_all_records()

        if not data:
            return {"max_article_no": 0, "max_cluster_num": 0}

        df = pd.DataFrame(data)

        # article_no 최대값
        max_article_no = 0
        if "article_no" in df.columns and len(df) > 0:
            # 숫자로 변환 가능한 값만 필터링
            article_nos = pd.to_numeric(df["article_no"], errors="coerce").dropna()
            if len(article_nos) > 0:
                max_article_no = int(article_nos.max())

        # cluster_id 최대 숫자 추출 (형식: {query}_c{5자리숫자})
        max_cluster_num = 0
        if "cluster_id" in df.columns and len(df) > 0:
            import re
            for cid in df["cluster_id"].dropna():
                if cid and str(cid).strip():
                    # {query}_c{숫자} 형식에서 숫자 추출
                    match = re.search(r'_c(\d+)$', str(cid))
                    if match:
                        cluster_num = int(match.group(1))
                        max_cluster_num = max(max_cluster_num, cluster_num)

        return {"max_article_no": max_article_no, "max_cluster_num": max_cluster_num}

    except Exception as e:
        print(f"  ⚠️  Sheets에서 최대값 가져오기 실패 ({sheet_name}): {e}")
        return {"max_article_no": 0, "max_cluster_num": 0}


def intermediate_sync(
    df_processed: pd.DataFrame,
    df_raw: pd.DataFrame,
    spreadsheet,
    stage_label: str,
    record_error_fn,
):
    """
    중간 동기화 패턴: Sheets에 직접 동기화 (upsert).

    Args:
        df_processed: 현재 단계까지 처리된 DataFrame
        df_raw: 원본 raw DataFrame (Sheets 동기화용)
        spreadsheet: gspread Spreadsheet 객체
        stage_label: 단계 라벨 (예: "중복 제거", "보도자료 탐지")
        record_error_fn: 에러 기록 함수
    """
    if not spreadsheet:
        return

    print(f"💾 중간 동기화 중 ({stage_label} 완료)...")

    try:
        from src.modules.export.sheets import sync_raw_and_processed
        sync_raw_and_processed(df_raw, df_processed, spreadsheet)
        print(f"✅ Sheets 동기화 완료 ({stage_label})")
    except Exception as e:
        record_error_fn(f"Sheets 동기화 실패 ({stage_label}): {e}")
