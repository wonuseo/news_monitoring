"""
sheets_helpers.py - Shared Google Sheets Utilities
워크시트 get-or-create, 중간 동기화 패턴
"""

import pandas as pd
from pathlib import Path


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


def intermediate_sync(
    df_processed: pd.DataFrame,
    df_raw: pd.DataFrame,
    result_csv_path: Path,
    spreadsheet,
    stage_label: str,
    save_csv_fn,
    record_error_fn,
):
    """
    중간 동기화 패턴: result.csv 병합 저장 + Sheets 동기화.

    Args:
        df_processed: 현재 단계까지 처리된 DataFrame
        df_raw: 원본 raw DataFrame (Sheets 동기화용)
        result_csv_path: result.csv 경로
        spreadsheet: gspread Spreadsheet 객체 (None이면 CSV만 저장)
        stage_label: 단계 라벨 (예: "중복 제거", "보도자료 탐지")
        save_csv_fn: CSV 저장 함수 (save_csv)
        record_error_fn: 에러 기록 함수
    """
    print(f"💾 중간 동기화 중 ({stage_label} 완료)...")

    # result.csv와 병합 저장
    if result_csv_path.exists():
        df_result_temp = pd.read_csv(result_csv_path, encoding='utf-8-sig')
        df_temp = pd.concat([df_result_temp, df_processed], ignore_index=True)
        df_temp = df_temp.drop_duplicates(subset=['link'], keep='last')
    else:
        df_temp = df_processed
    save_csv_fn(df_temp, result_csv_path)

    # Sheets 동기화
    if spreadsheet:
        try:
            from src.modules.export.sheets import sync_raw_and_processed
            sync_raw_and_processed(df_raw, df_temp, spreadsheet)
            print(f"✅ Sheets 동기화 완료 ({stage_label})")
        except Exception as e:
            record_error_fn(f"Sheets 동기화 실패 ({stage_label}): {e}")
