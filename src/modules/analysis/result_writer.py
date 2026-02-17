"""
result_writer.py - Classification Result Syncing to Google Sheets
분류 결과 Google Sheets 직접 동기화
"""

from typing import Optional
import pandas as pd


def sync_result_to_sheets(
    df_result: pd.DataFrame,
    raw_df: pd.DataFrame,
    spreadsheet,
    verbose: bool = True
) -> Optional[dict]:
    """
    분류 결과 DataFrame을 Google Sheets에 동기화 (upsert 방식)

    Args:
        df_result: 분류 결과 DataFrame
        raw_df: 원본 raw DataFrame
        spreadsheet: gspread Spreadsheet 객체
        verbose: 진행 메시지 출력 여부

    Returns:
        동기화 결과 딕셔너리 또는 None (실패시)
    """
    if not spreadsheet or df_result is None or len(df_result) == 0:
        return None

    try:
        from src.modules.export.sheets import sync_raw_and_processed

        sync_results = sync_raw_and_processed(raw_df, df_result, spreadsheet)

        added_count = sum(r.get('added', 0) for r in sync_results.values())
        updated_count = sum(r.get('updated', 0) for r in sync_results.values())

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
