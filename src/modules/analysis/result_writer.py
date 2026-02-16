"""
result_writer.py - Classification Result Saving & Syncing
분류 결과 실시간 저장 및 Google Sheets 동기화
"""

import csv
import os
from threading import Lock
from typing import List, Optional
import pandas as pd

csv_write_lock = Lock()
_csv_header_cache: dict[str, List[str]] = {}


def invalidate_csv_header_cache(result_csv_path: str) -> None:
    """CSV 헤더 캐시 무효화 (파일이 외부에서 재작성된 경우 호출)."""
    _csv_header_cache.pop(result_csv_path, None)


def save_result_to_csv_incremental(
    df: pd.DataFrame,
    idx: int,
    result_csv_path: str
) -> bool:
    """
    단일 분류 결과를 CSV에 실시간 append (멀티스레드 안전)

    기존 CSV 파일이 있으면 헤더 순서를 읽어 컬럼 정렬 후 append.
    DataFrame 컬럼 순서와 CSV 헤더 순서가 다를 수 있으므로 반드시 정렬 필요.

    Args:
        df: 전체 DataFrame
        idx: 저장할 행의 인덱스
        result_csv_path: 저장할 CSV 경로

    Returns:
        성공 여부
    """
    if not result_csv_path:
        return False

    try:
        with csv_write_lock:
            row_df = df.loc[[idx]].copy()
            file_exists = os.path.exists(result_csv_path)

            if file_exists:
                # 기존 CSV 헤더 순서 캐시 (파일당 1회만 읽기)
                if result_csv_path not in _csv_header_cache:
                    with open(result_csv_path, 'r', encoding='utf-8-sig') as f:
                        reader = csv.reader(f)
                        _csv_header_cache[result_csv_path] = next(reader)

                csv_cols = _csv_header_cache[result_csv_path]

                # DataFrame에만 있는 신규 컬럼 → 뒤에 추가
                for col in row_df.columns:
                    if col not in csv_cols:
                        csv_cols.append(col)

                # CSV 헤더 순서에 맞춰 컬럼 재정렬
                row_df = row_df.reindex(columns=csv_cols)

            row_df.to_csv(
                result_csv_path,
                mode='a' if file_exists else 'w',
                header=not file_exists,
                index=False,
                encoding='utf-8-sig' if not file_exists else 'utf-8'
            )

            # 새 파일 생성 시 헤더 캐시 설정
            if not file_exists:
                _csv_header_cache[result_csv_path] = list(row_df.columns)

        return True
    except Exception as e:
        print(f"⚠️  CSV 저장 실패 [idx={idx}]: {e}")
        return False


def sync_result_to_sheets(
    result_csv_path: str,
    raw_df: pd.DataFrame,
    spreadsheet,
    verbose: bool = True
) -> Optional[dict]:
    """
    result.csv 전체를 Google Sheets에 동기화 (upsert 방식)

    Args:
        result_csv_path: result.csv 경로
        raw_df: 원본 raw DataFrame
        spreadsheet: gspread Spreadsheet 객체
        verbose: 진행 메시지 출력 여부

    Returns:
        동기화 결과 딕셔너리 또는 None (실패시)
    """
    if not spreadsheet or not result_csv_path:
        return None

    try:
        from src.modules.export.sheets import sync_raw_and_processed

        if not os.path.exists(result_csv_path):
            return None

        df_result_current = pd.read_csv(result_csv_path, encoding='utf-8-sig')
        sync_results = sync_raw_and_processed(raw_df, df_result_current, spreadsheet)

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
