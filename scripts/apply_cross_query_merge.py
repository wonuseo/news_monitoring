"""
apply_cross_query_merge.py

새 임계값이 반영된 merge_cross_query_clusters를 현재 total_result에 적용.

변경되는 컬럼: cluster_id, source, cluster_summary
변경된 행만 Sheets에 batch 업데이트 (전체 재업로드 없음).

Usage:
    # 변경사항 미리보기 (Sheets 업데이트 없음)
    python scripts/apply_cross_query_merge.py --dry_run

    # 실제 적용
    python scripts/apply_cross_query_merge.py
"""

import os
import sys
import time
import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

TARGET_COLS = ["cluster_id", "source", "cluster_summary"]


def load_total_result(spreadsheet) -> tuple:
    """total_result 워크시트 → (df, worksheet, headers, link_to_row)"""
    ws = spreadsheet.worksheet("total_result")
    all_values = ws.get_all_values()
    if not all_values:
        raise RuntimeError("total_result 시트가 비어있음")

    headers = all_values[0]
    rows = all_values[1:]
    df = pd.DataFrame(rows, columns=headers)

    # Sheets 행 번호 매핑: link → 시트 행 번호 (헤더=1, 데이터 시작=2)
    link_col = "link"
    if link_col not in df.columns:
        raise RuntimeError("total_result에 link 컬럼 없음")

    link_to_sheet_row = {}
    for df_idx, row_values in enumerate(rows):
        link_val = row_values[headers.index(link_col)] if link_col in headers else ""
        if link_val:
            link_to_sheet_row[link_val] = df_idx + 2  # +2: 헤더행 + 0-base 보정

    return df, ws, headers, link_to_sheet_row


def run(dry_run: bool):
    from src.modules.export.sheets import connect_sheets
    from src.modules.analysis.source_verifier import merge_cross_query_clusters

    creds_path = os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH", "service-account.json")
    sheet_id = os.getenv("GOOGLE_SHEET_ID")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not os.path.exists(creds_path) or not sheet_id:
        print("❌ GOOGLE_SHEETS_CREDENTIALS_PATH 또는 GOOGLE_SHEET_ID 미설정")
        return

    # ── 1. Sheets 연결 + 데이터 로드 ────────────────────────────────────────
    print("📊 Google Sheets 연결 중...")
    sp = connect_sheets(creds_path, sheet_id)
    df, ws, headers, link_to_sheet_row = load_total_result(sp)
    print(f"  ✅ total_result: {len(df)}행 로드")

    # 필수 컬럼 보장
    for col in ["cluster_id", "source", "cluster_summary", "query", "title",
                "description", "pub_datetime", "media_domain", "news_category"]:
        if col not in df.columns:
            df[col] = ""

    # ── 2. 변경 전 스냅샷 ───────────────────────────────────────────────────
    before = df[TARGET_COLS].copy()

    # ── 3. 새 임계값으로 merge_cross_query_clusters 실행 ────────────────────
    print("\n🔀 새 임계값으로 Cross-query 병합 실행 중...")
    df_updated, stats = merge_cross_query_clusters(df, openai_key=openai_key)

    print(f"\n  병합 그룹: {stats['sv_cross_merged_groups']}개")
    print(f"  병합 기사: {stats['sv_cross_merged_articles']}개")

    # ── 4. 변경된 행 찾기 ───────────────────────────────────────────────────
    after = df_updated[TARGET_COLS]
    changed_mask = (before != after).any(axis=1)
    changed_df = df_updated[changed_mask].copy()
    changed_before = before[changed_mask].copy()

    print(f"\n  변경된 행: {len(changed_df)}개")

    if len(changed_df) == 0:
        print("  ℹ️  변경사항 없음. Sheets 업데이트 불필요.")
        return

    # ── 5. 변경사항 출력 ────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("변경 내역:")
    print("─" * 80)

    for df_idx in changed_df.index:
        row_before = changed_before.loc[df_idx]
        row_after = changed_df.loc[df_idx]
        title = str(df_updated.at[df_idx, "title"])[:60]
        link = str(df_updated.at[df_idx, "link"]) if "link" in df_updated.columns else ""
        sheet_row = link_to_sheet_row.get(link, "?")

        print(f"\n  [행 {sheet_row}] {title}")
        for col in TARGET_COLS:
            if str(row_before[col]) != str(row_after[col]):
                print(f"    {col}: '{row_before[col]}' → '{row_after[col]}'")

    if dry_run:
        print(f"\n{'─'*80}")
        print("  [DRY RUN] Sheets 업데이트 생략. --dry_run 없이 실행하면 적용됩니다.")
        return

    # ── 6. Sheets batch 업데이트 ────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print("☁️  Sheets 업데이트 중...")

    # 컬럼 → Sheets 열 번호 매핑
    col_indices = {}
    for col in TARGET_COLS:
        if col in headers:
            col_indices[col] = headers.index(col)
        else:
            # 컬럼이 없으면 헤더 끝에 추가
            ws.add_cols(1)
            headers.append(col)
            col_indices[col] = len(headers) - 1
            ws.update_cell(1, col_indices[col] + 1, col)
            print(f"  ℹ️  '{col}' 컬럼 신규 추가 (헤더 행)")

    # batch_data: [{range: "A2", values: [[val]]}, ...]
    batch_data = []
    for df_idx in changed_df.index:
        link = str(df_updated.at[df_idx, "link"]) if "link" in df_updated.columns else ""
        sheet_row = link_to_sheet_row.get(link)
        if sheet_row is None:
            print(f"  ⚠️  행 위치 찾기 실패: {link[:60]}")
            continue

        for col in TARGET_COLS:
            if str(before.at[df_idx, col]) != str(after.at[df_idx, col]):
                col_num = col_indices[col] + 1  # 1-based
                cell_addr = f"{_col_letter(col_num)}{sheet_row}"
                new_val = str(df_updated.at[df_idx, col]) if pd.notna(df_updated.at[df_idx, col]) else ""
                batch_data.append({
                    "range": cell_addr,
                    "values": [[new_val]]
                })

    if not batch_data:
        print("  ℹ️  업데이트할 셀 없음")
        return

    # gspread batch_update (최대 500셀 단위)
    BATCH_SIZE = 500
    total_cells = len(batch_data)
    updated = 0
    for i in range(0, total_cells, BATCH_SIZE):
        batch = batch_data[i:i + BATCH_SIZE]
        ws.batch_update(batch, value_input_option="RAW")
        updated += len(batch)
        print(f"  업데이트: {updated}/{total_cells}셀")
        if i + BATCH_SIZE < total_cells:
            time.sleep(1.0)

    print(f"\n✅ Sheets 업데이트 완료: {len(changed_df)}행 / {total_cells}셀")


def _col_letter(n: int) -> str:
    """1-based 컬럼 번호 → A, B, ..., Z, AA, ..."""
    result = ""
    while n > 0:
        n -= 1
        result = chr(65 + n % 26) + result
        n //= 26
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="새 Cross-query 임계값 Sheets 적용")
    parser.add_argument("--dry_run", action="store_true", help="변경사항 미리보기 (Sheets 업데이트 없음)")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
