#!/usr/bin/env python3
"""
migrate_source_verification.py - 기존 데이터 source 재검증 일회성 마이그레이션

Google Sheets total_result의 기존 기사에 대해:
1. 보도자료 클러스터를 LLM 분류 결과로 검증 (Part A)
2. 비클러스터 일반기사 중 같은 주제 그룹 발견 (Part B)
3. source 컬럼 업데이트 후 Sheets에 반영

사용법:
  python migrate_source_verification.py              # 실행
  python migrate_source_verification.py --dry_run     # 미리보기 (Sheets 업데이트 안 함)
"""

import os
import argparse
import pandas as pd
from dotenv import load_dotenv

from src.modules.export.sheets import connect_sheets, sync_to_sheets
from src.modules.analysis.source_verifier import verify_and_regroup_sources


def main():
    parser = argparse.ArgumentParser(description="기존 데이터 source 재검증 마이그레이션")
    parser.add_argument("--dry_run", action="store_true",
                        help="미리보기만 (Sheets 업데이트 안 함)")
    parser.add_argument("--sheets_id", type=str, default=None,
                        help="Google Sheets ID (.env 대신 사용)")
    args = parser.parse_args()

    load_dotenv()

    # Google Sheets 연결
    creds_path = os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH", "service-account.json")
    sheet_id = args.sheets_id or os.getenv("GOOGLE_SHEET_ID")

    if not os.path.exists(creds_path) or not sheet_id:
        print("Google Sheets 설정이 필요합니다.")
        print("  .env에 GOOGLE_SHEETS_CREDENTIALS_PATH, GOOGLE_SHEET_ID를 설정하세요.")
        return

    spreadsheet = connect_sheets(creds_path, sheet_id)
    if not spreadsheet:
        return

    # total_result 로드
    print("\n총 결과 데이터 로드 중...")
    try:
        worksheet = spreadsheet.worksheet("total_result")
        records = worksheet.get_all_records()
    except Exception as e:
        print(f"total_result 시트 로드 실패: {e}")
        return

    if not records:
        print("total_result 시트가 비어있습니다.")
        return

    df = pd.DataFrame(records)
    print(f"로드 완료: {len(df)}개 기사")

    # 변경 전 source 분포
    if "source" in df.columns:
        print(f"\n변경 전 source 분포:")
        print(df["source"].value_counts().to_string())
    else:
        print("\nsource 컬럼이 없습니다. 마이그레이션 불필요.")
        return

    # source 검증 및 주제 그룹화
    print("\n" + "=" * 60)
    df, stats = verify_and_regroup_sources(df)
    print("=" * 60)

    # 변경 후 source 분포
    print(f"\n변경 후 source 분포:")
    print(df["source"].value_counts().to_string())

    # 통계 요약
    total_changes = (
        stats.get("sv_reclassified_similar_topic", 0)
        + stats.get("sv_new_topic_articles", 0)
    )
    print(f"\n총 변경: {total_changes}개 기사")

    if stats.get("sv_llm_verified", 0) > 0 or stats.get("sv_llm_rejected", 0) > 0:
        print(f"  - LLM 경계선 검증: {stats['sv_llm_verified']}개 연결, "
              f"{stats['sv_llm_rejected']}개 거부")

    if total_changes == 0:
        print("변경 사항이 없습니다.")
        return

    # Sheets 업데이트
    if args.dry_run:
        print("\n[DRY RUN] Sheets 업데이트를 건너뜁니다.")
        return

    print("\nGoogle Sheets 업데이트 중...")
    result = sync_to_sheets(
        df, spreadsheet, "total_result",
        force_update_existing=True,
    )
    print(f"완료: 추가 {result.get('added', 0)}개, "
          f"업데이트 {result.get('updated', 0)}개, "
          f"건너뜀 {result.get('skipped', 0)}개")

    # 로컬 CSV 백업도 업데이트
    csv_path = "../data/result.csv"
    if os.path.exists(csv_path):
        print(f"\n로컬 CSV 백업 업데이트: {csv_path}")
        df_csv = pd.read_csv(csv_path, encoding="utf-8-sig")
        # source, cluster_id만 업데이트 (link 기준 merge)
        update_cols = ["link", "source"]
        if "cluster_id" in df.columns:
            update_cols.append("cluster_id")

        df_updates = df[update_cols].copy()
        df_csv = df_csv.drop(columns=["source", "cluster_id"], errors="ignore")
        df_csv = df_csv.merge(df_updates, on="link", how="left")
        df_csv.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"CSV 업데이트 완료: {len(df_csv)}개 기사")

    print("\n마이그레이션 완료!")


if __name__ == "__main__":
    main()
