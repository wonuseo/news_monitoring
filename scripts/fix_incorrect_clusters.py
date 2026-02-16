#!/usr/bin/env python3
"""
fix_incorrect_clusters.py - 잘못 묶인 클러스터 수정

특정 cluster_id를 해체하여 일반기사로 되돌립니다.

사용법:
  python fix_incorrect_clusters.py --cluster_ids "조선호텔_t00001,롯데호텔_t00001"
  python fix_incorrect_clusters.py --cluster_ids "조선호텔_t00001" --dry_run
"""

import os
import argparse
import pandas as pd
from dotenv import load_dotenv

from src.modules.export.sheets import connect_sheets, sync_to_sheets


def fix_clusters(df: pd.DataFrame, cluster_ids: list) -> pd.DataFrame:
    """
    지정된 cluster_id를 해체하여 일반기사로 되돌림.

    Args:
        df: 전체 데이터 DataFrame
        cluster_ids: 해체할 cluster_id 리스트

    Returns:
        수정된 DataFrame
    """
    df = df.copy()

    for cluster_id in cluster_ids:
        mask = df["cluster_id"] == cluster_id
        affected_count = mask.sum()

        if affected_count == 0:
            print(f"  ⚠️  '{cluster_id}' 클러스터를 찾을 수 없습니다.")
            continue

        # 기사 정보 출력
        print(f"\n  📌 '{cluster_id}' 클러스터 해체 ({affected_count}개 기사)")
        for idx, row in df[mask].iterrows():
            print(f"    - {row.get('title', 'N/A')[:60]}...")

        # cluster_id 제거, source를 일반기사로 변경
        df.loc[mask, "cluster_id"] = ""
        df.loc[mask, "source"] = "일반기사"
        df.loc[mask, "press_release_group"] = ""

    return df


def main():
    parser = argparse.ArgumentParser(description="잘못 묶인 클러스터 수정")
    parser.add_argument("--cluster_ids", type=str, required=True,
                        help="해체할 cluster_id (쉼표로 구분, 예: '조선호텔_t00001,롯데호텔_t00001')")
    parser.add_argument("--dry_run", action="store_true",
                        help="미리보기만 (Sheets 업데이트 안 함)")
    parser.add_argument("--sheets_id", type=str, default=None,
                        help="Google Sheets ID (.env 대신 사용)")
    args = parser.parse_args()

    load_dotenv()

    # cluster_ids 파싱
    cluster_ids = [cid.strip() for cid in args.cluster_ids.split(",")]
    print(f"\n해체 대상 클러스터: {cluster_ids}")

    # Google Sheets 연결
    creds_path = os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH", "service-account.json")
    sheet_id = args.sheets_id or os.getenv("GOOGLE_SHEET_ID")

    if not os.path.exists(creds_path) or not sheet_id:
        print("\n❌ Google Sheets 설정이 필요합니다.")
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
        print(f"❌ total_result 시트 로드 실패: {e}")
        return

    if not records:
        print("❌ total_result 시트가 비어있습니다.")
        return

    df = pd.DataFrame(records)
    print(f"✅ 로드 완료: {len(df)}개 기사")

    # 변경 전 상태
    print(f"\n변경 전 source 분포:")
    print(df["source"].value_counts().to_string())

    # 클러스터 수정
    print("\n" + "=" * 80)
    df_fixed = fix_clusters(df, cluster_ids)
    print("=" * 80)

    # 변경 후 상태
    print(f"\n변경 후 source 분포:")
    print(df_fixed["source"].value_counts().to_string())

    # Sheets 업데이트
    if args.dry_run:
        print("\n[DRY RUN] Sheets 업데이트를 건너뜁니다.")
        return

    print("\nGoogle Sheets 업데이트 중...")
    result = sync_to_sheets(
        df_fixed, spreadsheet, "total_result",
        force_update_existing=True,
    )
    print(f"✅ 완료: 추가 {result.get('added', 0)}개, "
          f"업데이트 {result.get('updated', 0)}개, "
          f"건너뜀 {result.get('skipped', 0)}개")

    # 로컬 CSV 백업도 업데이트
    csv_path = "../data/result.csv"
    if os.path.exists(csv_path):
        print(f"\n로컬 CSV 백업 업데이트: {csv_path}")
        df_csv = pd.read_csv(csv_path, encoding="utf-8-sig")

        # source, cluster_id, press_release_group 업데이트 (link 기준 merge)
        update_cols = ["link", "source", "cluster_id", "press_release_group"]
        df_updates = df_fixed[update_cols].copy()

        df_csv = df_csv.drop(columns=["source", "cluster_id", "press_release_group"], errors="ignore")
        df_csv = df_csv.merge(df_updates, on="link", how="left")
        df_csv.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"✅ CSV 업데이트 완료: {len(df_csv)}개 기사")

    print("\n✅ 클러스터 수정 완료!")


if __name__ == "__main__":
    main()
