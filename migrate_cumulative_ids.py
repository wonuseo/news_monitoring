"""
migrate_cumulative_ids.py - 기존 데이터의 article_no와 cluster_id를 cumulative하게 재할당

실행 방법:
    python migrate_cumulative_ids.py

동작:
1. Google Sheets의 total_result 데이터 읽기
2. pub_datetime 기준으로 정렬 (과거 → 최신)
3. article_no 재할당 (1부터 시작, 과거일수록 낮은 숫자)
4. cluster_id 재할당 (query별로 정렬 후 재할당, 형식: {query}_c{5자리숫자})
5. Google Sheets에 업데이트

주의: 이 작업은 기존 데이터를 변경합니다. 백업 권장.
"""

import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()


def migrate_cumulative_ids():
    """기존 데이터의 article_no와 cluster_id를 cumulative하게 재할당"""

    # Google Sheets 인증
    creds_path = os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH")
    sheet_id = os.getenv("GOOGLE_SHEET_ID")

    if not creds_path or not sheet_id:
        print("❌ GOOGLE_SHEETS_CREDENTIALS_PATH 또는 GOOGLE_SHEET_ID가 설정되지 않았습니다.")
        return

    print("🔄 Migration 시작: article_no와 cluster_id를 cumulative하게 재할당")
    print(f"   - Credentials: {creds_path}")
    print(f"   - Sheet ID: {sheet_id}")

    # Google Sheets 연결
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
    gc = gspread.authorize(creds)
    spreadsheet = gc.open_by_key(sheet_id)

    # total_result 데이터 읽기
    print("\n📖 total_result 데이터 읽기 중...")
    try:
        worksheet = spreadsheet.worksheet("total_result")
    except Exception as e:
        print(f"❌ total_result 워크시트를 찾을 수 없습니다: {e}")
        return

    data = worksheet.get_all_records()
    if not data:
        print("❌ total_result에 데이터가 없습니다.")
        return

    df = pd.DataFrame(data)
    original_count = len(df)
    print(f"   ✅ {original_count}개 행 읽기 완료")

    # pub_datetime 파싱 및 정렬
    print("\n🔧 pub_datetime 기준으로 정렬 중 (과거 → 최신)...")
    df["pub_datetime_parsed"] = pd.to_datetime(df["pub_datetime"], errors="coerce")
    df = df.sort_values("pub_datetime_parsed", ascending=True).reset_index(drop=True)
    print(f"   ✅ 정렬 완료")

    # article_no 재할당 (1부터 시작, 과거일수록 낮은 숫자)
    print("\n🔢 article_no 재할당 중...")
    df["article_no"] = range(1, len(df) + 1)
    print(f"   ✅ article_no: 1 ~ {len(df)}")

    # cluster_id 재할당 (query별로 정렬 후 재할당)
    print("\n🔢 cluster_id 재할당 중...")
    cluster_id_map = {}  # 기존 cluster_id → 새 cluster_id 매핑
    global_cluster_counter = 1  # 전체 클러스터 카운터

    for query in df["query"].unique():
        query_df = df[df["query"] == query]
        query_clusters = query_df[query_df["cluster_id"].notna() & (query_df["cluster_id"] != "")]["cluster_id"].unique()

        # query별 클러스터들을 해당 클러스터의 가장 오래된 기사 기준으로 정렬
        cluster_oldest = {}
        for cid in query_clusters:
            cluster_articles = query_df[query_df["cluster_id"] == cid]
            oldest_date = cluster_articles["pub_datetime_parsed"].min()
            cluster_oldest[cid] = oldest_date

        # 오래된 클러스터부터 정렬
        sorted_clusters = sorted(cluster_oldest.items(), key=lambda x: x[1])

        # 새 cluster_id 할당
        for old_cid, _ in sorted_clusters:
            new_cid = f"{query}_c{global_cluster_counter:05d}"
            cluster_id_map[old_cid] = new_cid
            global_cluster_counter += 1

    # cluster_id 업데이트
    df["cluster_id"] = df["cluster_id"].map(lambda x: cluster_id_map.get(x, x) if x and x != "" else "")

    total_clusters = len(cluster_id_map)
    print(f"   ✅ cluster_id: {total_clusters}개 클러스터 재할당 완료")

    # pub_datetime_parsed 컬럼 제거 (임시 컬럼)
    df = df.drop(columns=["pub_datetime_parsed"])

    # Google Sheets 업데이트
    print("\n📤 Google Sheets 업데이트 중...")
    print(f"   - 업데이트할 행: {len(df)}개")

    # 헤더 + 데이터 준비
    headers = df.columns.tolist()
    values = [headers] + df.fillna("").astype(str).values.tolist()

    # 기존 데이터 전체 삭제 후 새 데이터 쓰기
    worksheet.clear()
    worksheet.update(values, value_input_option="USER_ENTERED")

    print(f"   ✅ Google Sheets 업데이트 완료")

    # 결과 요약
    print("\n" + "="*60)
    print("✅ Migration 완료!")
    print("="*60)
    print(f"📊 처리 결과:")
    print(f"   - 전체 기사: {len(df)}개")
    print(f"   - article_no: 1 ~ {len(df)} (pub_datetime 기준 정렬)")
    print(f"   - cluster_id: {total_clusters}개 클러스터 재할당")
    print(f"   - 날짜 범위: {df['pub_datetime'].min()} ~ {df['pub_datetime'].max()}")
    print("="*60)


if __name__ == "__main__":
    try:
        migrate_cumulative_ids()
    except Exception as e:
        print(f"\n❌ Migration 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
