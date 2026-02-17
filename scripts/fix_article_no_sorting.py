"""
Fix article_no sorting and duplicates - One-time migration script

전체 result.csv를 pubDate 순서로 정렬하고 article_no를 재할당합니다.
- 중복 article_no 제거
- pubDate 오름차순 정렬 (과거 기사 = 낮은 번호)
- article_no를 정수형으로 변환 (5.0 → 5)

실행: python scripts/fix_article_no_sorting.py
"""

import pandas as pd
from pathlib import Path
import csv
from datetime import datetime

def main():
    print("=" * 80)
    print("article_no 재정렬 스크립트")
    print("=" * 80)
    print()

    # Load result.csv
    result_path = Path("data/result.csv")
    if not result_path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {result_path}")
        return

    print(f"📂 로딩: {result_path}")
    df = pd.read_csv(result_path, encoding='utf-8-sig')
    print(f"   총 {len(df)}개 기사")
    print()

    # Backup original file
    backup_path = result_path.parent / f"result_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    print(f"💾 백업 생성: {backup_path}")
    df.to_csv(backup_path, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)
    print()

    # Check current state
    print("현재 상태:")
    print(f"  - article_no 타입: {df['article_no'].dtype}")
    print(f"  - article_no 범위: {df['article_no'].min():.0f} ~ {df['article_no'].max():.0f}")
    unique_nos = df['article_no'].nunique()
    total_nos = len(df[df['article_no'].notna()])
    duplicates = total_nos - unique_nos
    print(f"  - 고유 article_no: {unique_nos}개")
    if duplicates > 0:
        print(f"  - 중복 article_no: {duplicates}개 ⚠️")
    print()

    # Sort by pubDate
    print("🔧 pubDate 기준 정렬 중...")
    df['_pub_dt_sort'] = pd.to_datetime(df['pub_datetime'], errors='coerce')

    # Check how many rows have valid dates
    valid_dates = df['_pub_dt_sort'].notna().sum()
    invalid_dates = len(df) - valid_dates
    print(f"   유효한 날짜: {valid_dates}개")
    if invalid_dates > 0:
        print(f"   ⚠️  유효하지 않은 날짜: {invalid_dates}개 (맨 뒤로 배치)")

    # Sort: ascending by date, NaN at last
    df = df.sort_values('_pub_dt_sort', ascending=True, na_position='last').reset_index(drop=True)
    df = df.drop(columns=['_pub_dt_sort'])
    print("   ✅ 정렬 완료 (오름차순: 과거 → 최신)")
    print()

    # Reassign article_no from 1
    print("🔧 article_no 재할당 중 (1부터 시작)...")
    df['article_no'] = range(1, len(df) + 1)

    # Convert to integer type
    df['article_no'] = df['article_no'].astype('Int64')
    print(f"   ✅ article_no 재할당 완료: 1 ~ {len(df)} (정수형)")
    print()

    # Save
    print(f"💾 저장 중: {result_path}")
    df.to_csv(result_path, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)
    print("   ✅ 저장 완료")
    print()

    # Verification
    print("검증:")
    df_check = pd.read_csv(result_path, encoding='utf-8-sig')
    df_check['pub_dt'] = pd.to_datetime(df_check['pub_datetime'], errors='coerce')

    print(f"  - 총 기사: {len(df_check)}개")
    print(f"  - article_no 타입: {df_check['article_no'].dtype}")
    print(f"  - article_no 범위: {df_check['article_no'].min()} ~ {df_check['article_no'].max()}")
    print(f"  - 고유 article_no: {df_check['article_no'].nunique()}개")

    # Check sorting
    is_sorted = df_check['pub_dt'].is_monotonic_increasing
    print(f"  - pubDate 정렬: {'✅ 오름차순' if is_sorted else '❌ 정렬 안됨'}")

    # Show sample
    print()
    print("샘플 (처음 10개):")
    print("-" * 80)
    print(df_check[['article_no', 'pub_datetime', 'title']].head(10).to_string(index=False))
    print()
    print("=" * 80)
    print("✅ 완료!")
    print(f"   백업 파일: {backup_path}")
    print(f"   결과 파일: {result_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
