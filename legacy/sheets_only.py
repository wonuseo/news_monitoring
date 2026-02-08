#!/usr/bin/env python3
"""
sheets_only.py - Google Sheets 업로드만 실행
기존 Excel 파일(raw.xlsx, result.xlsx)에서 데이터를 읽어 Google Sheets로 동기화
"""

import os
import pandas as pd
import argparse
from pathlib import Path
from dotenv import load_dotenv

from src.modules.export.sheets import connect_sheets, sync_raw_and_processed
from src.modules.processing.media_classify import update_media_directory


def main():
    parser = argparse.ArgumentParser(
        description="Google Sheets로 데이터 업로드만 실행",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python sheets_only.py
  python sheets_only.py --outdir data
  python sheets_only.py --sheets_id YOUR_SHEET_ID
        """
    )
    parser.add_argument("--outdir", type=str, default="data",
                       help="입력 디렉토리 (raw.xlsx, result.xlsx가 있는 폴더)")
    parser.add_argument("--sheets_id", type=str, default=None,
                       help="Google Sheets ID (기본: .env의 GOOGLE_SHEET_ID)")

    args = parser.parse_args()

    print("="*80)
    print("📤 Google Sheets 업로드만 실행")
    print("="*80)

    # 환경 변수 로드
    load_dotenv()

    # 입력 파일 확인
    outdir = Path(args.outdir)
    raw_path = outdir / "raw.xlsx"
    processed_path = outdir / "processed.xlsx"

    print(f"\n입력 디렉토리: {outdir}/")
    if not raw_path.exists():
        print(f"❌ {raw_path} 파일이 없습니다")
        return
    if not processed_path.exists():
        print(f"❌ {processed_path} 파일이 없습니다")
        return

    print(f"✅ raw.xlsx 발견 ({raw_path.stat().st_size / 1024:.1f} KB)")
    print(f"✅ processed.xlsx 발견 ({processed_path.stat().st_size / 1024:.1f} KB)")

    # Excel 파일 읽기
    print("\n📖 Excel 파일 읽는 중...")
    df_raw = pd.read_excel(raw_path)
    df_processed = pd.read_excel(processed_path)

    print(f"  - raw.xlsx: {len(df_raw)}개 행")
    print(f"  - processed.xlsx: {len(df_processed)}개 행")

    # Google Sheets 연결
    print("\n☁️  Google Sheets 연결 중...")
    spreadsheet = connect_sheets(
        os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH", "service-account.json"),
        args.sheets_id or os.getenv("GOOGLE_SHEET_ID")
    )

    if not spreadsheet:
        print("❌ Google Sheets 연결 실패")
        return

    # 업로드
    print("\n📊 Google Sheets 동기화 중...")
    sync_raw_and_processed(df_raw, df_processed, spreadsheet)

    # 미디어 디렉토리 업로드
    print("\n🏢 미디어 디렉토리 업로드 중...")
    # processed.xlsx에서 미디어 정보 추출 (중복 제거)
    media_info = {}
    for _, row in df_processed.iterrows():
        domain = row.get("media_domain", "")
        if domain and domain not in media_info:
            media_info[domain] = {
                "media_name": row.get("media_name", ""),
                "media_group": row.get("media_group", ""),
                "media_type": row.get("media_type", "")
            }

    if media_info:
        print(f"  📂 {len(media_info)}개 미디어 도메인 정보 업로드")
        update_media_directory(spreadsheet, media_info)
        print(f"  ✅ 미디어 디렉토리 업로드 완료")
    else:
        print(f"  ℹ️  미디어 정보가 없습니다")

    print("\n" + "="*80)
    print("✅ 업로드 완료!")
    print("="*80)


if __name__ == "__main__":
    main()
