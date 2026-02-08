#!/usr/bin/env python3
"""
main.py - News Monitoring System
뉴스 모니터링 시스템 메인 실행 파일
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# 모듈 임포트
from src.modules.collection.collect import OUR_BRANDS, COMPETITORS, collect_all_news
from src.modules.processing.process import (
    normalize_df, dedupe_df, detect_similar_articles,
    enrich_with_media_info, save_excel
)
from src.modules.analysis.classify import classify_all
from src.modules.export.report import generate_console_report, create_word_report
from src.modules.collection.scrape import collect_with_scraping, merge_api_and_scrape
from src.modules.enhancement.fulltext import batch_fetch_full_text
from src.modules.enhancement.looker_prep import add_time_series_columns
from src.modules.export.sheets import (
    connect_sheets, sync_raw_and_processed, sync_all_sheets,
    load_existing_links_from_sheets, filter_new_articles_from_sheets
)


def load_env():
    """환경 변수 로드"""
    load_dotenv()
    
    naver_id = os.getenv("NAVER_CLIENT_ID")
    naver_secret = os.getenv("NAVER_CLIENT_SECRET")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not naver_id or not naver_secret:
        raise ValueError("❌ .env 파일에 NAVER_CLIENT_ID 또는 NAVER_CLIENT_SECRET이 없습니다")
    if not openai_key:
        raise ValueError("❌ .env 파일에 OPENAI_API_KEY가 없습니다")
    
    return {
        "naver_id": naver_id,
        "naver_secret": naver_secret,
        "openai_key": openai_key
    }


def main():
    parser = argparse.ArgumentParser(
        description="뉴스 모니터링 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python main.py
  python main.py --display 200
  python main.py --scrape --start_date 2026-01-01 --end_date 2026-02-07
  python main.py --fulltext --fulltext_risk_levels 상,중
  python main.py --sheets --sheets_id YOUR_SHEET_ID
  python main.py --looker_prep
  python main.py --raw_only --sheets
        """
    )
    # 기존 옵션
    parser.add_argument("--display", type=int, default=100,
                       help="네이버 API에서 가져올 기사 수 (기본: 100)")
    parser.add_argument("--start", type=int, default=1,
                       help="네이버 API 시작 인덱스 (기본: 1)")
    parser.add_argument("--sort", type=str, default="date", choices=["date", "sim"],
                       help="정렬 방식: date(최신순) 또는 sim(관련도순) (기본: date)")
    parser.add_argument("--outdir", type=str, default="data",
                       help="출력 디렉토리 (기본: data)")
    parser.add_argument("--max_competitor_classify", type=int, default=20,
                       help="경쟁사별 분류할 최대 기사 수 (기본: 20)")
    parser.add_argument("--chunk_size", type=int, default=100,
                       help="AI 처리 시 청크 크기 (기본: 100)")
    parser.add_argument("--dry_run", action="store_true",
                       help="AI 분류 없이 테스트 실행")

    # 스크래핑 옵션
    parser.add_argument("--scrape", action="store_true",
                       help="Naver 뉴스 스크래핑 (날짜 범위 기반)")
    parser.add_argument("--start_date", type=str, default="2026-01-01",
                       help="스크래핑 시작 날짜 (YYYY-MM-DD, 기본: 2026-01-01)")
    parser.add_argument("--end_date", type=str, default="2026-02-07",
                       help="스크래핑 종료 날짜 (YYYY-MM-DD, 기본: 2026-02-07)")
    parser.add_argument("--max_scrape_pages", type=int, default=10,
                       help="스크래핑 최대 페이지 수 (기본: 10)")

    # 전문 스크래핑 옵션
    parser.add_argument("--fulltext", action="store_true",
                       help="기사 전문 스크래핑")
    parser.add_argument("--fulltext_risk_levels", type=str, default="상,중",
                       help="전문을 스크래핑할 위험도 (기본: 상,중)")
    parser.add_argument("--fulltext_max_articles", type=int, default=None,
                       help="최대 전문 스크래핑 기사 수 (기본: 무제한)")

    # Looker 준비 옵션
    parser.add_argument("--looker_prep", action="store_true",
                       help="Looker Studio용 시계열 컬럼 추가")

    # Google Sheets 옵션
    parser.add_argument("--sheets", action="store_true",
                       help="Google Sheets로 데이터 업로드")
    parser.add_argument("--sheets_id", type=str, default=None,
                       help="Google Sheets ID (기본: .env의 GOOGLE_SHEET_ID)")

    # API 페이지네이션 옵션
    parser.add_argument("--max_api_pages", type=int, default=9,
                       help="API 페이지네이션 최대 페이지 수 (기본: 9, 쿼터 90%% 안전 마진)")

    # Raw only 옵션
    parser.add_argument("--raw_only", action="store_true",
                       help="AI 분류 없이 API 수집 + Google Sheets 업로드만 실행")

    args = parser.parse_args()

    print("="*80)
    print("🚀 뉴스 모니터링 시스템 시작")
    print("="*80)
    print(f"\n설정:")
    print(f"  - 우리 브랜드: {', '.join(OUR_BRANDS)}")
    print(f"  - 경쟁사: {', '.join(COMPETITORS)}")
    if args.scrape:
        print(f"  - 수집 모드: API + 스크래핑 ({args.start_date} ~ {args.end_date})")
    else:
        print(f"  - 수집 모드: API만")
        print(f"  - 기사 수: {args.display}개/브랜드 (최대 {args.max_api_pages} 페이지)")
    print(f"  - 출력 디렉토리: {args.outdir}/")
    print(f"  - AI 청크 크기: {args.chunk_size}")
    if args.fulltext:
        print(f"  - 전문 스크래핑: {args.fulltext_risk_levels} (위험도)")
    if args.looker_prep:
        print(f"  - Looker 준비: 활성화")
    if args.sheets:
        print(f"  - Google Sheets 업로드: 활성화")
    if args.dry_run:
        print(f"  - 모드: DRY RUN (AI 분류 생략)")
    if args.raw_only:
        print(f"  - 모드: RAW ONLY (API 수집 + Sheets 업로드만)")
    print()
    
    # Step 0: 환경 설정
    env = load_env()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: 수집
    print("\n" + "="*80)
    print("STEP 1: 뉴스 수집")
    print("="*80)

    # Load existing articles from Google Sheets (if --sheets flag enabled)
    existing_links = set()
    if args.sheets:
        print("\n📊 Google Sheets 연결 중...")
        spreadsheet = connect_sheets(
            os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH", "service-account.json"),
            args.sheets_id or os.getenv("GOOGLE_SHEET_ID")
        )
        if spreadsheet:
            existing_links = load_existing_links_from_sheets(spreadsheet)

    if args.scrape:
        # 스크래핑 방식
        df_scrape = collect_with_scraping(
            OUR_BRANDS, COMPETITORS,
            args.start_date, args.end_date,
            args.max_scrape_pages
        )
        # API 방식도 동시에 수집 후 병합
        df_api = collect_all_news(
            OUR_BRANDS, COMPETITORS,
            args.display, args.max_api_pages, args.sort,
            env["naver_id"], env["naver_secret"]
        )
        df_raw = merge_api_and_scrape(df_api, df_scrape)
    else:
        # API 방식만 (기존 동작)
        df_raw = collect_all_news(
            OUR_BRANDS, COMPETITORS,
            args.display, args.max_api_pages, args.sort,
            env["naver_id"], env["naver_secret"]
        )

    # Filter new articles (skip duplicates from Google Sheets)
    if args.sheets and len(existing_links) > 0:
        df_raw = filter_new_articles_from_sheets(df_raw, existing_links)

    if len(df_raw) == 0:
        print("❌ 수집된 새 기사가 없습니다. 종료합니다.")
        return

    save_excel(df_raw, outdir / "raw.xlsx")

    # --raw_only 모드인 경우 처리/분류/리포트 스킵
    if args.raw_only:
        df_result = df_raw
    else:
        # Step 2: 처리
        print("\n" + "="*80)
        print("STEP 2: 데이터 처리")
        print("="*80)
        df_normalized = normalize_df(df_raw)
        df_processed = dedupe_df(df_normalized)
        df_processed = detect_similar_articles(df_processed, similarity_threshold=0.8)

        # 언론사 정보 추가
        if args.sheets and 'spreadsheet' in locals() and spreadsheet:
            df_processed = enrich_with_media_info(
                df_processed,
                spreadsheet=spreadsheet,
                openai_key=env["openai_key"]
            )
        else:
            # 호환성을 위해 빈 컬럼 추가
            df_processed["media_domain"] = ""
            df_processed["media_name"] = ""
            df_processed["media_group"] = ""
            df_processed["media_type"] = ""

        save_excel(df_processed, outdir / "processed.xlsx")

        # Step 3: 분류
        print("\n" + "="*80)
        print("STEP 3: AI 분류")
        print("="*80)
        df_classified = classify_all(
            df_processed,
            env["openai_key"],
            args.max_competitor_classify,
            args.chunk_size,
            args.dry_run
        )

        # Step 3.5: 전문 스크래핑 (선택적)
        if args.fulltext:
            print("\n" + "="*80)
            print("STEP 3.5: 기사 전문 스크래핑")
            print("="*80)
            risk_levels = [r.strip() for r in args.fulltext_risk_levels.split(",")]
            df_classified = batch_fetch_full_text(
                df_classified,
                risk_levels=risk_levels,
                max_articles=args.fulltext_max_articles
            )

        # Step 3.7: Looker 준비 (선택적)
        if args.looker_prep:
            print("\n" + "="*80)
            print("STEP 3.7: Looker Studio 준비")
            print("="*80)
            df_classified = add_time_series_columns(df_classified)

        df_result = df_classified

        # 결과 저장 (여러 시트)
        result_path = outdir / "result.xlsx"
        with pd.ExcelWriter(result_path, engine='openpyxl') as writer:
            df_result.to_excel(writer, sheet_name='전체데이터', index=False)

            # 우리 브랜드 부정 기사
            our_negative = df_result[(df_result["group"] == "OUR") & (df_result["sentiment"] == "부정")]
            our_negative.to_excel(writer, sheet_name='우리_부정', index=False)

            # 우리 브랜드 긍정 기사
            our_positive = df_result[(df_result["group"] == "OUR") & (df_result["sentiment"] == "긍정")]
            our_positive.to_excel(writer, sheet_name='우리_긍정', index=False)

            # 경쟁사
            competitor = df_result[df_result["group"] == "COMPETITOR"]
            competitor.to_excel(writer, sheet_name='경쟁사', index=False)

        print(f"💾 저장: {result_path}")

        # Step 4: 리포트 생성
        print("\n" + "="*80)
        print("STEP 4: 리포트 생성")
        print("="*80)

        # 콘솔 리포트
        generate_console_report(df_result)

        # Word 리포트
        word_path = outdir / "report.docx"
        create_word_report(df_result, word_path)

    # Step 5: Google Sheets 업로드 (선택적)
    if args.sheets:
        print("\n" + "="*80)
        print("STEP 5: Google Sheets 업로드")
        print("="*80)

        # Reuse spreadsheet connection from STEP 1 (or reconnect if needed)
        if 'spreadsheet' not in locals() or not spreadsheet:
            spreadsheet = connect_sheets(
                os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH", "service-account.json"),
                args.sheets_id or os.getenv("GOOGLE_SHEET_ID")
            )

        if spreadsheet:
            sync_raw_and_processed(df_raw, df_result, spreadsheet)
        else:
            print("⚠️  Google Sheets 연결 실패. Excel/Word 리포트는 생성되었습니다.")

    # 완료
    print("\n" + "="*80)
    print("✅ 모든 작업 완료!")
    print("="*80)
    print(f"\n생성된 파일:")
    print(f"  📊 {outdir}/raw.xlsx - 원본 데이터")
    if not args.raw_only:
        print(f"  📊 {outdir}/processed.xlsx - 정제된 데이터")
        print(f"  📊 {outdir}/result.xlsx - AI 분류 결과")
        print(f"  📄 {outdir}/report.docx - Word 리포트")
    if args.sheets:
        print(f"  ☁️  Google Sheets - 동기화 완료")
    print()


if __name__ == "__main__":
    main()
