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

# Pandas FutureWarning 방지
pd.set_option('future.no_silent_downcasting', True)

# 모듈 임포트
from src.modules.collection.collect import OUR_BRANDS, COMPETITORS, collect_all_news
from src.modules.processing.process import (
    normalize_df, dedupe_df, detect_similar_articles,
    enrich_with_media_info, save_csv, summarize_press_release_groups
)
from src.modules.analysis.classify import classify_all
from src.modules.analysis.hybrid import classify_hybrid, get_classification_stats, print_classification_stats
from src.modules.export.report import generate_console_report, create_word_report
from src.modules.collection.scrape import collect_with_scraping, merge_api_and_scrape
from src.modules.processing.fulltext import batch_fetch_full_text
from src.modules.processing.looker_prep import add_time_series_columns
from src.modules.export.sheets import (
    connect_sheets, sync_raw_and_processed, load_existing_links_from_sheets, filter_new_articles_from_sheets
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
  python main.py                                    # 전체 파이프라인 실행 (Sheets 자동 동기화)
  python main.py --display 200                      # API 결과 개수 지정
  python main.py --scrape --start_date 2026-01-01   # 날짜 범위 스크래핑
  python main.py --fulltext --fulltext_risk_levels 상,중  # 전문 스크래핑
  python main.py --sheets_id YOUR_SHEET_ID          # Google Sheets ID 지정
  python main.py --raw_only                         # 수집만 (Sheets 자동 동기화)

주의:
  - Google Sheets는 주 저장소입니다 (credentials 설정 권장)
  - CSV 파일은 troubleshooting 용도로 함께 저장됩니다
  - .env 파일에 GOOGLE_SHEETS_CREDENTIALS_PATH와 GOOGLE_SHEET_ID 설정 필요
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
    parser.add_argument("--max_workers", type=int, default=10,
                        help="병렬 처리 워커 수 (기본: 10, 권장: 5-15)")
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

    # Google Sheets 옵션 (항상 활성화됨)
    parser.add_argument("--sheets_id", type=str, default=None,
                        help="Google Sheets ID (.env의 GOOGLE_SHEET_ID 대신 사용)")

    # API 페이지네이션 옵션
    parser.add_argument("--max_api_pages", type=int, default=9,
                        help="API 페이지네이션 최대 페이지 수 (기본: 9, 쿼터 90%% 안전 마진)")

    # Raw only 옵션
    parser.add_argument("--raw_only", action="store_true",
                        help="AI 분류 없이 API 수집 + Google Sheets 업로드만 실행")

    # Preprocess only 옵션
    parser.add_argument("--preprocess_only", action="store_true",
                        help="수집 + 전처리까지만 실행 (AI 분류, 리포트 생략, Sheets 업로드는 실행)")

    # Legacy classify 옵션
    parser.add_argument("--legacy_classify", action="store_true",
                        help="레거시 분류 시스템 사용 (기본: 하이브리드 시스템)")

    args = parser.parse_args()

    print("=" * 80)
    print("🚀 뉴스 모니터링 시스템 시작")
    print("=" * 80)
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
    print(f"  - 병렬 처리 워커: {args.max_workers}개")
    if args.fulltext:
        print(f"  - 전문 스크래핑: {args.fulltext_risk_levels} (위험도)")
    if args.looker_prep:
        print(f"  - Looker 준비: 활성화")
    if args.dry_run:
        print(f"  - 모드: DRY RUN (AI 분류 생략)")
    if args.raw_only:
        print(f"  - 모드: RAW ONLY (API 수집 + Sheets 업로드만)")
    if args.preprocess_only:
        print(f"  - 모드: PREPROCESS ONLY (수집 + 전처리 + Sheets 업로드)")
    print()

    # Step 0: 환경 설정
    env = load_env()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    media_csv_path = outdir / "media_directory.csv"

    # Step 1: 수집
    print("\n" + "=" * 80)
    print("STEP 1: 뉴스 수집")
    print("=" * 80)

    # Google Sheets 연결 (주 저장소)
    existing_links = set()
    spreadsheet = None

    # Google Sheets 자동 연결 (credentials 필수 권장)
    creds_path = os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH", "service-account.json")
    sheet_id = args.sheets_id or os.getenv("GOOGLE_SHEET_ID")

    if os.path.exists(creds_path) and sheet_id:
        try:
            print("\n📊 Google Sheets 연결 중...")
            spreadsheet = connect_sheets(creds_path, sheet_id)
            if spreadsheet:
                existing_links = load_existing_links_from_sheets(spreadsheet)
                print("✅ Google Sheets 연결 성공 (주 저장소)")
                print("   CSV 파일은 troubleshooting용으로 함께 저장됩니다.")
        except Exception as e:
            print(f"⚠️  Google Sheets 연결 실패: {e}")
            print("   ⚠️  CSV 파일만 사용합니다 (troubleshooting 모드)")
    else:
        print("\n" + "="*80)
        print("⚠️  경고: Google Sheets 설정이 없습니다!")
        print("="*80)
        if not os.path.exists(creds_path):
            print(f"  credential 파일 없음: {creds_path}")
        if not sheet_id:
            print(f"  GOOGLE_SHEET_ID 환경변수 없음")
        print("\n  Google Sheets는 주 저장소입니다. 설정을 권장합니다.")
        print("  현재는 CSV 파일만 사용합니다 (troubleshooting 모드)")
        print("="*80 + "\n")

    raw_csv_path = outdir / "raw.csv"

    if args.scrape:
        # 스크래핑 방식
        df_scrape = collect_with_scraping(
            OUR_BRANDS, COMPETITORS,
            args.start_date, args.end_date,
            args.max_scrape_pages
        )
        # API 방식도 동시에 수집 후 병합 (raw.csv 기반 중복 체크)
        df_api = collect_all_news(
            OUR_BRANDS, COMPETITORS,
            args.display, args.max_api_pages, args.sort,
            env["naver_id"], env["naver_secret"],
            raw_csv_path=str(raw_csv_path)
        )
        df_raw_new = merge_api_and_scrape(df_api, df_scrape)
    else:
        # API 방식만 (raw.csv 기반 중복 체크)
        df_raw_new = collect_all_news(
            OUR_BRANDS, COMPETITORS,
            args.display, args.max_api_pages, args.sort,
            env["naver_id"], env["naver_secret"],
            raw_csv_path=str(raw_csv_path)
        )

    # API 수집 결과 확인
    if len(df_raw_new) == 0:
        print("\nℹ️  API에서 수집된 새로운 기사가 없습니다.")
    else:
        print(f"\n✅ API에서 {len(df_raw_new)}개 기사 수집 완료")

    # Filter new articles (skip duplicates from Google Sheets)
    if len(existing_links) > 0 and len(df_raw_new) > 0:
        df_raw_new = filter_new_articles_from_sheets(df_raw_new, existing_links)

    # raw.csv 업데이트 (append)
    if raw_csv_path.exists():
        df_raw_existing = pd.read_csv(raw_csv_path, encoding='utf-8-sig')
        df_raw = pd.concat([df_raw_existing, df_raw_new], ignore_index=True)
        df_raw = df_raw.drop_duplicates(subset=['link'], keep='last')
        print(f"📂 기존 raw.csv 업데이트: {len(df_raw_existing)} + {len(df_raw_new)} = {len(df_raw)}개 기사")
    else:
        df_raw = df_raw_new

    save_csv(df_raw, raw_csv_path)

    # STEP 1.5: 미처리/미분석 행 필터링 (result.csv 기준)
    result_csv_path = outdir / "result.csv"
    df_to_process = df_raw_new  # 기본: 신규 수집 기사만 처리

    if result_csv_path.exists():
        try:
            df_result_existing = pd.read_csv(result_csv_path, encoding='utf-8-sig')

            # 1. Link 기준 미처리 행 (새로 수집된 기사)
            if 'link' in df_result_existing.columns:
                processed_links = set(df_result_existing['link'].dropna().tolist())
                unprocessed_rows = df_raw[~df_raw['link'].isin(processed_links)]
            else:
                unprocessed_rows = df_raw

            # 2. 분석 필드 비어있는 행 (기존 기사 중 분석 안 된 것)
            analysis_cols_to_check = ['sentiment_final', 'danger_final', 'issue_category_final']
            if all(col in df_result_existing.columns for col in analysis_cols_to_check):
                # 분석 필드가 모두 비어있는 행 찾기
                missing_analysis = df_result_existing[
                    df_result_existing['sentiment_final'].isna() |
                    (df_result_existing['sentiment_final'] == "")
                ].copy()

                if len(missing_analysis) > 0:
                    # raw.csv에서 해당 링크들 찾기
                    missing_links = set(missing_analysis['link'].dropna().tolist())
                    reanalyze_rows = df_raw[df_raw['link'].isin(missing_links)]

                    # 미처리 행 + 재분석 행 합치기
                    df_to_process = pd.concat([unprocessed_rows, reanalyze_rows], ignore_index=True)
                    df_to_process = df_to_process.drop_duplicates(subset=['link'], keep='first')

                    print(f"\n📊 처리 상태 확인:")
                    print(f"  - 전체 raw.csv: {len(df_raw)}개")
                    print(f"  - 신규 미처리: {len(unprocessed_rows)}개")
                    print(f"  - 분석 누락: {len(reanalyze_rows)}개")
                    print(f"  - 총 처리 대상: {len(df_to_process)}개")
                else:
                    df_to_process = unprocessed_rows
                    print(f"\n📊 처리 상태 확인:")
                    print(f"  - 전체 raw.csv: {len(df_raw)}개")
                    print(f"  - 이미 처리됨: {len(processed_links)}개")
                    print(f"  - 미처리 행: {len(unprocessed_rows)}개")
            else:
                df_to_process = unprocessed_rows
                print(f"\n📊 처리 상태 확인:")
                print(f"  - 전체 raw.csv: {len(df_raw)}개")
                print(f"  - 미처리 행: {len(unprocessed_rows)}개")

        except Exception as e:
            print(f"⚠️  result.csv 로드 실패: {e}, 전체 처리 진행")
            df_to_process = df_raw
    else:
        print(f"\n📊 result.csv가 없습니다. 전체 raw.csv {len(df_raw)}개 기사 처리")
        df_to_process = df_raw  # 전체 raw.csv 처리

    if len(df_to_process) == 0:
        print("ℹ️  처리할 신규 기사가 없습니다.")

        # 기존 데이터가 있으면 Google Sheets 동기화 시도
        if spreadsheet and result_csv_path.exists():
            print("\n" + "=" * 80)
            print("STEP 5: Google Sheets 업로드 (기존 데이터)")
            print("=" * 80)
            try:
                df_result_existing = pd.read_csv(result_csv_path, encoding='utf-8-sig')
                sync_raw_and_processed(df_raw, df_result_existing, spreadsheet)
                print("✅ Google Sheets 동기화 완료")
            except Exception as e:
                print(f"⚠️  Google Sheets 업로드 실패: {e}")

        print("\n" + "=" * 80)
        print("✅ 작업 완료 (신규 처리 없음)")
        print("=" * 80)
        return

    # --raw_only 모드인 경우 처리/분류/리포트 스킵
    if args.raw_only:
        df_result = df_raw
    # --preprocess_only 모드인 경우 분류 스킵, 처리+리포트는 실행
    elif args.preprocess_only:
        # Step 2: 처리 (미처리 행만)
        print("\n" + "=" * 80)
        print("STEP 2: 데이터 처리 (미처리 행만)")
        print("=" * 80)
        df_normalized = normalize_df(df_to_process)
        df_processed = dedupe_df(df_normalized)
        df_processed = detect_similar_articles(df_processed, similarity_threshold=0.8)

        # 보도자료 그룹 요약 생성 (OpenAI)
        df_processed = summarize_press_release_groups(df_processed, env["openai_key"])

        # 언론사 정보 추가 (spreadsheet 있으면 자동 사용)
        media_csv_path = outdir / "media_directory.csv"
        df_processed = enrich_with_media_info(
            df_processed,
            spreadsheet=spreadsheet,  # None이면 CSV-only 모드
            openai_key=env["openai_key"],
            csv_path=media_csv_path
        )

        # 나머지 NaN → 공란 변환
        df_processed = df_processed.fillna("")

        # Looker Studio 준비 (항상 실행)
        print("\n🕒 Looker Studio 시계열 컬럼 추가 중...")
        df_processed = add_time_series_columns(df_processed)

        # 기존 result.csv와 병합
        if result_csv_path.exists():
            df_result_existing = pd.read_csv(result_csv_path, encoding='utf-8-sig')
            df_result = pd.concat([df_result_existing, df_processed], ignore_index=True)
            df_result = df_result.drop_duplicates(subset=['link'], keep='last')
            print(f"📂 기존 result.csv 업데이트: {len(df_result_existing)} + {len(df_processed)} = {len(df_result)}개 기사")
        else:
            df_result = df_processed

        # 결과 저장
        save_csv(df_result, result_csv_path)
    else:
        # Step 2: 처리 (미처리 행만)
        print("\n" + "=" * 80)
        print("STEP 2: 데이터 처리 (미처리 행만)")
        print("=" * 80)
        df_normalized = normalize_df(df_to_process)
        df_processed = dedupe_df(df_normalized)
        df_processed = detect_similar_articles(df_processed, similarity_threshold=0.8)

        # 보도자료 그룹 요약 생성 (OpenAI)
        df_processed = summarize_press_release_groups(df_processed, env["openai_key"])

        # 언론사 정보 추가 (spreadsheet 있으면 자동 사용)
        media_csv_path = outdir / "media_directory.csv"
        df_processed = enrich_with_media_info(
            df_processed,
            spreadsheet=spreadsheet,  # None이면 CSV-only 모드
            openai_key=env["openai_key"],
            csv_path=media_csv_path
        )

        # Step 3: 분류 (미처리 행만)
        print("\n" + "=" * 80)
        print("STEP 3: AI 분류")
        print("=" * 80)

        if args.legacy_classify:
            print("📚 레거시 분류 시스템 사용 중...")
            df_classified = classify_all(
                df_processed,
                env["openai_key"],
                args.max_competitor_classify,
                args.chunk_size,
                args.dry_run
            )
        else:
            print("🔬 하이브리드 분류 시스템 사용 중...")
            df_classified = classify_hybrid(
                df_processed,
                env["openai_key"],
                chunk_size=args.chunk_size,
                dry_run=args.dry_run,
                max_competitor_classify=args.max_competitor_classify,
                max_workers=args.max_workers,
                result_csv_path=str(result_csv_path)
            )

            # 통계 출력
            stats = get_classification_stats(df_classified)
            print_classification_stats(stats)

        # 보도자료 정보 및 언론사 정보 병합
        source_cols = ['link', 'source', 'group_id', 'press_release_group', 'media_domain', 'media_name', 'media_group',
                       'media_type']
        source_data = df_processed[source_cols].copy()

        # 기존 컬럼 제거 (중복 방지)
        cols_to_drop = [col for col in df_classified.columns if col in source_cols and col != 'link']
        df_classified = df_classified.drop(columns=cols_to_drop, errors='ignore')

        # merge
        df_classified = df_classified.merge(source_data, on='link', how='left')

        # 나머지 NaN → 공란 변환 (FutureWarning 방지)
        df_classified = df_classified.fillna("").infer_objects(copy=False)

        # Step 3.5: 전문 스크래핑 (선택적)
        if args.fulltext:
            print("\n" + "=" * 80)
            print("STEP 3.5: 기사 전문 스크래핑")
            print("=" * 80)
            risk_levels = [r.strip() for r in args.fulltext_risk_levels.split(",")]
            df_classified = batch_fetch_full_text(
                df_classified,
                risk_levels=risk_levels,
                max_articles=args.fulltext_max_articles
            )

        # Step 3.7: Looker Studio 준비 (항상 실행)
        print("\n🕒 Looker Studio 시계열 컬럼 추가 중...")
        df_classified = add_time_series_columns(df_classified)

        # 기존 result.csv와 병합
        if result_csv_path.exists():
            df_result_existing = pd.read_csv(result_csv_path, encoding='utf-8-sig')
            df_result = pd.concat([df_result_existing, df_classified], ignore_index=True)
            df_result = df_result.drop_duplicates(subset=['link'], keep='last')
            print(f"\n📂 기존 result.csv 업데이트: {len(df_result_existing)} + {len(df_classified)} = {len(df_result)}개 기사 (중복 제거 후)")
        else:
            df_result = df_classified

        # pub_datetime을 명시적으로 datetime으로 변환 (타입 불일치 방지)
        if 'pub_datetime' in df_result.columns:
            df_result['pub_datetime'] = pd.to_datetime(df_result['pub_datetime'], errors='coerce')

        # 결과 저장 (단일 CSV 파일)
        save_csv(df_result, result_csv_path)

        # Step 4: 리포트 생성
        print("\n" + "=" * 80)
        print("STEP 4: 리포트 생성")
        print("=" * 80)

        # 콘솔 리포트
        generate_console_report(df_result)

        # Word 리포트
        word_path = outdir / "report.docx"
        create_word_report(df_result, word_path)

    # Step 5: Google Sheets 동기화 (주 저장소)
    if spreadsheet:
        print("\n" + "=" * 80)
        print("STEP 5: Google Sheets 동기화 (주 저장소)")
        print("=" * 80)
        try:
            sync_raw_and_processed(df_raw, df_result, spreadsheet)
            print("✅ Google Sheets 동기화 완료")
        except Exception as e:
            print(f"❌ Google Sheets 동기화 실패: {e}")
            print("   ⚠️  CSV 파일만 저장되었습니다 (troubleshooting 모드)")
    else:
        print("\n" + "=" * 80)
        print("⚠️  Google Sheets 연결 없음")
        print("=" * 80)
        print("  주 저장소인 Google Sheets에 동기화되지 않았습니다.")
        print("  CSV 파일만 저장되었습니다 (troubleshooting 모드)")
        print("  .env 파일에 GOOGLE_SHEETS_CREDENTIALS_PATH 및 GOOGLE_SHEET_ID 설정을 권장합니다.")

    # 완료
    print("\n" + "=" * 80)
    print("✅ 모든 작업 완료!")
    print("=" * 80)
    print(f"\n생성된 파일:")
    if spreadsheet:
        print(f"  ☁️  Google Sheets - 동기화 완료 (주 저장소)")
    print(f"  📊 {outdir}/raw.csv - 원본 데이터 (troubleshooting)")
    if not args.raw_only:
        print(f"  📊 {outdir}/result.csv - AI 분류 결과 (troubleshooting)")
        print(f"  📄 {outdir}/report.docx - Word 리포트")
    if spreadsheet or not args.raw_only:
        print(f"  📂 {outdir}/media_directory.csv - 언론사 디렉토리")
    if not spreadsheet:
        print(f"\n  ⚠️  Google Sheets 미연결: CSV 파일만 저장됨")
    print()


if __name__ == "__main__":
    main()