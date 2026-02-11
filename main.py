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
    normalize_df, dedupe_df,
    enrich_with_media_info, save_csv
)
from src.modules.processing.press_release_detector import detect_similar_articles, summarize_press_release_groups
from src.modules.processing.looker_prep import add_time_series_columns
from src.modules.analysis.classify_llm import classify_llm, get_classification_stats, print_classification_stats
from src.modules.analysis.preset_pr import preset_press_release_values
from src.modules.analysis.keyword_extractor import extract_all_categories
from src.modules.export.report import generate_console_report
from src.modules.export.sheets import (
    connect_sheets, sync_raw_and_processed, load_existing_links_from_sheets, filter_new_articles_from_sheets
)
from src.modules.monitoring.logger import RunLogger, sync_logs_to_sheets


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
    # RunLogger 초기화
    logger = RunLogger()
    logger.start_stage("total")

    parser = argparse.ArgumentParser(
        description="뉴스 모니터링 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python main.py                                    # 전체 파이프라인 실행 (자사+경쟁사 전체 분석)
  python main.py --display 200                      # API 결과 개수 지정
  python main.py --extract_keywords                 # 카테고리별 키워드 추출
  python main.py --max_competitor_classify 20       # 경쟁사 분석 개수 제한
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
    parser.add_argument("--max_competitor_classify", type=int, default=0,
                        help="경쟁사별 분류할 최대 기사 수 (기본: 0=무제한, 전체 분석)")
    parser.add_argument("--chunk_size", type=int, default=100,
                        help="AI 처리 시 청크 크기 (기본: 100)")
    parser.add_argument("--max_workers", type=int, default=10,
                        help="병렬 처리 워커 수 (기본: 10, 권장: 5-15)")
    parser.add_argument("--dry_run", action="store_true",
                        help="AI 분류 없이 테스트 실행")

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

    # Keyword extraction 옵션
    parser.add_argument("--extract_keywords", action="store_true",
                        help="카테고리별 특징 키워드 추출 (kiwipiepy 형태소 분석 + Log-odds ratio)")
    parser.add_argument("--keyword_top_k", type=int, default=20,
                        help="키워드 추출 시 상위 K개 선택 (기본: 20)")

    args = parser.parse_args()

    # CLI args 로깅
    logger.log("cli_args", vars(args))

    print("=" * 80)
    print("🚀 뉴스 모니터링 시스템 시작")
    print("=" * 80)
    print(f"\n설정:")
    print(f"  - 우리 브랜드: {', '.join(OUR_BRANDS)}")
    print(f"  - 경쟁사: {', '.join(COMPETITORS)}")
    print(f"  - 수집 모드: Naver API")
    print(f"  - 기사 수: {args.display}개/브랜드 (최대 {args.max_api_pages} 페이지)")
    print(f"  - 출력 디렉토리: {args.outdir}/")
    if args.max_competitor_classify == 0:
        print(f"  - 분류 모드: 자사+경쟁사 전체 분석")
    else:
        print(f"  - 분류 모드: 자사 전체 + 경쟁사 최대 {args.max_competitor_classify}개/브랜드")
    print(f"  - AI 청크 크기: {args.chunk_size}")
    print(f"  - 병렬 처리 워커: {args.max_workers}개")
    if args.extract_keywords:
        print(f"  - 키워드 추출: 활성화 (상위 {args.keyword_top_k}개)")
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
    logger.start_stage("collection")
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

    # API 방식 수집 (raw.csv 기반 중복 체크)
    df_raw_new = collect_all_news(
        OUR_BRANDS, COMPETITORS,
        args.display, args.max_api_pages, args.sort,
        env["naver_id"], env["naver_secret"],
        raw_csv_path=str(raw_csv_path),
        spreadsheet=spreadsheet
    )

    # API 수집 결과 확인
    if len(df_raw_new) == 0:
        print("\nℹ️  API에서 수집된 새로운 기사가 없습니다.")
    else:
        print(f"\n✅ API에서 {len(df_raw_new)}개 기사 수집 완료")

    # Filter new articles (skip duplicates from Google Sheets)
    existing_links_skipped = 0
    if len(existing_links) > 0 and len(df_raw_new) > 0:
        before_filter = len(df_raw_new)
        df_raw_new = filter_new_articles_from_sheets(df_raw_new, existing_links)
        existing_links_skipped = before_filter - len(df_raw_new)

    # raw.csv 업데이트 (append)
    if raw_csv_path.exists():
        df_raw_existing = pd.read_csv(raw_csv_path, encoding='utf-8-sig')
        df_raw = pd.concat([df_raw_existing, df_raw_new], ignore_index=True)
        df_raw = df_raw.drop_duplicates(subset=['link'], keep='last')
        print(f"📂 기존 raw.csv 업데이트: {len(df_raw_existing)} + {len(df_raw_new)} = {len(df_raw)}개 기사")
    else:
        df_raw = df_raw_new

    save_csv(df_raw, raw_csv_path)

    # Google Sheets 즉시 동기화 (수집 직후)
    if spreadsheet and len(df_raw_new) > 0:
        print("\n📊 Google Sheets 즉시 동기화 중 (raw_data)...")
        try:
            from src.modules.export.sheets import sync_to_sheets
            sync_result = sync_to_sheets(df_raw, spreadsheet, "raw_data")
            print(f"✅ raw_data 시트 동기화 완료: {sync_result['added']}개 추가, {sync_result['skipped']}개 건너뜀")
        except Exception as e:
            print(f"⚠️  raw_data 시트 동기화 실패: {e}")

    # 수집 단계 메트릭
    articles_per_query = df_raw_new.groupby('query').size().to_dict() if 'query' in df_raw_new.columns else {}
    logger.log_dict({
        "articles_collected_total": len(df_raw_new),
        "articles_collected_per_query": articles_per_query,
        "existing_links_skipped": existing_links_skipped
    })
    logger.end_stage("collection")

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
            logger.start_stage("sheets_sync")
            print("\n" + "=" * 80)
            print("STEP 5: Google Sheets 업로드 (기존 데이터)")
            print("=" * 80)
            try:
                df_result_existing = pd.read_csv(result_csv_path, encoding='utf-8-sig')
                sync_results = sync_raw_and_processed(df_raw, df_result_existing, spreadsheet)
                print("✅ Google Sheets 동기화 완료")

                # Sheets 메트릭 수집
                logger.log_dict({
                    "sheets_sync_enabled": True,
                    "sheets_rows_uploaded_raw": sync_results.get("raw_data", {}).get("added", 0),
                    "sheets_rows_uploaded_result": sync_results.get("result", {}).get("added", 0)
                })
            except Exception as e:
                print(f"⚠️  Google Sheets 업로드 실패: {e}")
                logger.log("sheets_sync_enabled", False)
            logger.end_stage("sheets_sync")

        # 로그 저장
        logger.finalize()
        logs_csv_path = outdir / "logs" / "run_history.csv"
        logger.save_csv(str(logs_csv_path))

        # Sheets 로그 동기화
        if spreadsheet:
            sync_logs_to_sheets(str(logs_csv_path), spreadsheet)

        logger.print_summary()

        print("\n" + "=" * 80)
        print("✅ 작업 완료 (신규 처리 없음)")
        print("=" * 80)
        return

    # --raw_only 모드인 경우 처리/분류/리포트 스킵
    if args.raw_only:
        df_result = df_raw
        logger.log_dict({
            "articles_processed": 0,
            "duplicates_removed": 0,
            "press_releases_detected": 0,
            "press_release_groups": 0
        })
    # --preprocess_only 모드인 경우 분류 스킵, 처리+리포트는 실행
    elif args.preprocess_only:
        # Step 2: 처리 (미처리 행만)
        logger.start_stage("processing")
        print("\n" + "=" * 80)
        print("STEP 2: 데이터 처리 (미처리 행만)")
        print("=" * 80)
        # Step 2-1: Normalize
        df_normalized = normalize_df(df_to_process)
        before_dedupe = len(df_normalized)

        # Step 2-2: Deduplicate
        df_processed = dedupe_df(df_normalized)
        duplicates_removed = before_dedupe - len(df_processed)

        # Step 2-3: Detect similar articles (Press Release)
        df_processed = detect_similar_articles(df_processed)
        press_releases = len(df_processed[df_processed['source'] == '보도자료']) if 'source' in df_processed.columns else 0

        # Step 2-4: Summarize press release groups (OpenAI)
        print("\n📝 보도자료 그룹 요약 생성 중...")
        df_processed = summarize_press_release_groups(df_processed, env["openai_key"])

        # 중간 저장 (보도자료 요약 완료 후)
        print("💾 중간 저장 중 (보도자료 요약 완료)...")
        if result_csv_path.exists():
            df_result_temp = pd.read_csv(result_csv_path, encoding='utf-8-sig')
            df_temp = pd.concat([df_result_temp, df_processed], ignore_index=True)
            df_temp = df_temp.drop_duplicates(subset=['link'], keep='last')
        else:
            df_temp = df_processed
        save_csv(df_temp, result_csv_path)
        if spreadsheet:
            try:
                sync_raw_and_processed(df_raw, df_temp, spreadsheet)
                print("✅ Google Sheets 중간 동기화 완료")
            except Exception as e:
                print(f"⚠️  Sheets 동기화 실패: {e}")

        # Step 2-5: Media classification (OpenAI)
        print("\n🏢 언론사 정보 추가 중...")
        media_csv_path = outdir / "media_directory.csv"
        df_processed = enrich_with_media_info(
            df_processed,
            spreadsheet=spreadsheet,  # None이면 CSV-only 모드
            openai_key=env["openai_key"],
            csv_path=media_csv_path
        )

        # 중간 저장 (언론사 정보 완료 후)
        print("💾 중간 저장 중 (언론사 정보 완료)...")
        if result_csv_path.exists():
            df_result_temp = pd.read_csv(result_csv_path, encoding='utf-8-sig')
            df_temp = pd.concat([df_result_temp, df_processed], ignore_index=True)
            df_temp = df_temp.drop_duplicates(subset=['link'], keep='last')
        else:
            df_temp = df_processed
        save_csv(df_temp, result_csv_path)
        if spreadsheet:
            try:
                sync_raw_and_processed(df_raw, df_temp, spreadsheet)
                print("✅ Google Sheets 중간 동기화 완료")
            except Exception as e:
                print(f"⚠️  Sheets 동기화 실패: {e}")

        # 나머지 NaN → 공란 변환
        df_processed = df_processed.fillna("")

        # Step 2-6: Looker Studio time-series columns
        print("\n🕒 Looker Studio 시계열 컬럼 추가 중...")
        df_processed = add_time_series_columns(df_processed)

        # 전처리 메트릭
        press_release_groups = df_processed['group_id'].nunique() if 'group_id' in df_processed.columns else 0
        logger.log_dict({
            "articles_processed": len(df_processed),
            "duplicates_removed": duplicates_removed,
            "press_releases_detected": press_releases,
            "press_release_groups": press_release_groups
        })
        logger.end_stage("processing")

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

        # Google Sheets 즉시 동기화 (전처리 완료 직후)
        if spreadsheet:
            print("\n📊 Google Sheets 즉시 동기화 중 (전처리 결과)...")
            try:
                sync_results = sync_raw_and_processed(df_raw, df_result, spreadsheet)
                print("✅ 전처리 결과 Sheets 동기화 완료")
            except Exception as e:
                print(f"⚠️  전처리 결과 Sheets 동기화 실패: {e}")
    else:
        # Step 2: 처리 (미처리 행만)
        logger.start_stage("processing")
        print("\n" + "=" * 80)
        print("STEP 2: 데이터 처리 (미처리 행만)")
        print("=" * 80)

        # Step 2-1: Normalize
        df_normalized = normalize_df(df_to_process)
        before_dedupe = len(df_normalized)

        # Step 2-2: Deduplicate
        df_processed = dedupe_df(df_normalized)
        duplicates_removed = before_dedupe - len(df_processed)

        # Step 2-3: Detect similar articles (Press Release)
        df_processed = detect_similar_articles(df_processed)
        press_releases = len(df_processed[df_processed['source'] == '보도자료']) if 'source' in df_processed.columns else 0

        # Step 2-4: Summarize press release groups (OpenAI)
        print("\n📝 보도자료 그룹 요약 생성 중...")
        df_processed = summarize_press_release_groups(df_processed, env["openai_key"])

        # 중간 저장 (보도자료 요약 완료 후)
        print("💾 중간 저장 중 (보도자료 요약 완료)...")
        if result_csv_path.exists():
            df_result_temp = pd.read_csv(result_csv_path, encoding='utf-8-sig')
            df_temp = pd.concat([df_result_temp, df_processed], ignore_index=True)
            df_temp = df_temp.drop_duplicates(subset=['link'], keep='last')
        else:
            df_temp = df_processed
        save_csv(df_temp, result_csv_path)
        if spreadsheet:
            try:
                sync_raw_and_processed(df_raw, df_temp, spreadsheet)
                print("✅ Google Sheets 중간 동기화 완료")
            except Exception as e:
                print(f"⚠️  Sheets 동기화 실패: {e}")

        # Step 2-5: Media classification (OpenAI)
        print("\n🏢 언론사 정보 추가 중...")
        media_csv_path = outdir / "media_directory.csv"
        df_processed = enrich_with_media_info(
            df_processed,
            spreadsheet=spreadsheet,  # None이면 CSV-only 모드
            openai_key=env["openai_key"],
            csv_path=media_csv_path
        )

        # 중간 저장 (언론사 정보 완료 후)
        print("💾 중간 저장 중 (언론사 정보 완료)...")
        if result_csv_path.exists():
            df_result_temp = pd.read_csv(result_csv_path, encoding='utf-8-sig')
            df_temp = pd.concat([df_result_temp, df_processed], ignore_index=True)
            df_temp = df_temp.drop_duplicates(subset=['link'], keep='last')
        else:
            df_temp = df_processed
        save_csv(df_temp, result_csv_path)
        if spreadsheet:
            try:
                sync_raw_and_processed(df_raw, df_temp, spreadsheet)
                print("✅ Google Sheets 중간 동기화 완료")
            except Exception as e:
                print(f"⚠️  Sheets 동기화 실패: {e}")

        # 전처리 메트릭
        press_release_groups = df_processed['group_id'].nunique() if 'group_id' in df_processed.columns else 0
        logger.log_dict({
            "articles_processed": len(df_processed),
            "duplicates_removed": duplicates_removed,
            "press_releases_detected": press_releases,
            "press_release_groups": press_release_groups
        })
        logger.end_stage("processing")

        # Step 3: 분류 (미처리 행만)
        logger.start_stage("classification")
        print("\n" + "=" * 80)
        print("STEP 3: 분류")
        print("=" * 80)

        # Step 3-1: 보도자료 전처리 (LLM 스킵용 고정값 설정)
        df_processed = preset_press_release_values(df_processed)

        # Step 3-2: LLM 분류 (보도자료는 스킵)
        df_classified, llm_metrics = classify_llm(
            df_processed,
            env["openai_key"],
            chunk_size=args.chunk_size,
            dry_run=args.dry_run,
            max_competitor_classify=args.max_competitor_classify,
            max_workers=args.max_workers,
            result_csv_path=str(result_csv_path),
            spreadsheet=spreadsheet,
            raw_df=df_raw
        )

        # LLM 메트릭 로깅
        logger.log_dict(llm_metrics)

        # 통계 출력
        stats = get_classification_stats(df_classified)
        print_classification_stats(stats)

        # 보도자료 정보 및 언론사 정보 병합
        source_cols = ['link', 'source', 'cluster_id', 'press_release_group', 'media_domain', 'media_name', 'media_group',
                       'media_type']
        source_data = df_processed[source_cols].copy()

        # 기존 컬럼 제거 (중복 방지)
        cols_to_drop = [col for col in df_classified.columns if col in source_cols and col != 'link']
        df_classified = df_classified.drop(columns=cols_to_drop, errors='ignore')

        # merge
        df_classified = df_classified.merge(source_data, on='link', how='left')

        # 나머지 NaN → 공란 변환 (FutureWarning 방지)
        df_classified = df_classified.fillna("").infer_objects(copy=False)

        # Step 3.7: Looker Studio 준비 (항상 실행)
        print("\n🕒 Looker Studio 시계열 컬럼 추가 중...")
        df_classified = add_time_series_columns(df_classified)

        logger.end_stage("classification")

        # 기존 result.csv와 병합
        if result_csv_path.exists():
            df_result_existing = pd.read_csv(result_csv_path, encoding='utf-8-sig', on_bad_lines='skip', engine='python')
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

        # Google Sheets 즉시 동기화 (분류 완료 직후)
        if spreadsheet:
            print("\n📊 Google Sheets 즉시 동기화 중 (분류 결과)...")
            try:
                sync_results = sync_raw_and_processed(df_raw, df_result, spreadsheet)
                print("✅ 분류 결과 Sheets 동기화 완료")
            except Exception as e:
                print(f"⚠️  분류 결과 Sheets 동기화 실패: {e}")

        # Step 4: 리포트 생성
        print("\n" + "=" * 80)
        print("STEP 4: 리포트 생성")
        print("=" * 80)

        # 콘솔 리포트
        generate_console_report(df_result)

        # 분류 결과 메트릭
        our_brands_relevant = len(df_result[(df_result['group'] == 'OUR') & (df_result['brand_relevance'].isin(['관련', '언급']))]) if 'brand_relevance' in df_result.columns else 0
        our_brands_negative = len(df_result[(df_result['group'] == 'OUR') & (df_result['sentiment_stage'].isin(['부정 후보', '부정 확정']))]) if 'sentiment_stage' in df_result.columns else 0
        danger_high = len(df_result[df_result['danger_level'] == '상']) if 'danger_level' in df_result.columns else 0
        danger_medium = len(df_result[df_result['danger_level'] == '중']) if 'danger_level' in df_result.columns else 0
        competitor_articles = len(df_result[df_result['group'] == 'COMPETITOR']) if 'group' in df_result.columns else 0

        logger.log_dict({
            "our_brands_relevant": our_brands_relevant,
            "our_brands_negative": our_brands_negative,
            "danger_high": danger_high,
            "danger_medium": danger_medium,
            "competitor_articles": competitor_articles
        })

        # 키워드 추출 (옵션)
        if args.extract_keywords:
            print("\n" + "=" * 80)
            print("STEP 4.5: 카테고리별 키워드 추출")
            print("=" * 80)
            extract_all_categories(
                df=df_result,
                output_dir=outdir / "keywords",
                top_k=args.keyword_top_k,
                max_display=10,
                spreadsheet=spreadsheet  # Google Sheets 연결 전달
            )

    # Step 5: Google Sheets 동기화 (주 저장소)
    if spreadsheet:
        logger.start_stage("sheets_sync")
        print("\n" + "=" * 80)
        print("STEP 5: Google Sheets 동기화 (주 저장소)")
        print("=" * 80)
        try:
            sync_results = sync_raw_and_processed(df_raw, df_result, spreadsheet)
            print("✅ Google Sheets 동기화 완료")

            # Sheets 메트릭 수집
            logger.log_dict({
                "sheets_sync_enabled": True,
                "sheets_rows_uploaded_raw": sync_results.get("raw_data", {}).get("added", 0),
                "sheets_rows_uploaded_result": sync_results.get("result", {}).get("added", 0)
            })
        except Exception as e:
            print(f"❌ Google Sheets 동기화 실패: {e}")
            print("   ⚠️  CSV 파일만 저장되었습니다 (troubleshooting 모드)")
            logger.log("sheets_sync_enabled", False)
        logger.end_stage("sheets_sync")
    else:
        logger.log("sheets_sync_enabled", False)
        print("\n" + "=" * 80)
        print("⚠️  Google Sheets 연결 없음")
        print("=" * 80)
        print("  주 저장소인 Google Sheets에 동기화되지 않았습니다.")
        print("  CSV 파일만 저장되었습니다 (troubleshooting 모드)")
        print("  .env 파일에 GOOGLE_SHEETS_CREDENTIALS_PATH 및 GOOGLE_SHEET_ID 설정을 권장합니다.")

    # 로그 저장
    logger.end_stage("total")
    logger.finalize()

    logs_csv_path = outdir / "logs" / "run_history.csv"
    logger.save_csv(str(logs_csv_path))

    # Sheets 로그 동기화
    if spreadsheet:
        try:
            sync_logs_to_sheets(str(logs_csv_path), spreadsheet)
            logger.log("sheets_logs_uploaded", 1)
        except Exception as e:
            print(f"⚠️  로그 Sheets 동기화 실패: {e}")
            logger.log("sheets_logs_uploaded", 0)

    # 로그 요약 출력
    logger.print_summary()

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
        if args.extract_keywords:
            print(f"  📂 {outdir}/keywords/ - 카테고리별 키워드 CSV")
    if spreadsheet or not args.raw_only:
        print(f"  📂 {outdir}/media_directory.csv - 언론사 디렉토리")
    if not spreadsheet:
        print(f"\n  ⚠️  Google Sheets 미연결: CSV 파일만 저장됨")
    print()


if __name__ == "__main__":
    main()