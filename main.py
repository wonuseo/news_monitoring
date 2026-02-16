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
from src.modules.processing.reprocess_checker import (
    check_reprocess_targets, load_raw_data_from_sheets,
    clear_classified_at_for_targets, print_reprocess_stats,
)
from src.modules.processing.looker_prep import add_time_series_columns
from src.modules.analysis.classify_llm import classify_llm
from src.modules.analysis.result_writer import invalidate_csv_header_cache
from src.modules.analysis.classification_stats import get_classification_stats, print_classification_stats
from src.modules.analysis.llm_engine import set_error_callback as set_llm_error_callback
from src.modules.processing.press_release_detector import set_error_callback as set_pr_error_callback
from src.modules.processing.media_classify import set_error_callback as set_media_error_callback
from src.modules.analysis.classify_press_releases import classify_press_releases
from src.modules.analysis.source_verifier import verify_and_regroup_sources
from src.modules.analysis.keyword_extractor import extract_all_categories
from src.modules.export.report import generate_console_report
from src.modules.export.sheets import (
    connect_sheets,
    sync_raw_and_processed,
    load_existing_links_from_sheets,
    filter_new_articles_from_sheets,
    clean_all_bom_in_sheets,
)
from src.utils.sheets_helpers import intermediate_sync
from src.modules.monitoring.logger import RunLogger, sync_run_history_to_sheets


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


def run_preprocessing_pipeline(
    df_to_process, df_raw, result_csv_path, outdir,
    openai_key, spreadsheet, record_error
):
    """
    전처리 파이프라인 (Step 2-1 ~ 2-6).

    Args:
        df_to_process: 처리 대상 DataFrame
        df_raw: 원본 raw DataFrame (Sheets 동기화용)
        result_csv_path: result.csv 경로
        outdir: 출력 디렉토리
        openai_key: OpenAI API 키
        spreadsheet: gspread Spreadsheet 객체 (None이면 CSV-only)
        record_error: 에러 기록 함수

    Returns:
        (df_processed, metrics_dict)
    """
    sync_error_fn = lambda msg: record_error(msg, category="sheets_sync")

    # Step 2-1: Normalize (cumulative article_no numbering)
    df_normalized = normalize_df(df_to_process, spreadsheet)
    before_dedupe = len(df_normalized)

    # 날짜 필터링은 STEP 1.6에서 이미 수행됨
    filtered_count = 0

    # Step 2-2: Deduplicate
    df_processed = dedupe_df(df_normalized)
    duplicates_removed = before_dedupe - len(df_processed)

    # 중간 동기화 (중복 제거 완료 후)
    intermediate_sync(
        df_processed, df_raw, result_csv_path, spreadsheet,
        "중복 제거", save_csv, sync_error_fn
    )

    # Step 2-3: Detect similar articles (Press Release, cumulative cluster_id numbering)
    df_processed = detect_similar_articles(df_processed, spreadsheet)
    press_releases = len(df_processed[df_processed['source'] == '보도자료']) if 'source' in df_processed.columns else 0

    # 중간 동기화 (보도자료 탐지 완료 후)
    intermediate_sync(
        df_processed, df_raw, result_csv_path, spreadsheet,
        "보도자료 탐지", save_csv, sync_error_fn
    )

    # Step 2-4: Summarize press release groups (OpenAI)
    df_processed = summarize_press_release_groups(df_processed, openai_key)

    # 중간 동기화 (보도자료 요약 완료 후)
    intermediate_sync(
        df_processed, df_raw, result_csv_path, spreadsheet,
        "보도자료 요약", save_csv, sync_error_fn
    )

    # Step 2-5: Media classification (OpenAI)
    media_csv_path = outdir / "media_directory.csv"
    df_processed, media_stats = enrich_with_media_info(
        df_processed,
        spreadsheet=spreadsheet,
        openai_key=openai_key,
        csv_path=media_csv_path
    )

    # 중간 동기화 (언론사 분류 완료 후)
    intermediate_sync(
        df_processed, df_raw, result_csv_path, spreadsheet,
        "언론사 분류", save_csv, sync_error_fn
    )

    # Press release avg cluster size
    press_release_groups = df_processed['cluster_id'].nunique() if 'cluster_id' in df_processed.columns else 0
    avg_cluster_size = round(press_releases / press_release_groups, 1) if press_release_groups > 0 else 0

    # 메트릭 수집
    metrics = {
        "articles_processed": len(df_processed),
        "duplicates_removed": duplicates_removed,
        "articles_filtered_by_date": filtered_count,
        "press_releases_detected": press_releases,
        "press_release_groups": press_release_groups,
        "press_release_avg_cluster_size": avg_cluster_size,
    }
    metrics.update(media_stats)

    return df_processed, metrics


def main():
    # RunLogger 초기화
    logger = RunLogger()

    parser = argparse.ArgumentParser(
        description="뉴스 모니터링 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python main.py                                    # 전체 파이프라인 실행 (자사+경쟁사 전체 분석, 키워드 추출 자동)
  python main.py --display 200                      # API 결과 개수 지정
  python main.py --keyword_top_k 30                 # 키워드 추출 개수 조정 (기본: 20)
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
    parser.add_argument("--max_workers", type=int, default=3,
                        help="병렬 처리 워커 수 (기본: 3, 권장: 3-10)")
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
                        help="(더 이상 사용되지 않음, 항상 자동 실행됨) 카테고리별 특징 키워드 추출")
    parser.add_argument("--keyword_top_k", type=int, default=20,
                        help="키워드 추출 시 상위 K개 선택 (기본: 20)")

    # BOM 정리 옵션
    parser.add_argument("--clean_bom", action="store_true",
                        help="Google Sheets 전체 BOM 문자 정리 후 종료")

    # 재처리 전용 옵션
    parser.add_argument("--recheck_only", action="store_true",
                        help="API 수집 없이 재처리 대상만 검사/재처리 (Sheets 필수)")

    args = parser.parse_args()

    # run_mode 판별
    if args.dry_run:
        run_mode = "dry_run"
    elif args.raw_only:
        run_mode = "raw_only"
    elif args.preprocess_only:
        run_mode = "preprocess_only"
    elif args.recheck_only:
        run_mode = "recheck_only"
    else:
        run_mode = "full"

    # CLI args 로깅
    logger.log("run_mode", run_mode)
    logger.log("cli_args", vars(args))
    logger.log_event("run_started", {"cli_args": vars(args)}, stage="init")

    # current_stage for error callback propagation
    current_stage = "init"

    print("=" * 80)
    print("🚀 뉴스 모니터링 시스템 시작")
    print("=" * 80)
    print(f"\n설정:")
    print(f"  - 우리 브랜드: {', '.join(OUR_BRANDS)}")
    print(f"  - 경쟁사: {', '.join(COMPETITORS)}")
    print(f"  - 수집 모드: Naver API")
    print(f"  - 기사 수: {args.display}개/브랜드 (최대 {args.max_api_pages} 페이지)")
    print(f"  - 날짜 필터: 2026-02-01 이후만 분석")
    print(f"  - 출력 디렉토리: {args.outdir}/")
    if args.max_competitor_classify == 0:
        print(f"  - 분류 모드: 자사+경쟁사 전체 분석")
    else:
        print(f"  - 분류 모드: 자사 전체 + 경쟁사 최대 {args.max_competitor_classify}개/브랜드")
    print(f"  - AI 청크 크기: {args.chunk_size}")
    print(f"  - 병렬 처리 워커: {args.max_workers}개")
    print(f"  - 키워드 추출: 자동 실행 (상위 {args.keyword_top_k}개)")
    if args.dry_run:
        print(f"  - 모드: DRY RUN (AI 분류 생략)")
    if args.raw_only:
        print(f"  - 모드: RAW ONLY (API 수집 + Sheets 업로드만)")
    if args.preprocess_only:
        print(f"  - 모드: PREPROCESS ONLY (수집 + 전처리 + Sheets 업로드)")
    if args.recheck_only:
        print(f"  - 모드: RECHECK ONLY (API 수집 생략, Sheets 기준 재처리)")
    print()

    # Step 0: 환경 설정
    env = load_env()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    media_csv_path = outdir / "media_directory.csv"

    # Step 1: 수집
    current_stage = "collection"
    logger.start_stage("collection")
    print("\n" + "=" * 80)
    print("STEP 1: 뉴스 수집")
    print("=" * 80)

    # Google Sheets 연결 (주 저장소)
    existing_links = set()
    spreadsheet = None
    def record_error(message, data=None, category="system"):
        logger.log_error(message, data, category=category, stage=current_stage)
        if spreadsheet:
            logger.flush_all_to_sheets(spreadsheet)

    # Error callback registration for OpenAI wrappers (log failures only)
    set_llm_error_callback(lambda msg, data=None: record_error(msg, data, category="openai_api"))
    set_pr_error_callback(lambda msg, data=None: record_error(msg, data, category="openai_api"))
    set_media_error_callback(lambda msg, data=None: record_error(msg, data, category="openai_api"))

    # Google Sheets 자동 연결 (credentials 필수 권장)
    creds_path = os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH", "service-account.json")
    sheet_id = args.sheets_id or os.getenv("GOOGLE_SHEET_ID")

    if os.path.exists(creds_path) and sheet_id:
        try:
            print("\n📊 Google Sheets 연결 중...")
            spreadsheet = connect_sheets(creds_path, sheet_id)
            if spreadsheet:
                # Sheets 기존 BOM 자동 정리 (데이터 로드 전)
                print("🧹 Sheets BOM 문자 자동 정리 중...")
                bom_results = clean_all_bom_in_sheets(spreadsheet)
                bom_total = sum(bom_results.values())
                if bom_total > 0:
                    print(f"  ✅ {bom_total}개 셀 BOM 제거 완료")

                existing_links = load_existing_links_from_sheets(spreadsheet)
                print("✅ Google Sheets 연결 성공 (주 저장소)")
                print("   CSV 파일은 troubleshooting용으로 함께 저장됩니다.")
                logger.log_event("sheets_connected", {"sheet_id": sheet_id}, category="sheets_sync", stage="init")
                logger.flush_all_to_sheets(spreadsheet)
        except Exception as e:
            record_error(f"Google Sheets 연결 실패: {e}", {"sheet_id": sheet_id}, category="sheets_sync")
            print("   ⚠️  CSV 파일만 사용합니다 (troubleshooting 모드)")
            logger.log_event("sheets_connect_failed", {"error": str(e)}, category="sheets_sync", stage="init")
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
        logger.log_event("sheets_not_configured", {"credentials_path": creds_path, "sheet_id": sheet_id}, category="sheets_sync", stage="init")

    # --clean_bom 모드: Sheets 전체 BOM 정리 후 종료
    if args.clean_bom:
        if not spreadsheet:
            print("❌ --clean_bom 사용 시 Google Sheets 연결이 필요합니다.")
            print("  .env 파일에 GOOGLE_SHEETS_CREDENTIALS_PATH와 GOOGLE_SHEET_ID를 설정하세요.")
            return
        print("\n" + "=" * 80)
        print("BOM 문자 정리 모드")
        print("=" * 80)
        results = clean_all_bom_in_sheets(spreadsheet)
        total_cleaned = sum(results.values())
        print(f"\n✅ BOM 정리 완료: 총 {total_cleaned}개 셀 정리")
        return

    raw_csv_path = outdir / "raw.csv"

    # --recheck_only 모드: API 수집 생략, Sheets에서 raw_data 로드
    if args.recheck_only:
        if not spreadsheet:
            print("❌ --recheck_only 사용 시 Google Sheets 연결이 필요합니다.")
            print("  .env 파일에 GOOGLE_SHEETS_CREDENTIALS_PATH와 GOOGLE_SHEET_ID를 설정하세요.")
            return
        print("\n📋 --recheck_only 모드: API 수집 생략, Sheets raw_data 로드")
        df_raw = load_raw_data_from_sheets(spreadsheet)
        if len(df_raw) == 0:
            print("❌ Sheets raw_data가 비어있어 재처리할 수 없습니다.")
            return
        df_raw_new = pd.DataFrame(columns=df_raw.columns)
        save_csv(df_raw, raw_csv_path)
        logger.log_dict({
            "articles_collected_total": 0,
            "articles_collected_per_query": {},
            "existing_links_skipped": 0,
        })
        logger.end_stage("collection")
    else:
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
                msg_parts = [
                    f"{sync_result.get('attempted', 0)}개 시도",
                    f"{sync_result.get('added', 0)}개 추가"
                ]
                if sync_result.get('updated', 0) > 0:
                    msg_parts.append(f"{sync_result['updated']}개 업데이트")
                msg_parts.append(f"{sync_result.get('skipped', 0)}개 건너뜀")
                print(f"✅ raw_data 시트 동기화 완료: {', '.join(msg_parts)}")
                logger.log_event("sheets_sync_raw_data", sync_result, category="sheets_sync", stage="collection")
            except Exception as e:
                record_error(f"raw_data 시트 동기화 실패: {e}", category="sheets_sync")

        # 수집 단계 메트릭
        articles_per_query = df_raw_new.groupby('query').size().to_dict() if 'query' in df_raw_new.columns else {}
        logger.log_dict({
            "articles_collected_total": len(df_raw_new),
            "articles_collected_per_query": articles_per_query,
            "existing_links_skipped": existing_links_skipped
        })
        logger.log_event("collection_completed", {
            "articles_collected_total": len(df_raw_new),
            "articles_collected_per_query": articles_per_query,
            "existing_links_skipped": existing_links_skipped
        }, stage="collection")
        if spreadsheet:
            logger.flush_all_to_sheets(spreadsheet)
        logger.end_stage("collection")

    # STEP 1.5: 재처리 대상 검사
    result_csv_path = outdir / "result.csv"

    try:
        recheck = check_reprocess_targets(df_raw, spreadsheet, str(result_csv_path))
        print_reprocess_stats(recheck["stats"])

        df_reprocess = recheck["df_to_reprocess"]
        if len(df_reprocess) > 0:
            df_reprocess = clear_classified_at_for_targets(df_reprocess, recheck["reprocess_links"])
            df_to_process = pd.concat([df_raw_new, df_reprocess], ignore_index=True)
            df_to_process = df_to_process.drop_duplicates(subset=['link'], keep='first')
        else:
            df_to_process = df_raw_new

        # 재처리 메트릭 로깅
        logger.log_dict({
            "reprocess_targets_total": recheck["stats"]["total_reprocess_targets"],
            "reprocess_missing_from_result": recheck["stats"]["missing_from_result"],
            "reprocess_field_missing": recheck["stats"].get("field_missing", {}),
        })
    except Exception as e:
        record_error(f"재처리 대상 검사 실패: {e}", category="system")
        df_to_process = df_raw_new

    # STEP 1.6: 2026-02-01 이후 기사만 필터링
    if len(df_to_process) > 0 and 'pubDate' in df_to_process.columns:
        before_date_filter = len(df_to_process)
        df_to_process['pub_datetime_temp'] = pd.to_datetime(df_to_process['pubDate'], errors='coerce')
        df_to_process = df_to_process[df_to_process['pub_datetime_temp'] >= '2026-02-01'].copy()
        df_to_process = df_to_process.drop(columns=['pub_datetime_temp'])
        date_filtered = before_date_filter - len(df_to_process)
        print(f"🔧 날짜 필터링: {date_filtered}개 제외 (2026-02-01 이전), {len(df_to_process)}개 유지")

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

                # Sheets 메트릭 수집 (added + updated)
                raw_sync = sync_results.get("raw_data", {})
                result_sync = sync_results.get("total_result", {})
                logger.log_dict({
                    "sheets_sync_enabled": True,
                    "sheets_rows_uploaded_raw": raw_sync.get("added", 0) + raw_sync.get("updated", 0),
                    "sheets_rows_uploaded_result": result_sync.get("added", 0) + result_sync.get("updated", 0)
                })
            except Exception as e:
                record_error(f"Google Sheets 업로드 실패 (기존 데이터): {e}", category="sheets_sync")
                logger.log("sheets_sync_enabled", False)
            logger.end_stage("sheets_sync")
            logger.log_event("sheets_sync_completed", {
                "sheets_rows_uploaded_raw": logger.metrics.get("sheets_rows_uploaded_raw", 0),
                "sheets_rows_uploaded_result": logger.metrics.get("sheets_rows_uploaded_result", 0)
            }, category="sheets_sync", stage="sheets_sync")
            if spreadsheet:
                logger.flush_all_to_sheets(spreadsheet)

        # 로그 저장
        logger.finalize()
        logs_csv_path = outdir / "logs" / "run_history.csv"
        logger.save_csv(str(logs_csv_path))

        # Sheets 로그 동기화
        if spreadsheet:
            sync_run_history_to_sheets(str(logs_csv_path), spreadsheet)

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
            "articles_filtered_by_date": 0,
            "press_releases_detected": 0,
            "press_release_groups": 0
        })
    # --preprocess_only 모드인 경우 분류 스킵, 처리+리포트는 실행
    elif args.preprocess_only:
        # Step 2: 처리 (미처리 행만)
        current_stage = "processing"
        logger.start_stage("processing")
        print("\n" + "=" * 80)
        print("STEP 2: 데이터 처리 (미처리 행만)")
        print("=" * 80)

        df_processed, proc_metrics = run_preprocessing_pipeline(
            df_to_process, df_raw, result_csv_path, outdir,
            env["openai_key"], spreadsheet, record_error
        )

        # 나머지 NaN → 공란 변환
        df_processed = df_processed.fillna("")

        # Step 2-6: Looker Studio time-series columns
        print("\n🕒 Looker Studio 시계열 컬럼 추가 중...")
        df_processed = add_time_series_columns(df_processed)

        # 전처리 메트릭
        current_stage = "processing"
        logger.log_dict(proc_metrics)
        logger.end_stage("processing")
        logger.log_event("processing_completed", proc_metrics, stage="processing")
        if spreadsheet:
            logger.flush_all_to_sheets(spreadsheet)

        # 기존 result.csv와 병합
        if result_csv_path.exists():
            df_result_existing = pd.read_csv(result_csv_path, encoding='utf-8-sig')
            df_result = pd.concat([df_result_existing, df_processed], ignore_index=True)
            df_result = df_result.drop_duplicates(subset=['link'], keep='last')
            print(f"📂 기존 result.csv 업데이트: {len(df_result_existing)} + {len(df_processed)} = {len(df_result)}개 기사")
        else:
            df_result = df_processed

        # total_result_count
        logger.log("total_result_count", len(df_result))

        # 결과 저장
        save_csv(df_result, result_csv_path)
        invalidate_csv_header_cache(str(result_csv_path))

        # Google Sheets 즉시 동기화 (전처리 완료 직후)
        if spreadsheet:
            print("\n📊 Google Sheets 즉시 동기화 중 (전처리 결과)...")
            try:
                sync_results = sync_raw_and_processed(df_raw, df_result, spreadsheet)
                print("✅ 전처리 결과 Sheets 동기화 완료")
            except Exception as e:
                record_error(f"전처리 결과 Sheets 동기화 실패: {e}", category="sheets_sync")
    else:
        # Step 2: 처리 (미처리 행만)
        current_stage = "processing"
        logger.start_stage("processing")
        print("\n" + "=" * 80)
        print("STEP 2: 데이터 처리 (미처리 행만)")
        print("=" * 80)

        df_processed, proc_metrics = run_preprocessing_pipeline(
            df_to_process, df_raw, result_csv_path, outdir,
            env["openai_key"], spreadsheet, record_error
        )

        # 전처리 메트릭
        current_stage = "processing"
        logger.log_dict(proc_metrics)
        logger.end_stage("processing")

        # Step 3: 분류 (미처리 행만)
        print("\n" + "=" * 80)
        print("STEP 3: 분류")
        print("=" * 80)

        # Step 3-1: 보도자료 LLM 분류 (대표 기사 분석 -> 클러스터 공유)
        current_stage = "pr_classification"
        logger.start_stage("pr_classification")
        df_processed, pr_metrics = classify_press_releases(
            df_processed,
            env["openai_key"],
            chunk_size=args.chunk_size,
            max_workers=args.max_workers,
            result_csv_path=str(result_csv_path),
            spreadsheet=spreadsheet,
            raw_df=df_raw
        )
        logger.log_dict(pr_metrics)
        logger.end_stage("pr_classification")
        logger.log_event("pr_classification_completed", pr_metrics, stage="pr_classification")
        if spreadsheet:
            logger.flush_all_to_sheets(spreadsheet)

        # Step 3-2: LLM 분류 (보도자료는 스킵)
        current_stage = "general_classification"
        logger.start_stage("general_classification")
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
        logger.end_stage("general_classification")

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

        # 나머지 NaN -> 공란 변환 (FutureWarning 방지)
        df_classified = df_classified.fillna("").infer_objects(copy=False)

        # Step 3-3: Source 검증 및 주제 그룹화
        if not args.dry_run:
            current_stage = "source_verification"
            logger.start_stage("source_verification")
            df_classified, sv_metrics = verify_and_regroup_sources(df_classified, openai_key=env["openai_key"])
            logger.log_dict(sv_metrics)
            logger.end_stage("source_verification")
            logger.log_event("source_verification_completed", sv_metrics, stage="source_verification")
            if spreadsheet:
                logger.flush_all_to_sheets(spreadsheet)

        # Step 3.7: Looker Studio 준비 (항상 실행)
        print("\n🕒 Looker Studio 시계열 컬럼 추가 중...")
        df_classified = add_time_series_columns(df_classified)

        logger.log_event("classification_completed", {
            "articles_classified_llm": llm_metrics.get("articles_classified_llm", 0),
            "llm_api_calls": llm_metrics.get("llm_api_calls", 0),
            "press_releases_skipped": llm_metrics.get("press_releases_skipped", 0),
            "classification_errors": llm_metrics.get("classification_errors", 0)
        }, stage="general_classification")
        if spreadsheet:
            logger.flush_all_to_sheets(spreadsheet)

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
        invalidate_csv_header_cache(str(result_csv_path))

        # Google Sheets 즉시 동기화 (분류 완료 직후)
        if spreadsheet:
            print("\n📊 Google Sheets 즉시 동기화 중 (분류 결과)...")
            try:
                sync_results = sync_raw_and_processed(df_raw, df_result, spreadsheet)
                print("✅ 분류 결과 Sheets 동기화 완료")
            except Exception as e:
                record_error(f"분류 결과 Sheets 동기화 실패: {e}", category="sheets_sync")

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
            "competitor_articles": competitor_articles,
            "total_result_count": len(df_result),
        })

        # 키워드 추출 (자동 실행)
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
    current_stage = "sheets_sync"
    if spreadsheet:
        logger.start_stage("sheets_sync")
        print("\n" + "=" * 80)
        print("STEP 5: Google Sheets 동기화 (주 저장소)")
        print("=" * 80)
        try:
            sync_results = sync_raw_and_processed(df_raw, df_result, spreadsheet)
            print("✅ Google Sheets 동기화 완료")

            # Sheets 메트릭 수집 (added + updated)
            raw_sync = sync_results.get("raw_data", {})
            result_sync = sync_results.get("total_result", {})
            logger.log_dict({
                "sheets_sync_enabled": True,
                "sheets_rows_uploaded_raw": raw_sync.get("added", 0) + raw_sync.get("updated", 0),
                "sheets_rows_uploaded_result": result_sync.get("added", 0) + result_sync.get("updated", 0)
            })
        except Exception as e:
            record_error(f"Google Sheets 동기화 실패 (최종): {e}", category="sheets_sync")
            print("   ⚠️  CSV 파일만 저장되었습니다 (troubleshooting 모드)")
            logger.log("sheets_sync_enabled", False)
        logger.end_stage("sheets_sync")
        logger.log_event("sheets_sync_completed", {
            "sheets_rows_uploaded_raw": logger.metrics.get("sheets_rows_uploaded_raw", 0),
            "sheets_rows_uploaded_result": logger.metrics.get("sheets_rows_uploaded_result", 0)
        }, category="sheets_sync", stage="sheets_sync")
        if spreadsheet:
            logger.flush_all_to_sheets(spreadsheet)
    else:
        logger.log("sheets_sync_enabled", False)
        print("\n" + "=" * 80)
        print("⚠️  Google Sheets 연결 없음")
        print("=" * 80)
        print("  주 저장소인 Google Sheets에 동기화되지 않았습니다.")
        print("  CSV 파일만 저장되었습니다 (troubleshooting 모드)")
        print("  .env 파일에 GOOGLE_SHEETS_CREDENTIALS_PATH 및 GOOGLE_SHEET_ID 설정을 권장합니다.")

    # 로그 저장
    logger.finalize()

    logs_csv_path = outdir / "logs" / "run_history.csv"
    logger.save_csv(str(logs_csv_path))

    # Sheets run_history 동기화
    if spreadsheet:
        try:
            sync_run_history_to_sheets(str(logs_csv_path), spreadsheet)
        except Exception as e:
            record_error(f"run_history Sheets 동기화 실패: {e}", category="sheets_sync")

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
        print(f"  📂 {outdir}/keywords/ - 카테고리별 키워드 CSV")
    if spreadsheet or not args.raw_only:
        print(f"  📂 {outdir}/media_directory.csv - 언론사 디렉토리")
    if not spreadsheet:
        print(f"\n  ⚠️  Google Sheets 미연결: CSV 파일만 저장됨")
    print()


if __name__ == "__main__":
    main()
