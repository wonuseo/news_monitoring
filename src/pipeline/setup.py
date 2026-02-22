"""Pipeline setup: environment, Sheets connection, error callbacks."""

import os
from dotenv import load_dotenv

from src.modules.collection.collect import OUR_BRANDS, COMPETITORS
from src.modules.export.sheets import connect_sheets, load_existing_links_from_sheets, clean_all_bom_in_sheets
from src.modules.analysis.llm_engine import set_error_callback as set_llm_error_callback
from src.modules.processing.press_release_detector import set_error_callback as set_pr_error_callback
from src.modules.processing.media_classify import set_error_callback as set_media_error_callback


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


def print_banner(ctx):
    """Print startup banner with settings."""
    args = ctx.args

    print("=" * 80)
    print("🚀 뉴스 모니터링 시스템 시작")
    print("=" * 80)
    print(f"\n설정:")
    print(f"  - 우리 브랜드: {', '.join(OUR_BRANDS)}")
    print(f"  - 경쟁사: {', '.join(COMPETITORS)}")
    print(f"  - 수집 모드: Naver API")
    print(f"  - 기사 수: {args.display}개/브랜드 (최대 {args.max_api_pages} 페이지)")
    print(f"  - 날짜 필터: 2026-02-01 이후만 분석")
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


def connect_sheets_and_setup(ctx):
    """
    Connect to Google Sheets, register error callbacks, handle --clean_bom.

    Returns:
        bool: True if pipeline should continue, False if early exit (e.g. --clean_bom).
    """
    # Error callback registration for OpenAI wrappers (log failures only)
    error_callback = lambda msg, data=None: ctx.record_error(msg, data, category="openai_api")
    set_llm_error_callback(error_callback)
    set_pr_error_callback(error_callback)
    set_media_error_callback(error_callback)

    # Google Sheets 자동 연결 (credentials 필수 권장)
    creds_path = os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH", "service-account.json")
    sheet_id = ctx.args.sheets_id or os.getenv("GOOGLE_SHEET_ID")

    if os.path.exists(creds_path) and sheet_id:
        try:
            print("\n📊 Google Sheets 연결 중...")
            ctx.spreadsheet = connect_sheets(creds_path, sheet_id)
            if ctx.spreadsheet:
                print("✅ Google Sheets 연결 성공 (유일한 저장소)")
                ctx.logger.log_event("sheets_connected", {"sheet_id": sheet_id}, category="sheets_sync", stage="init")
                ctx.logger.flush_all_to_sheets(ctx.spreadsheet)
        except Exception as e:
            ctx.record_error(f"Google Sheets 연결 실패: {e}", {"sheet_id": sheet_id}, category="sheets_sync")
            ctx.sheets_required = False
            print("   ⚠️  Sheets 연결 실패: raw 수집만 가능합니다. 분류/분석은 Sheets 연결이 필요합니다.")
            ctx.logger.log_event("sheets_connect_failed", {"error": str(e)}, category="sheets_sync", stage="init")
    else:
        ctx.sheets_required = False
        print("\n" + "="*80)
        print("⚠️  경고: Google Sheets 설정이 없습니다!")
        print("="*80)
        if not os.path.exists(creds_path):
            print(f"  credential 파일 없음: {creds_path}")
        if not sheet_id:
            print(f"  GOOGLE_SHEET_ID 환경변수 없음")
        print("\n  Google Sheets는 유일한 저장소입니다.")
        print("  raw 수집만 가능합니다. 분류/분석은 Sheets 연결이 필요합니다.")
        print("="*80 + "\n")
        ctx.logger.log_event("sheets_not_configured", {"credentials_path": creds_path, "sheet_id": sheet_id}, category="sheets_sync", stage="init")

    # --clean_bom 모드: Sheets 전체 BOM 정리 후 종료
    if ctx.args.clean_bom:
        if not ctx.spreadsheet:
            print("❌ --clean_bom 사용 시 Google Sheets 연결이 필요합니다.")
            print("  .env 파일에 GOOGLE_SHEETS_CREDENTIALS_PATH와 GOOGLE_SHEET_ID를 설정하세요.")
            return False
        print("\n" + "=" * 80)
        print("BOM 문자 정리 모드")
        print("=" * 80)
        results = clean_all_bom_in_sheets(ctx.spreadsheet)
        total_cleaned = sum(results.values())
        print(f"\n✅ BOM 정리 완료: 총 {total_cleaned}개 셀 정리")
        return False

    return True
