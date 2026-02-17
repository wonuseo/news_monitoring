"""CLI argument parsing for News Monitoring System."""

import argparse


def parse_args():
    """Parse CLI arguments and return (args, run_mode)."""
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

    return args, run_mode
