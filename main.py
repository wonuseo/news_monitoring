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
from collect import OUR_BRANDS, COMPETITORS, collect_all_news
from process import normalize_df, dedupe_df, save_excel
from classify import classify_all
from report import generate_console_report, create_word_report


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
  python main.py --chunk_size 50 --outdir reports
  python main.py --dry_run  # AI 분류 없이 테스트
        """
    )
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
    
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 뉴스 모니터링 시스템 시작")
    print("="*80)
    print(f"\n설정:")
    print(f"  - 우리 브랜드: {', '.join(OUR_BRANDS)}")
    print(f"  - 경쟁사: {', '.join(COMPETITORS)}")
    print(f"  - 기사 수: {args.display}개/브랜드")
    print(f"  - 출력 디렉토리: {args.outdir}/")
    print(f"  - AI 청크 크기: {args.chunk_size}")
    if args.dry_run:
        print(f"  - 모드: DRY RUN (AI 분류 생략)")
    print()
    
    # Step 0: 환경 설정
    env = load_env()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: 수집
    print("\n" + "="*80)
    print("STEP 1: 뉴스 수집")
    print("="*80)
    df_raw = collect_all_news(
        OUR_BRANDS, COMPETITORS,
        args.display, args.start, args.sort,
        env["naver_id"], env["naver_secret"]
    )
    
    if len(df_raw) == 0:
        print("❌ 수집된 기사가 없습니다. 종료합니다.")
        return
    
    save_excel(df_raw, outdir / "raw.xlsx")
    
    # Step 2: 처리
    print("\n" + "="*80)
    print("STEP 2: 데이터 처리")
    print("="*80)
    df_normalized = normalize_df(df_raw)
    df_processed = dedupe_df(df_normalized)
    save_excel(df_processed, outdir / "processed.xlsx")
    
    # Step 3: 분류
    print("\n" + "="*80)
    print("STEP 3: AI 분류")
    print("="*80)
    df_result = classify_all(
        df_processed, 
        env["openai_key"], 
        args.max_competitor_classify,
        args.chunk_size,
        args.dry_run
    )
    
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
    
    # 완료
    print("\n" + "="*80)
    print("✅ 모든 작업 완료!")
    print("="*80)
    print(f"\n생성된 파일:")
    print(f"  📊 {outdir}/raw.xlsx - 원본 데이터")
    print(f"  📊 {outdir}/processed.xlsx - 정제된 데이터")
    print(f"  📊 {outdir}/result.xlsx - AI 분류 결과")
    print(f"  📄 {outdir}/report.docx - Word 리포트")
    print()


if __name__ == "__main__":
    main()
