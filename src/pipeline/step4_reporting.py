"""Step 4: Console report + keyword extraction + category discovery."""

from src.modules.export.report import generate_console_report
from src.modules.analysis.keyword_extractor import extract_all_categories
from src.modules.analysis.category_discovery import discover_new_categories


def run_reporting(ctx):
    """Generate reports and extract keywords."""
    if ctx.run_mode == "raw_only":
        return

    # Step 4: 리포트 생성
    print("\n" + "=" * 80)
    print("STEP 4: 리포트 생성")
    print("=" * 80)

    # 콘솔 리포트
    generate_console_report(ctx.df_result)

    # 분류 결과 메트릭
    our_brands_relevant = len(ctx.df_result[(ctx.df_result['group'] == 'OUR') & (ctx.df_result['brand_relevance'].isin(['관련', '언급']))]) if 'brand_relevance' in ctx.df_result.columns else 0
    our_brands_negative = len(ctx.df_result[(ctx.df_result['group'] == 'OUR') & (ctx.df_result['sentiment_stage'].isin(['부정 후보', '부정 확정']))]) if 'sentiment_stage' in ctx.df_result.columns else 0
    danger_high = len(ctx.df_result[ctx.df_result['danger_level'] == '상']) if 'danger_level' in ctx.df_result.columns else 0
    danger_medium = len(ctx.df_result[ctx.df_result['danger_level'] == '중']) if 'danger_level' in ctx.df_result.columns else 0
    competitor_articles = len(ctx.df_result[ctx.df_result['group'] == 'COMPETITOR']) if 'group' in ctx.df_result.columns else 0

    ctx.logger.log_dict({
        "our_brands_relevant": our_brands_relevant,
        "our_brands_negative": our_brands_negative,
        "danger_high": danger_high,
        "danger_medium": danger_medium,
        "competitor_articles": competitor_articles,
        "total_result_count": len(ctx.df_result),
    })

    # 키워드 추출 (자동 실행)
    print("\n" + "=" * 80)
    print("STEP 4.5: 카테고리별 키워드 추출")
    print("=" * 80)
    extract_all_categories(
        df=ctx.df_result,
        top_k=ctx.args.keyword_top_k,
        max_display=10,
        spreadsheet=ctx.spreadsheet
    )

    # 카테고리 발견 분석 (기타 기사 패턴 → 새 카테고리 제안)
    if not ctx.args.dry_run:
        print("\n" + "=" * 80)
        print("STEP 4.6: 카테고리 발견 분석")
        print("=" * 80)
        discover_new_categories(
            df=ctx.df_result,
            openai_key=ctx.env["openai_key"],
            logger=ctx.logger,
        )
