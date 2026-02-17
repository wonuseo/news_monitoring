#!/usr/bin/env python3
"""
main.py - News Monitoring System
뉴스 모니터링 시스템 메인 실행 파일
"""

import pandas as pd
from pathlib import Path

# Pandas FutureWarning 방지
pd.set_option('future.no_silent_downcasting', True)

# Pipeline imports
from src.cli import parse_args
from src.pipeline.context import PipelineContext
from src.pipeline.setup import load_env, connect_sheets_and_setup, print_banner
from src.pipeline.step1_collection import run_collection
from src.pipeline.step2_processing import run_processing
from src.pipeline.step3_classification import run_classification
from src.pipeline.step4_reporting import run_reporting
from src.pipeline.step5_sheets_sync import run_sheets_sync
from src.pipeline.finalize import finalize_pipeline
from src.modules.monitoring.logger import RunLogger


def main():
    # Parse CLI & detect run mode
    args, run_mode = parse_args()

    # Initialize logger & context
    logger = RunLogger()
    env = load_env()
    outdir = Path(args.outdir)

    ctx = PipelineContext(
        args=args,
        run_mode=run_mode,
        env=env,
        outdir=outdir,
        logger=logger,
    )

    # Log run metadata
    logger.log("run_mode", run_mode)
    logger.log("cli_args", vars(args))
    logger.log_event("run_started", {"cli_args": vars(args)}, stage="init")

    # Setup: Sheets connection, error callbacks, banner
    print_banner(ctx)
    should_continue = connect_sheets_and_setup(ctx)
    if not should_continue:  # --clean_bom exits here
        return

    # ── Sheets 미연결 시: raw 수집 → CSV 저장 → 종료 ──────────
    if ctx.spreadsheet is None:
        print("\n⚠️  Google Sheets 미연결: raw 수집 후 종료합니다.")
        _emergency_raw_collection(ctx)
        finalize_pipeline(ctx)
        return

    # ── 전체 파이프라인 (Sheets 연결 필수) ───────────────────
    # Step 1: Collection + Reprocess + Date filter
    has_articles = run_collection(ctx)
    if not has_articles:
        finalize_pipeline(ctx)
        return

    # Step 2: Processing (normalize, dedupe, press release, media)
    run_processing(ctx)

    # Step 3: Classification (PR clusters, general LLM, source verification)
    run_classification(ctx)

    # Step 4: Reporting + Keywords
    run_reporting(ctx)

    # Step 5: Google Sheets sync
    run_sheets_sync(ctx)

    # Finalize: logs, summary, completion banner
    finalize_pipeline(ctx)


def _emergency_raw_collection(ctx):
    """Sheets 미연결 시 Naver API raw 수집 → outdir/raw.csv 저장"""
    from src.modules.collection.collect import OUR_BRANDS, COMPETITORS, collect_all_news
    from src.modules.processing.process import save_csv

    ctx.current_stage = "collection"
    ctx.logger.start_stage("collection")

    outdir = ctx.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("STEP 1: 뉴스 수집 (비상 모드 - Sheets 미연결)")
    print("=" * 80)

    df_raw = collect_all_news(
        OUR_BRANDS, COMPETITORS,
        ctx.args.display, ctx.args.max_api_pages, ctx.args.sort,
        ctx.env["naver_id"], ctx.env["naver_secret"],
        existing_links=set(),
        spreadsheet=None
    )

    raw_csv_path = outdir / "raw.csv"
    if len(df_raw) > 0:
        save_csv(df_raw, raw_csv_path)
        print(f"\n✅ {len(df_raw)}개 기사 수집 완료")
        print(f"💾 raw.csv 저장: {raw_csv_path}")
    else:
        print("\nℹ️  수집된 기사가 없습니다.")

    ctx.df_raw = df_raw
    ctx.df_raw_new = df_raw

    articles_per_query = df_raw.groupby('query').size().to_dict() if 'query' in df_raw.columns else {}
    ctx.logger.log_dict({
        "articles_collected_total": len(df_raw),
        "articles_collected_per_query": articles_per_query,
        "existing_links_skipped": 0,
    })
    ctx.logger.end_stage("collection")


if __name__ == "__main__":
    main()
