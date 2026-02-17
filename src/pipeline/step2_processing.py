"""Step 2: Data processing (normalize, dedupe, press release detection, media classification)."""

from src.modules.processing.process import normalize_df, dedupe_df, enrich_with_media_info
from src.modules.processing.press_release_detector import detect_similar_articles, summarize_clusters
from src.utils.sheets_helpers import intermediate_sync


def run_processing(ctx):
    """Run preprocessing pipeline. Sets ctx.df_processed."""
    if ctx.run_mode == "raw_only":
        ctx.df_result = ctx.df_raw
        ctx.logger.log_dict({
            "articles_processed": 0,
            "duplicates_removed": 0,
            "articles_filtered_by_date": 0,
            "press_releases_detected": 0,
            "press_release_groups": 0
        })
        return

    ctx.current_stage = "processing"
    ctx.logger.start_stage("processing")
    print("\n" + "=" * 80)
    print("STEP 2: 데이터 처리 (미처리 행만)")
    print("=" * 80)

    ctx.df_processed, proc_metrics = run_preprocessing_pipeline(
        ctx.df_to_process, ctx.df_raw,
        ctx.env["openai_key"], ctx.spreadsheet, ctx.record_error
    )

    ctx.logger.log_dict(proc_metrics)
    ctx.logger.end_stage("processing")
    ctx.logger.log_event("processing_completed", proc_metrics, stage="processing")
    if ctx.spreadsheet:
        ctx.logger.flush_all_to_sheets(ctx.spreadsheet)


def run_preprocessing_pipeline(
    df_to_process, df_raw,
    openai_key, spreadsheet, record_error
):
    """
    전처리 파이프라인 (Step 2-1 ~ 2-6).

    Args:
        df_to_process: 처리 대상 DataFrame
        df_raw: 원본 raw DataFrame (Sheets 동기화용)
        openai_key: OpenAI API 키
        spreadsheet: gspread Spreadsheet 객체 (필수)
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
    intermediate_sync(df_processed, df_raw, spreadsheet, "중복 제거", sync_error_fn)

    # Step 2-3: Detect similar articles (Press Release, cumulative cluster_id numbering)
    df_processed = detect_similar_articles(df_processed, spreadsheet)
    press_releases = len(df_processed[df_processed['source'] == '보도자료']) if 'source' in df_processed.columns else 0

    # 중간 동기화 (보도자료 탐지 완료 후)
    intermediate_sync(df_processed, df_raw, spreadsheet, "보도자료 탐지", sync_error_fn)

    # Step 2-4: Summarize clusters (OpenAI) — 보도자료 클러스터 요약
    df_processed = summarize_clusters(df_processed, openai_key)

    # 중간 동기화 (보도자료 요약 완료 후)
    intermediate_sync(df_processed, df_raw, spreadsheet, "보도자료 요약", sync_error_fn)

    # Step 2-5: Media classification (OpenAI)
    df_processed, media_stats = enrich_with_media_info(
        df_processed,
        spreadsheet=spreadsheet,
        openai_key=openai_key,
    )

    # 중간 동기화 (언론사 분류 완료 후)
    intermediate_sync(df_processed, df_raw, spreadsheet, "언론사 분류", sync_error_fn)

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
