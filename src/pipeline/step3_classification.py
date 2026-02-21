"""Step 3: LLM classification (PR clusters, general, source verification)."""

import pandas as pd

from src.modules.analysis.classify_press_releases import classify_press_releases
from src.modules.analysis.classify_llm import classify_llm
from src.modules.analysis.source_verifier import verify_and_regroup_sources
from src.modules.analysis.reasoning_writer import ReasoningCollector
from src.modules.processing.press_release_detector import summarize_clusters
from src.modules.processing.looker_prep import add_time_series_columns
from src.modules.analysis.classification_stats import get_classification_stats, print_classification_stats
from src.modules.export.sheets import sync_raw_and_processed


def run_classification(ctx):
    """Run all classification steps. Sets ctx.df_result."""
    if ctx.run_mode == "raw_only":
        return  # Already set in step2

    if ctx.run_mode == "preprocess_only":
        _handle_preprocess_only(ctx)
        return

    # Full classification pipeline
    print("\n" + "=" * 80)
    print("STEP 3: 분류")
    print("=" * 80)

    # ReasoningCollector 초기화 (Sheets가 연결된 경우에만 수집)
    ctx.reasoning_collector = ReasoningCollector() if ctx.spreadsheet else None

    # Step 3-1: 보도자료 LLM 분류 (대표 기사 분석 -> 클러스터 공유)
    _classify_press_releases(ctx)

    # Step 3-2: LLM 분류 (보도자료는 스킵)
    _classify_general(ctx)

    # 통계 출력
    stats = get_classification_stats(ctx.df_processed)
    print_classification_stats(stats)

    # 보도자료 정보 및 언론사 정보 병합
    _merge_source_columns(ctx)

    # Step 3-3: Source 검증 및 주제 그룹화
    if not ctx.args.dry_run:
        _verify_sources(ctx)

        # Step 3-3 이후: 유사주제 클러스터 요약 (이미 요약된 보도자료 클러스터는 스킵)
        ctx.df_processed = summarize_clusters(ctx.df_processed, ctx.env["openai_key"])

    # Step 3.7: Looker Studio 준비 (항상 실행)
    _add_looker_columns(ctx)

    ctx.logger.log_event("classification_completed", {
        "articles_classified_llm": ctx.logger.metrics.get("articles_classified_llm", 0),
        "llm_api_calls": ctx.logger.metrics.get("llm_api_calls", 0),
        "press_releases_skipped": ctx.logger.metrics.get("press_releases_skipped", 0),
        "classification_errors": ctx.logger.metrics.get("classification_errors", 0)
    }, stage="general_classification")
    if ctx.spreadsheet:
        ctx.logger.flush_all_to_sheets(ctx.spreadsheet)

    # 기존 result.csv와 병합
    _merge_with_existing_result(ctx)

    # 결과 저장
    _save_and_sync(ctx)

    # reasoning 탭 업로드 (Sheets 연결 시)
    if ctx.reasoning_collector is not None and ctx.spreadsheet:
        print("\n📝 reasoning 탭 업로드 중...")
        ctx.reasoning_collector.flush_to_sheets(ctx.spreadsheet)


def _classify_press_releases(ctx):
    """Step 3-1: Press release cluster classification"""
    ctx.current_stage = "pr_classification"
    ctx.logger.start_stage("pr_classification")
    ctx.df_processed, pr_metrics = classify_press_releases(
        ctx.df_processed,
        ctx.env["openai_key"],
        chunk_size=ctx.args.chunk_size,
        max_workers=ctx.args.max_workers,
        spreadsheet=ctx.spreadsheet,
        raw_df=ctx.df_raw,
        reasoning_collector=ctx.reasoning_collector,
    )
    ctx.logger.log_dict(pr_metrics)
    ctx.logger.end_stage("pr_classification")
    ctx.logger.log_event("pr_classification_completed", pr_metrics, stage="pr_classification")
    if ctx.spreadsheet:
        ctx.logger.flush_all_to_sheets(ctx.spreadsheet)


def _classify_general(ctx):
    """Step 3-2: General LLM classification"""
    ctx.current_stage = "general_classification"
    ctx.logger.start_stage("general_classification")
    ctx.df_processed, llm_metrics = classify_llm(
        ctx.df_processed,
        ctx.env["openai_key"],
        chunk_size=ctx.args.chunk_size,
        dry_run=ctx.args.dry_run,
        max_competitor_classify=ctx.args.max_competitor_classify,
        max_workers=ctx.args.max_workers,
        spreadsheet=ctx.spreadsheet,
        raw_df=ctx.df_raw,
        reasoning_collector=ctx.reasoning_collector,
    )
    ctx.logger.log_dict(llm_metrics)
    ctx.logger.end_stage("general_classification")


def _merge_source_columns(ctx):
    """Merge source/media columns from preprocessing"""
    source_cols = ['link', 'source', 'cluster_id', 'cluster_summary', 'media_domain', 'media_name', 'media_group',
                   'media_type']
    source_data = ctx.df_processed[source_cols].copy()

    # 기존 컬럼 제거 (중복 방지)
    cols_to_drop = [col for col in ctx.df_processed.columns if col in source_cols and col != 'link']
    ctx.df_processed = ctx.df_processed.drop(columns=cols_to_drop, errors='ignore')

    # merge
    ctx.df_processed = ctx.df_processed.merge(source_data, on='link', how='left')

    # 나머지 NaN -> 공란 변환 (FutureWarning 방지)
    ctx.df_processed = ctx.df_processed.fillna("").infer_objects(copy=False)


def _verify_sources(ctx):
    """Step 3-3: Source verification and topic grouping"""
    ctx.current_stage = "source_verification"
    ctx.logger.start_stage("source_verification")
    ctx.df_processed, sv_metrics = verify_and_regroup_sources(ctx.df_processed, openai_key=ctx.env["openai_key"])
    ctx.logger.log_dict(sv_metrics)
    ctx.logger.end_stage("source_verification")
    ctx.logger.log_event("source_verification_completed", sv_metrics, stage="source_verification")
    if ctx.spreadsheet:
        ctx.logger.flush_all_to_sheets(ctx.spreadsheet)


def _add_looker_columns(ctx):
    """Add Looker Studio time-series columns"""
    print("\n🕒 Looker Studio 시계열 컬럼 추가 중...")
    ctx.df_processed = add_time_series_columns(ctx.df_processed)


def _merge_with_existing_result(ctx):
    """Merge with existing result from Sheets total_result"""
    if ctx.spreadsheet:
        try:
            worksheet = ctx.spreadsheet.worksheet("total_result")
            records = worksheet.get_all_records()
            if records:
                df_result_existing = pd.DataFrame(records)
                ctx.df_result = pd.concat([df_result_existing, ctx.df_processed], ignore_index=True)
                ctx.df_result = ctx.df_result.drop_duplicates(subset=['link'], keep='last')
                print(f"\n📂 Sheets total_result 업데이트: {len(df_result_existing)} + {len(ctx.df_processed)} = {len(ctx.df_result)}개 기사 (중복 제거 후)")
                return
        except Exception as e:
            ctx.record_error(f"기존 결과 로드 실패 (Sheets): {e}", category="sheets_sync")

    ctx.df_result = ctx.df_processed


def _save_and_sync(ctx):
    """Sync result to Google Sheets"""
    # pub_datetime을 명시적으로 datetime으로 변환 (타입 불일치 방지)
    if 'pub_datetime' in ctx.df_result.columns:
        ctx.df_result['pub_datetime'] = pd.to_datetime(ctx.df_result['pub_datetime'], errors='coerce')

    # Google Sheets 동기화 (분류 완료 직후)
    if ctx.spreadsheet:
        print("\n📊 Google Sheets 동기화 중 (분류 결과)...")
        try:
            sync_results = sync_raw_and_processed(ctx.df_raw, ctx.df_result, ctx.spreadsheet)
            result_sync = sync_results.get("total_result", {})
            ctx.logger.log_dict({
                "sheets_rows_uploaded_result": result_sync.get("added", 0) + result_sync.get("updated", 0),
            })
            print("✅ 분류 결과 Sheets 동기화 완료")
        except Exception as e:
            ctx.record_error(f"분류 결과 Sheets 동기화 실패: {e}", category="sheets_sync")


def _handle_preprocess_only(ctx):
    """Handle preprocess_only mode: add looker columns, merge, sync to Sheets"""
    # 나머지 NaN → 공란 변환
    ctx.df_processed = ctx.df_processed.fillna("")

    # Step 2-6: Looker Studio time-series columns
    print("\n🕒 Looker Studio 시계열 컬럼 추가 중...")
    ctx.df_processed = add_time_series_columns(ctx.df_processed)

    # 기존 Sheets total_result와 병합
    if ctx.spreadsheet:
        try:
            worksheet = ctx.spreadsheet.worksheet("total_result")
            records = worksheet.get_all_records()
            if records:
                df_result_existing = pd.DataFrame(records)
                ctx.df_result = pd.concat([df_result_existing, ctx.df_processed], ignore_index=True)
                ctx.df_result = ctx.df_result.drop_duplicates(subset=['link'], keep='last')
                print(f"📂 Sheets total_result 업데이트: {len(df_result_existing)} + {len(ctx.df_processed)} = {len(ctx.df_result)}개 기사")
            else:
                ctx.df_result = ctx.df_processed
        except Exception:
            ctx.df_result = ctx.df_processed
    else:
        ctx.df_result = ctx.df_processed

    # total_result_count
    ctx.logger.log("total_result_count", len(ctx.df_result))

    # Google Sheets 동기화 (전처리 완료 직후)
    if ctx.spreadsheet:
        print("\n📊 Google Sheets 동기화 중 (전처리 결과)...")
        try:
            sync_raw_and_processed(ctx.df_raw, ctx.df_result, ctx.spreadsheet)
            print("✅ 전처리 결과 Sheets 동기화 완료")
        except Exception as e:
            ctx.record_error(f"전처리 결과 Sheets 동기화 실패: {e}", category="sheets_sync")
