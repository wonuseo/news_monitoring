"""Step 1: News collection, reprocess check, date filtering."""

import pandas as pd

from src.modules.collection.collect import OUR_BRANDS, COMPETITORS, collect_all_news
from src.modules.processing.reprocess_checker import (
    check_reprocess_targets,
    load_raw_data_from_sheets,
    clear_classified_at_for_targets,
    print_reprocess_stats,
)
from src.modules.export.sheets import (
    load_existing_links_from_sheets,
    filter_new_articles_from_sheets,
    sync_raw_and_processed,
)


def run_collection(ctx) -> bool:
    """
    Collect news, check reprocess targets, filter by date.

    Sets ctx.df_raw, ctx.df_raw_new, ctx.df_to_process.
    Returns False if no articles to process (triggers early exit with sync).
    """
    ctx.current_stage = "collection"
    ctx.logger.start_stage("collection")
    print("\n" + "=" * 80)
    print("STEP 1: 뉴스 수집")
    print("=" * 80)

    # Load existing links from Sheets (for deduplication)
    existing_links = set()
    if ctx.spreadsheet:
        existing_links = load_existing_links_from_sheets(ctx.spreadsheet)

    if ctx.args.recheck_only:
        _collect_recheck_only(ctx)
    else:
        _collect_from_api(ctx, existing_links)

    # STEP 1.5a: Sheets 탭 내 중복 행 제거
    _dedup_sheets(ctx)

    # STEP 1.5: 재처리 대상 검사
    _reprocess_check(ctx)

    # STEP 1.6: 2026-02-01 이후 기사만 필터링
    _date_filter(ctx)

    if len(ctx.df_to_process) == 0:
        _handle_no_articles(ctx)
        return False

    return True


def _collect_recheck_only(ctx):
    """--recheck_only 모드: API 수집 생략, Sheets에서 raw_data 로드"""
    if not ctx.spreadsheet:
        print("❌ --recheck_only 사용 시 Google Sheets 연결이 필요합니다.")
        print("  .env 파일에 GOOGLE_SHEETS_CREDENTIALS_PATH와 GOOGLE_SHEET_ID를 설정하세요.")
        raise RuntimeError("--recheck_only requires Google Sheets connection")

    print("\n📋 --recheck_only 모드: API 수집 생략, Sheets raw_data 로드")
    ctx.df_raw = load_raw_data_from_sheets(ctx.spreadsheet)
    if len(ctx.df_raw) == 0:
        print("❌ Sheets raw_data가 비어있어 재처리할 수 없습니다.")
        raise RuntimeError("--recheck_only requires non-empty raw_data in Sheets")

    ctx.df_raw_new = pd.DataFrame(columns=ctx.df_raw.columns)

    ctx.logger.log_dict({
        "articles_collected_total": 0,
        "articles_collected_per_query": {},
        "existing_links_skipped": 0,
    })
    ctx.logger.end_stage("collection")


def _collect_from_api(ctx, existing_links):
    """API 방식 수집 (Sheets 기반 중복 체크)"""
    ctx.df_raw_new = collect_all_news(
        OUR_BRANDS, COMPETITORS,
        ctx.args.display, ctx.args.max_api_pages, ctx.args.sort,
        ctx.env["naver_id"], ctx.env["naver_secret"],
        existing_links=existing_links,
        spreadsheet=ctx.spreadsheet
    )

    # API 수집 결과 확인
    if len(ctx.df_raw_new) == 0:
        print("\nℹ️  API에서 수집된 새로운 기사가 없습니다.")
    else:
        print(f"\n✅ API에서 {len(ctx.df_raw_new)}개 기사 수집 완료")

    # Filter new articles (skip duplicates from Google Sheets)
    existing_links_skipped = 0
    if len(existing_links) > 0 and len(ctx.df_raw_new) > 0:
        before_filter = len(ctx.df_raw_new)
        ctx.df_raw_new = filter_new_articles_from_sheets(ctx.df_raw_new, existing_links)
        existing_links_skipped = before_filter - len(ctx.df_raw_new)

    # df_raw: 전체 (Sheets raw_data + 신규 기사) 메모리에서 병합
    if ctx.spreadsheet:
        df_raw_sheets = load_raw_data_from_sheets(ctx.spreadsheet)
        if len(df_raw_sheets) > 0:
            ctx.df_raw = pd.concat([df_raw_sheets, ctx.df_raw_new], ignore_index=True)
            ctx.df_raw = ctx.df_raw.drop_duplicates(subset=['link'], keep='last')
            print(f"📂 Sheets raw_data + 신규: {len(df_raw_sheets)} + {len(ctx.df_raw_new)} = {len(ctx.df_raw)}개 기사")
        else:
            ctx.df_raw = ctx.df_raw_new
    else:
        ctx.df_raw = ctx.df_raw_new

    # Google Sheets 즉시 동기화 (수집 직후) — _save_immediately에서 이미 동기화됨
    if ctx.spreadsheet and len(ctx.df_raw_new) > 0:
        print("\n📊 Google Sheets 즉시 동기화 중 (raw_data)...")
        try:
            from src.modules.export.sheets import sync_to_sheets
            sync_result = sync_to_sheets(ctx.df_raw, ctx.spreadsheet, "raw_data")
            msg_parts = [
                f"{sync_result.get('attempted', 0)}개 시도",
                f"{sync_result.get('added', 0)}개 추가"
            ]
            if sync_result.get('updated', 0) > 0:
                msg_parts.append(f"{sync_result['updated']}개 업데이트")
            msg_parts.append(f"{sync_result.get('skipped', 0)}개 건너뜀")
            print(f"✅ raw_data 시트 동기화 완료: {', '.join(msg_parts)}")
            ctx.logger.log_dict({
                "sheets_rows_uploaded_raw": sync_result.get("added", 0) + sync_result.get("updated", 0),
            })
            ctx.logger.log_event("sheets_sync_raw_data", sync_result, category="sheets_sync", stage="collection")
        except Exception as e:
            ctx.record_error(f"raw_data 시트 동기화 실패: {e}", category="sheets_sync")

    # 수집 단계 메트릭
    articles_per_query = ctx.df_raw_new.groupby('query').size().to_dict() if 'query' in ctx.df_raw_new.columns else {}
    ctx.logger.log_dict({
        "articles_collected_total": len(ctx.df_raw_new),
        "articles_collected_per_query": articles_per_query,
        "existing_links_skipped": existing_links_skipped
    })
    ctx.logger.log_event("collection_completed", {
        "articles_collected_total": len(ctx.df_raw_new),
        "articles_collected_per_query": articles_per_query,
        "existing_links_skipped": existing_links_skipped
    }, stage="collection")
    if ctx.spreadsheet:
        ctx.logger.flush_all_to_sheets(ctx.spreadsheet)
    ctx.logger.end_stage("collection")


def _dedup_sheets(ctx):
    """STEP 1.5a: Sheets 탭(raw_data, total_result) 내 중복 행 제거."""
    if not ctx.spreadsheet:
        return
    from src.modules.export.sheets import deduplicate_sheet

    print("\n🧹 Sheets 중복 행 검사 중...")
    for tab in ("raw_data", "total_result"):
        try:
            result = deduplicate_sheet(ctx.spreadsheet, tab)
            if result["removed"] > 0:
                print(f"  ✅ {tab}: 중복 {result['removed']}개 제거 ({result['before']} → {result['after']}개)")
            else:
                print(f"  ✅ {tab}: 중복 없음 ({result['after']}개)")
        except Exception as e:
            ctx.record_error(f"{tab} 중복 제거 실패: {e}", category="sheets_sync")


def _reprocess_check(ctx):
    """STEP 1.5: 재처리 대상 검사"""
    try:
        recheck = check_reprocess_targets(ctx.df_raw, ctx.spreadsheet)
        print_reprocess_stats(recheck["stats"])

        df_reprocess = recheck["df_to_reprocess"]
        if len(df_reprocess) > 0:
            df_reprocess = clear_classified_at_for_targets(df_reprocess, recheck["reprocess_links"])
            ctx.df_to_process = pd.concat([ctx.df_raw_new, df_reprocess], ignore_index=True)
            ctx.df_to_process = ctx.df_to_process.drop_duplicates(subset=['link'], keep='first')
        else:
            ctx.df_to_process = ctx.df_raw_new

        # 재처리 메트릭 로깅
        ctx.logger.log_dict({
            "reprocess_targets_total": recheck["stats"]["total_reprocess_targets"],
            "reprocess_missing_from_result": recheck["stats"]["missing_from_result"],
            "reprocess_field_missing": recheck["stats"].get("field_missing", {}),
        })
    except Exception as e:
        ctx.record_error(f"재처리 대상 검사 실패: {e}", category="system")
        ctx.df_to_process = ctx.df_raw_new


def _date_filter(ctx):
    """STEP 1.6: 2026-02-01 이후 기사만 필터링"""
    if len(ctx.df_to_process) > 0 and 'pubDate' in ctx.df_to_process.columns:
        before_date_filter = len(ctx.df_to_process)
        ctx.df_to_process['pub_datetime_temp'] = pd.to_datetime(ctx.df_to_process['pubDate'], errors='coerce')
        ctx.df_to_process = ctx.df_to_process[ctx.df_to_process['pub_datetime_temp'] >= '2026-02-01'].copy()
        ctx.df_to_process = ctx.df_to_process.drop(columns=['pub_datetime_temp'])
        date_filtered = before_date_filter - len(ctx.df_to_process)
        print(f"🔧 날짜 필터링: {date_filtered}개 제외 (2026-02-01 이전), {len(ctx.df_to_process)}개 유지")


def _handle_no_articles(ctx):
    """처리할 기사 없을 때 기존 데이터 Sheets 동기화 후 early exit"""
    print("ℹ️  처리할 신규 기사가 없습니다.")

    # 기존 데이터가 있으면 Google Sheets 동기화 시도
    if ctx.spreadsheet:
        ctx.logger.start_stage("sheets_sync")
        print("\n" + "=" * 80)
        print("STEP 5: Google Sheets 업로드 (기존 데이터)")
        print("=" * 80)
        try:
            # Sheets total_result 로드
            worksheet = ctx.spreadsheet.worksheet("total_result")
            records = worksheet.get_all_records()
            if records:
                df_result_existing = pd.DataFrame(records)
                sync_results = sync_raw_and_processed(ctx.df_raw, df_result_existing, ctx.spreadsheet)
                print("✅ Google Sheets 동기화 완료")

                raw_sync = sync_results.get("raw_data", {})
                result_sync = sync_results.get("total_result", {})
                ctx.logger.log_dict({
                    "sheets_sync_enabled": True,
                    "sheets_rows_uploaded_raw": raw_sync.get("added", 0) + raw_sync.get("updated", 0),
                    "sheets_rows_uploaded_result": result_sync.get("added", 0) + result_sync.get("updated", 0)
                })
            else:
                print("  ℹ️  total_result 시트가 비어있습니다.")
                ctx.logger.log("sheets_sync_enabled", True)
        except Exception as e:
            ctx.record_error(f"Google Sheets 업로드 실패 (기존 데이터): {e}", category="sheets_sync")
            ctx.logger.log("sheets_sync_enabled", False)
        ctx.logger.end_stage("sheets_sync")
        ctx.logger.log_event("sheets_sync_completed", {
            "sheets_rows_uploaded_raw": ctx.logger.metrics.get("sheets_rows_uploaded_raw", 0),
            "sheets_rows_uploaded_result": ctx.logger.metrics.get("sheets_rows_uploaded_result", 0)
        }, category="sheets_sync", stage="sheets_sync")
        ctx.logger.flush_all_to_sheets(ctx.spreadsheet)

    print("\n" + "=" * 80)
    print("✅ 작업 완료 (신규 처리 없음)")
    print("=" * 80)
    # 로그/요약/배너는 main.py에서 finalize_pipeline()이 처리함
