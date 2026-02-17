"""Step 5: Flush logger to Google Sheets (run_history / errors / events).

Sync already completed during collection (Step 1) and classification (Step 3).
This step only finalises the run_history schema metrics and flushes all
buffered log entries to Sheets.
"""


def run_sheets_sync(ctx):
    """Flush logger to Sheets. No data sync (already done in Steps 1 & 3)."""
    ctx.current_stage = "sheets_sync"
    ctx.logger.start_stage("sheets_sync")
    print("\n" + "=" * 80)
    print("STEP 5: 로그 저장 (run_history / errors / events)")
    print("=" * 80)

    ctx.logger.log_dict({
        "sheets_sync_enabled": bool(ctx.spreadsheet),
        "sheets_rows_uploaded_raw": ctx.logger.metrics.get("sheets_rows_uploaded_raw", 0),
        "sheets_rows_uploaded_result": ctx.logger.metrics.get("sheets_rows_uploaded_result", 0),
    })

    ctx.logger.end_stage("sheets_sync")
    ctx.logger.log_event("sheets_sync_completed", {
        "sheets_rows_uploaded_raw": ctx.logger.metrics.get("sheets_rows_uploaded_raw", 0),
        "sheets_rows_uploaded_result": ctx.logger.metrics.get("sheets_rows_uploaded_result", 0),
    }, category="sheets_sync", stage="sheets_sync")

    if ctx.spreadsheet:
        ctx.logger.flush_all_to_sheets(ctx.spreadsheet)
        print("✅ run_history / errors / events 저장 완료")
