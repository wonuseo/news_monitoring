"""Pipeline finalization: save logs to Sheets, print summary, completion banner."""


def finalize_pipeline(ctx):
    """Save logs to Sheets, print summary and completion banner."""
    ctx.logger.finalize()

    # Sheets에 run_history 저장
    if ctx.spreadsheet:
        try:
            ctx.logger.save_to_sheets(ctx.spreadsheet)
        except Exception as e:
            ctx.record_error(f"run_history Sheets 저장 실패: {e}", category="sheets_sync")

    # 로그 요약 출력
    ctx.logger.print_summary()

    # 완료 배너
    _print_completion_banner(ctx)


def _print_completion_banner(ctx):
    """Print completion banner."""
    print("\n" + "=" * 80)
    print("✅ 모든 작업 완료!")
    print("=" * 80)
    if ctx.spreadsheet:
        print(f"\n  ☁️  Google Sheets - 동기화 완료 (유일한 저장소)")
    elif not ctx.sheets_required:
        raw_csv = ctx.outdir / "raw.csv"
        print(f"\n  📊 {raw_csv} - raw 수집 결과 (Sheets 미연결 비상 저장)")
        print(f"  ⚠️  분류/분석 결과 없음 (Sheets 연결 필요)")
    print()
