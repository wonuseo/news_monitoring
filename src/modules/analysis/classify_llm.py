"""
classify_llm.py - LLM-Only Classification Orchestrator
병렬 처리 오케스트레이션
"""

import json
from datetime import datetime
from typing import Dict, Optional
import pandas as pd

from .llm_engine import load_prompts, analyze_article_llm
from .llm_orchestrator import run_chunked_parallel
from .result_writer import save_result_to_csv_incremental, sync_result_to_sheets
from src.utils.text_cleaning import clean_bom


def _process_single_article(
    idx: int,
    article: Dict,
    prompts_config: dict,
    openai_key: str
) -> Dict:
    """
    단일 기사 LLM 분석 (병렬 처리용 헬퍼 함수)

    Args:
        idx: DataFrame 인덱스
        article: {"title": ..., "description": ..., "query": ...}
        prompts_config: prompts.yaml 설정
        openai_key: OpenAI API 키

    Returns:
        {"idx": idx, "success": True/False, "result": {...}, "error": ..., "error_type": ...}
    """
    try:
        llm_result = analyze_article_llm(article, prompts_config, openai_key)
        return {
            "idx": idx,
            "success": True,
            "result": llm_result,
            "error": None,
            "error_type": None
        }
    except Exception as e:
        import traceback
        error_type = type(e).__name__
        error_msg = str(e)
        error_trace = traceback.format_exc()
        return {
            "idx": idx,
            "success": False,
            "result": None,
            "error": error_msg,
            "error_type": error_type,
            "error_trace": error_trace
        }


def classify_llm(
    df: pd.DataFrame,
    openai_key: str,
    chunk_size: int = 50,
    dry_run: bool = False,
    max_competitor_classify: int = 50,
    max_workers: int = 3,
    result_csv_path: Optional[str] = None,
    spreadsheet = None,
    raw_df: pd.DataFrame = None
) -> tuple[pd.DataFrame, Dict]:
    """
    LLM-only 분류 메인 함수 (병렬 처리 최적화)

    프로세스:
    1. 분류 대상 선택 (우리 브랜드 전체 + 경쟁사 상위 N개 또는 전체)
    2. LLM 분석 (ThreadPoolExecutor로 병렬 처리)
    3. 성공한 기사는 즉시 result.csv에 append
    4. 각 청크 완료 시 Google Sheets 동기화

    Args:
        df: 입력 DataFrame (title, description, query 필수)
        openai_key: OpenAI API 키
        chunk_size: 배치 크기 (청크당 기사 수)
        dry_run: True면 LLM 호출 생략
        max_competitor_classify: 경쟁사별 최대 분류 개수 (0이면 무제한)
        max_workers: 병렬 처리 워커 수 (기본값: 3)
        result_csv_path: 결과 저장 CSV 경로 (None이면 저장하지 않음)
        spreadsheet: Google Sheets 객체 (청크별 동기화용, 선택사항)
        raw_df: 원본 DataFrame (Sheets 동기화용, 선택사항)

    Returns:
        (분석 결과가 추가된 DataFrame, 메트릭 딕셔너리)
    """
    df = df.copy()

    # 메트릭 초기화
    metrics = {
        "articles_classified_llm": 0,
        "llm_api_calls": 0,
        "classification_errors": 0,
        "press_releases_skipped": 0,
    }

    # 초기화: 모든 결과 컬럼
    result_columns = [
        "brand_relevance",
        "brand_relevance_query_keywords",
        "sentiment_stage",
        "danger_level",
        "issue_category",
        "news_category",
        "news_keyword_summary"
    ]

    # 컬럼 초기화 (이미 존재하는 값은 보존)
    for col in result_columns:
        if col not in df.columns:
            df[col] = None

    if "classified_at" not in df.columns:
        df["classified_at"] = ""

    # Load prompts config
    print("\n🔧 LLM 분류 시작...")
    prompts_config = load_prompts()

    if dry_run:
        print("🔬 DRY RUN 모드: LLM 분석 생략")
        return df, metrics

    # ========================================
    # STEP 1: 분류 대상 선택
    # ========================================
    print("\n[1/2] 분류 대상 선택 중...")

    def already_classified(row) -> bool:
        value = clean_bom(row.get("classified_at", ""))
        return isinstance(value, str) and len(value.strip()) > 1

    def has_article_id(row) -> bool:
        value = clean_bom(row.get("article_id", ""))
        return isinstance(value, str) and len(value.strip()) >= 6

    indices_to_classify = []
    for idx, row in df.iterrows():
        if already_classified(row):
            continue
        if not has_article_id(row):
            continue
        indices_to_classify.append(idx)

    # 스킵된 기사 통계 (보도자료 전처리 등)
    skipped_count = sum(1 for _, row in df.iterrows() if already_classified(row))

    metrics["press_releases_skipped"] = skipped_count

    if len(indices_to_classify) == 0:
        if skipped_count > 0:
            print(f"ℹ️  전체 {len(df)}개 기사 중 {skipped_count}개는 이미 분류됨 (보도자료 등)")
            print("⚠️  LLM 분류 대상 없음")
        else:
            print("⚠️  분류 대상 없음")
        return df, metrics

    print(f"  선택된 기사: {len(indices_to_classify)}개")
    if skipped_count > 0:
        print(f"  - 스킵: {skipped_count}개 (이미 분류됨)")

    # ========================================
    # STEP 2: LLM 분석 (병렬 처리)
    # ========================================
    print(f"\n[2/2] LLM 분석 중 (청크 크기: {chunk_size}, 워커: {max_workers})...")

    timestamp = datetime.now().isoformat()
    # 실행 태스크 구성
    tasks = []
    for idx in indices_to_classify:
        article = {
            "query": df.at[idx, "query"],
            "group": df.at[idx, "group"] if "group" in df.columns else "",
            "title": df.at[idx, "title"],
            "description": df.at[idx, "description"]
        }
        tasks.append({"task_id": idx, "idx": idx, "article": article})

    def worker(task: Dict) -> Dict:
        return _process_single_article(
            task["idx"],
            task["article"],
            prompts_config,
            openai_key
        )

    def on_success(result: Dict):
        idx = result["idx"]
        llm_result = result["result"] or {}

        # LLM 결과를 DataFrame에 병합
        for col in result_columns:
            if col in llm_result:
                value = llm_result[col]
                # JSON serializable 타입으로 변환
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False)
                # 모든 BOM 및 invisible 문자 제거
                if isinstance(value, str):
                    value = clean_bom(value)
                df.at[idx, col] = value

        df.at[idx, "classified_at"] = timestamp

        # 성공한 기사 즉시 CSV에 저장
        save_result_to_csv_incremental(df, idx, result_csv_path)

    def on_failure(result: Dict, _chunk_fail_count: int, total_fail_count: int):
        idx = result.get("idx")
        if idx is None:
            idx = result.get("task_id")
        if idx is None:
            return

        # Fallback: 빈 값 사용
        for col in result_columns:
            df.at[idx, col] = ""
        df.at[idx, "classified_at"] = timestamp

        # 실패 에러 메시지 출력
        title = df.at[idx, 'title'] if pd.notna(df.at[idx, 'title']) else ""
        print(f"\n❌ 분류 실패 [idx={idx}]:")
        print(f"   제목: {title[:80]}...")
        print(f"   에러 타입: {result.get('error_type', 'Unknown')}")
        print(f"   에러 메시지: {result.get('error', 'Unknown error')}")
        # Traceback은 너무 길어서 첫 실패만 출력
        if total_fail_count == 1 and 'error_trace' in result:
            print(f"   상세 Traceback (첫 실패만):\n{result['error_trace']}")

    def sync_callback(_chunk_num: int, _total_chunks: int, _succ: int, _fail: int):
        if spreadsheet and result_csv_path and raw_df is not None:
            sync_result_to_sheets(result_csv_path, raw_df, spreadsheet, verbose=True)

    run_stats = run_chunked_parallel(
        tasks=tasks,
        worker_fn=worker,
        on_success=on_success,
        on_failure=on_failure,
        chunk_size=chunk_size,
        max_workers=max_workers,
        progress_desc="LLM 분석",
        unit="기사",
        inter_chunk_sleep=0.5,
        sync_callback=sync_callback,
    )

    # 전체 통계 출력
    total_processed = run_stats["processed"]
    total_success = run_stats["success"]
    total_failed = run_stats["failed"]
    overall_success_rate = (total_success / total_processed * 100) if total_processed > 0 else 0

    print(f"\n✅ LLM 분류 완료:")
    print(f"   총 처리: {total_processed}개 기사")
    print(f"   성공: {total_success}개 ({overall_success_rate:.1f}%)")
    print(f"   실패: {total_failed}개 ({100-overall_success_rate:.1f}%)")

    # 메트릭 업데이트
    metrics["articles_classified_llm"] = total_success
    metrics["llm_api_calls"] = total_success  # 1 article = 1 API call
    metrics["classification_errors"] = total_failed

    # 비용 추정 (gpt-5-nano: $0.00015/1K input tokens, $0.0006/1K output tokens)
    # 평균: ~1500 input tokens/article, ~300 output tokens/article
    avg_input_tokens_per_article = 1500
    avg_output_tokens_per_article = 300
    input_cost = (total_success * avg_input_tokens_per_article / 1000) * 0.00015
    output_cost = (total_success * avg_output_tokens_per_article / 1000) * 0.0006
    metrics["llm_cost_estimated"] = round(input_cost + output_cost, 4)

    return df, metrics
