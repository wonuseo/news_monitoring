"""
classify_press_releases.py - Press Release LLM Classification with Result Sharing
보도자료 대표 기사 선정 -> LLM 분석 -> 클러스터 내 결과 공유
"""

import json
from datetime import datetime
from typing import Dict, Optional, Tuple  # Optional kept for internal use

import pandas as pd

from .llm_engine import load_prompts, analyze_article_llm, analyze_article_negative_llm
from .task_runner import run_chunked_parallel
from src.modules.export.sheets import sync_result_to_sheets

EMPTY_PR_METRICS = {
    "pr_clusters_analyzed": 0,
    "pr_articles_propagated": 0,
    "pr_llm_success": 0,
    "pr_llm_failed": 0,
    "pr_cost_estimated": 0,
}


def select_representative_articles(df: pd.DataFrame) -> Dict[str, int]:
    """
    클러스터별 대표 기사 선정 (earliest pub_datetime)

    선정 기준:
    - cluster_id별로 가장 빠른 pub_datetime 기사 선택
    - pub_datetime 동일 시 DataFrame 순서 우선
    - null/NaT는 후순위

    Args:
        df: cluster_summary와 cluster_id가 있는 DataFrame

    Returns:
        {cluster_id: representative_index} 딕셔너리
    """
    # 보도자료 필터링
    pr_mask = (
        (df["source"] == "보도자료") &
        (df["cluster_id"].notna()) &
        (df["cluster_id"] != "")
    )
    df_pr = df[pr_mask].copy()

    if len(df_pr) == 0:
        return {}

    # pub_datetime 파싱 (실패 시 NaT)
    df_pr["pub_datetime_parsed"] = pd.to_datetime(df_pr["pub_datetime"], errors="coerce")

    representatives = {}
    for cluster_id, group in df_pr.groupby("cluster_id"):
        # 날짜순 정렬 (NaT는 마지막, 같은 날짜면 원본 순서)
        sorted_group = group.sort_values(
            by=["pub_datetime_parsed"],
            na_position="last"
        )
        representative_idx = sorted_group.index[0]
        representatives[cluster_id] = representative_idx

    return representatives


def classify_press_releases(
    df: pd.DataFrame,
    openai_key: str,
    chunk_size: int = 50,
    max_workers: int = 3,
    spreadsheet=None,
    raw_df: pd.DataFrame = None,
    reasoning_collector=None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    보도자료 LLM 분류 (대표 기사 분석 -> 클러스터 내 공유)

    처리 흐름:
    1. 보도자료 필터링 (source="보도자료", cluster_id 존재)
    2. 클러스터별 대표 기사 선정 (earliest pub_datetime)
    3. 대표 기사만 LLM 분석 (analyze_article_llm)
    4. 분류 결과를 같은 cluster_id 전체에 복사
    5. classified_at 타임스탬프 기록

    Args:
        df: 전처리된 DataFrame (cluster_id, source 컬럼 포함)
        openai_key: OpenAI API 키
        chunk_size: 청크 크기
        max_workers: 병렬 워커 수 (기본값: 3)
        spreadsheet: Google Sheets 객체 (청크 동기화용)
        raw_df: 원본 raw DataFrame (Sheets 동기화용)

    Returns:
        (df, pr_metrics) 튜플
    """
    df = df.copy()

    if "cluster_id" not in df.columns:
        print("⚠️  cluster_id 컬럼이 없습니다. 보도자료 분류 스킵")
        return df, EMPTY_PR_METRICS.copy()

    pr_mask = (
        (df["source"] == "보도자료") &
        (df["cluster_id"].notna()) &
        (df["cluster_id"] != "")
    )
    pr_count = pr_mask.sum()

    if pr_count == 0:
        print("ℹ️  보도자료가 없습니다. 분류 스킵")
        return df, EMPTY_PR_METRICS.copy()

    print(f"\n📋 보도자료 LLM 분류 시작: {pr_count}개 기사")

    representatives = select_representative_articles(df)
    print(f"  - 클러스터 수: {len(representatives)}개")

    try:
        prompts_config = load_prompts()
    except Exception as e:
        print(f"⚠️  prompts.yaml 로드 실패: {e}")
        return df, EMPTY_PR_METRICS.copy()

    # 1차 분류 결과 컬럼 (prompts.yaml 담당)
    result_columns = [
        "brand_relevance",
        "brand_relevance_query_keywords",
        "sentiment_stage",
        "news_category",
        "news_keyword_summary",
        "classified_at",
    ]
    for col in result_columns:
        if col not in df.columns:
            df[col] = ""

    # 2차 분류 결과 컬럼 (negative_prompts.yaml 담당) — 별도 초기화
    for col in ["danger_level", "issue_category"]:
        if col not in df.columns:
            df[col] = ""

    timestamp = datetime.now().isoformat()
    propagated_count = 0

    cluster_to_target_indices = {}
    for idx in df[pr_mask].index:
        cluster_id = df.at[idx, "cluster_id"]
        cluster_to_target_indices.setdefault(cluster_id, []).append(idx)

    tasks = []
    for cluster_id, rep_idx in representatives.items():
        if cluster_id not in cluster_to_target_indices:
            continue

        row = df.loc[rep_idx]
        article = {
            "query": row.get("query", ""),
            "group": row.get("group", ""),
            "title": row.get("title", ""),
            "description": row.get("description", ""),
        }
        tasks.append({
            "task_id": cluster_id,
            "cluster_id": cluster_id,
            "idx": rep_idx,
            "article": article,
        })

    if len(tasks) == 0:
        print("ℹ️  보도자료 대표 기사 분석 대상이 없습니다")
        return df, EMPTY_PR_METRICS.copy()

    print(f"  - LLM 호출 예정: {len(tasks)}회")
    print(f"  - 분석 중 (청크 크기: {chunk_size}, 워커: {max_workers})")

    def apply_cluster_result(cluster_id: str, result: Optional[Dict]):
        nonlocal propagated_count
        target_indices = cluster_to_target_indices.get(cluster_id, [])
        if not target_indices:
            return

        for idx in target_indices:
            if result:
                df.at[idx, "brand_relevance"] = result.get("brand_relevance", "")
                keywords = result.get("brand_relevance_query_keywords", [])
                df.at[idx, "brand_relevance_query_keywords"] = json.dumps(keywords, ensure_ascii=False)
                df.at[idx, "sentiment_stage"] = result.get("sentiment_stage", "")
                df.at[idx, "news_category"] = result.get("news_category", "")
                df.at[idx, "news_keyword_summary"] = result.get("news_keyword_summary", "")
            else:
                df.at[idx, "brand_relevance"] = ""
                df.at[idx, "brand_relevance_query_keywords"] = "[]"
                df.at[idx, "sentiment_stage"] = ""
                df.at[idx, "news_category"] = ""
                df.at[idx, "news_keyword_summary"] = ""

            df.at[idx, "classified_at"] = timestamp
            propagated_count += 1

    def worker(task: Dict) -> Dict:
        cluster_id = task["cluster_id"]
        try:
            result = analyze_article_llm(task["article"], prompts_config, openai_key)
            return {
                "task_id": cluster_id,
                "cluster_id": cluster_id,
                "idx": task["idx"],
                "success": bool(result),
                "result": result,
                "error": None if result else "Empty LLM response",
                "error_type": None if result else "EmptyResponse",
            }
        except Exception as e:
            import traceback
            return {
                "task_id": cluster_id,
                "cluster_id": cluster_id,
                "idx": task["idx"],
                "success": False,
                "result": None,
                "error": str(e),
                "error_type": type(e).__name__,
                "error_trace": traceback.format_exc(),
            }

    def on_success(result: Dict):
        cluster_result = result.get("result") or {}
        # 대표 기사 reasoning 수집 (_reasoning은 클러스터 전파에서 제외)
        _reasoning = cluster_result.pop("_reasoning", None)
        if reasoning_collector is not None and _reasoning:
            rep_idx = result.get("idx")
            if rep_idx is not None and "article_id" in df.columns:
                article_id = df.at[rep_idx, "article_id"]
                reasoning_collector.add_first_pass(str(article_id), timestamp, _reasoning)
        apply_cluster_result(result["cluster_id"], cluster_result if cluster_result else None)

    def on_failure(result: Dict, _chunk_fail_count: int, total_fail_count: int):
        cluster_id = result.get("cluster_id", result.get("task_id", "unknown"))
        rep_idx = result.get("idx")
        title = ""
        if rep_idx is not None and rep_idx in df.index:
            title = str(df.at[rep_idx, "title"]) if pd.notna(df.at[rep_idx, "title"]) else ""

        apply_cluster_result(cluster_id, None)
        print(f"\n❌ 보도자료 대표기사 분류 실패 [cluster={cluster_id}]:")
        print(f"   제목: {title[:80]}...")
        print(f"   에러 타입: {result.get('error_type', 'Unknown')}")
        print(f"   에러 메시지: {result.get('error', 'Unknown error')}")
        if total_fail_count == 1 and "error_trace" in result:
            print(f"   상세 Traceback (첫 실패만):\n{result['error_trace']}")

    def sync_callback(_chunk_num: int, _total_chunks: int, _succ: int, _fail: int):
        if spreadsheet and raw_df is not None:
            sync_result_to_sheets(df, raw_df, spreadsheet, verbose=True)

    run_stats = run_chunked_parallel(
        tasks=tasks,
        worker_fn=worker,
        on_success=on_success,
        on_failure=on_failure,
        chunk_size=chunk_size,
        max_workers=max_workers,
        progress_desc="보도자료 대표 LLM 분석",
        unit="클러스터",
        inter_chunk_sleep=0.5,
        sync_callback=sync_callback,
    )

    print(f"✅ 보도자료 1차 분류 완료: {propagated_count}개")
    print("  - 클러스터 기반 결과 공유")
    print(f"  - 대표 기사 LLM 분석: 성공 {run_stats['success']}개, 실패 {run_stats['failed']}개")

    # ========================================
    # Negative 2차 정밀 분석 (danger_level / issue_category)
    # ========================================
    negative_columns = ["danger_level", "issue_category"]
    negative_clusters = [
        cluster_id for cluster_id, rep_idx in representatives.items()
        if df.at[rep_idx, "sentiment_stage"] in ("부정 후보", "부정 확정")
    ]

    negative_success = 0
    if negative_clusters:
        print(f"\n[보도자료 Negative Pass] {len(negative_clusters)}개 클러스터 정밀 분석 중...")

        negative_tasks = []
        for cluster_id in negative_clusters:
            rep_idx = representatives[cluster_id]
            row = df.loc[rep_idx]
            article = {
                "query": row.get("query", ""),
                "group": row.get("group", ""),
                "title": row.get("title", ""),
                "description": row.get("description", ""),
            }
            initial_result = {
                "brand_relevance": df.at[rep_idx, "brand_relevance"],
                "sentiment_stage": df.at[rep_idx, "sentiment_stage"],
            }
            negative_tasks.append({
                "task_id": cluster_id,
                "cluster_id": cluster_id,
                "idx": rep_idx,
                "article": article,
                "initial_result": initial_result,
            })

        def negative_worker(task: Dict) -> Dict:
            cluster_id = task["cluster_id"]
            try:
                result = analyze_article_negative_llm(
                    task["article"], task["initial_result"], openai_key
                )
                return {
                    "task_id": cluster_id,
                    "cluster_id": cluster_id,
                    "success": bool(result),
                    "result": result,
                    "error": None if result else "Empty LLM response",
                }
            except Exception as e:
                import traceback
                return {
                    "task_id": cluster_id,
                    "cluster_id": cluster_id,
                    "success": False,
                    "result": None,
                    "error": str(e),
                    "error_trace": traceback.format_exc(),
                }

        def on_negative_success(result: Dict):
            nonlocal negative_success
            cluster_id = result["cluster_id"]
            neg_result = result.get("result") or {}

            # 대표 기사 2차 reasoning 수집
            _neg_reasoning = neg_result.pop("_reasoning", None)
            if reasoning_collector is not None and _neg_reasoning:
                rep_idx = representatives.get(cluster_id)
                if rep_idx is not None and "article_id" in df.columns:
                    article_id = df.at[rep_idx, "article_id"]
                    reasoning_collector.update_second_pass(str(article_id), _neg_reasoning)

            for idx in cluster_to_target_indices.get(cluster_id, []):
                for col in negative_columns:
                    if col in neg_result:
                        df.at[idx, col] = neg_result[col] if neg_result[col] is not None else ""
            negative_success += 1

        def on_negative_failure(result: Dict, _chunk_fail_count: int, _total_fail_count: int):
            pass  # 실패 시 1차 결과 유지

        run_chunked_parallel(
            tasks=negative_tasks,
            worker_fn=negative_worker,
            on_success=on_negative_success,
            on_failure=on_negative_failure,
            chunk_size=chunk_size,
            max_workers=max_workers,
            progress_desc="보도자료 부정 기사 정밀 분석",
            unit="클러스터",
            inter_chunk_sleep=0.5,
        )

        print(f"✅ 보도자료 Negative Pass 완료: {negative_success}/{len(negative_clusters)}개 클러스터 정밀 분석")

    pr_metrics = {
        "pr_clusters_analyzed": len(tasks),
        "pr_articles_propagated": propagated_count,
        "pr_llm_success": run_stats["success"],
        "pr_llm_failed": run_stats["failed"],
        "pr_negative_clusters": len(negative_clusters),
        "pr_negative_success": negative_success,
        "pr_cost_estimated": 0,  # Cost tracking can be added later
    }

    return df, pr_metrics
