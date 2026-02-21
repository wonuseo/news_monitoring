"""
source_verifier.py - Source Verification & Topic Grouping (진입점)

세 모듈을 순서대로 실행하는 얇은 오케스트레이터:
  - cluster_verifier   : Part A  — 보도자료 클러스터 LLM 검증
  - cross_query_merger : Part A-2 — Cross-query 클러스터 병합
  - topic_grouper      : Part B  — 비클러스터 기사 주제 그룹화

Source Labels:
  - 보도자료: 브랜드 공식 배포 보도자료 (LLM 클러스터 검증으로 판단)
  - 유사주제: 독립 기사, 같은 주제 (클러스터됐지만 보도자료 기준 미달)
  - 일반기사: 독립 기사 (기본값)
"""

import os
from typing import Dict, Tuple

import pandas as pd

from .cluster_verifier import llm_verify_cluster, verify_press_release_clusters
from .cross_query_merger import merge_cross_query_clusters
from .topic_grouper import discover_topic_groups, llm_verify_topic_similarity
from ._sv_common import llm_judge_component_representative, determine_verified_source

__all__ = [
    "verify_and_regroup_sources",
    "llm_verify_cluster",
    "verify_press_release_clusters",
    "merge_cross_query_clusters",
    "discover_topic_groups",
    "llm_verify_topic_similarity",
    "llm_judge_component_representative",
    "determine_verified_source",
]


def verify_and_regroup_sources(
    df: pd.DataFrame,
    openai_key: str = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Entry point: Part A → A-2 → B 순서 실행 + 일관성 검증.

    Args:
        df: 분류 완료된 DataFrame
        openai_key: OpenAI API 키 (None이면 환경변수 fallback)

    Returns:
        (df, combined_stats) 튜플
    """
    if openai_key is None:
        openai_key = os.getenv("OPENAI_API_KEY")

    print("\n📋 Source 검증 및 주제 그룹화 시작")

    # Part A: 보도자료 클러스터 LLM 검증
    df, verify_stats = verify_press_release_clusters(df, openai_key=openai_key)
    print(f"  Part A: 보도자료 검증 - {verify_stats['sv_clusters_verified']}개 클러스터 검증")
    if verify_stats["sv_clusters_verified"] > 0:
        print(f"    - 보도자료 유지: {verify_stats['sv_kept_press_release']}개")
        print(f"    - 유사주제 재분류: {verify_stats['sv_reclassified_similar_topic']}개")

    # Part A-2: Cross-query 클러스터 병합
    df, cross_stats = merge_cross_query_clusters(df, openai_key=openai_key)
    print(f"  Part A-2: Cross-query 병합 - {cross_stats['sv_cross_merged_groups']}개 그룹, "
          f"{cross_stats['sv_cross_merged_articles']}개 기사")

    # Part B: 비클러스터 기사 주제 그룹화
    df, topic_stats = discover_topic_groups(df, openai_key=openai_key)
    print(f"  Part B: 주제 그룹 발견 - {topic_stats['sv_new_topic_groups']}개 그룹, "
          f"{topic_stats['sv_new_topic_articles']}개 기사")
    if topic_stats.get("sv_llm_verified", 0) > 0 or topic_stats.get("sv_llm_rejected", 0) > 0:
        print(f"    - LLM 경계선 검증: {topic_stats['sv_llm_verified']}개 연결, "
              f"{topic_stats['sv_llm_rejected']}개 거부")

    combined = {**verify_stats, **cross_stats, **topic_stats}

    # 일관성 검증
    if "cluster_id" in df.columns and "cluster_summary" in df.columns:
        # Rule 1: source=="일반기사" → cluster_id="", cluster_summary=""
        general_mask = df["source"] == "일반기사"
        r1_count = (general_mask & (df["cluster_id"].astype(str).str.strip() != "")).sum()
        if r1_count > 0:
            df.loc[general_mask, "cluster_id"] = ""
            df.loc[general_mask, "cluster_summary"] = ""
            print(f"  일관성 Rule 1: 일반기사 {r1_count}건의 cluster_id/cluster_summary 초기화")

        # Rule 2: 같은 cluster_id → 같은 cluster_summary (첫 비어있지 않은 값 전파)
        clustered = df["cluster_id"].astype(str).str.strip() != ""
        r2_count = 0
        if clustered.any():
            for cid, cgroup in df[clustered].groupby("cluster_id"):
                summaries = cgroup["cluster_summary"].dropna().astype(str)
                summaries = summaries[summaries.str.strip() != ""]
                if len(summaries) > 0:
                    first_val = summaries.iloc[0]
                    mismatch = clustered & (df["cluster_id"] == cid) & (df["cluster_summary"] != first_val)
                    fix_count = mismatch.sum()
                    if fix_count > 0:
                        df.loc[mismatch, "cluster_summary"] = first_val
                        r2_count += fix_count
        if r2_count > 0:
            print(f"  일관성 Rule 2: {r2_count}건의 cluster_summary 전파")

        # Rule 3: cluster_id 있는데 cluster_summary 없으면 경고
        r3_mask = clustered & (df["cluster_summary"].astype(str).str.strip() == "")
        r3_count = r3_mask.sum()
        if r3_count > 0:
            print(f"  일관성 Rule 3: {r3_count}건 cluster_summary 누락 (summarize_clusters 호출 예정)")

    if "source" in df.columns:
        source_dist = df["source"].value_counts().to_dict()
        print(f"  Source 분포: {source_dist}")

    print("✅ Source 검증 및 주제 그룹화 완료")
    return df, combined
