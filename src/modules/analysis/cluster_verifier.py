"""
cluster_verifier.py - Part A: 보도자료 클러스터 LLM 검증

클러스터별 LLM 호출로 보도자료/유사주제 구분.
LLM 실패 시 규칙 기반 fallback (determine_verified_source).
"""

from typing import Dict, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from src.modules.analysis.llm_engine import (
    load_source_verifier_prompts,
    render_prompt,
    call_openai_structured,
)
from ._sv_common import _get_sv_model, determine_verified_source


def llm_verify_cluster(
    cluster_df: pd.DataFrame,
    query: str,
    cluster_summary: str,
    openai_key: str,
) -> Optional[str]:
    """
    LLM으로 클러스터 단위 보도자료/유사주제 검증 (1 cluster = 1 API call).

    Returns:
        "보도자료" / "유사주제" 또는 None (실패 시)
    """
    prompts = load_source_verifier_prompts()
    cv_config = prompts.get("cluster_verification", {})

    system_prompt = cv_config.get("system", "")
    user_template = cv_config.get("user_prompt_template", "")
    response_schema = cv_config.get("response_schema", {})

    if not system_prompt or not user_template:
        return None

    titles = cluster_df["title"].tolist() if "title" in cluster_df.columns else []
    titles_display = titles[:10]
    titles_list = "\n".join(f"- {t}" for t in titles_display)
    if len(titles) > 10:
        titles_list += f"\n- ... 외 {len(titles) - 10}개"

    classified = cluster_df[cluster_df["brand_relevance"].astype(str).str.strip() != ""]
    rep = classified.iloc[0] if len(classified) > 0 else cluster_df.iloc[0]

    context = {
        "query": query,
        "cluster_summary": cluster_summary if cluster_summary else "없음",
        "article_count": str(len(cluster_df)),
        "titles_list": titles_list,
        "brand_relevance": str(rep.get("brand_relevance", "")),
        "sentiment_stage": str(rep.get("sentiment_stage", "")),
        "news_category": str(rep.get("news_category", "")),
    }

    user_prompt = render_prompt(user_template, context)
    model = _get_sv_model()

    try:
        result = call_openai_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=response_schema,
            openai_key=openai_key,
            model=model,
            label="클러스터검증",
            schema_name="cluster_verification_result",
        )
        if result and "source_type" in result:
            return result["source_type"]
        return None
    except Exception as e:
        print(f"  ⚠️  LLM 클러스터 검증 실패: {e}")
        return None


def verify_press_release_clusters(
    df: pd.DataFrame,
    openai_key: str = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Part A: 클러스터 단위 LLM 검증으로 보도자료/유사주제 구분.

    Returns:
        (df, stats) 튜플
    """
    df = df.copy()
    stats = {
        "sv_clusters_verified": 0,
        "sv_kept_press_release": 0,
        "sv_reclassified_similar_topic": 0,
    }

    if "source" not in df.columns:
        return df, stats

    pr_mask = df["source"] == "보도자료"
    if pr_mask.sum() == 0:
        return df, stats

    if "cluster_id" not in df.columns:
        return df, stats

    pr_df = df[pr_mask]
    cluster_groups = list(pr_df.groupby("cluster_id", dropna=False))
    stats["sv_clusters_verified"] = len(cluster_groups)
    total_cl = len(cluster_groups)

    pbar = tqdm(total=total_cl, desc="    PR 클러스터 검증", unit="클러스터", leave=False)

    for cl_idx, (cluster_id, cluster_df) in enumerate(cluster_groups, 1):
        print(f"    [{cl_idx}/{total_cl}] 클러스터 '{cluster_id}' ({len(cluster_df)}개 기사) 검증 중...", end=" ")
        verified_source = None

        if openai_key:
            query = str(cluster_df["query"].iloc[0]) if "query" in cluster_df.columns else ""
            cs = str(cluster_df["cluster_summary"].iloc[0]) if "cluster_summary" in cluster_df.columns else ""
            verified_source = llm_verify_cluster(cluster_df, query, cs, openai_key)

        if verified_source is None:
            rep = cluster_df.iloc[0]
            verified_source = determine_verified_source(
                brand_relevance=str(rep.get("brand_relevance", "")),
                sentiment_stage=str(rep.get("sentiment_stage", "")),
                news_category=str(rep.get("news_category", "")),
                date_spread_days=0,
            )

        for idx in cluster_df.index:
            df.at[idx, "source"] = verified_source

        if verified_source == "보도자료":
            stats["sv_kept_press_release"] += len(cluster_df)
            print("→ 보도자료")
        else:
            stats["sv_reclassified_similar_topic"] += len(cluster_df)
            print("→ 유사주제")

        pbar.set_postfix({
            "보도자료유지": stats["sv_kept_press_release"],
            "유사주제전환": stats["sv_reclassified_similar_topic"],
        })
        pbar.update(1)

    pbar.close()
    return df, stats
