"""
topic_grouper.py - Part B: 비클러스터 일반기사 주제 그룹화

news_keyword_summary 토큰 Jaccard 유사도 + news_category 일치 +
경계선 케이스 LLM 검증으로 같은 주제를 다루는 기사를 그룹화.
"""

import os
from collections import deque
from typing import Dict, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from src.modules.analysis.llm_engine import (
    load_source_verifier_prompts,
    render_prompt,
    call_openai_structured,
)
from ._sv_common import (
    _tokenize_summary,
    _jaccard_similarity,
    _select_representative_index,
    _build_article_payload,
    llm_judge_component_representative,
    _get_sv_model,
    TOPIC_JACCARD_LOW_THRESHOLD,
    TOPIC_JACCARD_HIGH_THRESHOLD,
)


def llm_verify_topic_similarity(
    summary_a: str,
    summary_b: str,
    title_a: Optional[str] = None,
    title_b: Optional[str] = None,
    openai_key: str = None,
) -> bool:
    """
    LLM을 사용하여 두 기사가 같은 주제를 다루는지 판단.

    경계선 케이스 (Jaccard 0.35~0.50)에서만 호출하여 비용 최소화.
    현재 discover_topic_groups에서는 컴포넌트 단위 검증(llm_judge_component_representative)을
    사용하므로 이 함수는 미사용 상태이나 추후 활용을 위해 보존.
    """
    if not openai_key:
        openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("  ⚠️  OPENAI_API_KEY 없음, 보수적으로 False 반환")
        return False

    prompts = load_source_verifier_prompts()
    ts_config = prompts.get("topic_similarity", {})

    system_prompt = ts_config.get("system", "")
    user_template = ts_config.get("user_prompt_template", "")
    response_schema = ts_config.get("response_schema", {})

    if not system_prompt or not user_template:
        print("  ⚠️  topic_similarity prompt 없음, 보수적으로 False 반환")
        return False

    context_a = f"제목: {title_a}\n요약: {summary_a}" if title_a else f"요약: {summary_a}"
    context_b = f"제목: {title_b}\n요약: {summary_b}" if title_b else f"요약: {summary_b}"

    user_prompt = render_prompt(user_template, {"context_a": context_a, "context_b": context_b})
    model = _get_sv_model()

    try:
        result = call_openai_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=response_schema,
            openai_key=openai_key,
            model=model,
            label="주제유사도",
            schema_name="topic_similarity_result",
        )
        if result and "same_topic" in result:
            return result["same_topic"]
        print("  ⚠️  LLM 응답 파싱 실패, 보수적으로 False 반환")
        return False
    except Exception as e:
        print(f"  ⚠️  LLM 검증 실패: {e}, 보수적으로 False 반환")
        return False


def discover_topic_groups(
    df: pd.DataFrame,
    openai_key: str = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Part B: 비클러스터 일반기사 중 같은 주제 그룹 발견.

    알고리즘:
    1. 비클러스터 일반기사 필터
    2. query별 그룹핑
    3. news_category 일치 필수
    4. Jaccard similarity 3단계 분류:
       - >= 0.50: 확실히 같은 주제 → 즉시 연결
       - < 0.35: 확실히 다른 주제 → 건너뜀
       - 0.35 ~ 0.50: 경계선 케이스 → LLM 검증
    5. BFS connected components → 2+ 멤버만
    6. cluster_id format: "{query}_t{counter:05d}"
    7. source: 유사주제

    Returns:
        (df, stats) 튜플
    """
    df = df.copy()
    stats = {
        "sv_new_topic_groups": 0,
        "sv_new_topic_articles": 0,
        "sv_llm_verified": 0,
        "sv_llm_rejected": 0,
    }

    required = {"source", "news_keyword_summary", "news_category", "pub_datetime"}
    if not required.issubset(df.columns):
        return df, stats

    general_mask = df["source"] == "일반기사"
    if "news_category" in df.columns:
        general_mask = general_mask & (df["news_category"] != "비관련")

    if general_mask.sum() < 2:
        return df, stats

    if "cluster_id" not in df.columns:
        df["cluster_id"] = ""

    if "query" not in df.columns:
        df["query"] = ""

    df_general = df[general_mask].copy()
    query_groups = list(df_general.groupby("query", dropna=False))
    total_queries = len(query_groups)

    for q_idx, (query, q_group) in enumerate(query_groups, 1):
        if len(q_group) < 2:
            continue

        print(f"    [{q_idx}/{total_queries}] Query '{query}' ({len(q_group)}개 일반기사) 주제 그룹 탐색 중...")
        indices = q_group.index.tolist()

        token_cache = {}
        cat_cache = {}
        for idx in indices:
            summary_val = df.at[idx, "news_keyword_summary"]
            token_cache[idx] = set() if pd.isna(summary_val) else _tokenize_summary(str(summary_val))

            cat_val = df.at[idx, "news_category"]
            cat_cache[idx] = None if (pd.isna(cat_val) or cat_val == "") else str(cat_val)

        valid_indices = [i for i in indices if len(token_cache[i]) > 0 and cat_cache[i] is not None]
        if len(valid_indices) < 2:
            continue

        adjacency = {i: [] for i in valid_indices}
        borderline_edges = set()
        llm_verified_count = 0
        llm_rejected_count = 0
        n_valid = len(valid_indices)
        total_pairs = n_valid * (n_valid - 1) // 2

        pbar = tqdm(total=total_pairs, desc=f"      [{q_idx}/{total_queries}] 주제 그룹 탐색", unit="쌍", leave=False)

        for i in range(n_valid):
            for j in range(i + 1, n_valid):
                idx_a, idx_b = valid_indices[i], valid_indices[j]
                if cat_cache[idx_a] != cat_cache[idx_b]:
                    pbar.update(1)
                    continue

                sim = _jaccard_similarity(token_cache[idx_a], token_cache[idx_b])

                if sim >= TOPIC_JACCARD_HIGH_THRESHOLD:
                    adjacency[idx_a].append(idx_b)
                    adjacency[idx_b].append(idx_a)
                elif sim < TOPIC_JACCARD_LOW_THRESHOLD:
                    pass
                else:
                    a, b = (idx_a, idx_b) if idx_a < idx_b else (idx_b, idx_a)
                    borderline_edges.add((a, b))

                pbar.update(1)

        pbar.close()

        if borderline_edges:
            borderline_adj = {idx: [] for idx in valid_indices}
            for a, b in borderline_edges:
                borderline_adj[a].append(b)
                borderline_adj[b].append(a)

            borderline_components = []
            border_visited = set()
            for start in valid_indices:
                if start in border_visited or len(borderline_adj[start]) == 0:
                    continue
                component = []
                queue = deque([start])
                while queue:
                    node = queue.popleft()
                    if node in border_visited:
                        continue
                    border_visited.add(node)
                    component.append(node)
                    for neighbor in borderline_adj[node]:
                        if neighbor not in border_visited:
                            queue.append(neighbor)
                if len(component) >= 2:
                    borderline_components.append(component)

            if borderline_components and openai_key:
                comp_pbar = tqdm(
                    total=len(borderline_components),
                    desc=f"      [{q_idx}/{total_queries}] 경계선 컴포넌트 검증",
                    unit="컴포넌트",
                    leave=False,
                )
                for component in borderline_components:
                    rep_idx = _select_representative_index(df, component)
                    if rep_idx is None:
                        comp_pbar.update(1)
                        continue

                    is_same = llm_judge_component_representative(
                        _build_article_payload(df, rep_idx),
                        openai_key=openai_key,
                        mode="topic_group_borderline",
                    )

                    if is_same:
                        llm_verified_count += 1
                        comp_size = len(component)
                        for a_pos in range(comp_size):
                            for b_pos in range(a_pos + 1, comp_size):
                                a_idx = component[a_pos]
                                b_idx = component[b_pos]
                                if b_idx not in adjacency[a_idx]:
                                    adjacency[a_idx].append(b_idx)
                                if a_idx not in adjacency[b_idx]:
                                    adjacency[b_idx].append(a_idx)
                    else:
                        llm_rejected_count += 1

                    comp_pbar.update(1)
                comp_pbar.close()
            elif borderline_components:
                llm_rejected_count += len(borderline_components)

        stats["sv_llm_verified"] += llm_verified_count
        stats["sv_llm_rejected"] += llm_rejected_count
        if llm_verified_count > 0 or llm_rejected_count > 0:
            print(
                f"    Query '{query}': LLM 경계선 컴포넌트 검증 "
                f"{llm_verified_count}개 승인, {llm_rejected_count}개 거부"
            )

        visited = set()
        components = []
        for start in valid_indices:
            if start in visited:
                continue
            component = []
            queue = deque([start])
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                for neighbor in adjacency[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            if len(component) >= 2:
                components.append(component)

        query_str = str(query) if query else "unknown"
        query_prefix = query_str.split("|")[0] if "|" in query_str else query_str

        existing_topic_ids = df["cluster_id"][
            df["cluster_id"].str.startswith(f"{query_prefix}_t", na=False)
        ]
        if len(existing_topic_ids) > 0:
            max_num = max(
                int(tid.split("_t")[-1])
                for tid in existing_topic_ids
                if tid.split("_t")[-1].isdigit()
            )
            counter = max_num + 1
        else:
            counter = 1

        for component in components:
            cid = f"{query_prefix}_t{counter:05d}"
            counter += 1

            for idx in component:
                df.at[idx, "cluster_id"] = cid
                df.at[idx, "source"] = "유사주제"

            stats["sv_new_topic_groups"] += 1
            stats["sv_new_topic_articles"] += len(component)

    return df, stats
