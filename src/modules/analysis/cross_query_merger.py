"""
cross_query_merger.py - Part A-2: Cross-query 클러스터 병합

서로 다른 query로 수집된 동일 보도자료 클러스터를 TF-IDF cosine + Jaccard로 병합.
Part A(클러스터 검증) 이후, Part B(주제 그룹화) 이전에 실행.
"""

from collections import deque
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from ._sv_common import (
    _clean_html,
    _tokenize_simple,
    _extract_media_key,
    _jaccard_similarity,
    _select_representative_index,
    _build_article_payload,
    llm_judge_component_representative,
    CROSS_TITLE_COS_THRESHOLD,
    CROSS_TITLE_JAC_THRESHOLD,
    CROSS_DESC_COS_THRESHOLD,
    CROSS_DESC_JAC_THRESHOLD,
    CROSS_TITLE_JAC_STANDALONE,
    CROSS_TITLE_COS_BORDERLINE,
    CROSS_TITLE_JAC_BORDERLINE_MIN,
    CROSS_DESC_COS_BORDERLINE,
    CROSS_DESC_JAC_BORDERLINE_MIN,
    CROSS_BORDERLINE_DAY_HINT_MAX,
    CROSS_TITLE_JAC_BORDERLINE_STANDALONE,
)


def merge_cross_query_clusters(
    df: pd.DataFrame,
    openai_key: str = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Cross-query 클러스터 병합.

    알고리즘:
    1. 후보 수집: 각 cluster별 대표 기사 + 미클러스터 일반기사
    2. TF-IDF char n-gram 벡터화 (cross-query)
    3. Skip mask: 같은 cluster, 같은 query+미클러스터 쌍 제외
    4. Auto-merge + LLM 경계선 검증
    5. BFS connected components → 병합 처리

    Returns:
        (df, stats) 튜플
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    df = df.copy()
    stats = {
        "sv_cross_merged_groups": 0,
        "sv_cross_merged_articles": 0,
    }

    required = {"source", "cluster_id", "title", "description"}
    if not required.issubset(df.columns):
        return df, stats

    # ── 1. 후보 수집 ───────────────────────────────────────────────────────────
    candidates: List[Dict] = []

    clustered_mask = df["cluster_id"].astype(str).str.strip() != ""
    if clustered_mask.any():
        for cid, cgroup in df[clustered_mask].groupby("cluster_id"):
            if "pub_datetime" in df.columns:
                repr_idx = cgroup["pub_datetime"].astype(str).idxmin()
            else:
                repr_idx = cgroup.index[0]
            candidates.append({
                "repr_idx": repr_idx,
                "member_indices": cgroup.index.tolist(),
                "cluster_id": str(cid),
                "query": str(cgroup["query"].iloc[0]) if "query" in cgroup.columns else "",
                "source": str(cgroup["source"].iloc[0]),
            })

    unclustered_mask = (~clustered_mask) & (df["source"] == "일반기사")
    if "news_category" in df.columns:
        unclustered_mask = unclustered_mask & (df["news_category"] != "비관련")
    for idx in df[unclustered_mask].index:
        candidates.append({
            "repr_idx": idx,
            "member_indices": [idx],
            "cluster_id": "",
            "query": str(df.at[idx, "query"]) if "query" in df.columns else "",
            "source": "일반기사",
        })

    n = len(candidates)
    if n < 2:
        return df, stats

    # ── 2. 텍스트 전처리 + TF-IDF ──────────────────────────────────────────────
    repr_indices = [c["repr_idx"] for c in candidates]
    title_texts = [_clean_html(str(df.at[idx, "title"])) for idx in repr_indices]
    desc_texts = [_clean_html(str(df.at[idx, "description"])) for idx in repr_indices]
    repr_media_keys = [_extract_media_key(df, idx) for idx in repr_indices]
    if "pub_datetime" in df.columns:
        repr_pub_dt = [pd.to_datetime(df.at[idx, "pub_datetime"], errors="coerce", utc=True) for idx in repr_indices]
    else:
        repr_pub_dt = [pd.NaT for _ in repr_indices]

    title_toksets = [_tokenize_simple(t) for t in title_texts]
    desc_toksets = [_tokenize_simple(d) for d in desc_texts]

    try:
        vec_title = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1)
        vec_desc = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1)
        title_corpus = [t if t.strip() else " " for t in title_texts]
        desc_corpus = [d if d.strip() else " " for d in desc_texts]
        X_title = vec_title.fit_transform(title_corpus)
        X_desc = vec_desc.fit_transform(desc_corpus)
        sim_title = cosine_similarity(X_title)
        sim_desc = cosine_similarity(X_desc)
    except Exception as e:
        print(f"  ⚠️  Cross-query TF-IDF 실패: {e}")
        return df, stats

    # ── 3-5. Skip mask + 유사도 기반 adjacency ────────────────────────────────
    adjacency = {i: [] for i in range(n)}
    borderline_pair_scores: Dict[Tuple[int, int], float] = {}
    total_pairs = n * (n - 1) // 2

    print(f"    Cross-query 후보 {n}개, 비교 쌍 {total_pairs}개 처리 중...")
    pbar = tqdm(total=total_pairs, desc="    Cross-query 병합", unit="쌍", leave=False)

    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = candidates[i], candidates[j]

            if ci["cluster_id"] and ci["cluster_id"] == cj["cluster_id"]:
                pbar.update(1)
                continue

            if ci["query"] == cj["query"] and not ci["cluster_id"] and not cj["cluster_id"]:
                pbar.update(1)
                continue

            t_cos = sim_title[i, j]
            d_cos = sim_desc[i, j]
            t_jac = _jaccard_similarity(title_toksets[i], title_toksets[j])
            d_jac = _jaccard_similarity(desc_toksets[i], desc_toksets[j])

            if t_jac >= CROSS_TITLE_JAC_STANDALONE:
                adjacency[i].append(j)
                adjacency[j].append(i)
                pbar.update(1)
                continue

            if t_cos >= CROSS_TITLE_COS_THRESHOLD and t_jac >= CROSS_TITLE_JAC_THRESHOLD:
                adjacency[i].append(j)
                adjacency[j].append(i)
                pbar.update(1)
                continue

            if d_cos >= CROSS_DESC_COS_THRESHOLD and d_jac >= CROSS_DESC_JAC_THRESHOLD:
                adjacency[i].append(j)
                adjacency[j].append(i)
                pbar.update(1)
                continue

            jac_standalone_borderline = (
                CROSS_TITLE_JAC_BORDERLINE_STANDALONE <= t_jac < CROSS_TITLE_JAC_STANDALONE
            )
            title_borderline = (
                CROSS_TITLE_COS_BORDERLINE[0] <= t_cos < CROSS_TITLE_COS_BORDERLINE[1]
                and t_jac >= CROSS_TITLE_JAC_BORDERLINE_MIN
            )
            desc_borderline = (
                CROSS_DESC_COS_BORDERLINE[0] <= d_cos < CROSS_DESC_COS_BORDERLINE[1]
                and d_jac >= CROSS_DESC_JAC_BORDERLINE_MIN
            )

            if jac_standalone_borderline:
                score = float(t_jac)
                key = (i, j)
                prev = borderline_pair_scores.get(key)
                if prev is None or score > prev:
                    borderline_pair_scores[key] = score
            elif title_borderline or desc_borderline:
                same_media = (
                    bool(repr_media_keys[i])
                    and bool(repr_media_keys[j])
                    and repr_media_keys[i] == repr_media_keys[j]
                )
                day_hint = False
                if pd.notna(repr_pub_dt[i]) and pd.notna(repr_pub_dt[j]):
                    day_diff = abs((repr_pub_dt[i] - repr_pub_dt[j]).total_seconds()) / 86400.0
                    day_hint = day_diff <= CROSS_BORDERLINE_DAY_HINT_MAX

                if same_media or day_hint:
                    title_score = (0.7 * float(t_cos) + 0.3 * float(t_jac)) if title_borderline else -1.0
                    desc_score = (0.7 * float(d_cos) + 0.3 * float(d_jac)) if desc_borderline else -1.0
                    score = max(title_score, desc_score)
                    key = (i, j)
                    prev = borderline_pair_scores.get(key)
                    if prev is None or score > prev:
                        borderline_pair_scores[key] = score

            pbar.update(1)

    pbar.close()

    # ── 경계선 컴포넌트 LLM 검증 ──────────────────────────────────────────────
    llm_calls = 0
    llm_component_accepted = 0
    if borderline_pair_scores:
        borderline_adj = {i: [] for i in range(n)}
        for i, j in borderline_pair_scores.keys():
            borderline_adj[i].append(j)
            borderline_adj[j].append(i)

        borderline_components = []
        border_visited = set()
        for start in range(n):
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
                desc="    Cross-query 경계선 컴포넌트",
                unit="컴포넌트",
                leave=False,
            )
            for component in borderline_components:
                rep_row_indices = [candidates[ci]["repr_idx"] for ci in component]
                rep_idx = _select_representative_index(df, rep_row_indices)
                if rep_idx is None:
                    comp_pbar.update(1)
                    continue

                llm_calls += 1
                should_merge = llm_judge_component_representative(
                    _build_article_payload(df, rep_idx),
                    openai_key=openai_key,
                    mode="cross_query_borderline",
                )

                if should_merge:
                    llm_component_accepted += 1
                    comp_size = len(component)
                    for a in range(comp_size):
                        for b in range(a + 1, comp_size):
                            i, j = component[a], component[b]
                            if j not in adjacency[i]:
                                adjacency[i].append(j)
                            if i not in adjacency[j]:
                                adjacency[j].append(i)

                comp_pbar.update(1)
            comp_pbar.close()
        elif borderline_components:
            print("    ⚠️  OPENAI_API_KEY 없음: Cross-query 경계선 컴포넌트 LLM 검증 스킵")

    # ── 6. BFS connected components ────────────────────────────────────────────
    visited = set()
    components = []
    for start in range(n):
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

    # ── 7. 병합 처리 ──────────────────────────────────────────────────────────
    cross_counter = 1
    for component in components:
        comp_candidates = [candidates[ci] for ci in component]

        existing_cids = sorted([c["cluster_id"] for c in comp_candidates if c["cluster_id"]])
        target_cid = existing_cids[0] if existing_cids else f"cross_m{cross_counter:05d}"
        if not existing_cids:
            cross_counter += 1

        sources_in_comp = {c["source"] for c in comp_candidates}
        target_source = "보도자료" if "보도자료" in sources_in_comp else "유사주제"

        target_cs = ""
        if "cluster_summary" in df.columns:
            for c in comp_candidates:
                for midx in c["member_indices"]:
                    val = str(df.at[midx, "cluster_summary"]).strip()
                    if val and val != "nan":
                        target_cs = val
                        break
                if target_cs:
                    break

        total_members = 0
        for c in comp_candidates:
            for midx in c["member_indices"]:
                df.at[midx, "cluster_id"] = target_cid
                df.at[midx, "source"] = target_source
                if target_cs and "cluster_summary" in df.columns:
                    df.at[midx, "cluster_summary"] = target_cs
                total_members += 1

        stats["sv_cross_merged_groups"] += 1
        stats["sv_cross_merged_articles"] += total_members

    if llm_calls > 0:
        print(
            f"    Cross-query 경계선 컴포넌트 LLM 검증: "
            f"{llm_calls}회 호출, {llm_component_accepted}개 컴포넌트 승인"
        )

    return df, stats
