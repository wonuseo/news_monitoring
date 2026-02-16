"""
source_verifier.py - Source Verification & Topic Grouping

LLM 분류 결과 + LLM 클러스터 검증을 활용한 보도자료 클러스터 검증 및 비클러스터 기사 주제 그룹화.

Source Labels:
- 보도자료: 브랜드 공식 배포 보도자료 (LLM 클러스터 검증으로 판단)
- 유사주제: 독립 기사, 같은 주제 (클러스터됐지만 보도자료 기준 미달)
- 일반기사: 독립 기사 (기본값)
"""

import re
import os
from collections import deque
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from src.modules.analysis.llm_engine import (
    load_source_verifier_prompts,
    render_prompt,
    call_openai_structured,
)
from src.utils.openai_client import load_api_models


PR_CATEGORIES = {
    "PR/보도자료", "상품/오퍼링", "제휴/파트너십",
    "브랜드/마케팅", "이벤트/프로모션", "시설/오픈",
    "사업/실적", "ESG/사회",
}

# Topic grouping thresholds
TOPIC_JACCARD_LOW_THRESHOLD = 0.35   # 이하: 확실히 다른 주제
TOPIC_JACCARD_HIGH_THRESHOLD = 0.50  # 이상: 확실히 같은 주제
# 0.35 ~ 0.50 사이: LLM 검증 필요 (경계선 케이스)

# Cross-query merge thresholds (STEP 2 대비 약간 완화, 날짜 제약 제거)
CROSS_TITLE_COS_THRESHOLD = 0.65
CROSS_TITLE_JAC_THRESHOLD = 0.15
CROSS_DESC_COS_THRESHOLD = 0.55
CROSS_DESC_JAC_THRESHOLD = 0.08
# LLM 경계선 범위
CROSS_TITLE_COS_BORDERLINE = (0.50, 0.65)  # [low, high)
CROSS_DESC_COS_BORDERLINE = (0.40, 0.55)   # [low, high)


def determine_verified_source(
    brand_relevance: str,
    sentiment_stage: str,
    news_category: str,
    date_spread_days: float,
) -> str:
    """
    규칙 기반 source 검증 (LLM 클러스터 검증 실패 시 fallback).

    보도자료 유지 조건 (AND):
    - brand_relevance=="관련"
    - sentiment_stage NOT IN ["부정 후보", "부정 확정"]
    - sentiment_stage=="긍정" OR (sentiment_stage=="중립" AND news_category in PR_CATEGORIES)

    Edge case:
    - brand_relevance=="판단 필요" → 보도자료 유지
    - 빈 값 (LLM 실패) → 보도자료 유지

    Returns: "보도자료" / "유사주제"
    """
    # Edge case: LLM 미분류 또는 판단 필요 → 보도자료 유지
    if not brand_relevance or brand_relevance == "판단 필요":
        return "보도자료"

    # 보도자료 유지 조건
    if brand_relevance == "관련":
        # 부정은 절대 보도자료 아님 (명시적 체크)
        if sentiment_stage in ["부정 후보", "부정 확정"]:
            return "유사주제"

        if sentiment_stage == "긍정":
            return "보도자료"
        if sentiment_stage == "중립" and news_category in PR_CATEGORIES:
            return "보도자료"

    # 보도자료 기준 미달 → 유사주제
    return "유사주제"


def _get_sv_model() -> str:
    """source_verification 모델 로드."""
    api_models = load_api_models()
    return api_models.get("source_verification", "gpt-4o-mini")


def llm_verify_cluster(
    cluster_df: pd.DataFrame,
    query: str,
    press_release_group: str,
    openai_key: str,
) -> Optional[str]:
    """
    LLM으로 클러스터 단위 보도자료/유사주제 검증 (1 cluster = 1 API call).

    Args:
        cluster_df: 클러스터 내 기사 DataFrame
        query: 검색 브랜드
        press_release_group: 클러스터 요약
        openai_key: OpenAI API 키

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

    # 기사 제목 목록 (최대 10개)
    titles = cluster_df["title"].tolist() if "title" in cluster_df.columns else []
    titles_display = titles[:10]
    titles_list = "\n".join(f"- {t}" for t in titles_display)
    if len(titles) > 10:
        titles_list += f"\n- ... 외 {len(titles) - 10}개"

    # 대표 기사 LLM 분류 결과 (첫 번째 분류 완료 기사)
    classified = cluster_df[cluster_df["brand_relevance"].astype(str).str.strip() != ""]
    if len(classified) > 0:
        rep = classified.iloc[0]
    else:
        rep = cluster_df.iloc[0]

    brand_relevance = str(rep.get("brand_relevance", ""))
    sentiment_stage = str(rep.get("sentiment_stage", ""))
    news_category = str(rep.get("news_category", ""))

    context = {
        "query": query,
        "press_release_group": press_release_group if press_release_group else "없음",
        "article_count": str(len(cluster_df)),
        "titles_list": titles_list,
        "brand_relevance": brand_relevance,
        "sentiment_stage": sentiment_stage,
        "news_category": news_category,
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

    각 cluster_id별로 LLM이 보도자료인지 유사주제인지 판단.
    LLM 실패 시 규칙 기반 fallback (determine_verified_source).

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

    # cluster_id별 그룹핑
    pr_df = df[pr_mask]
    cluster_groups = pr_df.groupby("cluster_id", dropna=False)
    stats["sv_clusters_verified"] = len(cluster_groups)

    for cluster_id, cluster_df in cluster_groups:
        verified_source = None

        # LLM 클러스터 검증 시도
        if openai_key:
            query = str(cluster_df["query"].iloc[0]) if "query" in cluster_df.columns else ""
            prg = str(cluster_df["press_release_group"].iloc[0]) if "press_release_group" in cluster_df.columns else ""
            verified_source = llm_verify_cluster(cluster_df, query, prg, openai_key)

        # LLM 실패 시 규칙 기반 fallback (대표 기사 기준)
        if verified_source is None:
            rep = cluster_df.iloc[0]
            verified_source = determine_verified_source(
                brand_relevance=str(rep.get("brand_relevance", "")),
                sentiment_stage=str(rep.get("sentiment_stage", "")),
                news_category=str(rep.get("news_category", "")),
                date_spread_days=0,
            )

        # 클러스터 내 모든 기사에 결과 적용
        for idx in cluster_df.index:
            df.at[idx, "source"] = verified_source

        if verified_source == "보도자료":
            stats["sv_kept_press_release"] += len(cluster_df)
        else:
            stats["sv_reclassified_similar_topic"] += len(cluster_df)

    return df, stats


def _tokenize_summary(summary: str) -> set:
    """news_keyword_summary를 공백으로 토큰화."""
    if not summary or not isinstance(summary, str):
        return set()
    return set(summary.strip().split())


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


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
    YAML prompt + Responses API (call_openai_structured) 사용.

    Args:
        summary_a: 기사 A의 news_keyword_summary
        summary_b: 기사 B의 news_keyword_summary
        title_a: 기사 A의 제목 (optional)
        title_b: 기사 B의 제목 (optional)
        openai_key: OpenAI API 키

    Returns:
        True if 같은 주제, False otherwise
    """
    if not openai_key:
        openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print(f"  ⚠️  OPENAI_API_KEY 없음, 보수적으로 False 반환")
        return False

    prompts = load_source_verifier_prompts()
    ts_config = prompts.get("topic_similarity", {})

    system_prompt = ts_config.get("system", "")
    user_template = ts_config.get("user_prompt_template", "")
    response_schema = ts_config.get("response_schema", {})

    if not system_prompt or not user_template:
        print(f"  ⚠️  topic_similarity prompt 없음, 보수적으로 False 반환")
        return False

    # Context 구성
    context_a = f"제목: {title_a}\n요약: {summary_a}" if title_a else f"요약: {summary_a}"
    context_b = f"제목: {title_b}\n요약: {summary_b}" if title_b else f"요약: {summary_b}"

    context = {
        "context_a": context_a,
        "context_b": context_b,
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
            label="주제유사도",
            schema_name="topic_similarity_result",
        )

        if result and "same_topic" in result:
            return result["same_topic"]

        print(f"  ⚠️  LLM 응답 파싱 실패, 보수적으로 False 반환")
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

    news_keyword_summary 토큰 Jaccard 유사도 + news_category 일치 +
    경계선 케이스 LLM 검증으로 같은 주제를 다루는 기사를 그룹화.

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
        "sv_llm_verified": 0,  # LLM이 같은 주제로 판단한 쌍
        "sv_llm_rejected": 0,  # LLM이 다른 주제로 판단한 쌍
    }

    # 필수 컬럼 확인
    required = {"source", "news_keyword_summary", "news_category", "pub_datetime"}
    if not required.issubset(df.columns):
        return df, stats

    # 비클러스터 일반기사만 대상 (비관련 카테고리 제외)
    general_mask = df["source"] == "일반기사"

    # news_category가 "비관련"인 기사는 주제 그룹화에서 제외
    if "news_category" in df.columns:
        general_mask = general_mask & (df["news_category"] != "비관련")

    if general_mask.sum() < 2:
        return df, stats

    # cluster_id 컬럼 보장
    if "cluster_id" not in df.columns:
        df["cluster_id"] = ""

    # query 컬럼이 없으면 전체를 하나의 그룹으로
    if "query" not in df.columns:
        df["query"] = ""

    df_general = df[general_mask].copy()

    # query별 처리
    for query, q_group in df_general.groupby("query", dropna=False):
        if len(q_group) < 2:
            continue

        indices = q_group.index.tolist()

        # 토큰화 + 카테고리 캐싱 (nan 값 명시적 처리)
        token_cache = {}
        cat_cache = {}
        for idx in indices:
            # news_keyword_summary 처리 (nan → 빈 set)
            summary_val = df.at[idx, "news_keyword_summary"]
            if pd.isna(summary_val):
                token_cache[idx] = set()
            else:
                token_cache[idx] = _tokenize_summary(str(summary_val))

            # news_category 처리 (nan → None)
            cat_val = df.at[idx, "news_category"]
            if pd.isna(cat_val) or cat_val == "":
                cat_cache[idx] = None
            else:
                cat_cache[idx] = str(cat_val)

        # 유효 토큰이 있는 기사만 (LLM 분류 성공한 기사)
        valid_indices = [i for i in indices if len(token_cache[i]) > 0 and cat_cache[i] is not None]
        if len(valid_indices) < 2:
            continue

        # 인접 리스트 구성 (category 일치 + Jaccard + LLM 경계선 검증)
        adjacency = {i: [] for i in valid_indices}
        llm_verified_count = 0
        llm_rejected_count = 0

        for i in range(len(valid_indices)):
            for j in range(i + 1, len(valid_indices)):
                idx_a, idx_b = valid_indices[i], valid_indices[j]
                # news_category 일치 필수 (None 체크 불필요 - valid_indices에서 이미 필터링됨)
                if cat_cache[idx_a] != cat_cache[idx_b]:
                    continue

                sim = _jaccard_similarity(token_cache[idx_a], token_cache[idx_b])

                # 확실히 같은 주제 (high threshold 이상)
                if sim >= TOPIC_JACCARD_HIGH_THRESHOLD:
                    adjacency[idx_a].append(idx_b)
                    adjacency[idx_b].append(idx_a)
                # 확실히 다른 주제 (low threshold 이하)
                elif sim < TOPIC_JACCARD_LOW_THRESHOLD:
                    continue
                # 경계선 케이스 (0.35 ~ 0.50): LLM 검증
                else:
                    summary_a = str(df.at[idx_a, "news_keyword_summary"])
                    summary_b = str(df.at[idx_b, "news_keyword_summary"])
                    title_a = str(df.at[idx_a, "title"]) if "title" in df.columns else None
                    title_b = str(df.at[idx_b, "title"]) if "title" in df.columns else None

                    is_same = llm_verify_topic_similarity(
                        summary_a, summary_b, title_a, title_b,
                        openai_key=openai_key,
                    )

                    if is_same:
                        adjacency[idx_a].append(idx_b)
                        adjacency[idx_b].append(idx_a)
                        llm_verified_count += 1
                    else:
                        llm_rejected_count += 1

        # LLM 검증 통계 누적 및 출력
        stats["sv_llm_verified"] += llm_verified_count
        stats["sv_llm_rejected"] += llm_rejected_count
        if llm_verified_count > 0 or llm_rejected_count > 0:
            print(f"    Query '{query}': LLM 경계선 검증 {llm_verified_count}개 연결, {llm_rejected_count}개 거부")

        # BFS connected components
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

        # query 문자열 정리 (파이프 포함 시 첫 번째)
        query_str = str(query) if query else "unknown"
        query_prefix = query_str.split("|")[0] if "|" in query_str else query_str

        # 기존 topic cluster counter 확인
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

        # 클러스터 할당
        for component in components:
            cid = f"{query_prefix}_t{counter:05d}"
            counter += 1

            for idx in component:
                df.at[idx, "cluster_id"] = cid
                df.at[idx, "source"] = "유사주제"

            stats["sv_new_topic_groups"] += 1
            stats["sv_new_topic_articles"] += len(component)

    return df, stats


def _clean_html(s: str) -> str:
    """HTML 태그 제거 (press_release_detector 방식 재활용)."""
    if not isinstance(s, str):
        return ""
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("&quot;", " ").replace("&amp;", "&")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokenize_simple(s: str, min_token_len: int = 2) -> set:
    """간단 토큰화 → set (Jaccard용)."""
    if not isinstance(s, str) or not s.strip():
        return set()
    s = s.lower()
    toks = re.findall(r"[가-힣a-z0-9]+", s)
    return {t for t in toks if len(t) >= min_token_len}


def merge_cross_query_clusters(
    df: pd.DataFrame,
    openai_key: str = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Cross-query 클러스터 병합: 서로 다른 query로 수집된 동일 보도자료 클러스터를 병합.

    Part A(클러스터 검증) 이후, Part B(주제 그룹화) 이전에 실행.
    STEP 2(press_release_detector)에서 query별로 분리된 클러스터 + 미클러스터 일반기사를
    TF-IDF cosine + Jaccard 유사도로 cross-query 병합.

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

    # 필수 컬럼 확인
    required = {"source", "cluster_id", "title", "description"}
    if not required.issubset(df.columns):
        return df, stats

    # ─── 1. 후보 수집 ───────────────────────────────────────────────────────
    candidates: List[Dict] = []  # {repr_idx, member_indices, cluster_id, query, source}

    # 1a. 기존 클러스터별 대표 기사 (가장 빠른 pub_datetime)
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

    # 1b. 미클러스터 일반기사 (비관련 제외)
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

    # ─── 2. 텍스트 전처리 + TF-IDF ──────────────────────────────────────────
    repr_indices = [c["repr_idx"] for c in candidates]
    title_texts = [_clean_html(str(df.at[idx, "title"])) for idx in repr_indices]
    desc_texts = [_clean_html(str(df.at[idx, "description"])) for idx in repr_indices]

    # Jaccard용 token set
    title_toksets = [_tokenize_simple(t) for t in title_texts]
    desc_toksets = [_tokenize_simple(d) for d in desc_texts]

    # TF-IDF 벡터화
    try:
        vec_title = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1)
        vec_desc = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1)

        # 빈 문서 처리: 공백 하나라도 넣어야 TfidfVectorizer가 작동
        title_corpus = [t if t.strip() else " " for t in title_texts]
        desc_corpus = [d if d.strip() else " " for d in desc_texts]

        X_title = vec_title.fit_transform(title_corpus)
        X_desc = vec_desc.fit_transform(desc_corpus)

        sim_title = cosine_similarity(X_title)
        sim_desc = cosine_similarity(X_desc)
    except Exception as e:
        print(f"  ⚠️  Cross-query TF-IDF 실패: {e}")
        return df, stats

    # ─── 3-5. Skip mask + 유사도 기반 adjacency ─────────────────────────────
    adjacency = {i: [] for i in range(n)}
    llm_calls = 0

    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = candidates[i], candidates[j]

            # Skip: 같은 cluster에 속한 쌍
            if ci["cluster_id"] and ci["cluster_id"] == cj["cluster_id"]:
                continue

            # Skip: 같은 query이면서 둘 다 미클러스터 (STEP 2에서 이미 처리)
            if ci["query"] == cj["query"] and not ci["cluster_id"] and not cj["cluster_id"]:
                continue

            t_cos = sim_title[i, j]
            d_cos = sim_desc[i, j]
            t_jac = _jaccard_similarity(title_toksets[i], title_toksets[j])
            d_jac = _jaccard_similarity(desc_toksets[i], desc_toksets[j])

            # Auto-merge: title 기준
            if t_cos >= CROSS_TITLE_COS_THRESHOLD and t_jac >= CROSS_TITLE_JAC_THRESHOLD:
                adjacency[i].append(j)
                adjacency[j].append(i)
                continue

            # Auto-merge: description 기준
            if d_cos >= CROSS_DESC_COS_THRESHOLD and d_jac >= CROSS_DESC_JAC_THRESHOLD:
                adjacency[i].append(j)
                adjacency[j].append(i)
                continue

            # LLM 경계선 검증
            title_borderline = (CROSS_TITLE_COS_BORDERLINE[0] <= t_cos < CROSS_TITLE_COS_BORDERLINE[1])
            desc_borderline = (CROSS_DESC_COS_BORDERLINE[0] <= d_cos < CROSS_DESC_COS_BORDERLINE[1])

            if title_borderline or desc_borderline:
                # desc를 summary로 전달 (llm_verify_topic_similarity 재활용)
                title_a = str(df.at[ci["repr_idx"], "title"]) if "title" in df.columns else None
                title_b = str(df.at[cj["repr_idx"], "title"]) if "title" in df.columns else None
                desc_a = desc_texts[i][:200]
                desc_b = desc_texts[j][:200]

                is_same = llm_verify_topic_similarity(
                    summary_a=desc_a, summary_b=desc_b,
                    title_a=title_a, title_b=title_b,
                    openai_key=openai_key,
                )
                llm_calls += 1

                if is_same:
                    adjacency[i].append(j)
                    adjacency[j].append(i)

    # ─── 6. BFS connected components ────────────────────────────────────────
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

    # ─── 7. 병합 처리 ───────────────────────────────────────────────────────
    cross_counter = 1
    for component in components:
        comp_candidates = [candidates[ci] for ci in component]

        # target cluster_id: 기존 cluster_id 중 가장 작은 것
        existing_cids = sorted([c["cluster_id"] for c in comp_candidates if c["cluster_id"]])
        if existing_cids:
            target_cid = existing_cids[0]
        else:
            target_cid = f"cross_m{cross_counter:05d}"
            cross_counter += 1

        # target source: component 내 "보도자료" 있으면 "보도자료", 아니면 "유사주제"
        sources_in_comp = {c["source"] for c in comp_candidates}
        target_source = "보도자료" if "보도자료" in sources_in_comp else "유사주제"

        # target press_release_group: 기존 값 중 첫 번째
        target_prg = ""
        if "press_release_group" in df.columns:
            for c in comp_candidates:
                for midx in c["member_indices"]:
                    val = str(df.at[midx, "press_release_group"]).strip()
                    if val and val != "nan":
                        target_prg = val
                        break
                if target_prg:
                    break

        # 모든 member_indices에 대해 업데이트
        total_members = 0
        for c in comp_candidates:
            for midx in c["member_indices"]:
                df.at[midx, "cluster_id"] = target_cid
                df.at[midx, "source"] = target_source
                if target_prg and "press_release_group" in df.columns:
                    df.at[midx, "press_release_group"] = target_prg
                total_members += 1

        stats["sv_cross_merged_groups"] += 1
        stats["sv_cross_merged_articles"] += total_members

    if llm_calls > 0:
        print(f"    Cross-query LLM 경계선 검증: {llm_calls}회 호출")

    return df, stats


def verify_and_regroup_sources(
    df: pd.DataFrame,
    openai_key: str = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Entry point: Part A (보도자료 클러스터 LLM 검증) + Part B (주제 그룹 발견) 실행.

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

    # 통합 통계
    combined = {**verify_stats, **cross_stats, **topic_stats}

    # Source 분포 출력
    if "source" in df.columns:
        source_dist = df["source"].value_counts().to_dict()
        print(f"  Source 분포: {source_dist}")

    print("✅ Source 검증 및 주제 그룹화 완료")
    return df, combined
