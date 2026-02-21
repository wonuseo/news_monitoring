"""
_sv_common.py - Source Verifier 공유 유틸리티

cluster_verifier, cross_query_merger, topic_grouper에서 공통으로 사용하는
상수, 임계값, 헬퍼 함수 모음.
"""

import re
from typing import Dict, List, Optional
from urllib.parse import urlparse

import pandas as pd

from src.modules.analysis.llm_engine import load_prompts, analyze_article_llm
from src.utils.openai_client import load_api_models

# ── 보도자료 카테고리 ──────────────────────────────────────────────────────────
PR_CATEGORIES = {
    "PR/보도자료", "상품/오퍼링", "제휴/파트너십",
    "브랜드/마케팅", "이벤트/프로모션", "시설/오픈",
    "사업/실적", "ESG/사회",
}


# ── 임계값: config/thresholds.yaml에서 로드, 없으면 하드코딩 fallback ──────────
def _load_sv_thresholds():
    try:
        from src.utils.config import load_config
        return load_config("thresholds")
    except Exception:
        return {}


_sv_thr = _load_sv_thresholds()
_tg = _sv_thr.get("topic_grouping", {})
_cq = _sv_thr.get("cross_query_merge", {})
_bl = _cq.get("borderline", {})

# Topic grouping thresholds (config/thresholds.yaml → topic_grouping)
TOPIC_JACCARD_LOW_THRESHOLD  = _tg.get("jaccard_low",  0.35)
TOPIC_JACCARD_HIGH_THRESHOLD = _tg.get("jaccard_high", 0.50)

# Cross-query merge thresholds (config/thresholds.yaml → cross_query_merge)
CROSS_TITLE_COS_THRESHOLD         = _cq.get("title_cosine",            0.65)
CROSS_TITLE_JAC_THRESHOLD         = _cq.get("title_jaccard",           0.15)
CROSS_DESC_COS_THRESHOLD          = _cq.get("desc_cosine",             0.55)
CROSS_DESC_JAC_THRESHOLD          = _cq.get("desc_jaccard",            0.08)
CROSS_TITLE_JAC_STANDALONE        = _cq.get("title_jaccard_standalone", 0.80)
CROSS_TITLE_COS_BORDERLINE        = (_bl.get("title_cos_low",  0.58), _bl.get("title_cos_high", 0.65))
CROSS_TITLE_JAC_BORDERLINE_MIN    = _bl.get("title_jac_min",   0.20)
CROSS_DESC_COS_BORDERLINE         = (_bl.get("desc_cos_low",   0.48), _bl.get("desc_cos_high",  0.55))
CROSS_DESC_JAC_BORDERLINE_MIN     = _bl.get("desc_jac_min",    0.12)
CROSS_BORDERLINE_DAY_HINT_MAX     = _bl.get("day_hint_max",    1)
CROSS_TITLE_JAC_BORDERLINE_STANDALONE = _bl.get("title_jac_standalone", 0.40)


# ── 캐시 ───────────────────────────────────────────────────────────────────────
_article_prompts_cache: Optional[dict] = None


def _get_article_prompts() -> Optional[dict]:
    """article 분류 프롬프트 캐시 로드."""
    global _article_prompts_cache
    if _article_prompts_cache is not None:
        return _article_prompts_cache
    try:
        _article_prompts_cache = load_prompts()
    except Exception as e:
        print(f"  ⚠️  article prompts 로드 실패: {e}")
        _article_prompts_cache = None
    return _article_prompts_cache


def _get_sv_model() -> str:
    """source_verification 모델 로드."""
    api_models = load_api_models()
    return api_models.get("source_verification", "gpt-4o-mini")


# ── 데이터 변환 헬퍼 ──────────────────────────────────────────────────────────
def _build_article_payload(df: pd.DataFrame, idx: int) -> Dict[str, str]:
    """DataFrame row index를 analyze_article_llm 입력 형태로 변환."""
    return {
        "query": str(df.at[idx, "query"]) if "query" in df.columns else "",
        "group": str(df.at[idx, "group"]) if "group" in df.columns else "",
        "title": str(df.at[idx, "title"]) if "title" in df.columns else "",
        "description": str(df.at[idx, "description"]) if "description" in df.columns else "",
    }


def _select_representative_index(df: pd.DataFrame, row_indices: List[int]) -> Optional[int]:
    """
    컴포넌트 대표 기사 선택:
    1) 가장 이른 pub_datetime
    2) 동일/결측이면 정보량(title+description 길이) 높은 기사
    """
    if not row_indices:
        return None

    has_pub = "pub_datetime" in df.columns
    has_title = "title" in df.columns
    has_desc = "description" in df.columns

    best_idx = None
    best_key = None

    for idx in row_indices:
        dt = pd.to_datetime(df.at[idx, "pub_datetime"], errors="coerce", utc=True) if has_pub else pd.NaT
        has_dt = 0 if pd.notna(dt) else 1
        dt_key = dt.value if pd.notna(dt) else float("inf")
        info_len = (
            len(str(df.at[idx, "title"])) if has_title else 0
        ) + (
            len(str(df.at[idx, "description"])) if has_desc else 0
        )
        key = (has_dt, dt_key, -info_len, idx)
        if best_key is None or key < best_key:
            best_key = key
            best_idx = idx

    return best_idx


def _extract_media_key(df: pd.DataFrame, idx: int) -> str:
    """same_media 판별용 키 추출 (media_domain 우선, 없으면 link 도메인)."""
    if "media_domain" in df.columns:
        media_domain = str(df.at[idx, "media_domain"]).strip().lower()
        if media_domain and media_domain != "nan":
            return media_domain

    if "link" in df.columns:
        link = str(df.at[idx, "link"]).strip()
        if link and link != "nan":
            try:
                return urlparse(link).netloc.lower()
            except Exception:
                return ""

    return ""


# ── 텍스트 처리 ───────────────────────────────────────────────────────────────
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


def _clean_html(s: str) -> str:
    """HTML 태그 제거."""
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


# ── LLM 헬퍼 ──────────────────────────────────────────────────────────────────
def llm_judge_component_representative(
    article: Dict[str, str],
    openai_key: str,
    mode: str = "cross_query_borderline",
) -> bool:
    """
    경계선 컴포넌트 대표 기사 1건만 LLM 분류해 컴포넌트 승인 여부 결정.

    mode:
      - press_release_borderline: 보도자료성 엄격 게이트
      - cross_query_borderline: cross-query 병합 게이트
      - topic_group_borderline: 일반기사 주제 그룹 게이트
    """
    if not openai_key:
        return False

    prompts = _get_article_prompts()
    if not prompts:
        return False

    try:
        result = analyze_article_llm(article, prompts, openai_key)
    except Exception as e:
        print(f"  ⚠️  컴포넌트 대표기사 LLM 실패: {e}")
        return False

    if not result:
        return False

    brand_rel = str(result.get("brand_relevance", "")).strip()
    sentiment = str(result.get("sentiment_stage", "")).strip()
    news_category = str(result.get("news_category", "")).strip()

    if mode == "press_release_borderline":
        if brand_rel != "관련":
            return False
        if sentiment in {"부정 후보", "부정 확정"}:
            return False
        return news_category in PR_CATEGORIES

    if mode == "cross_query_borderline":
        return brand_rel in {"관련", "언급"} and news_category != "비관련"

    # topic_group_borderline
    return brand_rel in {"관련", "언급"} and news_category != "비관련"


def determine_verified_source(
    brand_relevance: str,
    sentiment_stage: str,
    news_category: str,
    date_spread_days: float,
) -> str:
    """
    규칙 기반 source 검증 (LLM 클러스터 검증 실패 시 fallback).

    Returns: "보도자료" / "유사주제"
    """
    if not brand_relevance or brand_relevance == "판단 필요":
        return "보도자료"

    if brand_relevance == "관련":
        if sentiment_stage in ["부정 후보", "부정 확정"]:
            return "유사주제"
        if sentiment_stage == "긍정":
            return "보도자료"
        if sentiment_stage == "중립" and news_category in PR_CATEGORIES:
            return "보도자료"

    return "유사주제"
