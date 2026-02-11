"""
press_release_detector.py - Press Release Detection & Summarization
보도자료 검출, 클러스터링, 요약 (TF-IDF + Cosine Similarity + OpenAI)
"""

import re
import json
import time
import yaml
import requests
import pandas as pd
from pathlib import Path
from typing import Dict, List


def _extract_response_text(result: Dict) -> str:
    text = result.get("output_text", "")
    if isinstance(text, str) and text.strip():
        return text.strip()

    for section in result.get("output", []):
        contents = section.get("content", [])
        for item in contents:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "output_text" and isinstance(item.get("text"), str):
                return item["text"].strip()
            # Some payloads embed raw text without explicit type
            raw_text = item.get("text")
            if isinstance(raw_text, str) and raw_text.strip():
                return raw_text.strip()

    # Fallback to legacy choices (if any)
    for choice in result.get("choices", []):
        message = choice.get("message", {})
        content = message.get("content", "")
        if isinstance(content, str) and content.strip():
            return content.strip()

    return ""


def _trim_json_object(payload: str) -> str:
    # Remove wrapping markdown fences, keep inner braces/brackets
    cleaned = re.sub(r"```(?:json)?\s*", "", payload)
    cleaned = cleaned.replace("`", "")
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and start < end:
        json_snippet = cleaned[start:end + 1]
        trailing = cleaned[end + 1:]
        trailing_match = re.search(r"\]", trailing)
        if trailing_match:
            json_snippet += trailing[: trailing_match.end()]
        return json_snippet
    return cleaned


def _repair_json_text(payload: str) -> str:
    cleaned = payload.strip()
    cleaned = re.sub(r",\s*([\]\}])", r"\1", cleaned)
    return cleaned


def _parse_summaries_from_result(result: Dict) -> List[Dict]:
    text = _extract_response_text(result)
    if not text:
        print("❌ JSON 파싱 실패: API 응답에 텍스트가 없습니다")
        print(f"   전체 응답 구조: {json.dumps(result, indent=2, ensure_ascii=False)[:500]}")
        raise ValueError("Responses API 반환 body에 텍스트가 없습니다.")

    trimmed = _trim_json_object(text)

    last_exc: Exception | None = None
    parsed = None
    candidates = [trimmed, _repair_json_text(trimmed)]

    for i, candidate in enumerate(candidates, 1):
        try:
            parsed = json.loads(candidate)
            break
        except json.JSONDecodeError as exc:
            last_exc = exc
            print(f"❌ JSON 파싱 시도 {i} 실패: {exc}")
            print(f"   시도한 텍스트 (앞 300자): {candidate[:300]}")

    if parsed is None:
        print("❌ 모든 JSON 파싱 시도 실패")
        print(f"   원본 텍스트 (앞 500자): {text[:500]}")
        print(f"   정제된 텍스트 (앞 500자): {trimmed[:500]}")
        raise last_exc or ValueError("Responses API JSON을 파싱할 수 없습니다.")

    if isinstance(parsed, dict):
        summaries = parsed.get("summaries")
    elif isinstance(parsed, list):
        summaries = parsed
    else:
        print(f"❌ 예상하지 못한 JSON 구조: {type(parsed)}")
        print(f"   파싱된 객체: {json.dumps(parsed, indent=2, ensure_ascii=False)[:500]}")
        raise ValueError("Responses API가 요약 목록을 포함하지 않습니다.")

    if not isinstance(summaries, list):
        print(f"❌ summaries가 리스트가 아님: {type(summaries)}")
        print(f"   summaries 값: {summaries}")
        raise ValueError("Responses API가 summaries 리스트를 반환하지 않았습니다.")

    return summaries

OPENAI_API_URL = "https://api.openai.com/v1/responses"


def load_api_models() -> dict:
    """api_models.yaml에서 모델 설정 로드"""
    yaml_path = Path(__file__).parent.parent.parent / "api_models.yaml"
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get("models", {})
    except FileNotFoundError:
        return {"press_release_summary": "gpt-5-nano"}


def detect_similar_articles(
    df: pd.DataFrame,
    day_window: int = 3,
    strong_start_day: int = 4,
    thr_title_cos: float = 0.70,
    thr_title_jac: float = 0.18,
    thr_desc_cos: float = 0.60,
    thr_desc_jac: float = 0.10,
    strong_desc_cos: float = 0.85,
    strong_desc_jac: float = 0.35,
    min_token_len: int = 2
) -> pd.DataFrame:
    """
    내용 기반 유사도 검사로 보도자료 식별
    - 날짜 기반 차등 임계값 적용 (Δdays ≤ 3 vs Δdays ≥ 4)
    - TF-IDF Cosine + Jaccard 이중 유사도 측정
    - Title/Description 별도 처리
    - Query별 독립 클러스터링
    - BFS로 Connected Components 그룹화
    - 모든 기사 유지 (비파괴적 라벨링만 수행)

    Args:
        df: 정규화 및 중복 제거된 DataFrame (query, pub_datetime 컬럼 필수)
        day_window: 기본 규칙 적용 날짜 범위 (기본값: 3일)
        strong_start_day: 초강유사 규칙 시작 날짜 (기본값: 4일)
        thr_title_cos: Title Cosine 임계값 (기본값: 0.70)
        thr_title_jac: Title Jaccard 임계값 (기본값: 0.18)
        thr_desc_cos: Description Cosine 임계값 (기본값: 0.60)
        thr_desc_jac: Description Jaccard 임계값 (기본값: 0.10)
        strong_desc_cos: 초강유사 Description Cosine 임계값 (기본값: 0.85)
        strong_desc_jac: 초강유사 Description Jaccard 임계값 (기본값: 0.35)
        min_token_len: 토큰 최소 길이 (기본값: 2)

    Returns:
        'source', 'cluster_id' 컬럼이 추가된 DataFrame
    """
    print("🔍 유사 기사 검색 중...")
    print(f"  - 기본 규칙(Δdays≤{day_window}): title {thr_title_cos}/{thr_title_jac}, desc {thr_desc_cos}/{thr_desc_jac}")
    print(f"  - 초강유사(Δdays≥{strong_start_day}): desc {strong_desc_cos}/{strong_desc_jac}")

    df = df.copy()

    # 컬럼 초기화
    df["source"] = "일반기사"
    df["cluster_id"] = ""

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        from collections import deque
    except ImportError as e:
        print(f"⚠️  필수 패키지가 설치되지 않았습니다 ({e}). 유사도 검사를 건너뜁니다.")
        return df

    # 필수 컬럼 확인
    if "query" not in df.columns:
        print("⚠️  'query' 컬럼이 없습니다. 유사도 검사를 건너뜁니다.")
        return df

    # 유틸 함수들
    def clean_html(s: str) -> str:
        """HTML 태그 제거"""
        if not isinstance(s, str):
            return ""
        s = re.sub(r"<[^>]+>", " ", s)
        s = s.replace("&quot;", " ").replace("&amp;", "&")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def tokenize_simple(s: str):
        """간단 토큰화 (한글/영문/숫자 덩어리)"""
        if not isinstance(s, str) or not s.strip():
            return []
        s = s.lower()
        toks = re.findall(r"[가-힣a-z0-9]+", s)
        return [t for t in toks if len(t) >= min_token_len]

    def jaccard(a: set, b: set) -> float:
        """Jaccard 유사도"""
        if not a and not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

    def to_ordinal_safe(ts):
        """datetime → ordinal, NaT면 np.nan"""
        if pd.isna(ts):
            return np.nan
        try:
            return pd.to_datetime(ts).to_pydatetime().date().toordinal()
        except:
            return np.nan

    # 텍스트 전처리
    df["title_clean"] = df["title"].fillna("").map(clean_html)
    df["desc_clean"] = df["description"].fillna("").map(clean_html)

    # 날짜 변환 (pub_datetime → ordinal)
    df["pub_dt_ordinal"] = df["pub_datetime"].map(to_ordinal_safe)

    # 빈 텍스트 제거
    valid_mask = (df["title_clean"].str.len() > 0) | (df["desc_clean"].str.len() > 0)
    if valid_mask.sum() < 2:
        print(f"✅ 유효한 기사가 {valid_mask.sum()}개로 너무 적습니다. 유사도 검사 스킵")
        df = df.drop(columns=["title_clean", "desc_clean", "pub_dt_ordinal"])
        return df

    # Jaccard용 토큰 셋
    df["title_tokset"] = df["title_clean"].map(lambda x: set(tokenize_simple(x)))
    df["desc_tokset"] = df["desc_clean"].map(lambda x: set(tokenize_simple(x)))

    # Query별 클러스터링
    total_press_release = 0
    total_clusters = 0

    for query, group_df in df.groupby("query"):
        cluster_id_counter = 1  # Query별로 1부터 시작
        idxs = group_df.index.tolist()
        m = len(idxs)

        # 단독 기사
        if m == 1:
            continue

        try:
            # TF-IDF 벡터화 (Title/Description 별도)
            vec_title = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2)
            vec_desc = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2)

            X_title = vec_title.fit_transform(df.loc[idxs, "title_clean"])
            X_desc = vec_desc.fit_transform(df.loc[idxs, "desc_clean"])

            # Cosine similarity matrices
            sim_t = cosine_similarity(X_title)
            sim_d = cosine_similarity(X_desc)

            # 날짜 차이(일) 행렬
            ordinals = df.loc[idxs, "pub_dt_ordinal"].to_numpy()
            dd = np.abs(ordinals[:, None] - ordinals[None, :])  # NaN 포함 가능

            close_mask = (dd <= day_window)  # Δdays ≤ 3
            far_mask = (dd >= strong_start_day)  # Δdays ≥ 4

            # Adjacency list
            adj = [set() for _ in range(m)]

            # 토큰 셋 캐시
            title_sets = df.loc[idxs, "title_tokset"].tolist()
            desc_sets = df.loc[idxs, "desc_tokset"].tolist()

            # (1) Δdays ≤ 3: Title 규칙
            cand_title = np.triu((sim_t >= thr_title_cos) & close_mask, k=1)
            ti, tj = np.where(cand_title)
            for i, j in zip(ti, tj):
                if jaccard(title_sets[i], title_sets[j]) >= thr_title_jac:
                    adj[i].add(j)
                    adj[j].add(i)

            # (2) Δdays ≤ 3: Description 규칙
            cand_desc = np.triu((sim_d >= thr_desc_cos) & close_mask, k=1)
            di, dj = np.where(cand_desc)
            for i, j in zip(di, dj):
                if jaccard(desc_sets[i], desc_sets[j]) >= thr_desc_jac:
                    adj[i].add(j)
                    adj[j].add(i)

            # (3) Δdays ≥ 4: 초강유사 규칙
            cand_strong = np.triu((sim_d >= strong_desc_cos) & far_mask, k=1)
            si, sj = np.where(cand_strong)
            for i, j in zip(si, sj):
                if jaccard(desc_sets[i], desc_sets[j]) >= strong_desc_jac:
                    adj[i].add(j)
                    adj[j].add(i)

            # Connected components (BFS)
            visited = [False] * m
            for i in range(m):
                if visited[i]:
                    continue

                queue = deque([i])
                visited[i] = True
                comp = [i]

                while queue:
                    u = queue.popleft()
                    for v in adj[u]:
                        if not visited[v]:
                            visited[v] = True
                            queue.append(v)
                            comp.append(v)

                # 2개 이상인 클러스터만 처리
                if len(comp) < 2:
                    continue

                members_global = [idxs[k] for k in comp]
                cid = f"{query}_c{cluster_id_counter:05d}"
                cluster_id_counter += 1
                total_clusters += 1

                for gidx in members_global:
                    df.at[gidx, "source"] = "보도자료"
                    df.at[gidx, "cluster_id"] = cid
                    total_press_release += 1

        except Exception as e:
            print(f"⚠️  '{query}' 처리 중 오류 발생: {e}")
            continue

    # 임시 컬럼 제거
    df = df.drop(columns=["title_clean", "desc_clean", "pub_dt_ordinal", "title_tokset", "desc_tokset"])

    unique_count = len(df) - total_press_release
    print(f"✅ {total_press_release}개 기사를 '보도자료'로 표시 ({total_clusters}개 클러스터), {unique_count}개 기사는 독립 기사")

    return df


def _call_openai_summarize_batch(
    articles: List[Dict],
    openai_key: str,
    retry: bool = True
) -> Dict[str, str]:
    """OpenAI API로 기사 배치 요약"""
    articles_text = "\n".join([
        f"[{a['cluster_id']}]\n제목: {a['title']}\n설명: {a['description'][:200]}"
        for a in articles
    ])

    prompt = f"""각 기사를 5단어 내외로 요약하세요.

기사:
{articles_text}

IMPORTANT: JSON 객체만 출력하세요. 다른 설명이나 마크다운 없이 JSON만 응답:
{{"summaries":[{{"cluster_id":"롯데호텔_c00001","summary":"신라호텔 개장"}},{{"cluster_id":"롯데호텔_c00002","summary":"호텔 리모델링"}}]}}

규칙:
- 반드시 JSON 객체 형식으로만 출력
- 기사의 핵심 주체를 반드시 포함할 것
- 5단어 이내 한국어 요약
- 명사형 위주
- 코드 블록(```)이나 설명 텍스트 금지"""

    # api_models.yaml에서 모델 로드
    api_models = load_api_models()
    model = api_models.get("press_release_summary", "gpt-5-nano")

    try:
        response = requests.post(
            OPENAI_API_URL,
            headers={"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "press_release_summaries",
                        "strict": False,
                        "schema": {
                            "type": "object",
                            "required": ["summaries"],
                            "additionalProperties": True,
                            "properties": {
                                "summaries": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "required": ["cluster_id", "summary"],
                                        "additionalProperties": True,
                                        "properties": {
                                            "cluster_id": {"type": "string"},
                                            "summary": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
            },
            timeout=60
        )

        if response.status_code == 429 and retry:
            print("  (Rate limit, 5초 대기 후 재시도)")
            time.sleep(5)
            return _call_openai_summarize_batch(articles, openai_key, retry=False)

        if response.status_code != 200:
            print(f"  (API 오류 {response.status_code}, 기본값 사용)")
            return {a["cluster_id"]: a["title"][:15] for a in articles}

        result = response.json()
        try:
            summaries = _parse_summaries_from_result(result)
            return {
                item["cluster_id"]: item["summary"]
                for item in summaries
                if isinstance(item, dict) and "cluster_id" in item and "summary" in item
            }
        except (ValueError, json.JSONDecodeError, KeyError) as e:
            if retry:
                print(f"  (JSON 파싱 실패: {type(e).__name__}, 재시도)")
                time.sleep(2)
                return _call_openai_summarize_batch(articles, openai_key, retry=False)
            print(f"  (최종 파싱 실패, 기본값 사용)")
            return {a["cluster_id"]: a["title"][:15] for a in articles}

    except Exception as e:
        print(f"  (오류: {e}, 기본값 사용)")
        return {a["cluster_id"]: a["title"][:15] for a in articles}


def summarize_press_release_groups(
    df: pd.DataFrame,
    openai_key: str
) -> pd.DataFrame:
    """
    보도자료 그룹별로 가장 이른 기사를 OpenAI로 요약

    Args:
        df: cluster_id, pub_datetime, title, description 컬럼 포함
        openai_key: OpenAI API 키

    Returns:
        press_release_group 컬럼이 추가된 DataFrame
    """
    print("📝 보도자료 그룹 요약 생성 중...")
    df = df.copy()
    df["press_release_group"] = ""

    press_release_mask = (df["source"] == "보도자료") & (df["cluster_id"] != "")
    if press_release_mask.sum() == 0:
        print("  ℹ️  보도자료 그룹이 없습니다.")
        return df

    try:
        # 그룹별 가장 이른 기사 선택
        pr_df = df[press_release_mask].copy()
        pr_df["pub_datetime_parsed"] = pd.to_datetime(pr_df["pub_datetime"], errors="coerce")
        earliest_articles = pr_df.sort_values("pub_datetime_parsed").groupby("cluster_id").first().reset_index()

        # OpenAI 배치 요약 (20개씩)
        articles_to_summarize = [
            {"cluster_id": row["cluster_id"], "title": row["title"], "description": row["description"]}
            for _, row in earliest_articles.iterrows()
        ]

        group_summaries = {}
        for i in range(0, len(articles_to_summarize), 20):
            batch = articles_to_summarize[i:i+20]
            batch_summaries = _call_openai_summarize_batch(batch, openai_key)
            group_summaries.update(batch_summaries)

        # press_release_group 업데이트
        for idx, row in df[press_release_mask].iterrows():
            cluster_id = row["cluster_id"]
            if cluster_id in group_summaries:
                df.at[idx, "press_release_group"] = group_summaries[cluster_id]

        print(f"✅ {len(group_summaries)}개 보도자료 그룹 요약 완료")
        return df

    except Exception as e:
        print(f"⚠️  보도자료 그룹 요약 중 오류: {e}")
        return df
