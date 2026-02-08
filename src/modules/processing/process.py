"""
process.py - Data Processing Module
데이터 정규화, 중복 제거, Excel 저장
"""

import re
import json
import time
import requests
from html import unescape
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd


def strip_html(text: str) -> str:
    """HTML 태그 제거 및 엔티티 디코딩"""
    if not text:
        return ""
    # HTML 태그 제거
    text = re.sub(r'<[^>]+>', '', text)
    # HTML 엔티티 디코딩
    text = unescape(text)
    return text.strip()


def parse_pubdate(pubdate_str: str) -> Optional[str]:
    """
    네이버 pubDate를 ISO 형식으로 변환
    
    예: 'Mon, 03 Feb 2026 10:30:00 +0900' → '2026-02-03T10:30:00+09:00'
    """
    if not pubdate_str:
        return None
    
    try:
        dt = datetime.strptime(pubdate_str, "%a, %d %b %Y %H:%M:%S %z")
        return dt.isoformat()
    except Exception as e:
        print(f"⚠️  날짜 파싱 실패 '{pubdate_str}': {e}")
        return None


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    데이터 정규화
    - HTML 태그 제거
    - 날짜 파싱
    """
    print("🔧 데이터 정규화 중...")
    df = df.copy()

    # NaN 값을 빈 문자열로 변환 후 strip_html 적용
    df["title"] = df["title"].fillna("").apply(strip_html)
    df["description"] = df["description"].fillna("").apply(strip_html)
    df["pub_datetime"] = df["pubDate"].fillna("").apply(parse_pubdate)
    
    print(f"✅ {len(df)}개 기사 정규화 완료")
    return df


def dedupe_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    중복 제거
    - originallink 또는 link를 기준으로 중복 제거
    - 최신 기사만 유지
    """
    print("🔧 중복 기사 제거 중...")
    df = df.copy()

    # Primary key 생성
    df["pk"] = df["originallink"].where(df["originallink"].str.strip() != "", df["link"])

    # 날짜 기준 정렬 (최신 → 오래된)
    df["pub_datetime_sort"] = pd.to_datetime(df["pub_datetime"], errors="coerce")
    df = df.sort_values("pub_datetime_sort", ascending=False, na_position="last")

    # 중복 제거 (최신 것만 유지)
    original_count = len(df)
    df_deduped = df.drop_duplicates(subset=["pk"], keep="first")

    # 임시 컬럼 제거
    df_deduped = df_deduped.drop(columns=["pk", "pub_datetime_sort"])

    removed = original_count - len(df_deduped)
    print(f"✅ 중복 {removed}개 제거, {len(df_deduped)}개 기사 유지")
    return df_deduped.reset_index(drop=True)


def enrich_with_media_info(
    df: pd.DataFrame,
    spreadsheet=None,
    openai_key: str = None,
    csv_path: Path = None
) -> pd.DataFrame:
    """
    DataFrame에 언론사 정보 추가 (wrapper for media_classify.add_media_columns)

    Args:
        df: 처리된 DataFrame
        spreadsheet: gspread Spreadsheet 객체 (선택사항)
        openai_key: OpenAI API 키 (선택사항)
        csv_path: media_directory CSV 경로 (선택사항)

    Returns:
        언론사 정보 컬럼이 추가된 DataFrame
    """
    try:
        from src.modules.processing.media_classify import add_media_columns
        return add_media_columns(df, spreadsheet, openai_key, csv_path)
    except ImportError:
        print("⚠️  media_classify 모듈을 로드할 수 없습니다.")
        return df
    except Exception as e:
        print(f"⚠️  언론사 정보 추가 중 오류: {e}")
        return df


def detect_similar_articles(
    df: pd.DataFrame,
    similarity_threshold: float = 0.8,
    min_text_length: int = 10
) -> pd.DataFrame:
    """
    내용 기반 유사도 검사로 보도자료 식별
    - TF-IDF + 코사인 유사도로 중복 기사 감지
    - 유사도 >= threshold인 기사들을 '보도자료'로 표시
    - Union-Find로 그룹화
    - 모든 기사 유지 (비파괴적 라벨링만 수행)

    Args:
        df: 정규화 및 중복 제거된 DataFrame
        similarity_threshold: 코사인 유사도 임계값 (기본값: 0.8)
        min_text_length: 최소 텍스트 길이 (기본값: 10자)

    Returns:
        'source', 'group_id' 컬럼이 추가된 DataFrame
    """
    print("🔍 유사 기사 검색 중...")
    df = df.copy()

    # 컬럼 초기화
    df["source"] = "일반기사"
    df["group_id"] = ""

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("⚠️  scikit-learn이 설치되지 않았습니다. 유사도 검사를 건너뜁니다.")
        return df

    # 제목과 설명을 결합하여 검색 텍스트 생성
    df["search_text"] = (
        df["title"].fillna("") + " " + df["description"].fillna("")
    ).str.strip()

    # 최소 길이 조건을 만족하는 기사 필터링
    valid_mask = df["search_text"].str.len() >= min_text_length
    valid_indices = df[valid_mask].index.tolist()

    if len(valid_indices) < 2:
        print(f"✅ 검색 대상 기사가 {len(valid_indices)}개로 너무 적습니다. 유사도 검사 스킵")
        df = df.drop(columns=["search_text"])
        return df

    try:
        # TF-IDF 벡터화 (한글 최적화: 문자 단위 n-gram 사용)
        print(f"  - 유사도 임계값: {similarity_threshold}")
        print(f"  - 검색 대상: {len(valid_indices)}개 기사")

        vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(2, 4),
            min_df=2,
            max_df=0.9,
            max_features=5000
        )

        # 유효한 기사만 벡터화
        valid_texts = df.loc[valid_indices, "search_text"].tolist()
        tfidf_matrix = vectorizer.fit_transform(valid_texts)

        # 코사인 유사도 계산 (희소 행렬 사용)
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Union-Find로 유사 기사 그룹화
        parent = {i: i for i in range(len(valid_indices))}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # 유사한 기사들을 같은 그룹으로 묶기
        for i in range(len(valid_indices)):
            for j in range(i + 1, len(valid_indices)):
                if similarity_matrix[i, j] >= similarity_threshold:
                    union(i, j)

        # 그룹별 기사 수집
        groups = {}
        for i in range(len(valid_indices)):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)

        # source, group_id 컬럼 업데이트
        # 2개 이상의 기사가 있는 그룹만 필터링하고 정렬
        sorted_groups = sorted(
            [(gid, indices) for gid, indices in groups.items() if len(indices) > 1],
            key=lambda x: min(x[1])  # 첫 번째 기사의 인덱스 기준 정렬
        )

        # 1부터 순차적으로 그룹 ID 할당
        for new_id, (old_id, indices) in enumerate(sorted_groups, start=1):
            for i in indices:
                idx = valid_indices[i]
                df.at[idx, "source"] = "보도자료"
                df.at[idx, "group_id"] = f"group_{new_id}"

        similar_count = sum(len(indices) for indices in groups.values() if len(indices) > 1)
        unique_count = len(valid_indices) - similar_count

        print(f"✅ {similar_count}개 기사를 '보도자료'로 표시 ({len(groups)}개 그룹), {unique_count}개 기사는 독립 기사")

    except MemoryError:
        print("⚠️  메모리 부족으로 유사도 검사를 건너뜁니다.")
    except Exception as e:
        print(f"⚠️  유사도 검사 중 오류 발생: {e}")

    # 임시 컬럼 제거
    df = df.drop(columns=["search_text"])
    return df


def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    """CSV 파일로 저장 (UTF-8 BOM 인코딩)"""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"💾 저장: {filepath}")
    except Exception as e:
        print(f"❌ CSV 저장 실패 ({filepath}): {e}")
        print("  → 디스크 공간 또는 권한을 확인하세요.")


def _call_openai_summarize_batch(
    articles: List[Dict],
    openai_key: str,
    retry: bool = True
) -> Dict[str, str]:
    """OpenAI API로 기사 배치 요약"""
    articles_text = "\n".join([
        f"[{a['group_id']}]\n제목: {a['title']}\n설명: {a['description'][:200]}"
        for a in articles
    ])

    prompt = f"""각 기사를 3단어 내외로 요약하세요.

기사:
{articles_text}

JSON 배열 형식으로만 응답:
[{{"group_id":"group_0","summary":"요약"}},...]

규칙:
- 3단어 이내
- 명사형 위주
- JSON만 출력"""

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": min(len(articles) * 30, 4096)  # 최소 30토큰/항목, 최대 4096
            },
            timeout=60
        )

        if response.status_code == 429 and retry:
            print("  (Rate limit, 5초 대기 후 재시도)")
            time.sleep(5)
            return _call_openai_summarize_batch(articles, openai_key, retry=False)

        if response.status_code != 200:
            print(f"  (API 오류 {response.status_code}, 기본값 사용)")
            return {a["group_id"]: a["title"][:15] for a in articles}

        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

        try:
            summaries = json.loads(content)
            return {item["group_id"]: item["summary"] for item in summaries}
        except json.JSONDecodeError:
            if retry:
                print("  (JSON 파싱 실패, 재시도)")
                time.sleep(2)
                return _call_openai_summarize_batch(articles, openai_key, retry=False)
            return {a["group_id"]: a["title"][:15] for a in articles}

    except Exception as e:
        print(f"  (오류: {e}, 기본값 사용)")
        return {a["group_id"]: a["title"][:15] for a in articles}


def summarize_press_release_groups(
    df: pd.DataFrame,
    openai_key: str
) -> pd.DataFrame:
    """
    보도자료 그룹별로 가장 이른 기사를 OpenAI로 요약

    Args:
        df: group_id, pub_datetime, title, description 컬럼 포함
        openai_key: OpenAI API 키

    Returns:
        press_release_group 컬럼이 추가된 DataFrame
    """
    print("📝 보도자료 그룹 요약 생성 중...")
    df = df.copy()
    df["press_release_group"] = ""

    press_release_mask = (df["source"] == "보도자료") & (df["group_id"] != "")
    if press_release_mask.sum() == 0:
        print("  ℹ️  보도자료 그룹이 없습니다.")
        return df

    try:
        # 그룹별 가장 이른 기사 선택
        pr_df = df[press_release_mask].copy()
        pr_df["pub_datetime_parsed"] = pd.to_datetime(pr_df["pub_datetime"], errors="coerce")
        earliest_articles = pr_df.sort_values("pub_datetime_parsed").groupby("group_id").first().reset_index()

        # OpenAI 배치 요약 (100개씩)
        articles_to_summarize = [
            {"group_id": row["group_id"], "title": row["title"], "description": row["description"]}
            for _, row in earliest_articles.iterrows()
        ]

        group_summaries = {}
        for i in range(0, len(articles_to_summarize), 100):
            batch = articles_to_summarize[i:i+100]
            batch_summaries = _call_openai_summarize_batch(batch, openai_key)
            group_summaries.update(batch_summaries)

        # press_release_group 업데이트
        for idx, row in df[press_release_mask].iterrows():
            group_id = row["group_id"]
            if group_id in group_summaries:
                df.at[idx, "press_release_group"] = group_summaries[group_id]

        print(f"✅ {len(group_summaries)}개 보도자료 그룹 요약 완료")
        return df

    except Exception as e:
        print(f"⚠️  보도자료 그룹 요약 중 오류: {e}")
        return df


def save_excel(df: pd.DataFrame, filepath: Path, sheet_name: str = "data") -> None:
    """Excel 파일로 저장"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(filepath, index=False, sheet_name=sheet_name, engine='openpyxl')
    print(f"💾 저장: {filepath}")
