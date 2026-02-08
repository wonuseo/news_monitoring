"""
process.py - Data Processing Module
데이터 정규화, 중복 제거, Excel 저장
"""

import re
from html import unescape
from datetime import datetime
from pathlib import Path
from typing import Optional
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
    
    df["title"] = df["title"].apply(strip_html)
    df["description"] = df["description"].apply(strip_html)
    df["pub_datetime"] = df["pubDate"].apply(parse_pubdate)
    
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
    openai_key: str = None
) -> pd.DataFrame:
    """
    DataFrame에 언론사 정보 추가 (wrapper for media_classify.add_media_columns)

    Args:
        df: 처리된 DataFrame
        spreadsheet: gspread Spreadsheet 객체 (선택사항)
        openai_key: OpenAI API 키 (선택사항)

    Returns:
        언론사 정보 컬럼이 추가된 DataFrame
    """
    try:
        from src.modules.enhancement.media_classify import add_media_columns
        return add_media_columns(df, spreadsheet, openai_key)
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
    - 모든 기사 유지 (비파괴적 라벨링만 수행)

    Args:
        df: 정규화 및 중복 제거된 DataFrame
        similarity_threshold: 코사인 유사도 임계값 (기본값: 0.8)
        min_text_length: 최소 텍스트 길이 (기본값: 10자)

    Returns:
        'source' 컬럼이 추가된 DataFrame
    """
    print("🔍 유사 기사 검색 중...")
    df = df.copy()

    # source 컬럼 초기화 (모두 빈 문자열)
    df["source"] = ""

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

        # 유사 기사 그룹 식별
        similar_groups = set()
        for i in range(len(valid_indices)):
            for j in range(i + 1, len(valid_indices)):
                if similarity_matrix[i, j] >= similarity_threshold:
                    # 유사한 쌍 발견 - 두 기사 모두 '보도자료'로 표시
                    idx_i = valid_indices[i]
                    idx_j = valid_indices[j]
                    similar_groups.add(idx_i)
                    similar_groups.add(idx_j)

        # source 컬럼 업데이트
        for idx in similar_groups:
            df.at[idx, "source"] = "보도자료"

        similar_count = len(similar_groups)
        unique_count = len(valid_indices) - similar_count

        print(f"✅ {similar_count}개 기사를 '보도자료'로 표시, {unique_count}개 기사는 독립 기사")

    except MemoryError:
        print("⚠️  메모리 부족으로 유사도 검사를 건너뜁니다.")
    except Exception as e:
        print(f"⚠️  유사도 검사 중 오류 발생: {e}")

    # 임시 컬럼 제거
    df = df.drop(columns=["search_text"])
    return df


def save_excel(df: pd.DataFrame, filepath: Path, sheet_name: str = "data") -> None:
    """Excel 파일로 저장"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(filepath, index=False, sheet_name=sheet_name, engine='openpyxl')
    print(f"💾 저장: {filepath}")
