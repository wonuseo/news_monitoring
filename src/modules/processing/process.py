"""
process.py - Data Processing Module
데이터 정규화, 중복 제거, CSV 저장
"""

import re
import hashlib
from html import unescape
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd

# 보도자료 검출/요약은 별도 모듈로 분리
from .press_release_detector import detect_similar_articles, summarize_press_release_groups


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
    - article_id (영구 식별자) 추가
    - article_no (순차 번호) 추가
    """
    print("🔧 데이터 정규화 중...")
    df = df.copy()

    # NaN 값을 빈 문자열로 변환 후 strip_html 적용
    df["title"] = df["title"].fillna("").apply(strip_html)
    df["description"] = df["description"].fillna("").apply(strip_html)
    df["pub_datetime"] = df["pubDate"].fillna("").apply(parse_pubdate)

    # article_id: link 기반 MD5 해시 (영구 식별자, 시스템용)
    df["article_id"] = df["link"].fillna("").apply(
        lambda x: hashlib.md5(str(x).encode()).hexdigest()[:12] if x else ""
    )

    # article_no: 순차 번호 (사람이 읽는 번호, 검토용)
    df["article_no"] = range(1, len(df) + 1)

    print(f"✅ {len(df)}개 기사 정규화 완료 (article_id, article_no 추가)")
    return df


def dedupe_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    중복 제거 (query 병합 포함)
    - originallink 또는 link를 기준으로 중복 제거
    - 최신 기사만 유지하되, query는 파이프(|)로 병합
    - 예: "롯데호텔|호텔롯데" (복수 브랜드에서 수집된 경우)
    """
    print("🔧 중복 기사 제거 중...")
    df = df.copy()

    # Primary key 생성
    df["pk"] = df["originallink"].where(df["originallink"].str.strip() != "", df["link"])

    # 날짜 기준 정렬 (최신 → 오래된)
    df["pub_datetime_sort"] = pd.to_datetime(df["pub_datetime"], errors="coerce")
    df = df.sort_values("pub_datetime_sort", ascending=False, na_position="last")

    # query 병합 (같은 pk의 query들을 파이프로 구분)
    query_merged = df.groupby("pk")["query"].apply(
        lambda x: "|".join(sorted(set(x.dropna().astype(str))))
    ).to_dict()

    # query 컬럼 업데이트
    df["query"] = df["pk"].map(query_merged)

    # 중복 제거 (최신 것만 유지)
    original_count = len(df)
    df_deduped = df.drop_duplicates(subset=["pk"], keep="first")

    # 복수 브랜드 수집 통계
    multi_brand_count = sum(1 for q in df_deduped["query"] if "|" in str(q))

    # 임시 컬럼 제거
    df_deduped = df_deduped.drop(columns=["pk", "pub_datetime_sort"])

    removed = original_count - len(df_deduped)
    print(f"✅ 중복 {removed}개 제거, {len(df_deduped)}개 기사 유지")
    if multi_brand_count > 0:
        print(f"   📌 복수 브랜드 수집: {multi_brand_count}개 기사 (query 파이프(|) 구분)")
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


# detect_similar_articles 함수는 similarity_detector.py로 이동
# 위에서 import됨


def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    """
    CSV 파일로 저장 (UTF-8 BOM 인코딩, 모든 필드 quoting)
    BOM 문자(\ufeff) 및 invisible 문자를 데이터에서 제거
    """
    import csv
    import re
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # 모든 BOM 및 invisible 문자 제거 (데이터 내부)
        df_clean = df.copy()

        # 정규식으로 모든 invisible/제어 문자 일괄 제거
        invisible_pattern = re.compile(
            r'[\ufeff\ufffe'
            r'\u200b-\u200f'
            r'\u2028-\u202f'
            r'\u2060\u180e'
            r'\u00a0\u3000\u00ad'
            r'\ufff9-\ufffc'
            r'\x00-\x08\x0b\x0c\x0e-\x1f'
            r'\x7f-\x9f]'
        )

        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].apply(
                    lambda x: invisible_pattern.sub('', str(x)).strip()
                    if pd.notna(x) and x != '' else x
                )

        df_clean.to_csv(filepath, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)
        print(f"💾 저장: {filepath}")
    except Exception as e:
        print(f"❌ CSV 저장 실패 ({filepath}): {e}")
        print("  → 디스크 공간 또는 권한을 확인하세요.")


# summarize_press_release_groups() 함수는 press_release_detector.py로 이동
# 위에서 import됨


