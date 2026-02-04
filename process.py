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


def save_excel(df: pd.DataFrame, filepath: Path, sheet_name: str = "data") -> None:
    """Excel 파일로 저장"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(filepath, index=False, sheet_name=sheet_name, engine='openpyxl')
    print(f"💾 저장: {filepath}")
