"""
looker_prep.py - Looker Studio Preparation Module
Looker Studio 대시보드 작성을 위한 시계열 컬럼 추가
"""

import pandas as pd
from datetime import datetime


def add_time_series_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Looker Studio 대시보드용 시계열 컬럼 추가

    Args:
        df: 분류된 DataFrame (pub_datetime 컬럼 필요)

    Returns:
        다음 컬럼이 추가된 DataFrame:
        - date_only: YYYY-MM-DD 형식
        - week_number: YYYY-WW 형식 (ISO 8601 주차)
        - month: YYYY-MM 형식
        - article_count: 항상 1
    """
    df = df.copy()

    print("⏰ Looker 시계열 컬럼 추가 중...")

    # pub_datetime을 datetime으로 변환
    if "pub_datetime" not in df.columns:
        print("  ⚠️  pub_datetime 컬럼이 없습니다")
        df["date_only"] = ""
        df["week_number"] = ""
        df["month"] = ""
        df["article_count"] = 1
        return df

    # datetime으로 변환 (에러는 NaT로 처리)
    df["pub_datetime"] = pd.to_datetime(df["pub_datetime"], errors="coerce")

    # date_only: YYYY-MM-DD
    df["date_only"] = df["pub_datetime"].dt.strftime("%Y-%m-%d").fillna("")

    # week_number: YYYY-WW (ISO 8601)
    # isocalendar()을 사용하여 ISO 8601 주차 계산
    df["week_number"] = df["pub_datetime"].dt.isocalendar().year.astype(str) + "-" + \
                        df["pub_datetime"].dt.isocalendar().week.astype(str).str.zfill(2)
    df["week_number"] = df["week_number"].where(df["pub_datetime"].notna(), "")

    # month: YYYY-MM
    df["month"] = df["pub_datetime"].dt.strftime("%Y-%m").fillna("")

    # article_count: 항상 1 (Looker의 COUNT 집계용)
    df["article_count"] = 1

    # 통계
    valid_dates = df["pub_datetime"].notna().sum()
    print(f"  ✅ 완료: {valid_dates}개 기사에 시계열 컬럼 추가")

    return df
