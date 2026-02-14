"""
preset_pr.py - Press Release Preset Values
보도자료(press_release_group 있는 기사)에 고정값 자동 채우기
LLM 분류 전에 실행되어 보도자료는 LLM 스킵
"""

import json
from datetime import datetime
import pandas as pd


def preset_press_release_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    press_release_group이 있는 기사(보도자료)에 고정값 자동 설정

    고정값:
    - brand_relevance: "관련"
    - brand_relevance_query_keywords: [query 값]
    - sentiment_stage: "중립"
    - danger_level: ""
    - issue_category: ""
    - news_category: "PR/보도자료"
    - news_keyword_summary: press_release_group 값 복사
    - classified_at: 전처리 시간 (ISO 형식)

    Args:
        df: press_release_group 컬럼 포함한 DataFrame

    Returns:
        고정값이 채워진 DataFrame
    """
    df = df.copy()

    # press_release_group 컬럼 확인
    if "press_release_group" not in df.columns:
        print("⚠️  press_release_group 컬럼이 없습니다. 보도자료 전처리 스킵")
        return df

    # 보도자료 필터링 (press_release_group이 비어있지 않은 행)
    pr_mask = (df["press_release_group"].notna()) & (df["press_release_group"] != "")
    pr_count = pr_mask.sum()

    if pr_count == 0:
        print("ℹ️  보도자료가 없습니다. 전처리 스킵")
        return df

    print(f"\n📋 보도자료 전처리 시작: {pr_count}개 기사")

    # 현재 시간 (ISO 형식)
    timestamp = datetime.now().isoformat()

    # 고정값 채우기
    for idx in df[pr_mask].index:
        # 1. brand_relevance: "관련" 고정
        df.at[idx, "brand_relevance"] = "관련"

        # 2. brand_relevance_query_keywords: [query 값] (JSON 배열)
        query_value = df.at[idx, "query"] if pd.notna(df.at[idx, "query"]) else ""
        df.at[idx, "brand_relevance_query_keywords"] = json.dumps([query_value], ensure_ascii=False)

        # 3. sentiment_stage: "중립"
        df.at[idx, "sentiment_stage"] = "중립"

        # 4. danger_level: "" (빈칸)
        df.at[idx, "danger_level"] = ""

        # 5. issue_category: "" (빈칸)
        df.at[idx, "issue_category"] = ""

        # 6. news_category: "PR/보도자료"
        df.at[idx, "news_category"] = "PR/보도자료"

        # 7. news_keyword_summary: press_release_group 값 복사
        pr_group = df.at[idx, "press_release_group"]
        df.at[idx, "news_keyword_summary"] = pr_group if pd.notna(pr_group) else ""

        # 8. classified_at: 전처리 시간
        df.at[idx, "classified_at"] = timestamp

    print(f"✅ 보도자료 {pr_count}개 전처리 완료")
    print(f"  - news_category: PR/보도자료")
    print(f"  - sentiment_stage: 중립")
    print(f"  - brand_relevance: 관련")

    return df
