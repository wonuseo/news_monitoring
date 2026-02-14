"""
classification_stats.py - Classification Statistics Generation & Reporting
분류 결과 통계 생성 및 출력
"""

from typing import Dict
import pandas as pd


def get_classification_stats(df: pd.DataFrame) -> Dict:
    """
    분류 통계 생성

    Args:
        df: 분석 결과 DataFrame

    Returns:
        통계 딕셔너리
    """
    stats = {}

    # Sentiment distribution
    if "sentiment_stage" in df.columns:
        sentiment_counts = df["sentiment_stage"].value_counts().to_dict()
        stats["sentiment_stage"] = sentiment_counts

    # Danger distribution
    if "danger_level" in df.columns:
        danger_counts = df["danger_level"].value_counts().to_dict()
        stats["danger_level"] = danger_counts

    # Brand relevance distribution
    if "brand_relevance" in df.columns:
        relevance_counts = df["brand_relevance"].value_counts().to_dict()
        stats["brand_relevance"] = relevance_counts

    # Issue category distribution
    if "issue_category" in df.columns:
        category_counts = df["issue_category"].value_counts().to_dict()
        stats["issue_category"] = category_counts

    # News category distribution
    if "news_category" in df.columns:
        news_cat_counts = df["news_category"].value_counts().to_dict()
        stats["news_category"] = news_cat_counts

    return stats


def print_classification_stats(stats: Dict):
    """
    분류 통계 출력

    Args:
        stats: get_classification_stats() 결과
    """
    print("\n📊 분류 통계:")

    if "sentiment_stage" in stats:
        print("\n  감정 단계:")
        for sentiment, count in stats["sentiment_stage"].items():
            print(f"    - {sentiment}: {count}개")

    if "danger_level" in stats:
        print("\n  위험도:")
        for danger, count in stats["danger_level"].items():
            print(f"    - {danger}: {count}개")

    if "brand_relevance" in stats:
        print("\n  브랜드 관련성:")
        for relevance, count in stats["brand_relevance"].items():
            print(f"    - {relevance}: {count}개")

    if "issue_category" in stats:
        print("\n  이슈 카테고리 (상위 5개):")
        for category, count in sorted(stats["issue_category"].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"    - {category}: {count}개")

    if "news_category" in stats:
        print("\n  뉴스 카테고리 (상위 5개):")
        for category, count in sorted(stats["news_category"].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"    - {category}: {count}개")
