"""
report.py - Report Generation Module
콘솔 리포트 생성
"""

import pandas as pd

from src.utils.group_labels import is_competitor_group, is_our_group


def generate_console_report(df: pd.DataFrame) -> None:
    """콘솔에 수치 요약 출력 (상세 리포트는 Google Sheets 참고)"""
    sentiment_col = "sentiment_stage"
    danger_col = "danger_level"

    our_negative = len(df[(df["group"].map(is_our_group)) & (df[sentiment_col].str.contains("부정", case=False, na=False))]) if sentiment_col in df.columns else 0
    our_positive = len(df[(df["group"].map(is_our_group)) & (df[sentiment_col].str.contains("긍정", case=False, na=False))]) if sentiment_col in df.columns else 0
    comp_total = len(df[df["group"].map(is_competitor_group)]) if "group" in df.columns else 0
    danger_high = len(df[df[danger_col] == "상"]) if danger_col in df.columns else 0
    danger_medium = len(df[df[danger_col] == "중"]) if danger_col in df.columns else 0

    print(f"  우리 브랜드: 부정 {our_negative}건 (위험 상:{danger_high} / 중:{danger_medium}) | 긍정 {our_positive}건 | 경쟁사: {comp_total}건")
