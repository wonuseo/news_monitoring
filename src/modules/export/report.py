"""
report.py - Report Generation Module
콘솔 리포트 생성
"""

import pandas as pd

from src.utils.group_labels import is_competitor_group, is_our_group


def generate_console_report(df: pd.DataFrame) -> None:
    """콘솔에 요약 리포트 출력"""
    print("\n" + "="*80)
    print("📊 뉴스 모니터링 리포트")
    print("="*80)

    # 컬럼 이름 (LLM 분류 시스템)
    sentiment_col = "sentiment_stage"
    danger_col = "danger_level"
    category_col = "issue_category"
    news_category_col = "news_category"

    # Section A: 우리 브랜드 - 부정 기사 (위험도 높은 순)
    print("\n[A] 우리 브랜드 - 부정 기사 (위험도 높은 순)")
    print("-" * 80)
    our_negative = df[(df["group"].map(is_our_group)) & (df[sentiment_col].str.contains("부정", case=False, na=False))].copy()

    # 위험도 정렬 (HIGH > MEDIUM > LOW)
    risk_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "상": 0, "중": 1, "하": 2, "-": 3, "": 4}
    our_negative["risk_order"] = our_negative[danger_col].map(risk_order).fillna(4)
    our_negative = our_negative.sort_values(["risk_order", "pub_datetime"], ascending=[True, False])
    our_negative = our_negative.head(10)

    if len(our_negative) == 0:
        print("✅ 부정 기사 없음")
    else:
        for idx, row in our_negative.iterrows():
            risk_display = row.get(danger_col, "-")
            category_display = row.get(category_col, "-")
            news_summary = row.get("news_keyword_summary", "")
            print(f"\n[{risk_display}] {row['query']} | {category_display}")
            print(f"제목: {row['title']}")
            if news_summary and news_summary not in ["-", ""]:
                print(f"요약: {news_summary}")
            print(f"날짜: {row['pubDate']}")
            print(f"링크: {row['link']}")

    # Section B: 우리 브랜드 - 긍정 기사
    print("\n" + "-" * 80)
    print("[B] 우리 브랜드 - 긍정 기사 (최신순)")
    print("-" * 80)
    our_positive = df[(df["group"].map(is_our_group)) & (df[sentiment_col].str.contains("긍정", case=False, na=False))]
    our_positive = our_positive.sort_values("pub_datetime", ascending=False).head(10)

    if len(our_positive) == 0:
        print("긍정 기사 없음")
    else:
        for idx, row in our_positive.iterrows():
            category_display = row.get(category_col, "-")
            print(f"\n{row['query']} | {category_display}")
            print(f"제목: {row['title']}")
            print(f"날짜: {row['pubDate']}")
            print(f"링크: {row['link']}")

    # Section C: 경쟁사 하이라이트
    print("\n" + "-" * 80)
    print("[C] 경쟁사 하이라이트 (최신순)")
    print("-" * 80)
    comp = df[df["group"].map(is_competitor_group)]
    comp = comp.sort_values("pub_datetime", ascending=False).head(10)

    if len(comp) == 0:
        print("경쟁사 기사 없음")
    else:
        for idx, row in comp.iterrows():
            sentiment_val = row.get(sentiment_col, "")
            sentiment_icon = ""
            if "긍정" in str(sentiment_val):
                sentiment_icon = "😊"
            elif "부정" in str(sentiment_val):
                sentiment_icon = "😟"
            else:
                sentiment_icon = "😐"

            category_display = row.get(category_col, "-")
            news_summary = row.get("news_keyword_summary", "")
            risk_display = row.get(danger_col, "")
            risk_info = f" | 위험도: {risk_display}" if risk_display not in ["-", "", None] else ""
            print(f"\n{sentiment_icon} {row['query']} | {category_display}{risk_info}")
            print(f"제목: {row['title']}")
            if news_summary and news_summary not in ["-", ""]:
                print(f"요약: {news_summary}")
            print(f"날짜: {row['pubDate']}")
            print(f"링크: {row['link']}")

    print("\n" + "="*80)
