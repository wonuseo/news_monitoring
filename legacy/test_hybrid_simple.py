#!/usr/bin/env python3
"""
test_hybrid_simple.py - 하이브리드 시스템 간단 테스트
"""

import os
import pandas as pd
from dotenv import load_dotenv
from src.modules.analysis.hybrid import classify_hybrid, print_classification_stats, get_classification_stats

# Load environment
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Sample articles for testing
test_articles = [
    {
        "title": "롯데호텔 화재 발생, 투숙객 20명 대피",
        "description": "서울 중구 롯데호텔에서 화재가 발생해 투숙객 20명이 긴급 대피했다. 소방당국은 화재 원인을 조사 중이다.",
        "query": "롯데호텔",
        "group": "OUR",
        "link": "http://test1.com",
        "pub_datetime": "2026-02-08T10:00:00"
    },
    {
        "title": "신라호텔, 고객정보 유출 의혹 제기",
        "description": "신라호텔의 고객 개인정보가 외부로 유출되었다는 의혹이 제기됐다. 호텔 측은 사실무근이라고 반박했다.",
        "query": "신라호텔",
        "group": "COMPETITOR",
        "link": "http://test2.com",
        "pub_datetime": "2026-02-08T11:00:00"
    },
    {
        "title": "호텔롯데, 새로운 F&B 브랜드 론칭",
        "description": "호텔롯데가 새로운 프리미엄 F&B 브랜드를 선보인다. 업계에서는 긍정적인 반응을 보이고 있다.",
        "query": "호텔롯데",
        "group": "OUR",
        "link": "http://test3.com",
        "pub_datetime": "2026-02-08T12:00:00"
    }
]

# Create DataFrame
df = pd.DataFrame(test_articles)

print("="*80)
print("하이브리드 시스템 테스트")
print("="*80)
print(f"\n테스트 기사: {len(df)}개\n")

for idx, row in df.iterrows():
    print(f"[{idx+1}] {row['title']}")

print("\n" + "="*80)
print("분석 시작...")
print("="*80)

# Run hybrid classification
df_result = classify_hybrid(
    df,
    openai_key=openai_key,
    chunk_size=10,
    dry_run=False,
    max_competitor_classify=50
)

# Print results
print("\n" + "="*80)
print("분석 결과")
print("="*80)

for idx, row in df_result.iterrows():
    print(f"\n[{idx+1}] {row['title']}")
    print(f"  - Brand Scope (RB): {row['brand_scope_rb']}")
    print(f"  - Sentiment (RB): {row['sentiment_rb']}")
    print(f"  - Sentiment (LLM): {row['sentiment_llm']}")
    print(f"  - Sentiment (Final): {row['sentiment_final']} (confidence: {row['sentiment_final_confidence']})")
    print(f"  - Danger (RB): {row['danger_rb']} (score: {row['risk_score_rb']})")
    print(f"  - Danger (LLM): {row['danger_llm']}")
    print(f"  - Danger (Final): {row['danger_final']} (confidence: {row['danger_final_confidence']})")
    print(f"  - Issue Category: {row['issue_category_rb']}")
    print(f"  - Decision Rule: {row['sentiment_final_decision_rule']}")

# Print statistics
stats = get_classification_stats(df_result)
print_classification_stats(stats)

# Save to CSV for inspection
output_path = "data/test_hybrid_result.csv"
df_result.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n💾 결과 저장: {output_path}")

print("\n✅ 테스트 완료!")
