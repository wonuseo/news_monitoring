#!/usr/bin/env python3
"""
test_category_llm.py - Category LLM 테스트 스크립트
"""

import os
import pandas as pd
from dotenv import load_dotenv
from src.modules.analysis.rule_engine import analyze_batch_rb, load_rules
from src.modules.analysis.llm_engine import analyze_article_llm, load_prompts

# 환경 변수 로드
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# 설정 로드
rules = load_rules()
prompts_config = load_prompts()

# raw.csv에서 샘플 5개 로드
df = pd.read_csv("data/raw.csv", encoding="utf-8-sig").head(5)
print(f"📊 {len(df)}개 기사 로드\n")

# Rule-Based 분석
articles = df[["title", "description"]].fillna("").to_dict("records")
rb_results = analyze_batch_rb(articles, rules)
print("✅ Rule-Based 분석 완료\n")

# LLM 분석 (1개만 테스트)
print("🤖 LLM 분석 시작...\n")
for idx in range(min(1, len(df))):  # 1개만 테스트
    print(f"[{idx+1}/{len(df)}] 분석 중...")
    article = articles[idx]
    rb_result = rb_results[idx]

    print(f"  제목: {article['title'][:50]}...")
    print(f"  RB 결과:")
    print(f"    - brand_scope_rb: {rb_result.get('brand_scope_rb', '')}")
    print(f"    - sentiment_rb: {rb_result.get('sentiment_rb', '')}")
    print(f"    - issue_category_rb: {rb_result.get('issue_category_rb', '')}")
    print(f"    - coverage_themes_rb: {rb_result.get('coverage_themes_rb', [])}")

    # LLM 분석
    llm_result = analyze_article_llm(article, rb_result, prompts_config, openai_key)

    print(f"\n  LLM 결과:")
    print(f"    - sentiment_llm: {llm_result.get('sentiment_llm', '')}")
    print(f"    - sentiment_final: {llm_result.get('sentiment_final', '')}")
    print(f"    - sentiment_final_decision_rule: {llm_result.get('sentiment_final_decision_rule', '')}")

    print(f"\n  Category LLM 결과:")
    print(f"    - issue_category_llm: {llm_result.get('issue_category_llm', '')}")
    print(f"    - coverage_themes_llm: {llm_result.get('coverage_themes_llm', [])}")
    print(f"    - category_llm_confidence: {llm_result.get('category_llm_confidence', 0):.2f}")
    print(f"    - category_llm_rationale: {llm_result.get('category_llm_rationale', '')[:100]}...")

    print(f"\n  Category Final 결과:")
    print(f"    - issue_category_final: {llm_result.get('issue_category_final', '')}")
    print(f"    - coverage_themes_final: {llm_result.get('coverage_themes_final', [])}")
    print(f"    - category_final_confidence: {llm_result.get('category_final_confidence', 0):.2f}")
    print(f"    - category_final_decision_rule: {llm_result.get('category_final_decision_rule', '')}")
    print(f"    - category_final_rationale: {llm_result.get('category_final_rationale', '')[:100]}...")

    print("\n" + "="*80 + "\n")

print("✅ 테스트 완료!")
