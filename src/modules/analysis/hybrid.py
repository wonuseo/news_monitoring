"""
hybrid.py - Hybrid Analysis Orchestrator
Rule-Based + LLM 하이브리드 분석 시스템
"""

import time
import json
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

from .rule_engine import load_rules, analyze_batch_rb
from .llm_engine import load_prompts, analyze_article_llm


def classify_hybrid(
    df: pd.DataFrame,
    openai_key: str,
    chunk_size: int = 50,
    dry_run: bool = False,
    max_competitor_classify: int = 50
) -> pd.DataFrame:
    """
    하이브리드 분석 메인 함수

    3단계 프로세스:
    1. Rule-Based 분석 (전체 기사)
    2. LLM 분석 (조건부: 우리 브랜드 전체 + 경쟁사 상위 N개)
    3. 결과 병합 및 DataFrame 반환

    Args:
        df: 입력 DataFrame (title, description 필수)
        openai_key: OpenAI API 키
        chunk_size: LLM 배치 크기
        dry_run: True면 LLM 호출 생략 (Rule-Based만)
        max_competitor_classify: 경쟁사별 최대 분류 개수

    Returns:
        분석 결과가 추가된 DataFrame
    """
    df = df.copy()

    # 초기화: 모든 결과 컬럼
    rb_columns = [
        "brand_mentions",
        "brand_scope_rb",
        "sentiment_rb",
        "danger_rb",
        "risk_score_rb",
        "issue_category_rb",
        "coverage_themes_rb",
        "reason_codes_rb",
        "matched_rules_rb",
        "score_breakdown_rb"
    ]

    llm_columns = [
        "sentiment_llm",
        "sentiment_llm_confidence",
        "sentiment_llm_evidence",
        "sentiment_llm_rationale",
        "sentiment_final",
        "sentiment_final_confidence",
        "sentiment_final_decision_rule",
        "sentiment_final_evidence",
        "sentiment_final_rationale",
        "danger_llm",
        "danger_llm_confidence",
        "danger_llm_evidence",
        "danger_llm_rationale",
        "danger_final",
        "danger_final_confidence",
        "danger_final_decision_rule",
        "danger_final_evidence",
        "danger_final_rationale",
        "issue_category_llm",
        "coverage_themes_llm",
        "category_llm_confidence",
        "category_llm_evidence",
        "category_llm_rationale",
        "issue_category_final",
        "coverage_themes_final",
        "category_final_confidence",
        "category_final_decision_rule",
        "category_final_evidence",
        "category_final_rationale"
    ]

    # 컬럼 초기화
    for col in rb_columns + llm_columns:
        df[col] = None

    df["classified_at"] = ""

    # Load configs
    print("\n🔧 하이브리드 분석 시작...")
    rules = load_rules()
    prompts_config = load_prompts()

    # ========================================
    # STEP 1: Rule-Based 분석 (전체 기사)
    # ========================================
    print("\n[1/3] Rule-Based 분석 중...")
    articles = df[["title", "description"]].fillna("").to_dict("records")
    rb_results = analyze_batch_rb(articles, rules)

    # RB 결과를 DataFrame에 병합
    for idx, rb_result in enumerate(rb_results):
        for col in rb_columns:
            if col in rb_result:
                # JSON serializable 타입으로 변환
                value = rb_result[col]
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False)
                df.at[idx, col] = value

    print(f"✅ Rule-Based 분석 완료: {len(df)}개 기사")

    if dry_run:
        print("🔬 DRY RUN 모드: LLM 분석 생략")
        return df

    # ========================================
    # STEP 2: LLM 분석 대상 선택
    # ========================================
    print("\n[2/3] LLM 분석 대상 선택 중...")

    # 분류 대상 필터링 (우리 브랜드 전체 + 경쟁사 상위 N개)
    indices_to_classify = []
    competitor_count = {}

    for idx, row in df.iterrows():
        group = row.get("group", "")

        # 우리 브랜드는 전부
        if group == "OUR":
            indices_to_classify.append(idx)
        # 경쟁사는 각 브랜드당 최신 N개
        elif group == "COMPETITOR":
            query = row.get("query", "")
            count = competitor_count.get(query, 0)
            if count < max_competitor_classify:
                competitor_count[query] = count + 1
                indices_to_classify.append(idx)

    if len(indices_to_classify) == 0:
        print("⚠️  LLM 분석 대상 없음")
        return df

    print(f"  선택된 기사: {len(indices_to_classify)}개")
    print(f"  - 우리 브랜드: {sum(1 for idx in indices_to_classify if df.at[idx, 'group'] == 'OUR')}개")
    print(f"  - 경쟁사: {sum(1 for idx in indices_to_classify if df.at[idx, 'group'] == 'COMPETITOR')}개")

    # ========================================
    # STEP 3: LLM 분석 (청크 단위)
    # ========================================
    print(f"\n[3/3] LLM 분석 중 (청크 크기: {chunk_size})...")

    timestamp = datetime.now().isoformat()
    total = len(indices_to_classify)
    total_chunks = (total + chunk_size - 1) // chunk_size

    for chunk_start in range(0, total, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total)
        chunk_indices = indices_to_classify[chunk_start:chunk_end]
        chunk_num = (chunk_start // chunk_size) + 1

        print(f"\n--- 청크 {chunk_num}/{total_chunks} ({len(chunk_indices)}개 기사) ---")

        for i, idx in enumerate(chunk_indices):
            article = {
                "title": df.at[idx, "title"],
                "description": df.at[idx, "description"]
            }

            # RB 결과 파싱 (JSON string → dict/list)
            rb_result = {}
            for col in rb_columns:
                value = df.at[idx, col]
                if pd.isna(value):
                    continue
                # JSON string이면 파싱
                if isinstance(value, str) and (value.startswith("{") or value.startswith("[")):
                    try:
                        rb_result[col] = json.loads(value)
                    except:
                        rb_result[col] = value
                else:
                    rb_result[col] = value

            # LLM 분석 수행
            try:
                llm_result = analyze_article_llm(article, rb_result, prompts_config, openai_key)

                # LLM 결과를 DataFrame에 병합
                for col in llm_columns:
                    if col in llm_result:
                        value = llm_result[col]
                        # JSON serializable 타입으로 변환
                        if isinstance(value, (dict, list)):
                            value = json.dumps(value, ensure_ascii=False)
                        df.at[idx, col] = value

                df.at[idx, "classified_at"] = timestamp

                # Progress indicator
                if (i + 1) % 10 == 0:
                    print(f"  진행: {i + 1}/{len(chunk_indices)}개 완료")

                # Rate limiting
                time.sleep(0.5)

            except Exception as e:
                print(f"  ⚠️  기사 {idx} LLM 분석 실패: {e}")
                # Fallback: RB 결과 사용
                df.at[idx, "sentiment_final"] = df.at[idx, "sentiment_rb"]
                df.at[idx, "sentiment_final_decision_rule"] = f"LLM failed: {str(e)}"
                df.at[idx, "danger_final"] = df.at[idx, "danger_rb"]
                df.at[idx, "danger_final_decision_rule"] = f"LLM failed: {str(e)}"
                df.at[idx, "classified_at"] = timestamp

        print(f"  ✅ 청크 {chunk_num}/{total_chunks} 완료")
        time.sleep(1)

    print(f"\n✅ 하이브리드 분석 완료: {total}개 기사 처리")

    return df


def get_classification_stats(df: pd.DataFrame) -> Dict:
    """
    분류 통계 생성

    Args:
        df: 분석 결과 DataFrame

    Returns:
        통계 딕셔너리
    """
    stats = {}

    # Sentiment distribution (Final)
    if "sentiment_final" in df.columns:
        sentiment_counts = df["sentiment_final"].value_counts().to_dict()
        stats["sentiment_final"] = sentiment_counts

    # Danger distribution (Final)
    if "danger_final" in df.columns:
        danger_counts = df["danger_final"].value_counts().to_dict()
        stats["danger_final"] = danger_counts

    # Brand scope distribution
    if "brand_scope_rb" in df.columns:
        scope_counts = df["brand_scope_rb"].value_counts().to_dict()
        stats["brand_scope_rb"] = scope_counts

    # Issue category distribution
    if "issue_category_rb" in df.columns:
        category_counts = df["issue_category_rb"].value_counts().to_dict()
        stats["issue_category_rb"] = category_counts

    # Average confidence scores
    if "sentiment_final_confidence" in df.columns:
        avg_sentiment_conf = df["sentiment_final_confidence"].apply(
            lambda x: float(x) if pd.notna(x) and x != "" else 0
        ).mean()
        stats["avg_sentiment_confidence"] = round(avg_sentiment_conf, 3)

    if "danger_final_confidence" in df.columns:
        avg_danger_conf = df["danger_final_confidence"].apply(
            lambda x: float(x) if pd.notna(x) and x != "" else 0
        ).mean()
        stats["avg_danger_confidence"] = round(avg_danger_conf, 3)

    return stats


def print_classification_stats(stats: Dict):
    """
    분류 통계 출력

    Args:
        stats: get_classification_stats() 결과
    """
    print("\n📊 분류 통계:")

    if "sentiment_final" in stats:
        print("\n  Sentiment (Final):")
        for sentiment, count in stats["sentiment_final"].items():
            print(f"    - {sentiment}: {count}개")

    if "danger_final" in stats:
        print("\n  Danger (Final):")
        for danger, count in stats["danger_final"].items():
            print(f"    - {danger}: {count}개")

    if "brand_scope_rb" in stats:
        print("\n  Brand Scope:")
        for scope, count in stats["brand_scope_rb"].items():
            print(f"    - {scope}: {count}개")

    if "issue_category_rb" in stats:
        print("\n  Issue Category:")
        for category, count in sorted(stats["issue_category_rb"].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"    - {category}: {count}개")

    if "avg_sentiment_confidence" in stats:
        print(f"\n  평균 Sentiment 신뢰도: {stats['avg_sentiment_confidence']:.2%}")

    if "avg_danger_confidence" in stats:
        print(f"  평균 Danger 신뢰도: {stats['avg_danger_confidence']:.2%}")
