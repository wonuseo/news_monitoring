"""
category_discovery.py - 카테고리 발견 메커니즘

issue_category="기타"로 분류된 기사들을 LLM으로 분석하여,
반복되는 패턴을 발견하면 새 카테고리 후보를 제안한다.

사용 흐름:
  1. 매 실행 후 "기타" 기사들을 자동 수집
  2. LLM이 패턴 분석 → 새 카테고리 후보 제안
  3. 콘솔 출력 + Sheets events 탭에 category_suggestion 태그로 기록
  4. 사람이 검토 → prompts.yaml labels.issue_category_kr에 추가
  5. 다음 실행부터 자동 반영 (Python 코드 수정 불필요)
"""

import json
from typing import List, Dict, Optional

import pandas as pd

from src.modules.analysis.llm_engine import call_openai_structured, load_prompts

# "기타" 기사가 이 수 이상일 때만 분석 실행 (소수 노이즈 방지)
MIN_GITA_COUNT = 3

_DISCOVERY_SCHEMA = {
    "type": "object",
    "required": ["suggestions"],
    "additionalProperties": False,
    "properties": {
        "suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["category_name", "count", "rationale", "article_ids"],
                "additionalProperties": False,
                "properties": {
                    "category_name": {"type": "string"},
                    "count": {"type": "integer"},
                    "rationale": {"type": "string"},
                    "article_ids": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            }
        }
    }
}


def discover_new_categories(
    df: pd.DataFrame,
    openai_key: str,
    logger=None,
) -> Optional[List[Dict]]:
    """
    issue_category="기타" 기사들을 분석하여 새 카테고리 후보를 제안.

    Args:
        df: 분류 완료된 전체 DataFrame (issue_category 컬럼 필요)
        openai_key: OpenAI API 키
        logger: RunLogger 인스턴스 (None이면 Sheets 기록 생략)

    Returns:
        제안 리스트 [{"category_name": ..., "count": ..., "rationale": ..., "article_ids": [...]}, ...]
        또는 None (기타 기사 부족 / 제안 없음)
    """
    if "issue_category" not in df.columns:
        return None

    df_gita = df[df["issue_category"] == "기타"].copy()

    if len(df_gita) < MIN_GITA_COUNT:
        print(f"\n💡 카테고리 발견: '기타' {len(df_gita)}건 (최소 {MIN_GITA_COUNT}건 미만, 분석 생략)")
        return None

    print(f"\n💡 카테고리 발견 분석 중 ('기타' {len(df_gita)}건)...")

    # 현재 카테고리 목록을 prompts.yaml에서 읽어 LLM에 전달
    prompts_config = load_prompts()
    current_cats = prompts_config.get("labels", {}).get("issue_category_kr", [])
    current_cats_str = ", ".join(current_cats)

    # 기사 목록 구성: article_id + title + keyword_summary
    article_lines = []
    for _, row in df_gita.iterrows():
        article_id = str(row.get("article_id", ""))
        title = str(row.get("title", ""))
        summary = str(row.get("news_keyword_summary", "")) or str(row.get("description", ""))[:80]
        article_lines.append(f"- [{article_id}] {title} / {summary}")

    articles_text = "\n".join(article_lines)

    system_prompt = f"""너는 뉴스 카테고리 분석 전문가다.
호텔 브랜드 뉴스 모니터링 시스템에서 '기타'로 분류된 기사들을 분석하여,
반복되는 패턴이 있다면 새로운 issue_category 후보를 제안한다.

현재 issue_category 목록: {current_cats_str}

제안 규칙:
- 기존 카테고리와 명확히 다른 패턴만 제안한다
- 최소 2건 이상의 기사가 해당하는 패턴만 제안한다
- 카테고리명은 한국어 10자 이내, 슬래시(/) 구분 형식 권장 (예: "환경/소음")
- "기타"라는 이름 사용 금지
- 패턴이 없으면 suggestions를 빈 배열로 반환한다"""

    user_prompt = f"""다음 '기타' 분류 기사들을 분석하라:

{articles_text}

반복되는 패턴을 발견하면 새 카테고리를 제안하라.
article_ids에는 해당 패턴에 속하는 기사의 article_id를 넣어라."""

    result = call_openai_structured(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_schema=_DISCOVERY_SCHEMA,
        openai_key=openai_key,
        label="카테고리발견",
        schema_name="category_suggestions"
    )

    if result is None:
        print("  ⚠️  카테고리 발견 분석 실패 (LLM 오류)")
        return None

    suggestions = result.get("suggestions", [])

    _print_suggestions(suggestions, len(df_gita))

    if logger and suggestions:
        logger.log_event(
            message="category_suggestion",
            data={
                "gita_count": len(df_gita),
                "suggestions_count": len(suggestions),
                "suggestions": suggestions,
            },
            category="category_suggestion",
            stage="reporting"
        )

    return suggestions if suggestions else None


def _print_suggestions(suggestions: List[Dict], gita_count: int) -> None:
    """카테고리 제안 결과를 콘솔에 출력."""
    print(f"\n{'─' * 60}")
    print(f"💡 카테고리 발견 결과 ('기타' {gita_count}건 분석)")
    print(f"{'─' * 60}")

    if not suggestions:
        print("  ✅ 새 카테고리 제안 없음 (기존 분류 체계로 충분)")
    else:
        for s in suggestions:
            name = s.get("category_name", "")
            count = s.get("count", 0)
            rationale = s.get("rationale", "")
            ids = s.get("article_ids", [])
            ids_preview = ", ".join(ids[:3]) + ("..." if len(ids) > 3 else "")
            print(f"  ★ \"{name}\" ({count}건)")
            print(f"    이유: {rationale}")
            print(f"    기사: {ids_preview}")
        print(f"\n  → prompts.yaml labels.issue_category_kr에 추가 후 적용됩니다")

    print(f"{'─' * 60}")
