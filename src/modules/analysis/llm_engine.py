"""
llm_engine.py - LLM-Based Analysis Engine (Simplified)
prompts.yaml 기반 단일 OpenAI 호출로 전체 분류 수행
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.openai_client import (
    OPENAI_API_URL,
    set_error_callback,
    load_api_models,
    call_openai_with_retry,
    extract_response_text,
)


# JSON Schema 캐싱 (매번 생성 비용 제거)
_RESPONSE_SCHEMA = {
    "type": "object",
    "required": [
        "reasoning",
        "brand_relevance",
        "brand_relevance_query_keywords",
        "sentiment_stage",
        "danger_level",
        "issue_category",
        "news_category",
        "news_keyword_summary"
    ],
    "additionalProperties": False,
    "properties": {
        "reasoning": {
            "type": "object",
            "description": "분석 추론 과정 (내부용, 저장하지 않음)",
            "required": [
                "article_subject",
                "brand_role",
                "subject_test",
                "sentiment_rationale"
            ],
            "additionalProperties": False,
            "properties": {
                "article_subject": {
                    "type": "string",
                    "description": "기사의 핵심 주체 (예: 'TS손해보험', '롯데호텔')"
                },
                "brand_role": {
                    "type": "string",
                    "enum": ["주체", "배경장소", "나열", "무관"],
                    "description": "query 브랜드의 기사 내 역할"
                },
                "subject_test": {
                    "type": "string",
                    "description": "주체 테스트 결과: 브랜드명 제거 시 기사 주제 변화 여부"
                },
                "sentiment_rationale": {
                    "type": "string",
                    "description": "감정 판단 근거 1문장"
                }
            }
        },
        "brand_relevance": {
            "type": "string",
            "enum": ["관련", "언급", "무관", "판단 필요"]
        },
        "brand_relevance_query_keywords": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 0,
            "maxItems": 3
        },
        "sentiment_stage": {
            "type": "string",
            "enum": ["긍정", "중립", "부정 후보", "부정 확정"]
        },
        "danger_level": {
            "type": ["string", "null"],
            "enum": ["상", "중", "하", None]
        },
        "issue_category": {
            "type": ["string", "null"],
            "enum": [
                "안전/사고",
                "위생/식음",
                "보안/개인정보/IT",
                "법무/규제",
                "고객 분쟁",
                "서비스 품질/운영",
                "가격/상업",
                "노무/인사",
                "거버넌스/윤리",
                "평판/PR",
                "기타",
                None
            ]
        },
        "news_category": {
            "type": "string",
            "enum": [
                "PR/보도자료",
                "사업/실적",
                "브랜드/마케팅",
                "상품/오퍼링",
                "제휴/파트너십",
                "이벤트/프로모션",
                "시설/오픈",
                "고객 경험",
                "운영/기술",
                "인사/조직",
                "리스크/위기",
                "ESG/사회",
                "기타"
            ]
        },
        "news_keyword_summary": {
            "type": "string",
            "maxLength": 50
        }
    }
}


def load_prompts(yaml_path: Path = None) -> dict:
    """
    prompts.yaml 로드

    Args:
        yaml_path: prompts.yaml 경로 (기본값: 현재 디렉토리)

    Returns:
        prompts 딕셔너리
    """
    if yaml_path is None:
        yaml_path = Path(__file__).parent / "prompts.yaml"

    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# Source verifier prompts 캐싱
_source_verifier_prompts_cache: Optional[dict] = None


def load_source_verifier_prompts(yaml_path: Path = None) -> dict:
    """
    source_verifier_prompts.yaml 로드 (캐싱 지원)

    Args:
        yaml_path: YAML 경로 (기본값: 현재 디렉토리)

    Returns:
        prompts 딕셔너리
    """
    global _source_verifier_prompts_cache
    if _source_verifier_prompts_cache is not None:
        return _source_verifier_prompts_cache

    if yaml_path is None:
        yaml_path = Path(__file__).parent / "source_verifier_prompts.yaml"

    with open(yaml_path, 'r', encoding='utf-8') as f:
        _source_verifier_prompts_cache = yaml.safe_load(f)

    return _source_verifier_prompts_cache


def render_prompt(template: str, context: Dict) -> str:
    """
    간단한 {{variable}} 치환 렌더링

    Args:
        template: 프롬프트 템플릿
        context: 변수 딕셔너리

    Returns:
        렌더링된 프롬프트
    """
    result = template
    for key, value in context.items():
        placeholder = f"{{{{{key}}}}}"
        result = result.replace(placeholder, str(value))
    return result


def call_openai_structured(
    system_prompt: str,
    user_prompt: str,
    response_schema: dict,
    openai_key: str,
    model: str = None,
    max_retries: int = 5,
    label: str = "LLM분류",
    schema_name: str = "analysis_result"
) -> Optional[Dict]:
    """
    OpenAI Structured Output 호출 (재시도 지원)

    Args:
        system_prompt: 시스템 프롬프트
        user_prompt: 유저 프롬프트
        response_schema: JSON schema for response format
        openai_key: OpenAI API 키
        model: 모델명 (None이면 api_models.yaml에서 로드)
        max_retries: 최대 재시도 횟수 (기본: 5)
        label: 로그 라벨 (기본: "LLM분류")
        schema_name: JSON schema name (기본: "analysis_result")

    Returns:
        파싱된 JSON 응답 또는 None (실패시)
    """
    if model is None:
        api_models = load_api_models()
        model = api_models.get("article_classification", "gpt-5-nano")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_key}"
    }

    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]}
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": response_schema
            }
        }
    }

    response = call_openai_with_retry(
        OPENAI_API_URL, headers, payload,
        max_retries=max_retries, label=label
    )

    if response is None:
        return None

    data = response.json()
    content = extract_response_text(data)

    if not content:
        print("  Responses API 응답에서 output_text를 찾을 수 없습니다.")
        return None

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"  JSON 파싱 실패: {e}")
        return None


def build_response_schema() -> dict:
    """
    JSON Schema 반환 (캐싱된 상수 사용)

    OpenAI Structured Output은 스키마 순서대로 생성하므로,
    reasoning을 먼저 생성하게 하여 분류 품질을 향상시킨다.

    Returns:
        JSON Schema 딕셔너리
    """
    return _RESPONSE_SCHEMA


def _post_process_result(result: Dict) -> Dict:
    """
    LLM 출력의 논리적 일관성 보장을 위한 후처리

    규칙:
    1. brand_relevance="무관" → sentiment="중립", danger=null, issue=null
    2. danger_level은 (관련/언급) + (부정 후보/부정 확정)일 때만 유효
    3. reasoning 필드 제거

    Args:
        result: LLM 원본 응답 딕셔너리

    Returns:
        후처리된 결과 (reasoning 제거됨)
    """
    if not result:
        return result

    # reasoning 필드 제거 (LLM 내부 추론용으로만 사용)
    result.pop("reasoning", None)

    brand_rel = result.get("brand_relevance", "")
    sentiment = result.get("sentiment_stage", "")

    # 규칙 1: 무관 → 중립, danger/issue null, news_category=비관련
    if brand_rel == "무관":
        result["sentiment_stage"] = "중립"
        result["danger_level"] = None
        result["issue_category"] = None
        result["news_category"] = "비관련"

    # 규칙 2: danger_level은 (관련/언급) + (부정 후보/부정 확정)일 때만
    if brand_rel not in ("관련", "언급") or sentiment not in ("부정 후보", "부정 확정"):
        result["danger_level"] = None
        result["issue_category"] = None

    return result


def analyze_article_llm(
    article: Dict,
    prompts_config: dict,
    openai_key: str
) -> Optional[Dict]:
    """
    단일 기사 LLM 분석 (단일 API 호출, CoT reasoning + 후처리)

    Args:
        article: {"title": ..., "description": ..., "query": ..., "group": ...}
        prompts_config: prompts.yaml 딕셔너리
        openai_key: OpenAI API 키

    Returns:
        {
            "brand_relevance": "관련",
            "brand_relevance_query_keywords": ["롯데호텔"],
            "sentiment_stage": "부정 확정",
            "danger_level": "상",
            "issue_category": "안전/사고",
            "news_category": "리스크/위기",
            "news_keyword_summary": "화재 사고 대피"
        }
    """
    system_prompt = prompts_config.get("system", "")
    user_template = prompts_config.get("user_prompt_template", "")

    # User prompt context (group 포함)
    context = {
        "query": article.get("query", ""),
        "group": article.get("group", ""),
        "title": article.get("title", ""),
        "description": article.get("description", "")
    }

    user_prompt = render_prompt(user_template, context)

    # Call OpenAI
    model = prompts_config.get("model", None)
    response_schema = build_response_schema()

    result = call_openai_structured(
        system_prompt, user_prompt, response_schema,
        openai_key, model=model
    )

    # 후처리: reasoning 제거 + 논리적 일관성 보장
    result = _post_process_result(result)

    return result


def analyze_batch_llm(
    articles: List[Dict],
    prompts_config: dict,
    openai_key: str
) -> List[Dict]:
    """
    배치 LLM 분석 (단순 순차 처리)

    Args:
        articles: [{"title": ..., "description": ..., "query": ...}, ...]
        prompts_config: prompts.yaml 딕셔너리
        openai_key: OpenAI API 키

    Returns:
        [분석 결과 딕셔너리, ...]
    """
    results = []
    for article in articles:
        result = analyze_article_llm(article, prompts_config, openai_key)
        results.append(result if result else {})

    return results
