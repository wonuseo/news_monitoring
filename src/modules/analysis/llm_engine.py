"""
llm_engine.py - LLM-Based Analysis Engine (Simplified)
prompts.yaml 기반 단일 OpenAI 호출로 전체 분류 수행
"""

import copy
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


# 1차 분류 스키마 템플릿
# brand_relevance, sentiment_stage, news_category, news_keyword_summary 담당
# danger_level / issue_category는 2차(negative_prompts.yaml)에서 처리
_SCHEMA_TEMPLATE = {
    "type": "object",
    "required": [
        "reasoning",
        "brand_relevance",
        "brand_relevance_query_keywords",
        "sentiment_stage",
        "news_category",
        "news_keyword_summary"
    ],
    "additionalProperties": False,
    "properties": {
        "reasoning": {
            "type": "object",
            "description": "분석 추론 과정 (reasoning 탭에 저장됨)",
            "required": [
                "article_subject",
                "brand_role",
                "subject_test",
                "sentiment_rationale",
                "news_category_rationale"
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
                },
                "news_category_rationale": {
                    "type": "string",
                    "description": "뉴스 카테고리 선택 근거 1문장"
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
        "news_category": {
            "type": "string",
            "enum": []  # populated by build_response_schema() from prompts.yaml
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
        yaml_path = Path(__file__).parents[3] / "config" / "prompts.yaml"

    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# Negative prompts 캐싱
_negative_prompts_cache: Optional[dict] = None

# Source verifier prompts 캐싱
_source_verifier_prompts_cache: Optional[dict] = None


def load_negative_prompts(yaml_path: Path = None) -> dict:
    """
    negative_prompts.yaml 로드 (캐싱 지원)

    Args:
        yaml_path: YAML 경로 (기본값: config/negative_prompts.yaml)

    Returns:
        negative prompts 딕셔너리
    """
    global _negative_prompts_cache
    if _negative_prompts_cache is not None:
        return _negative_prompts_cache

    if yaml_path is None:
        yaml_path = Path(__file__).parents[3] / "config" / "negative_prompts.yaml"

    with open(yaml_path, 'r', encoding='utf-8') as f:
        _negative_prompts_cache = yaml.safe_load(f)

    return _negative_prompts_cache


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
        yaml_path = Path(__file__).parents[3] / "config" / "source_verifier_prompts.yaml"

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


# 2차 분류 스키마 템플릿 (danger_level / issue_category만 담당)
# sentiment_stage는 1차에서 확정되므로 여기서는 출력하지 않음
_NEGATIVE_SCHEMA_TEMPLATE = {
    "type": "object",
    "required": ["reasoning", "danger_level", "issue_category"],
    "additionalProperties": False,
    "properties": {
        "reasoning": {
            "type": "object",
            "description": "위기 리스크 분석 추론 (내부용, 저장하지 않음)",
            "required": ["severity_analysis", "attribution_analysis", "spread_signals", "final_judgment"],
            "additionalProperties": False,
            "properties": {
                "severity_analysis": {
                    "type": "string",
                    "description": "사건 심각도 분석 (사상자/재산피해/법적제재 규모)"
                },
                "attribution_analysis": {
                    "type": "string",
                    "description": "브랜드 귀속/책임 가능성 분석 (과실/은폐/재발 프레이밍)"
                },
                "spread_signals": {
                    "type": "string",
                    "description": "확산 신호 분석 (유사기사/집단피해/미디어 집중도)"
                },
                "final_judgment": {
                    "type": "string",
                    "description": "3가지 축 종합 최종 판단 근거 1문장"
                }
            }
        },
        "danger_level": {
            "type": ["string", "null"],
            "enum": ["상", "중", "하", None]
        },
        "issue_category": {
            "type": ["string", "null"],
            "enum": []  # populated by build_negative_response_schema() from negative_prompts.yaml
        }
    }
}

# 런타임 스키마 캐시 (첫 호출 시 prompts.yaml에서 enum 주입 후 재사용)
_schema_cache: Optional[dict] = None
_negative_schema_cache: Optional[dict] = None


def build_response_schema(prompts_config: dict = None) -> dict:
    """
    prompts.yaml labels 섹션에서 issue_category / news_category enum을 주입한
    JSON Schema를 반환한다. 첫 호출 시 빌드 후 메모리 캐시.

    prompts.yaml labels만 수정하면 Python 코드 변경 없이 카테고리 추가/수정 가능.

    Args:
        prompts_config: 이미 로드된 prompts dict (None이면 자동 로드)

    Returns:
        JSON Schema 딕셔너리
    """
    global _schema_cache
    if _schema_cache is not None:
        return _schema_cache

    if prompts_config is None:
        prompts_config = load_prompts()

    labels = prompts_config.get("labels", {})
    news_cats = labels.get("news_category_kr", [
        "PR/보도자료", "사업/실적", "브랜드/마케팅", "상품/오퍼링",
        "제휴/파트너십", "이벤트/프로모션", "시설/오픈", "고객 경험",
        "운영/기술", "인사/조직", "리스크/위기", "ESG/사회", "기타",
    ])

    schema = copy.deepcopy(_SCHEMA_TEMPLATE)
    schema["properties"]["news_category"]["enum"] = news_cats

    _schema_cache = schema
    return _schema_cache


def build_negative_response_schema(prompts_config: dict = None) -> dict:
    """
    negative 분석용 JSON Schema 빌드 (issue_category enum 주입).
    첫 호출 시 빌드 후 메모리 캐시.

    Args:
        prompts_config: 이미 로드된 prompts dict (None이면 자동 로드)

    Returns:
        JSON Schema 딕셔너리
    """
    global _negative_schema_cache
    if _negative_schema_cache is not None:
        return _negative_schema_cache

    if prompts_config is None:
        prompts_config = load_prompts()

    labels = prompts_config.get("labels", {})
    issue_cats = labels.get("issue_category_kr", [
        "안전/사고", "위생/식음", "보안/개인정보/IT", "법무/규제",
        "고객 분쟁", "서비스 품질/운영", "상품/서비스 철수",
        "가격/상업", "노무/인사", "거버넌스/윤리", "평판/PR", "기타",
    ])

    schema = copy.deepcopy(_NEGATIVE_SCHEMA_TEMPLATE)
    schema["properties"]["issue_category"]["enum"] = issue_cats + [None]

    _negative_schema_cache = schema
    return _negative_schema_cache


def _post_process_result(result: Dict) -> Dict:
    """
    1차 LLM 출력의 논리적 일관성 보장을 위한 후처리

    규칙:
    1. brand_relevance="무관" → sentiment="중립", news_category="비관련"
    2. reasoning 필드 제거
    (danger_level / issue_category는 2차 패스에서 처리하므로 여기서 관여하지 않음)

    Args:
        result: LLM 원본 응답 딕셔너리

    Returns:
        후처리된 결과 (reasoning 제거됨)
    """
    if not result:
        return result

    # reasoning 필드를 _reasoning으로 보존 (reasoning_writer가 수집)
    result["_reasoning"] = result.pop("reasoning", None)

    # 무관 → sentiment 강제 중립, news_category=비관련
    if result.get("brand_relevance") == "무관":
        result["sentiment_stage"] = "중립"
        result["news_category"] = "비관련"

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
            "news_category": "리스크/위기",
            "news_keyword_summary": "롯데호텔 화재 사고 대피"
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
    response_schema = build_response_schema(prompts_config)

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


def analyze_article_negative_llm(
    article: Dict,
    initial_result: Dict,
    openai_key: str
) -> Optional[Dict]:
    """
    부정 기사 2차 정밀 분석 (config/negative_prompts.yaml 사용)

    1차 분류에서 부정 후보/부정 확정으로 판정된 기사에 대해 별도로 호출한다.
    danger_level, issue_category를 정밀 평가.

    Args:
        article: {"title": ..., "description": ..., "query": ..., "group": ...}
        initial_result: 1차 분류 결과 {"brand_relevance": ..., "sentiment_stage": ...}
        openai_key: OpenAI API 키

    Returns:
        {"danger_level": "상", "issue_category": "안전/사고"}
        또는 None (실패시)
    """
    negative_config = load_negative_prompts()
    system_prompt = negative_config.get("system", "")
    user_template = negative_config.get("user_prompt_template", "")

    if not system_prompt or not user_template:
        return None

    context = {
        "query": article.get("query", ""),
        "group": article.get("group", ""),
        "title": article.get("title", ""),
        "description": article.get("description", ""),
        "brand_relevance": initial_result.get("brand_relevance", ""),
        "initial_sentiment": initial_result.get("sentiment_stage", ""),
    }

    user_prompt = render_prompt(user_template, context)

    model = negative_config.get("model", None)
    response_schema = build_negative_response_schema(negative_config)

    result = call_openai_structured(
        system_prompt, user_prompt, response_schema,
        openai_key, model=model, label="부정분석"
    )

    if not result:
        return None

    # reasoning 필드를 _reasoning으로 보존 (reasoning_writer가 수집)
    result["_reasoning"] = result.pop("reasoning", None)
    return result
