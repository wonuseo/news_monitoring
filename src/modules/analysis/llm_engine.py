"""
llm_engine.py - LLM-Based Analysis Engine
prompts.yaml 기반 OpenAI Structured Output 호출
"""

import json
import time
import re
import yaml
import requests
from pathlib import Path
from typing import Dict, List, Optional


OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"


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


def render_prompt(template: str, context: Dict) -> str:
    """
    Jinja2-style 템플릿 렌더링 (간단한 {{variable}} 치환)

    Args:
        template: 프롬프트 템플릿
        context: 변수 딕셔너리

    Returns:
        렌더링된 프롬프트
    """
    result = template
    for key, value in context.items():
        placeholder = f"{{{{{key}}}}}"
        # Convert lists/dicts to JSON string
        if isinstance(value, (list, dict)):
            value_str = json.dumps(value, ensure_ascii=False, indent=2)
        else:
            value_str = str(value)
        result = result.replace(placeholder, value_str)
    return result


def call_openai_structured(
    system_prompt: str,
    user_prompt: str,
    response_schema: dict,
    openai_key: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    retry: bool = True
) -> Optional[Dict]:
    """
    OpenAI Structured Output 호출

    Args:
        system_prompt: 시스템 프롬프트
        user_prompt: 유저 프롬프트
        response_schema: JSON schema for response format
        openai_key: OpenAI API 키
        model: 모델명
        temperature: 온도
        retry: 재시도 여부

    Returns:
        파싱된 JSON 응답 또는 None (실패시)
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_key}"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "analysis_result",
                "strict": True,
                "schema": response_schema
            }
        }
    }

    try:
        response = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=60)

        if response.status_code == 401:
            raise RuntimeError("OpenAI API 인증 실패 (401). API 키를 확인하세요.")
        elif response.status_code == 429:
            print("⚠️  OpenAI 요청 한도 초과 (429). 5초 대기 중...")
            time.sleep(5)
            if retry:
                return call_openai_structured(
                    system_prompt, user_prompt, response_schema,
                    openai_key, model, temperature, retry=False
                )
            else:
                raise RuntimeError("OpenAI 요청 한도 초과 (재시도 후에도 실패)")
        elif response.status_code >= 500:
            raise RuntimeError(f"OpenAI 서버 오류 ({response.status_code})")

        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"].strip()
        result = json.loads(content)

        return result

    except (json.JSONDecodeError, KeyError, requests.exceptions.RequestException) as e:
        print(f"⚠️  OpenAI 호출 오류: {e}")
        if retry:
            print("재시도 중...")
            time.sleep(2)
            return call_openai_structured(
                system_prompt, user_prompt, response_schema,
                openai_key, model, temperature, retry=False
            )
        else:
            return None


def analyze_sentiment_llm(
    article: Dict,
    rb_result: Dict,
    prompts_config: dict,
    openai_key: str
) -> Optional[Dict]:
    """
    LLM 기반 감정 분석 (독립 판단)

    Args:
        article: {"title": ..., "description": ...}
        rb_result: Rule-Based 결과
        prompts_config: prompts.yaml 딕셔너리
        openai_key: OpenAI API 키

    Returns:
        {
            "sentiment_llm": "NEGATIVE_CONFIRMED",
            "confidence": 0.95,
            "evidence": [...],
            "rationale": "..."
        }
    """
    prompt_config = prompts_config["prompts"]["sentiment_llm"]
    system_prompt = prompt_config["system"]

    # User prompt context
    context = {
        "policy_sentiment": prompts_config["policy_text"]["sentiment"],
        "title": article.get("title", ""),
        "description": article.get("description", ""),
        "brand_mentions": rb_result.get("brand_mentions", {}),
        "brand_scope_rb": rb_result.get("brand_scope_rb", ""),
        "sentiment_rb": rb_result.get("sentiment_rb", ""),
        "reason_codes_rb": rb_result.get("reason_codes_rb", []),
        "matched_rules_rb": rb_result.get("matched_rules_rb", []),
        "schema_sentiment_llm": prompts_config["output_schemas"]["sentiment_llm"]
    }

    user_prompt = render_prompt(prompt_config["user"], context)

    # Call OpenAI
    response_schema = prompts_config["output_schemas"]["sentiment_llm"]
    result = call_openai_structured(
        system_prompt, user_prompt, response_schema,
        openai_key, model="gpt-4o-mini", temperature=0.2
    )

    return result


def analyze_sentiment_final(
    article: Dict,
    rb_result: Dict,
    llm_result: Dict,
    prompts_config: dict,
    openai_key: str
) -> Optional[Dict]:
    """
    LLM 기반 최종 감정 판단 (RB vs LLM 조정)

    Args:
        article: {"title": ..., "description": ...}
        rb_result: Rule-Based 결과
        llm_result: LLM 결과
        prompts_config: prompts.yaml 딕셔너리
        openai_key: OpenAI API 키

    Returns:
        {
            "sentiment_final": "NEGATIVE_CONFIRMED",
            "confidence": 0.95,
            "decision_rule": "...",
            "evidence": [...],
            "rationale": "..."
        }
    """
    prompt_config = prompts_config["prompts"]["sentiment_final"]
    system_prompt = prompt_config["system"]

    # User prompt context
    context = {
        "policy_sentiment": prompts_config["policy_text"]["sentiment"],
        "title": article.get("title", ""),
        "description": article.get("description", ""),
        "sentiment_rb": rb_result.get("sentiment_rb", ""),
        "sentiment_llm": llm_result.get("sentiment_llm", "") if llm_result else "",
        "brand_scope_rb": rb_result.get("brand_scope_rb", ""),
        "reason_codes_rb": rb_result.get("reason_codes_rb", []),
        "matched_rules_rb": rb_result.get("matched_rules_rb", []),
        "schema_sentiment_final": prompts_config["output_schemas"]["sentiment_final"]
    }

    user_prompt = render_prompt(prompt_config["user"], context)

    # Call OpenAI
    response_schema = prompts_config["output_schemas"]["sentiment_final"]
    result = call_openai_structured(
        system_prompt, user_prompt, response_schema,
        openai_key, model="gpt-4o-mini", temperature=0.2
    )

    return result


def analyze_danger_llm(
    article: Dict,
    rb_result: Dict,
    sentiment_final: str,
    prompts_config: dict,
    openai_key: str
) -> Optional[Dict]:
    """
    LLM 기반 위험도 분석 (독립 판단)

    Args:
        article: {"title": ..., "description": ...}
        rb_result: Rule-Based 결과
        sentiment_final: 최종 감정 결과
        prompts_config: prompts.yaml 딕셔너리
        openai_key: OpenAI API 키

    Returns:
        {
            "danger_llm": "D3",
            "confidence": 0.9,
            "evidence": [...],
            "rationale": "..."
        }
    """
    # Danger는 BRAND_TARGETED + NEGATIVE_* 조건에서만 실행
    brand_scope = rb_result.get("brand_scope_rb", "")
    if brand_scope != "BRAND_TARGETED":
        return None

    if sentiment_final not in ["NEGATIVE_CANDIDATE", "NEGATIVE_CONFIRMED"]:
        return None

    prompt_config = prompts_config["prompts"]["danger_llm"]
    system_prompt = prompt_config["system"]

    # User prompt context
    context = {
        "policy_danger": prompts_config["policy_text"]["danger"],
        "title": article.get("title", ""),
        "description": article.get("description", ""),
        "brand_scope_rb": brand_scope,
        "sentiment_final": sentiment_final,
        "reason_codes_rb": rb_result.get("reason_codes_rb", []),
        "matched_rules_rb": rb_result.get("matched_rules_rb", []),
        "danger_rb": rb_result.get("danger_rb", ""),
        "risk_score_rb": rb_result.get("risk_score_rb", 0),
        "score_breakdown_rb": rb_result.get("score_breakdown_rb", {}),
        "schema_danger_llm": prompts_config["output_schemas"]["danger_llm"]
    }

    user_prompt = render_prompt(prompt_config["user"], context)

    # Call OpenAI
    response_schema = prompts_config["output_schemas"]["danger_llm"]
    result = call_openai_structured(
        system_prompt, user_prompt, response_schema,
        openai_key, model="gpt-4o-mini", temperature=0.2
    )

    return result


def analyze_danger_final(
    article: Dict,
    rb_result: Dict,
    llm_result: Dict,
    prompts_config: dict,
    openai_key: str
) -> Optional[Dict]:
    """
    LLM 기반 최종 위험도 판단 (RB vs LLM 조정)

    Args:
        article: {"title": ..., "description": ...}
        rb_result: Rule-Based 결과
        llm_result: LLM 결과 (danger_llm)
        prompts_config: prompts.yaml 딕셔너리
        openai_key: OpenAI API 키

    Returns:
        {
            "danger_final": "D3",
            "confidence": 0.95,
            "decision_rule": "...",
            "evidence": [...],
            "rationale": "..."
        }
    """
    if llm_result is None:
        return None

    prompt_config = prompts_config["prompts"]["danger_final"]
    system_prompt = prompt_config["system"]

    # User prompt context
    context = {
        "policy_danger": prompts_config["policy_text"]["danger"],
        "title": article.get("title", ""),
        "description": article.get("description", ""),
        "danger_rb": rb_result.get("danger_rb", ""),
        "risk_score_rb": rb_result.get("risk_score_rb", 0),
        "danger_llm": llm_result.get("danger_llm", "") if llm_result else "",
        "score_breakdown_rb": rb_result.get("score_breakdown_rb", {}),
        "reason_codes_rb": rb_result.get("reason_codes_rb", []),
        "schema_danger_final": prompts_config["output_schemas"]["danger_final"]
    }

    user_prompt = render_prompt(prompt_config["user"], context)

    # Call OpenAI
    response_schema = prompts_config["output_schemas"]["danger_final"]
    result = call_openai_structured(
        system_prompt, user_prompt, response_schema,
        openai_key, model="gpt-4o-mini", temperature=0.2
    )

    return result


def analyze_category_llm(
    article: Dict,
    rb_result: Dict,
    sentiment_final: str,
    danger_final: str,
    prompts_config: dict,
    openai_key: str
) -> Optional[Dict]:
    """
    LLM 기반 카테고리 분석 (독립 판단)

    Args:
        article: {"title": ..., "description": ...}
        rb_result: Rule-Based 결과
        sentiment_final: 최종 감정 결과
        danger_final: 최종 위험도 결과
        prompts_config: prompts.yaml 딕셔너리
        openai_key: OpenAI API 키

    Returns:
        {
            "issue_category_llm": "Safety / Incident",
            "coverage_themes_llm": ["Risk / Crisis", "Operations / Technology"],
            "confidence": 0.9,
            "evidence": [...],
            "rationale": "..."
        }
    """
    prompt_config = prompts_config["prompts"]["category_llm"]
    system_prompt = prompt_config["system"]

    # User prompt context
    context = {
        "policy_category": prompts_config["policy_text"]["category"],
        "title": article.get("title", ""),
        "description": article.get("description", ""),
        "brand_scope_rb": rb_result.get("brand_scope_rb", ""),
        "sentiment_final": sentiment_final,
        "danger_final": danger_final,
        "reason_codes_rb": rb_result.get("reason_codes_rb", []),
        "issue_category_rb": rb_result.get("issue_category_rb", ""),
        "coverage_themes_rb": rb_result.get("coverage_themes_rb", []),
        "matched_rules_rb": rb_result.get("matched_rules_rb", []),
        "schema_category_llm": prompts_config["output_schemas"]["category_llm"]
    }

    user_prompt = render_prompt(prompt_config["user"], context)

    # Call OpenAI
    response_schema = prompts_config["output_schemas"]["category_llm"]
    result = call_openai_structured(
        system_prompt, user_prompt, response_schema,
        openai_key, model="gpt-4o-mini", temperature=0.2
    )

    return result


def analyze_category_final(
    article: Dict,
    rb_result: Dict,
    llm_result: Dict,
    prompts_config: dict,
    openai_key: str
) -> Optional[Dict]:
    """
    LLM 기반 최종 카테고리 판단 (RB vs LLM 조정)

    Args:
        article: {"title": ..., "description": ...}
        rb_result: Rule-Based 결과
        llm_result: LLM 결과 (category_llm)
        prompts_config: prompts.yaml 딕셔너리
        openai_key: OpenAI API 키

    Returns:
        {
            "issue_category_final": "Safety / Incident",
            "coverage_themes_final": ["Risk / Crisis"],
            "confidence": 0.95,
            "decision_rule": "...",
            "evidence": [...],
            "rationale": "..."
        }
    """
    if llm_result is None:
        return None

    prompt_config = prompts_config["prompts"]["category_final"]
    system_prompt = prompt_config["system"]

    # User prompt context
    context = {
        "policy_category": prompts_config["policy_text"]["category"],
        "title": article.get("title", ""),
        "description": article.get("description", ""),
        "issue_category_rb": rb_result.get("issue_category_rb", ""),
        "coverage_themes_rb": rb_result.get("coverage_themes_rb", []),
        "issue_category_llm": llm_result.get("issue_category_llm", "") if llm_result else "",
        "coverage_themes_llm": llm_result.get("coverage_themes_llm", []) if llm_result else [],
        "reason_codes_rb": rb_result.get("reason_codes_rb", []),
        "matched_rules_rb": rb_result.get("matched_rules_rb", []),
        "schema_category_final": prompts_config["output_schemas"]["category_final"]
    }

    user_prompt = render_prompt(prompt_config["user"], context)

    # Call OpenAI
    response_schema = prompts_config["output_schemas"]["category_final"]
    result = call_openai_structured(
        system_prompt, user_prompt, response_schema,
        openai_key, model="gpt-4o-mini", temperature=0.2
    )

    return result


def analyze_article_llm(
    article: Dict,
    rb_result: Dict,
    prompts_config: dict,
    openai_key: str
) -> Dict:
    """
    단일 기사 LLM 분석 (메인 함수)

    5단계 프로세스:
    1. sentiment_llm (독립 판단)
    2. sentiment_final (RB vs LLM 조정)
    3. danger_llm → danger_final (조건부)
    4. category_llm (독립 판단)
    5. category_final (RB vs LLM 조정)

    Args:
        article: {"title": ..., "description": ...}
        rb_result: Rule-Based 결과
        prompts_config: prompts.yaml 딕셔너리
        openai_key: OpenAI API 키

    Returns:
        {
            "sentiment_llm": "...",
            "sentiment_llm_confidence": 0.95,
            "sentiment_llm_evidence": [...],
            "sentiment_llm_rationale": "...",
            "sentiment_final": "...",
            "sentiment_final_confidence": 0.95,
            "sentiment_final_decision_rule": "...",
            "sentiment_final_evidence": [...],
            "sentiment_final_rationale": "...",
            "danger_llm": "...",
            "danger_llm_confidence": 0.9,
            "danger_llm_evidence": [...],
            "danger_llm_rationale": "...",
            "danger_final": "...",
            "danger_final_confidence": 0.95,
            "danger_final_decision_rule": "...",
            "danger_final_evidence": [...],
            "danger_final_rationale": "...",
            "issue_category_llm": "...",
            "coverage_themes_llm": [...],
            "category_llm_confidence": 0.9,
            "category_llm_evidence": [...],
            "category_llm_rationale": "...",
            "issue_category_final": "...",
            "coverage_themes_final": [...],
            "category_final_confidence": 0.95,
            "category_final_decision_rule": "...",
            "category_final_evidence": [...],
            "category_final_rationale": "..."
        }
    """
    result = {}

    # Step 1: Sentiment LLM
    sentiment_llm = analyze_sentiment_llm(article, rb_result, prompts_config, openai_key)
    if sentiment_llm:
        result["sentiment_llm"] = sentiment_llm.get("sentiment_llm", "")
        result["sentiment_llm_confidence"] = sentiment_llm.get("confidence", 0.0)
        result["sentiment_llm_evidence"] = sentiment_llm.get("evidence", [])
        result["sentiment_llm_rationale"] = sentiment_llm.get("rationale", "")
    else:
        result["sentiment_llm"] = ""
        result["sentiment_llm_confidence"] = 0.0
        result["sentiment_llm_evidence"] = []
        result["sentiment_llm_rationale"] = ""

    # Step 2: Sentiment Final
    sentiment_final = analyze_sentiment_final(article, rb_result, sentiment_llm, prompts_config, openai_key)
    if sentiment_final:
        result["sentiment_final"] = sentiment_final.get("sentiment_final", "")
        result["sentiment_final_confidence"] = sentiment_final.get("confidence", 0.0)
        result["sentiment_final_decision_rule"] = sentiment_final.get("decision_rule", "")
        result["sentiment_final_evidence"] = sentiment_final.get("evidence", [])
        result["sentiment_final_rationale"] = sentiment_final.get("rationale", "")
    else:
        result["sentiment_final"] = rb_result.get("sentiment_rb", "")  # Fallback to RB
        result["sentiment_final_confidence"] = 0.0
        result["sentiment_final_decision_rule"] = "LLM failed, used RB"
        result["sentiment_final_evidence"] = []
        result["sentiment_final_rationale"] = ""

    # Step 3: Danger LLM (conditional)
    danger_llm = analyze_danger_llm(
        article, rb_result, result["sentiment_final"], prompts_config, openai_key
    )
    if danger_llm:
        result["danger_llm"] = danger_llm.get("danger_llm", "")
        result["danger_llm_confidence"] = danger_llm.get("confidence", 0.0)
        result["danger_llm_evidence"] = danger_llm.get("evidence", [])
        result["danger_llm_rationale"] = danger_llm.get("rationale", "")
    else:
        result["danger_llm"] = ""
        result["danger_llm_confidence"] = 0.0
        result["danger_llm_evidence"] = []
        result["danger_llm_rationale"] = ""

    # Step 4: Danger Final (conditional)
    if danger_llm:
        danger_final = analyze_danger_final(article, rb_result, danger_llm, prompts_config, openai_key)
        if danger_final:
            result["danger_final"] = danger_final.get("danger_final", "")
            result["danger_final_confidence"] = danger_final.get("confidence", 0.0)
            result["danger_final_decision_rule"] = danger_final.get("decision_rule", "")
            result["danger_final_evidence"] = danger_final.get("evidence", [])
            result["danger_final_rationale"] = danger_final.get("rationale", "")
        else:
            result["danger_final"] = rb_result.get("danger_rb", "")  # Fallback to RB
            result["danger_final_confidence"] = 0.0
            result["danger_final_decision_rule"] = "LLM failed, used RB"
            result["danger_final_evidence"] = []
            result["danger_final_rationale"] = ""
    else:
        result["danger_final"] = ""
        result["danger_final_confidence"] = 0.0
        result["danger_final_decision_rule"] = ""
        result["danger_final_evidence"] = []
        result["danger_final_rationale"] = ""

    # Step 5: Category LLM
    category_llm = analyze_category_llm(
        article, rb_result, result["sentiment_final"], result["danger_final"],
        prompts_config, openai_key
    )
    if category_llm:
        result["issue_category_llm"] = category_llm.get("issue_category_llm", "")
        result["coverage_themes_llm"] = category_llm.get("coverage_themes_llm", [])
        result["category_llm_confidence"] = category_llm.get("confidence", 0.0)
        result["category_llm_evidence"] = category_llm.get("evidence", [])
        result["category_llm_rationale"] = category_llm.get("rationale", "")
    else:
        result["issue_category_llm"] = ""
        result["coverage_themes_llm"] = []
        result["category_llm_confidence"] = 0.0
        result["category_llm_evidence"] = []
        result["category_llm_rationale"] = ""

    # Step 6: Category Final
    category_final = analyze_category_final(article, rb_result, category_llm, prompts_config, openai_key)
    if category_final:
        result["issue_category_final"] = category_final.get("issue_category_final", "")
        result["coverage_themes_final"] = category_final.get("coverage_themes_final", [])
        result["category_final_confidence"] = category_final.get("confidence", 0.0)
        result["category_final_decision_rule"] = category_final.get("decision_rule", "")
        result["category_final_evidence"] = category_final.get("evidence", [])
        result["category_final_rationale"] = category_final.get("rationale", "")
    else:
        result["issue_category_final"] = rb_result.get("issue_category_rb", "")  # Fallback to RB
        result["coverage_themes_final"] = rb_result.get("coverage_themes_rb", [])
        result["category_final_confidence"] = 0.0
        result["category_final_decision_rule"] = "LLM failed, used RB"
        result["category_final_evidence"] = []
        result["category_final_rationale"] = ""

    return result
