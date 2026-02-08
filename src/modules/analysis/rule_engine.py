"""
rule_engine.py - Rule-Based Analysis Engine
rules.yaml 기반 브랜드 스코프, 감정, 위험도, 카테고리 판단
"""

import re
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def load_rules(yaml_path: Path = None) -> dict:
    """
    rules.yaml 로드

    Args:
        yaml_path: rules.yaml 경로 (기본값: 현재 디렉토리)

    Returns:
        rules 딕셔너리
    """
    if yaml_path is None:
        yaml_path = Path(__file__).parent / "rules.yaml"

    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def find_brand_mentions(text: str, rules: dict) -> Dict[str, List[str]]:
    """
    텍스트에서 브랜드 언급 찾기

    Args:
        text: 제목 + 설명 결합 텍스트
        rules: rules.yaml 딕셔너리

    Returns:
        {"our": [...], "competitors": [...]}
    """
    brands_config = rules.get("brands", {})
    our_brands = brands_config.get("our", [])
    competitor_brands = brands_config.get("competitors", [])

    mentions = {"our": [], "competitors": []}

    for brand in our_brands:
        if brand in text:
            mentions["our"].append(brand)

    for brand in competitor_brands:
        if brand in text:
            mentions["competitors"].append(brand)

    return mentions


def detect_brand_scope(title: str, description: str, rules: dict) -> Dict:
    """
    브랜드 스코프 판단 (BRAND_TARGETED / BRAND_MENTIONED / VENUE_ONLY)

    Args:
        title: 기사 제목
        description: 기사 설명
        rules: rules.yaml 딕셔너리

    Returns:
        {
            "brand_scope_rb": "BRAND_TARGETED",
            "brand_mentions": {"our": [...], "competitors": [...]},
            "matched_cues": [...]
        }
    """
    combined_text = f"{title} {description}"
    brand_mentions = find_brand_mentions(combined_text, rules)

    scope_config = rules.get("brand_scope", {})
    require_brand_mention = scope_config.get("require_brand_mention", True)
    targeted_cues = scope_config.get("targeted_cues_regex", [])

    # 브랜드 언급 없으면 VENUE_ONLY
    if require_brand_mention:
        if not brand_mentions["our"] and not brand_mentions["competitors"]:
            return {
                "brand_scope_rb": "VENUE_ONLY",
                "brand_mentions": brand_mentions,
                "matched_cues": []
            }

    # Targeted cue 검사 (제목/설명 중 브랜드 언급과 같은 필드에 있어야 함)
    matched_cues = []
    is_targeted = False

    for cue_regex in targeted_cues:
        # 제목에서 체크
        if brand_mentions["our"] or brand_mentions["competitors"]:
            for brand in brand_mentions["our"] + brand_mentions["competitors"]:
                if brand in title and re.search(cue_regex, title):
                    matched_cues.append(cue_regex)
                    is_targeted = True
                    break

            # 설명에서 체크
            if not is_targeted:
                for brand in brand_mentions["our"] + brand_mentions["competitors"]:
                    if brand in description and re.search(cue_regex, description):
                        matched_cues.append(cue_regex)
                        is_targeted = True
                        break

    if is_targeted:
        brand_scope = "BRAND_TARGETED"
    elif brand_mentions["our"] or brand_mentions["competitors"]:
        brand_scope = "BRAND_MENTIONED"
    else:
        brand_scope = "VENUE_ONLY"

    return {
        "brand_scope_rb": brand_scope,
        "brand_mentions": brand_mentions,
        "matched_cues": list(set(matched_cues))
    }


def detect_sentiment(text: str, rules: dict) -> Dict:
    """
    감정 판단 (POSITIVE / NEUTRAL / NEGATIVE_CANDIDATE / NEGATIVE_CONFIRMED)

    Args:
        text: 제목 + 설명 결합 텍스트
        rules: rules.yaml 딕셔너리

    Returns:
        {
            "sentiment_rb": "NEGATIVE_CONFIRMED",
            "matched_rules_rb": [...],
            "trigger_family": "incident_regex"
        }
    """
    sentiment_config = rules.get("sentiment", {})
    priority_order = sentiment_config.get("priority_order", [
        "POSITIVE", "NEGATIVE_CONFIRMED", "NEGATIVE_CANDIDATE", "NEUTRAL"
    ])

    matched_sentiments = {}

    # POSITIVE 체크
    positive_triggers = sentiment_config.get("positive_triggers_regex", [])
    for regex in positive_triggers:
        if re.search(regex, text):
            matched_sentiments["POSITIVE"] = matched_sentiments.get("POSITIVE", []) + [regex]

    # NEGATIVE_CONFIRMED 체크 (5개 카테고리)
    negative_confirmed = sentiment_config.get("negative_confirmed_triggers", {})
    for family_name, regex_list in negative_confirmed.items():
        for regex in regex_list:
            if re.search(regex, text):
                matched_sentiments["NEGATIVE_CONFIRMED"] = matched_sentiments.get("NEGATIVE_CONFIRMED", []) + [f"{family_name}:{regex}"]

    # NEGATIVE_CANDIDATE 체크 (4개 카테고리)
    negative_candidate = sentiment_config.get("negative_candidate_triggers", {})
    for family_name, regex_list in negative_candidate.items():
        for regex in regex_list:
            if re.search(regex, text):
                matched_sentiments["NEGATIVE_CANDIDATE"] = matched_sentiments.get("NEGATIVE_CANDIDATE", []) + [f"{family_name}:{regex}"]

    # Priority order에 따라 최종 sentiment 결정
    final_sentiment = "NEUTRAL"
    matched_rules = []

    for sentiment in priority_order:
        if sentiment in matched_sentiments:
            final_sentiment = sentiment
            matched_rules = matched_sentiments[sentiment]
            break

    return {
        "sentiment_rb": final_sentiment,
        "matched_rules_rb": matched_rules
    }


def calculate_danger_score(text: str, brand_scope: str, sentiment: str, rules: dict) -> Dict:
    """
    위험도 점수 계산 및 등급 판단 (D1 / D2 / D3)

    Args:
        text: 제목 + 설명 결합 텍스트
        brand_scope: BRAND_TARGETED/MENTIONED/VENUE_ONLY
        sentiment: POSITIVE/NEUTRAL/NEGATIVE_CANDIDATE/NEGATIVE_CONFIRMED
        rules: rules.yaml 딕셔너리

    Returns:
        {
            "danger_rb": "D3",
            "risk_score_rb": 75,
            "score_breakdown_rb": {...}
        }
    """
    danger_config = rules.get("danger", {})

    # Danger 계산 조건 체크
    if not danger_config.get("enabled", True):
        return {
            "danger_rb": "",
            "risk_score_rb": 0,
            "score_breakdown_rb": {}
        }

    apply_when = danger_config.get("apply_when", {})
    required_scopes = apply_when.get("brand_scope", [])
    required_sentiments = apply_when.get("sentiment", [])

    if brand_scope not in required_scopes or sentiment not in required_sentiments:
        return {
            "danger_rb": "",
            "risk_score_rb": 0,
            "score_breakdown_rb": {}
        }

    # 점수 계산
    score = 0
    breakdown = {}

    score_components = danger_config.get("score_components", {})

    # 1. Hard trigger (50점)
    hard_trigger_config = score_components.get("hard_trigger", {})
    hard_trigger_points = hard_trigger_config.get("points", 50)
    hard_trigger_regex = hard_trigger_config.get("regex", [])
    hard_trigger_matched = False

    for regex in hard_trigger_regex:
        if re.search(regex, text):
            score += hard_trigger_points
            breakdown["hard_trigger"] = hard_trigger_points
            hard_trigger_matched = True
            break

    # 2. High risk category (20점)
    high_risk_config = score_components.get("high_risk_category", {})
    high_risk_points = high_risk_config.get("points", 20)
    trigger_families = high_risk_config.get("match_any_trigger_families", [])

    sentiment_config = rules.get("sentiment", {})
    negative_confirmed = sentiment_config.get("negative_confirmed_triggers", {})

    for family in trigger_families:
        if family in negative_confirmed:
            for regex in negative_confirmed[family]:
                if re.search(regex, text):
                    score += high_risk_points
                    breakdown["high_risk_category"] = high_risk_points
                    break
            if "high_risk_category" in breakdown:
                break

    # 3. Attribution (15점)
    attribution_config = score_components.get("attribution", {})
    attribution_points = attribution_config.get("points", 15)
    attribution_regex = attribution_config.get("regex", [])

    for regex in attribution_regex:
        if re.search(regex, text):
            score += attribution_points
            breakdown["attribution"] = attribution_points
            break

    # 4. Amplification terms (5점)
    amplification_config = score_components.get("amplification_terms", {})
    amplification_points = amplification_config.get("points", 5)
    amplification_regex = amplification_config.get("regex", [])

    for regex in amplification_regex:
        if re.search(regex, text):
            score += amplification_points
            breakdown["amplification_terms"] = amplification_points
            break

    # 등급 결정 (D3 > D2 > D1)
    thresholds = danger_config.get("thresholds", {})
    d3_config = thresholds.get("D3", {})
    d2_config = thresholds.get("D2", {})
    d1_config = thresholds.get("D1", {})

    # Hard trigger override for D3
    if hard_trigger_matched and d3_config.get("hard_trigger_override", True):
        danger_level = "D3"
    elif score >= d3_config.get("score_min", 50):
        danger_level = "D3"
    elif score >= d2_config.get("score_min", 20):
        danger_level = "D2"
    elif score >= d1_config.get("score_min", 0):
        danger_level = "D1"
    else:
        danger_level = ""

    return {
        "danger_rb": danger_level,
        "risk_score_rb": score,
        "score_breakdown_rb": breakdown
    }


def detect_categories(text: str, rules: dict) -> Dict:
    """
    카테고리 및 테마 분류

    Args:
        text: 제목 + 설명 결합 텍스트
        rules: rules.yaml 딕셔너리

    Returns:
        {
            "issue_category_rb": "Safety / Incident",
            "coverage_themes_rb": ["Risk / Crisis", "Operations / Technology"]
        }
    """
    categorization = rules.get("categorization", {})

    # Issue category (top 1)
    issue_config = categorization.get("issue_category", {})
    issue_categories = issue_config.get("categories", {})
    top_k_issue = issue_config.get("top_k", 1)

    issue_scores = {}
    for category, config in issue_categories.items():
        score = config.get("score", 0)
        regex_list = config.get("regex", [])

        for regex in regex_list:
            if re.search(regex, text):
                issue_scores[category] = score
                break

    # Sort by score and take top k
    sorted_issues = sorted(issue_scores.items(), key=lambda x: x[1], reverse=True)
    top_issue = sorted_issues[0][0] if sorted_issues else ""

    # Coverage theme (top 2)
    theme_config = categorization.get("coverage_theme", {})
    theme_categories = theme_config.get("categories", {})
    top_k_theme = theme_config.get("top_k", 2)

    theme_scores = {}
    for theme, config in theme_categories.items():
        score = config.get("score", 0)
        regex_list = config.get("regex", [])

        for regex in regex_list:
            if re.search(regex, text):
                theme_scores[theme] = score
                break

    # Sort by score and take top k
    sorted_themes = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)
    top_themes = [theme for theme, _ in sorted_themes[:top_k_theme]]

    return {
        "issue_category_rb": top_issue,
        "coverage_themes_rb": top_themes
    }


def extract_reason_codes(text: str, rules: dict) -> List[str]:
    """
    Reason codes 추출

    Args:
        text: 제목 + 설명 결합 텍스트
        rules: rules.yaml 딕셔너리

    Returns:
        ["DATA_BREACH", "INVESTIGATION"]
    """
    reason_codes_config = rules.get("reason_codes", {})
    mappings = reason_codes_config.get("mappings", {})

    matched_codes = []
    for code, regex in mappings.items():
        if re.search(regex, text):
            matched_codes.append(code)

    return matched_codes


def analyze_article_rb(article: Dict, rules: dict = None) -> Dict:
    """
    단일 기사 Rule-Based 분석 (메인 함수)

    Args:
        article: {"title": ..., "description": ...}
        rules: rules.yaml 딕셔너리 (None이면 자동 로드)

    Returns:
        {
            "brand_mentions": {...},
            "brand_scope_rb": "BRAND_TARGETED",
            "sentiment_rb": "NEGATIVE_CONFIRMED",
            "danger_rb": "D3",
            "risk_score_rb": 75,
            "issue_category_rb": "Safety / Incident",
            "coverage_themes_rb": [...],
            "reason_codes_rb": [...],
            "matched_rules_rb": [...],
            "score_breakdown_rb": {...}
        }
    """
    if rules is None:
        rules = load_rules()

    title = article.get("title", "")
    description = article.get("description", "")
    combined_text = f"{title} {description}"

    # 1. Brand scope
    scope_result = detect_brand_scope(title, description, rules)

    # 2. Sentiment
    sentiment_result = detect_sentiment(combined_text, rules)

    # 3. Danger (only if conditions met)
    danger_result = calculate_danger_score(
        combined_text,
        scope_result["brand_scope_rb"],
        sentiment_result["sentiment_rb"],
        rules
    )

    # 4. Categories
    category_result = detect_categories(combined_text, rules)

    # 5. Reason codes
    reason_codes = extract_reason_codes(combined_text, rules)

    # Merge all results
    return {
        "brand_mentions": scope_result["brand_mentions"],
        "brand_scope_rb": scope_result["brand_scope_rb"],
        "sentiment_rb": sentiment_result["sentiment_rb"],
        "danger_rb": danger_result["danger_rb"],
        "risk_score_rb": danger_result["risk_score_rb"],
        "issue_category_rb": category_result["issue_category_rb"],
        "coverage_themes_rb": category_result["coverage_themes_rb"],
        "reason_codes_rb": reason_codes,
        "matched_rules_rb": sentiment_result["matched_rules_rb"],
        "score_breakdown_rb": danger_result["score_breakdown_rb"]
    }


def analyze_batch_rb(articles: List[Dict], rules: dict = None) -> List[Dict]:
    """
    배치 Rule-Based 분석

    Args:
        articles: [{"title": ..., "description": ...}, ...]
        rules: rules.yaml 딕셔너리 (None이면 자동 로드)

    Returns:
        [RB 결과 딕셔너리, ...]
    """
    if rules is None:
        rules = load_rules()

    results = []
    for article in articles:
        result = analyze_article_rb(article, rules)
        results.append(result)

    return results
