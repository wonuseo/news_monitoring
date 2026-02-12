"""
classify.py - AI Classification Module
AI를 통한 카테고리화 및 위험도 판단
1단계: 감정 분석 (긍정/중립/부정)
2단계: 카테고리 분류
3단계: 부정 기사에 대해서만 위험도 평가 (상/중/하)
"""

import json
import time
import random
import requests
import uuid
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd


OPENAI_API_URL = "https://api.openai.com/v1/responses"

_error_callback = None


def set_error_callback(callback):
    """Register error logger callback: fn(message: str, data: dict)."""
    global _error_callback
    _error_callback = callback

# 한국어 카테고리
CATEGORIES = [
    "법률/규제",      # 법적 문제, 규제, 소송
    "보안/데이터",    # 해킹, 데이터 유출
    "안전/사고",      # 화재, 사고, 안전 문제
    "재무/실적",      # 실적, 주가, 투자
    "제품/서비스",    # 신규 서비스, 리뉴얼
    "평판/SNS",       # 여론, SNS 이슈
    "운영/기타",      # 일반 운영, 인사
]


def call_openai_batch(articles: List[Dict], task_type: str, openai_key: str,
                      retry: bool = True, retry_count: int = 0,
                      max_retries: int = 5, base_wait: int = 15,
                      request_id: str = None) -> Dict[int, Dict]:
    """
    OpenAI API 배치 호출
    
    Args:
        articles: 기사 목록 [{"title": ..., "description": ...}, ...]
        task_type: "sentiment" / "categorize" / "risk_assess"
        openai_key: OpenAI API 키
        retry: 재시도 여부
    
    Returns:
        {idx: {"sentiment": "긍정", ...}} 형태의 결과
    """
    if task_type == "sentiment":
        # 1단계: 감정 분석
        articles_text = ""
        for idx, article in enumerate(articles):
            articles_text += f"\n[{idx}]\n제목: {article['title']}\n설명: {article['description'][:150]}\n"
        
        prompt = f"""당신은 뉴스 감정 분석 전문가입니다. 각 기사를 긍정/중립/부정으로 분류하세요.

기사 목록:{articles_text}

JSON 객체만 반환하세요:
{{
  "results": [
    {{"id": 0, "sentiment": "긍정"}},
    {{"id": 1, "sentiment": "부정"}}
  ]
}}

감정 분류 기준:
- 긍정: 좋은 소식, 성과, 수상, 성장 등
- 중립: 일반 소식, 사실 전달
- 부정: 사고, 논란, 비판, 손실 등

JSON 객체만 출력하세요."""
        
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["results"],
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["id", "sentiment"],
                        "properties": {
                            "id": {"type": "integer"},
                            "sentiment": {"type": "string", "enum": ["긍정", "중립", "부정"]}
                        }
                    }
                }
            }
        }
    
    elif task_type == "categorize":
        # 2단계: 카테고리 분류
        articles_text = ""
        for idx, article in enumerate(articles):
            articles_text += f"\n[{idx}] {article['sentiment']}\n제목: {article['title']}\n"
        
        prompt = f"""당신은 호텔 업계 뉴스 분석가입니다. 각 기사를 카테고리로 분류하세요.

기사 목록:{articles_text}

JSON 객체만 반환하세요:
{{
  "results": [
    {{"id": 0, "category": "제품/서비스"}},
    {{"id": 1, "category": "안전/사고"}}
  ]
}}

카테고리 (하나만 선택):
- 법률/규제: 법적 문제, 규제, 소송, 조사
- 보안/데이터: 해킹, 정보 유출, 개인정보 침해
- 안전/사고: 화재, 사망, 부상, 안전 문제
- 재무/실적: 실적 발표, 주가, 투자, 재무 성과
- 제품/서비스: 신규 오픈, 리뉴얼, 제휴, 프로모션
- 평판/SNS: 여론, SNS 이슈, 불매운동, 고객 불만
- 운영/기타: 일반 운영, 인사, 기타

JSON 객체만 출력하세요."""
        
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["results"],
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["id", "category"],
                        "properties": {
                            "id": {"type": "integer"},
                            "category": {"type": "string", "enum": CATEGORIES}
                        }
                    }
                }
            }
        }
    
    else:  # risk_assess
        # 3단계: 위험도 평가 (부정 기사만)
        articles_text = ""
        for idx, article in enumerate(articles):
            articles_text += f"\n[{idx}] {article['category']}\n제목: {article['title']}\n"
        
        prompt = f"""당신은 호텔 업계 리스크 분석가입니다. 각 부정 기사의 위험도를 평가하세요.

부정 기사 목록:{articles_text}

JSON 객체만 반환하세요:
{{
  "results": [
    {{"id": 0, "risk_level": "상", "reason": "화재 사고"}},
    {{"id": 1, "risk_level": "중", "reason": "고객 불만"}}
  ]
}}

위험도 기준:
- 상: 즉각 대응 필요. 사망/화재/대규모 유출/영업정지/집단소송 등
- 중: 모니터링 필요. 고객 불만 확산/규제 조사/평판 손상 등
- 하: 경미한 영향. 소규모 불만/일반 논란 등

이유(reason)는 3-5단어로 간결하게 작성하세요.

JSON 객체만 출력하세요."""
        
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["results"],
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["id", "risk_level", "reason"],
                        "properties": {
                            "id": {"type": "integer"},
                            "risk_level": {"type": "string", "enum": ["상", "중", "하"]},
                            "reason": {"type": "string", "maxLength": 30}
                        }
                    }
                }
            }
        }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_key}"
    }
    
    payload = {
        "model": "gpt-5-nano",
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": "당신은 정확한 JSON 응답만 제공하는 어시스턴트입니다."}]},
            {"role": "user", "content": [{"type": "input_text", "text": prompt}]}
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": f"classify_{task_type}",
                "strict": True,
                "schema": schema
            }
        },
    }
    
    request_id = request_id or uuid.uuid4().hex[:8]
    current_retry = retry_count

    while True:
        try:
            response = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=60)

            if response.status_code == 401:
                raise RuntimeError("OpenAI API 인증 실패 (401). API 키를 확인하세요.")
            elif response.status_code == 429:
                if retry and current_retry < max_retries - 1:
                    wait_time = base_wait * (2 ** current_retry)
                    retry_after = response.headers.get("Retry-After")
                    retry_after_seconds = None
                    if retry_after:
                        try:
                            retry_after_seconds = float(retry_after)
                        except ValueError:
                            retry_after_seconds = None
                    if retry_after_seconds is not None:
                        wait_time = retry_after_seconds
                    wait_time = max(wait_time, 10)
                    jitter = random.uniform(0, 6)
                    wait_time += jitter
                    if _error_callback:
                        _error_callback(
                            "OpenAI 요청 한도 초과 (429)",
                            {
                                "request_id": request_id,
                                "status": 429,
                                "retry_after": retry_after_seconds,
                                "wait_time": round(wait_time, 1),
                                "attempt": current_retry + 1,
                            }
                        )
                    if retry_after_seconds is not None:
                        print(f"⚠️  OpenAI 요청 한도 초과 (429) [req:{request_id}] Retry-After={retry_after_seconds:.1f}s, jitter={jitter:.1f}s → {wait_time:.1f}초 대기 후 재시도... (retry {current_retry + 1}/{max_retries})")
                    else:
                        print(f"⚠️  OpenAI 요청 한도 초과 (429) [req:{request_id}] Retry-After 없음, jitter={jitter:.1f}s → {wait_time:.1f}초 대기 후 재시도... (retry {current_retry + 1}/{max_retries})")
                    time.sleep(wait_time)
                    current_retry += 1
                    continue
                if _error_callback:
                    _error_callback(
                        "OpenAI 요청 한도 초과 (429) - 재시도 실패",
                        {"request_id": request_id, "status": 429, "attempts": max_retries}
                    )
                raise RuntimeError("OpenAI 요청 한도 초과 (재시도 후에도 실패)")
            elif response.status_code >= 500:
                if _error_callback:
                    _error_callback(
                        "OpenAI 서버 오류",
                        {"request_id": request_id, "status": response.status_code}
                    )
                raise RuntimeError(f"OpenAI 서버 오류 ({response.status_code})")

            response.raise_for_status()
            data = response.json()
            content = data.get("output_text", "").strip()
            if not content:
                output = data.get("output", [])
                if output:
                    contents = output[0].get("content", [])
                    for item in contents:
                        if item.get("type") == "output_text" and item.get("text"):
                            content = item["text"].strip()
                            break

            if not content:
                raise KeyError("Responses API 응답에서 output_text를 찾을 수 없습니다.")

            parsed = json.loads(content)
            results = parsed.get("results", [])

            # 리스트를 딕셔너리로 변환 {id: result}
            result_dict = {}
            for item in results:
                idx = item.get("id")
                if idx is not None:
                    result_dict[idx] = item

            return result_dict

        except (json.JSONDecodeError, KeyError, requests.exceptions.RequestException) as e:
            print(f"⚠️  배치 {task_type} 오류 [req:{request_id}]: {e}")
            if _error_callback:
                _error_callback(
                    f"OpenAI 배치 {task_type} 오류",
                    {"request_id": request_id, "error": str(e), "attempt": current_retry + 1}
                )
            if retry and current_retry < max_retries - 1:
                print(f"재시도 중... [req:{request_id} retry:{current_retry + 1}/{max_retries}]")
                time.sleep(2)
                current_retry += 1
                continue
            if _error_callback:
                _error_callback(
                    f"OpenAI 배치 {task_type} 실패",
                    {"request_id": request_id, "error": str(e), "attempt": current_retry + 1}
                )
            return {}


def should_classify(row: Dict, max_competitor_classify: int, 
                   competitor_count: Dict[str, int]) -> bool:
    """분류 대상 여부 판단"""
    # 우리 브랜드는 전부 분류
    if row.get("group") == "OUR":
        return True
    
    # 경쟁사는 각 브랜드당 최신 N개만 분류
    if row.get("group") == "COMPETITOR":
        query = row.get("query", "")
        count = competitor_count.get(query, 0)
        if count < max_competitor_classify:
            competitor_count[query] = count + 1
            return True
    
    return False


def classify_all(df: pd.DataFrame, openai_key: str, 
                max_competitor_classify: int, chunk_size: int, 
                dry_run: bool) -> pd.DataFrame:
    """
    3단계 분류 프로세스
    1. 감정 분석 (전체)
    2. 카테고리 분류 (전체)
    3. 위험도 평가 (부정 기사만)
    """
    df = df.copy()
    
    # 컬럼 초기화
    df["sentiment"] = ""
    df["category"] = ""
    df["risk_level"] = ""
    df["reason"] = ""
    df["classified_at"] = ""
    
    if dry_run:
        print("🔬 DRY RUN 모드: AI 분류 생략")
        return df
    
    # 분류 대상 수집
    competitor_count = {}
    articles_to_classify = []
    indices_to_classify = []
    
    for idx, row in df.iterrows():
        if should_classify(row.to_dict(), max_competitor_classify, competitor_count):
            articles_to_classify.append({
                'title': row['title'],
                'description': row['description']
            })
            indices_to_classify.append(idx)
    
    if len(articles_to_classify) == 0:
        print("⚠️  분류할 기사가 없습니다")
        return df
    
    total = len(articles_to_classify)
    print(f"\n🤖 AI 분류 시작: {total}개 기사 (청크 크기: {chunk_size})")
    
    timestamp = datetime.now().isoformat()
    
    # 청크 단위로 처리
    for chunk_start in range(0, total, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total)
        chunk_articles = articles_to_classify[chunk_start:chunk_end]
        chunk_indices = indices_to_classify[chunk_start:chunk_end]
        chunk_num = (chunk_start // chunk_size) + 1
        total_chunks = (total + chunk_size - 1) // chunk_size
        
        print(f"\n--- 청크 {chunk_num}/{total_chunks} ({len(chunk_articles)}개 기사) ---")
        
        # Step 1: 감정 분석
        print("  [1/3] 감정 분석 중...")
        sentiment_results = call_openai_batch(chunk_articles, "sentiment", openai_key)
        
        for i, idx in enumerate(chunk_indices):
            if i in sentiment_results:
                sentiment = sentiment_results[i].get("sentiment", "중립")
                df.at[idx, "sentiment"] = sentiment
                chunk_articles[i]["sentiment"] = sentiment
            else:
                df.at[idx, "sentiment"] = "중립"
                chunk_articles[i]["sentiment"] = "중립"
        
        time.sleep(1)
        
        # Step 2: 카테고리 분류
        print("  [2/3] 카테고리 분류 중...")
        category_results = call_openai_batch(chunk_articles, "categorize", openai_key)
        
        for i, idx in enumerate(chunk_indices):
            if i in category_results:
                category = category_results[i].get("category", "운영/기타")
                if category not in CATEGORIES:
                    category = "운영/기타"
                df.at[idx, "category"] = category
                chunk_articles[i]["category"] = category
            else:
                df.at[idx, "category"] = "운영/기타"
                chunk_articles[i]["category"] = "운영/기타"
        
        time.sleep(1)
        
        # Step 3: 위험도 평가 (부정 기사만)
        negative_articles = [art for art in chunk_articles if art.get("sentiment") == "부정"]
        negative_indices_map = {
            i: chunk_indices[i] 
            for i, art in enumerate(chunk_articles) 
            if art.get("sentiment") == "부정"
        }
        
        if len(negative_articles) > 0:
            print(f"  [3/3] 위험도 평가 중 (부정 기사 {len(negative_articles)}개)...")
            risk_results = call_openai_batch(negative_articles, "risk_assess", openai_key)
            
            for i, idx in negative_indices_map.items():
                # negative_articles의 인덱스는 0부터 시작하므로 매핑 필요
                neg_idx = list(negative_indices_map.keys()).index(i)
                if neg_idx in risk_results:
                    risk_level = risk_results[neg_idx].get("risk_level", "하")
                    reason = risk_results[neg_idx].get("reason", "평가 실패")
                    df.at[idx, "risk_level"] = risk_level
                    df.at[idx, "reason"] = reason
                else:
                    df.at[idx, "risk_level"] = "하"
                    df.at[idx, "reason"] = "평가 실패"
        else:
            print("  [3/3] 부정 기사 없음, 위험도 평가 생략")
        
        # 긍정/중립 기사는 위험도 없음
        for i, idx in enumerate(chunk_indices):
            if df.at[idx, "sentiment"] != "부정":
                df.at[idx, "risk_level"] = "-"
                df.at[idx, "reason"] = "-"
        
        # 타임스탬프 추가
        for idx in chunk_indices:
            df.at[idx, "classified_at"] = timestamp
        
        print(f"  ✅ 청크 {chunk_num}/{total_chunks} 완료")
        time.sleep(1)
    
    print(f"\n✅ AI 분류 완료: {total}개 기사 처리")
    return df
