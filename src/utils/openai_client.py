"""
openai_client.py - Shared OpenAI API Utilities
에러 콜백, 모델 로딩(캐싱), 재시도/Rate Limit 처리, 응답 파싱
"""

import json
import time
import random
import uuid
import yaml
import requests
from pathlib import Path
from typing import Dict, Optional, Callable, Any


OPENAI_API_URL = "https://api.openai.com/v1/responses"

# ── Error callback management ──────────────────────────────────────
_error_callback: Optional[Callable] = None


def set_error_callback(callback: Optional[Callable]):
    """Register a global error logger callback: fn(message: str, data: dict)."""
    global _error_callback
    _error_callback = callback


def get_error_callback() -> Optional[Callable]:
    return _error_callback


def notify_error(message: str, data: Optional[Dict] = None):
    """Call the registered error callback if present."""
    if _error_callback:
        _error_callback(message, data)


# ── API model loading (cached) ─────────────────────────────────────
_api_models_cache: Optional[Dict] = None

_DEFAULT_MODELS = {
    "article_classification": "gpt-5-nano",
    "media_classification": "gpt-5-nano",
    "press_release_summary": "gpt-5-nano",
}


def load_api_models(yaml_path: Path = None) -> dict:
    """
    api_models.yaml 로드 (캐싱 지원).
    첫 호출 시 파일을 읽고 이후 캐시를 반환.
    """
    global _api_models_cache
    if _api_models_cache is not None:
        return _api_models_cache

    if yaml_path is None:
        yaml_path = Path(__file__).parent.parent / "api_models.yaml"

    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            _api_models_cache = config.get("models", {})
    except FileNotFoundError:
        print(f"  api_models.yaml 파일을 찾을 수 없습니다. 기본 모델 사용")
        _api_models_cache = _DEFAULT_MODELS.copy()

    return _api_models_cache


# ── Response text extraction ───────────────────────────────────────
def extract_response_text(result: Dict) -> str:
    """
    OpenAI Responses API 응답에서 텍스트 추출.

    우선순위:
    1. output_text (최상위)
    2. output[0].content[*] (output_text 타입)
    3. choices[*].message.content (legacy fallback)
    """
    # 1) output_text
    text = result.get("output_text", "")
    if isinstance(text, str) and text.strip():
        return text.strip()

    # 2) output[0].content
    for section in result.get("output", []):
        contents = section.get("content", [])
        for item in contents:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "output_text" and isinstance(item.get("text"), str):
                return item["text"].strip()
            raw_text = item.get("text")
            if isinstance(raw_text, str) and raw_text.strip():
                return raw_text.strip()

    # 3) Legacy choices fallback
    for choice in result.get("choices", []):
        message = choice.get("message", {})
        content = message.get("content", "")
        if isinstance(content, str) and content.strip():
            return content.strip()

    return ""


# ── Retry with rate-limit handling ─────────────────────────────────
def call_openai_with_retry(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    *,
    max_retries: int = 5,
    base_wait_429: int = 15,
    base_wait_5xx: int = 8,
    base_wait_other: int = 2,
    timeout: int = 60,
    request_id: str = None,
    label: str = "OpenAI",
) -> Optional[requests.Response]:
    """
    OpenAI API 호출 + 재시도 (429/5xx 처리, Retry-After 파싱, 지수 백오프, jitter).

    Returns:
        성공 시 Response 객체 (status 200), 모든 재시도 실패 시 None.
        401/400은 RuntimeError를 raise.
    """
    request_id = request_id or uuid.uuid4().hex[:8]

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)

            if response.status_code == 200:
                return response

            if response.status_code == 401:
                raise RuntimeError("OpenAI API 인증 실패 (401). API 키를 확인하세요.")

            if response.status_code == 400:
                try:
                    error_detail = response.json()
                except json.JSONDecodeError:
                    error_detail = response.text[:200]
                notify_error(
                    f"{label} 요청 오류 (400)",
                    {"request_id": request_id, "status": 400, "detail": error_detail, "attempt": attempt + 1}
                )
                raise RuntimeError(f"OpenAI 요청 오류 (400): {error_detail}")

            if response.status_code == 429:
                print(f"  {label} 요청 한도 초과 (429) [req:{request_id}] - 시도 {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    wait_time = _compute_wait(response, base_wait_429, attempt, request_id, label)
                    time.sleep(wait_time)
                    continue
                else:
                    notify_error(
                        f"{label} 요청 한도 초과 (429) - 재시도 실패",
                        {"request_id": request_id, "status": 429, "attempts": max_retries}
                    )
                    return None

            if response.status_code >= 500:
                print(f"  {label} 서버 오류 ({response.status_code}) - 시도 {attempt + 1}/{max_retries}")
                notify_error(
                    f"{label} 서버 오류",
                    {"request_id": request_id, "status": response.status_code, "attempt": attempt + 1}
                )
                if attempt < max_retries - 1:
                    wait_time = base_wait_5xx * (2 ** attempt)
                    print(f"   {wait_time}초 대기 후 재시도...")
                    time.sleep(wait_time)
                    continue
                else:
                    notify_error(
                        f"{label} 서버 오류 - 재시도 실패",
                        {"request_id": request_id, "status": response.status_code, "attempts": max_retries}
                    )
                    return None

            # Other non-200 status
            print(f"  {label} API 오류 ({response.status_code})")
            notify_error(
                f"{label} API 오류",
                {"request_id": request_id, "status": response.status_code}
            )
            return None

        except (requests.exceptions.RequestException,) as e:
            print(f"  {label} 호출 오류 - 시도 {attempt + 1}/{max_retries}: {e}")
            notify_error(
                f"{label} 호출 오류 (예외)",
                {"request_id": request_id, "error": str(e), "attempt": attempt + 1}
            )
            if attempt < max_retries - 1:
                wait_time = base_wait_other * (2 ** attempt)
                print(f"   {wait_time}초 대기 후 재시도...")
                time.sleep(wait_time)
                continue
            else:
                print(f"  {label} {max_retries}회 재시도 후에도 실패: {e}")
                notify_error(
                    f"{label} 호출 실패 (예외)",
                    {"request_id": request_id, "error": str(e)}
                )
                return None

    return None


def _compute_wait(response, base_wait: int, attempt: int, request_id: str, label: str) -> float:
    """Retry-After 파싱 + 지수 백오프 + jitter 계산"""
    wait_time = base_wait * (2 ** attempt)
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

    notify_error(
        f"{label} 요청 한도 초과 (429)",
        {
            "request_id": request_id,
            "status": 429,
            "retry_after": retry_after_seconds,
            "wait_time": round(wait_time, 1),
            "attempt": attempt + 1,
        }
    )

    if retry_after_seconds is not None:
        print(f"   Retry-After={retry_after_seconds:.1f}s, jitter={jitter:.1f}s -> {wait_time:.1f}초 대기 후 재시도... [req:{request_id}]")
    else:
        print(f"   Retry-After 없음, jitter={jitter:.1f}s -> {wait_time:.1f}초 대기 후 재시도... [req:{request_id}]")

    return wait_time
