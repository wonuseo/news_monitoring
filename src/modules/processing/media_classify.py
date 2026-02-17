"""
media_classify.py - Media Outlet Classification Module
언론사 분류 및 매체 정보 추가
"""

import json
import uuid
from typing import Dict, List, Optional
from urllib.parse import urlparse
from pathlib import Path
import pandas as pd

from src.utils.openai_client import (
    OPENAI_API_URL,
    set_error_callback,
    load_api_models,
    call_openai_with_retry,
    extract_response_text,
    notify_error,
)
from src.utils.sheets_helpers import get_or_create_worksheet


def extract_domain_safe(url: str) -> str:
    """
    URL에서 도메인 추출 (www. 제거, 에러 안전)

    Args:
        url: 기사 URL

    Returns:
        도메인 (예: "chosun.com") 또는 빈 문자열

    Examples:
        https://www.chosun.com/article/123 → chosun.com
        https://woman.chosun.com/article/456 → woman.chosun.com
        invalid-url → ""
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.replace('www.', '')
        return domain if domain else ""
    except Exception:
        return ""


def load_media_directory(spreadsheet=None) -> Dict[str, Dict]:
    """
    media_directory 로드 (Google Sheets만 사용)

    Args:
        spreadsheet: gspread Spreadsheet 객체 (선택사항)

    Returns:
        {domain: {"media_name": ..., "media_group": ..., "media_type": ...}} 형태의 딕셔너리
    """
    if spreadsheet:
        try:
            worksheet = spreadsheet.worksheet("media_directory")
            existing_data = worksheet.get_all_records()

            if existing_data:
                media_dir = {}
                for row in existing_data:
                    domain = row.get("domain", "").strip()
                    if domain:
                        media_dir[domain] = {
                            "media_name": row.get("media_name", ""),
                            "media_group": row.get("media_group", ""),
                            "media_type": row.get("media_type", "")
                        }
                print(f"📂 media_directory (Google Sheets): {len(media_dir)}개 도메인 로드")
                return media_dir
        except Exception as e:
            print(f"  ⚠️  Google Sheets media_directory 로드 실패: {e}")

    print("  ℹ️  media_directory가 없습니다. 신규 생성합니다.")
    return {}


def classify_media_outlets_batch(
    domains: List[str],
    openai_key: str,
    max_retries: int = 5,
) -> Dict[str, Dict]:
    """
    OpenAI API를 사용하여 언론사 정보 분류 (배치 처리)

    Args:
        domains: 분류할 도메인 목록
        openai_key: OpenAI API 키
        max_retries: 최대 재시도 횟수

    Returns:
        {domain: {"media_name": ..., "media_group": ..., "media_type": ...}} 형태의 딕셔너리
    """
    if not domains:
        return {}

    # 도메인 목록을 텍스트로 변환
    domain_list = "\n".join(domains)

    prompt = f"""한국 언론사 정보를 분류하세요.

도메인:
{domain_list}

JSON만 반환:
{{"classifications":[{{"domain":"example.com","media_name":"예시언론","media_group":"예시그룹","media_type":"종합지"}}]}}

분류:
- 종합지/경제지/IT전문지/방송사/통신사/인터넷신문/기타
- group은 모르면 media_name 동일
- JSON만"""

    headers = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json"
    }

    # api_models.yaml에서 모델 로드
    api_models = load_api_models()
    model = api_models.get("media_classification", "gpt-5-nano")

    data = {
        "model": model,
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": prompt}]}
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "media_classification_batch",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["classifications"],
                    "properties": {
                        "classifications": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["domain", "media_name", "media_group", "media_type"],
                                "properties": {
                                    "domain": {"type": "string"},
                                    "media_name": {"type": "string"},
                                    "media_group": {"type": "string"},
                                    "media_type": {
                                        "type": "string",
                                        "enum": ["종합지", "경제지", "IT전문지", "방송사", "통신사", "인터넷신문", "기타"]
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "max_output_tokens": min(len(domains) * 100, 8000)  # 최소 100토큰/도메인, 최대 8000
    }

    request_id = uuid.uuid4().hex[:8]
    print(f"  🤖 OpenAI 분류: {len(domains)}개 신규 도메인", end="", flush=True)

    response = call_openai_with_retry(
        OPENAI_API_URL, headers, data,
        max_retries=max_retries, request_id=request_id, label="언론사분류"
    )

    if response is None:
        print(" (재시도 실패, 기본값 사용)")
        return _fallback_classification(domains)

    result = response.json()
    content = extract_response_text(result)

    try:
        parsed = json.loads(content)
        classifications = parsed.get("classifications", [])
    except json.JSONDecodeError as e:
        print(f" (JSON 파싱 실패: {str(e)[:50]}, 기본값 사용)")
        notify_error(
            "OpenAI 언론사 분류 파싱 실패",
            {"request_id": request_id, "error": f"{type(e).__name__}: {e}"}
        )
        return _fallback_classification(domains)

    # 도메인 기준 딕셔너리로 변환
    media_info = {}
    for item in classifications:
        domain = item.get("domain", "")
        if domain:
            media_info[domain] = {
                "media_name": item.get("media_name", domain),
                "media_group": item.get("media_group", domain),
                "media_type": item.get("media_type", "기타")
            }

    print(f" ✅")
    return media_info


def _fallback_classification(domains: List[str]) -> Dict[str, Dict]:
    """
    OpenAI 호출 실패 시 기본값 사용

    Args:
        domains: 도메인 목록

    Returns:
        {domain: {media_name: domain, media_group: domain, media_type: "기타"}}
    """
    return {
        domain: {
            "media_name": domain,
            "media_group": domain,
            "media_type": "기타"
        }
        for domain in domains
    }


def update_media_directory(spreadsheet=None, new_entries: Dict[str, Dict] = None) -> None:
    """
    media_directory 업데이트 (Google Sheets만)

    Args:
        spreadsheet: gspread Spreadsheet 객체 (선택사항)
        new_entries: {domain: {"media_name": ..., "media_group": ..., "media_type": ...}}
    """
    if not new_entries:
        return

    if spreadsheet:
        try:
            worksheet = get_or_create_worksheet(spreadsheet, "media_directory", rows=1, cols=4)
            existing_rows = worksheet.row_values(1)
            if not existing_rows:
                worksheet.append_row(["domain", "media_name", "media_group", "media_type"])

            rows_to_add = [
                [domain, info.get("media_name", ""), info.get("media_group", ""), info.get("media_type", "")]
                for domain, info in new_entries.items()
            ]

            if rows_to_add:
                worksheet.append_rows(rows_to_add)

            print(f"✅ media_directory (Google Sheets): {len(new_entries)}개 신규 도메인 추가")
        except Exception as e:
            print(f"⚠️  Google Sheets 업데이트 실패: {e}")


def add_media_columns(
    df: pd.DataFrame,
    spreadsheet=None,
    openai_key: str = None,
):
    """
    DataFrame에 언론사 정보 컬럼 추가

    Args:
        df: 처리된 DataFrame (originallink 컬럼 필요)
        spreadsheet: gspread Spreadsheet 객체 (선택사항)
        openai_key: OpenAI API 키 (선택사항)

    Returns:
        (df, media_stats) 튜플:
        - df: 4개의 새로운 컬럼이 추가된 DataFrame
        - media_stats: {"media_domains_total", "media_domains_new", "media_domains_cached"}
    """
    empty_stats = {"media_domains_total": 0, "media_domains_new": 0, "media_domains_cached": 0}

    print("🏢 언론사 정보 추가 중...")
    df = df.copy()

    # 컬럼 초기화
    df["media_domain"] = ""
    df["media_name"] = ""
    df["media_group"] = ""
    df["media_type"] = ""

    # originallink 컬럼이 없으면 조기 반환
    if "originallink" not in df.columns:
        print("  ⚠️  originallink 컬럼이 없습니다.")
        return df, empty_stats

    try:
        # 도메인 추출
        df["media_domain"] = df["originallink"].apply(extract_domain_safe)

        # 고유한 도메인 목록
        unique_domains = df[df["media_domain"] != ""]["media_domain"].unique()
        unique_domains = [d for d in unique_domains if d]

        if not unique_domains:
            print("  ⚠️  추출된 도메인이 없습니다.")
            return df, empty_stats

        # media_directory 로드 (Google Sheets)
        existing_media = load_media_directory(spreadsheet=spreadsheet)
        new_domains = [d for d in unique_domains if d not in existing_media]

        # 신규 도메인 분류 (OpenAI) - 배치 처리 (100개씩)
        if new_domains and openai_key:
            new_media = {}
            for i in range(0, len(new_domains), 100):
                batch = new_domains[i:i+100]
                batch_result = classify_media_outlets_batch(batch, openai_key)
                new_media.update(batch_result)

            existing_media.update(new_media)

            # media_directory 업데이트 (Sheets만)
            update_media_directory(spreadsheet=spreadsheet, new_entries=new_media)

        # DataFrame에 정보 추가
        for idx, row in df.iterrows():
            domain = row["media_domain"]
            if domain in existing_media:
                info = existing_media[domain]
                df.at[idx, "media_name"] = info.get("media_name", "")
                df.at[idx, "media_group"] = info.get("media_group", "")
                df.at[idx, "media_type"] = info.get("media_type", "")

        # 통계
        has_info = (df["media_name"] != "").sum()
        print(f"✅ 완료: {len(df)}개 기사에 언론사 정보 추가")
        print(f"  - 기존 디렉토리: {len(existing_media) - len(new_domains)}개")
        if new_domains:
            print(f"  - 신규 분류: {len(new_domains)}개")

        media_stats = {
            "media_domains_total": len(unique_domains),
            "media_domains_new": len(new_domains),
            "media_domains_cached": len(unique_domains) - len(new_domains),
        }
        return df, media_stats

    except Exception as e:
        print(f"⚠️  언론사 정보 추가 중 오류: {e}")
        return df, empty_stats
