"""
media_classify.py - Media Outlet Classification Module
언론사 분류 및 매체 정보 추가
"""

import json
import time
import random
import uuid
import yaml
import requests
from typing import Dict, List, Optional
from urllib.parse import urlparse
from pathlib import Path
import pandas as pd

OPENAI_API_URL = "https://api.openai.com/v1/responses"

_error_callback = None


def set_error_callback(callback):
    """Register error logger callback: fn(message: str, data: dict)."""
    global _error_callback
    _error_callback = callback


def load_api_models() -> dict:
    """api_models.yaml에서 모델 설정 로드"""
    yaml_path = Path(__file__).parent.parent.parent / "api_models.yaml"
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get("models", {})
    except FileNotFoundError:
        return {"media_classification": "gpt-5-nano"}


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


def load_media_directory(spreadsheet=None, csv_path: Path = None) -> Dict[str, Dict]:
    """
    media_directory 로드 (Google Sheets 또는 CSV). 우선순위: Sheets > CSV > 빈 dict

    Args:
        spreadsheet: gspread Spreadsheet 객체 (선택사항)
        csv_path: CSV 파일 경로 (선택사항)

    Returns:
        {domain: {"media_name": ..., "media_group": ..., "media_type": ...}} 형태의 딕셔너리
    """
    # 1. Google Sheets 시도
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
            print(f"  ⚠️  Google Sheets 로드 실패: {e}, CSV 시도")

    # 2. CSV 시도
    if csv_path and csv_path.exists():
        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')

            media_dir = {}
            for _, row in df.iterrows():
                domain = str(row.get("domain", "")).strip()
                if domain:
                    media_dir[domain] = {
                        "media_name": str(row.get("media_name", "")),
                        "media_group": str(row.get("media_group", "")),
                        "media_type": str(row.get("media_type", ""))
                    }
            print(f"📂 media_directory (CSV): {len(media_dir)}개 도메인 로드")
            return media_dir
        except Exception as e:
            print(f"  ⚠️  CSV 로드 실패: {e}")

    # 3. 빈 dict
    print("  ℹ️  media_directory가 없습니다. 신규 생성합니다.")
    return {}


def classify_media_outlets_batch(
    domains: List[str],
    openai_key: str,
    retry: bool = True,
    retry_count: int = 0,
    max_retries: int = 5,
    base_wait: int = 15,
    request_id: str = None
) -> Dict[str, Dict]:
    """
    OpenAI API를 사용하여 언론사 정보 분류 (배치 처리)

    Args:
        domains: 분류할 도메인 목록
        openai_key: OpenAI API 키
        retry: 재시도 여부

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

    request_id = request_id or uuid.uuid4().hex[:8]
    current_retry = retry_count

    while True:
        try:
            print(f"  🤖 OpenAI 분류: {len(domains)}개 신규 도메인", end="", flush=True)
            response = requests.post(
                OPENAI_API_URL,
                headers=headers,
                json=data,
                timeout=60
            )

            if response.status_code == 429:  # Rate limit
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
                        print(f" (Rate limit [req:{request_id}] Retry-After={retry_after_seconds:.1f}s, jitter={jitter:.1f}s → {wait_time:.1f}초 대기 후 재시도, retry {current_retry + 1}/{max_retries})")
                    else:
                        print(f" (Rate limit [req:{request_id}] Retry-After 없음, jitter={jitter:.1f}s → {wait_time:.1f}초 대기 후 재시도, retry {current_retry + 1}/{max_retries})")
                    time.sleep(wait_time)
                    current_retry += 1
                    continue
                print(" (Rate limit 초과, 기본값 사용)")
                if _error_callback:
                    _error_callback(
                        "OpenAI 언론사 분류 429 - 재시도 실패",
                        {"request_id": request_id, "status": 429, "attempts": max_retries}
                    )
                return _fallback_classification(domains)

            if response.status_code != 200:
                print(f" (API 오류 {response.status_code}, 기본값 사용)")
                if _error_callback:
                    _error_callback(
                        "OpenAI 언론사 분류 API 오류",
                        {"request_id": request_id, "status": response.status_code}
                    )
                return _fallback_classification(domains)

            result = response.json()
            content = result.get("output_text", "").strip()
            if not content:
                output = result.get("output", [])
                if output:
                    contents = output[0].get("content", [])
                    for item in contents:
                        if item.get("type") == "output_text" and item.get("text"):
                            content = item["text"].strip()
                            break

            try:
                parsed = json.loads(content)
                classifications = parsed.get("classifications", [])
            except json.JSONDecodeError as e:
                if _error_callback:
                    _error_callback(
                        "OpenAI 언론사 분류 파싱 실패",
                        {"request_id": request_id, "error": f"{type(e).__name__}: {e}", "attempt": current_retry + 1}
                    )
                if retry and current_retry < max_retries - 1:
                    print(f" (JSON 파싱 실패 [req:{request_id}]: {str(e)[:50]}, 2초 대기 후 재시도 {current_retry + 1}/{max_retries})")
                    time.sleep(2)
                    current_retry += 1
                    continue
                else:
                    print(f" (JSON 파싱 실패: {str(e)[:50]}, 기본값 사용)")
                    if _error_callback:
                        _error_callback(
                            "OpenAI 언론사 분류 파싱 실패",
                            {"request_id": request_id, "error": f"{type(e).__name__}: {e}", "attempt": current_retry + 1}
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

        except requests.exceptions.Timeout:
            print(" (Timeout, 기본값 사용)")
            if _error_callback:
                _error_callback(
                    "OpenAI 언론사 분류 Timeout",
                    {"request_id": request_id}
                )
            return _fallback_classification(domains)
        except Exception as e:
            print(f" (오류: {e}, 기본값 사용)")
            if _error_callback:
                _error_callback(
                    "OpenAI 언론사 분류 실패(예외)",
                    {"request_id": request_id, "error": str(e)}
                )
            return _fallback_classification(domains)


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


def update_media_directory(spreadsheet=None, new_entries: Dict[str, Dict] = None, csv_path: Path = None) -> None:
    """
    media_directory 업데이트 (Google Sheets 및/또는 CSV)

    Args:
        spreadsheet: gspread Spreadsheet 객체 (선택사항)
        new_entries: {domain: {"media_name": ..., "media_group": ..., "media_type": ...}}
        csv_path: CSV 파일 경로 (선택사항)
    """
    if not new_entries:
        return

    # 1. Google Sheets 업데이트 (배치 처리로 rate limit 방지)
    if spreadsheet:
        try:
            try:
                worksheet = spreadsheet.worksheet("media_directory")
            except:
                worksheet = spreadsheet.add_worksheet(title="media_directory", rows=1, cols=4)
                worksheet.append_row(["domain", "media_name", "media_group", "media_type"])

            # 배치 업데이트: 모든 행을 한 번에 추가
            rows_to_add = [
                [domain, info.get("media_name", ""), info.get("media_group", ""), info.get("media_type", "")]
                for domain, info in new_entries.items()
            ]

            # append_rows로 한 번에 추가 (1 write request)
            if rows_to_add:
                worksheet.append_rows(rows_to_add)

            print(f"✅ media_directory (Google Sheets): {len(new_entries)}개 신규 도메인 추가")
        except Exception as e:
            print(f"⚠️  Google Sheets 업데이트 실패: {e}")

    # 2. CSV 업데이트 (항상 실행)
    if csv_path:
        try:
            if csv_path.exists():
                existing_df = pd.read_csv(csv_path, encoding='utf-8-sig')
            else:
                existing_df = pd.DataFrame(columns=["domain", "media_name", "media_group", "media_type"])

            new_rows = [
                {"domain": domain, "media_name": info.get("media_name", ""),
                 "media_group": info.get("media_group", ""), "media_type": info.get("media_type", "")}
                for domain, info in new_entries.items()
            ]

            new_df = pd.DataFrame(new_rows)
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
            updated_df = updated_df.drop_duplicates(subset=["domain"], keep="last")

            csv_path.parent.mkdir(parents=True, exist_ok=True)
            updated_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

            print(f"✅ media_directory (CSV): {len(new_entries)}개 신규 도메인 추가")
        except Exception as e:
            print(f"⚠️  CSV 저장 실패: {e}")


def add_media_columns(
    df: pd.DataFrame,
    spreadsheet=None,
    openai_key: str = None,
    csv_path: Path = None
) -> pd.DataFrame:
    """
    DataFrame에 언론사 정보 컬럼 추가

    Args:
        df: 처리된 DataFrame (originallink 컬럼 필요)
        spreadsheet: gspread Spreadsheet 객체 (선택사항)
        openai_key: OpenAI API 키 (선택사항)
        csv_path: media_directory CSV 경로 (선택사항)

    Returns:
        4개의 새로운 컬럼이 추가된 DataFrame:
        - media_domain: 추출된 도메인
        - media_name: 언론사명
        - media_group: 언론사 그룹
        - media_type: 매체 분류
    """
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
        return df

    try:
        # 도메인 추출
        df["media_domain"] = df["originallink"].apply(extract_domain_safe)

        # 고유한 도메인 목록
        unique_domains = df[df["media_domain"] != ""]["media_domain"].unique()
        unique_domains = [d for d in unique_domains if d]

        if not unique_domains:
            print("  ⚠️  추출된 도메인이 없습니다.")
            return df

        # media_directory 로드 (Google Sheets 또는 CSV)
        existing_media = load_media_directory(spreadsheet=spreadsheet, csv_path=csv_path)
        new_domains = [d for d in unique_domains if d not in existing_media]

        # 신규 도메인 분류 (OpenAI) - 배치 처리 (100개씩)
        if new_domains and openai_key:
            new_media = {}
            for i in range(0, len(new_domains), 100):
                batch = new_domains[i:i+100]
                batch_result = classify_media_outlets_batch(batch, openai_key)
                new_media.update(batch_result)

            existing_media.update(new_media)

            # media_directory 업데이트 (Sheets + CSV)
            update_media_directory(spreadsheet=spreadsheet, new_entries=new_media, csv_path=csv_path)

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

        return df

    except Exception as e:
        print(f"⚠️  언론사 정보 추가 중 오류: {e}")
        return df
