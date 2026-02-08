"""
media_classify.py - Media Outlet Classification Module
언론사 분류 및 매체 정보 추가
"""

import json
import time
import requests
from typing import Dict, List, Optional
from urllib.parse import urlparse
import pandas as pd


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


def load_media_directory(spreadsheet) -> Dict[str, Dict]:
    """
    Google Sheets에서 media_directory 시트 로드

    Args:
        spreadsheet: gspread Spreadsheet 객체

    Returns:
        {domain: {"media_name": ..., "media_group": ..., "media_type": ...}} 형태의 딕셔너리
    """
    try:
        # 워크시트 선택 (없으면 빈 dict 반환)
        try:
            worksheet = spreadsheet.worksheet("media_directory")
        except:
            print("  ℹ️  'media_directory' 워크시트가 없습니다. 신규 생성합니다.")
            return {}

        # 모든 데이터 읽기
        existing_data = worksheet.get_all_records()

        if not existing_data:
            print("  ℹ️  'media_directory' 워크시트가 비어있습니다.")
            return {}

        # 딕셔너리로 변환
        media_dir = {}
        for row in existing_data:
            domain = row.get("domain", "").strip()
            if domain:
                media_dir[domain] = {
                    "media_name": row.get("media_name", ""),
                    "media_group": row.get("media_group", ""),
                    "media_type": row.get("media_type", "")
                }

        print(f"📂 media_directory: {len(media_dir)}개 도메인 로드")
        return media_dir

    except Exception as e:
        print(f"⚠️  media_directory 로드 실패: {e}")
        return {}


def classify_media_outlets_batch(
    domains: List[str],
    openai_key: str,
    retry: bool = True
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

    prompt = f"""당신은 한국 언론사 분류 전문가입니다. 각 도메인의 언론사 정보를 분류하세요.

도메인 목록:
{domain_list}

JSON 배열만 반환하세요:
[
  {{
    "domain": "chosun.com",
    "media_name": "조선일보",
    "media_group": "조선미디어그룹",
    "media_type": "종합지"
  }},
  ...
]

media_type 분류 기준:
- 종합지: 조선일보, 중앙일보, 동아일보 등 일반 종합 일간지
- 경제지: 한국경제, 매일경제, 서울경제 등 경제 전문지
- IT전문지: 블로터, 전자신문, 디지털타임스 등
- 방송사: KBS, MBC, SBS, JTBC 등
- 통신사: 연합뉴스, 뉴시스, 뉴스1 등
- 인터넷신문: 오마이뉴스, 프레시안, 미디어오늘 등 온라인 전용
- 기타: 위 분류에 해당하지 않는 경우

media_group 규칙:
- 알려진 그룹이 있으면 기재 (예: 조선미디어그룹, 중앙일보그룹)
- 독립 언론사는 media_name과 동일하게 기재
- 불명확하면 media_name과 동일하게 기재

JSON 배열만 출력하세요."""

    headers = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": len(domains) * 80
    }

    try:
        print(f"  🤖 OpenAI 분류: {len(domains)}개 신규 도메인", end="", flush=True)
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )

        if response.status_code == 429:  # Rate limit
            if retry:
                print(" (Rate limit, 5초 대기 후 재시도)")
                time.sleep(5)
                return classify_media_outlets_batch(domains, openai_key, retry=False)
            else:
                print(" (Rate limit 초과, 기본값 사용)")
                return _fallback_classification(domains)

        if response.status_code != 200:
            print(f" (API 오류 {response.status_code}, 기본값 사용)")
            return _fallback_classification(domains)

        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

        # JSON 추출
        try:
            classifications = json.loads(content)
        except json.JSONDecodeError:
            if retry:
                print(" (JSON 파싱 실패, 2초 대기 후 재시도)")
                time.sleep(2)
                return classify_media_outlets_batch(domains, openai_key, retry=False)
            else:
                print(" (JSON 파싱 실패, 기본값 사용)")
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
        return _fallback_classification(domains)
    except Exception as e:
        print(f" (오류: {e}, 기본값 사용)")
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


def update_media_directory(spreadsheet, new_entries: Dict[str, Dict]) -> None:
    """
    Google Sheets의 media_directory에 신규 도메인 추가

    Args:
        spreadsheet: gspread Spreadsheet 객체
        new_entries: {domain: {"media_name": ..., "media_group": ..., "media_type": ...}}
    """
    if not new_entries:
        return

    try:
        # 워크시트 가져오기 (없으면 생성)
        try:
            worksheet = spreadsheet.worksheet("media_directory")
        except:
            worksheet = spreadsheet.add_worksheet(title="media_directory", rows=1, cols=4)
            # 헤더 추가
            worksheet.append_row(["domain", "media_name", "media_group", "media_type"])

        # 신규 항목 추가
        for domain, info in new_entries.items():
            row = [
                domain,
                info.get("media_name", ""),
                info.get("media_group", ""),
                info.get("media_type", "")
            ]
            worksheet.append_row(row)

        print(f"✅ media_directory: {len(new_entries)}개 신규 도메인 추가")

    except Exception as e:
        print(f"⚠️  media_directory 업데이트 실패: {e}")
        print("  → Google Sheets 업로드 시 자동으로 추가됩니다.")


def add_media_columns(
    df: pd.DataFrame,
    spreadsheet=None,
    openai_key: str = None
) -> pd.DataFrame:
    """
    DataFrame에 언론사 정보 컬럼 추가

    Args:
        df: 처리된 DataFrame (originallink 컬럼 필요)
        spreadsheet: gspread Spreadsheet 객체 (선택사항)
        openai_key: OpenAI API 키 (선택사항)

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

        # media_directory 로드 (Sheets 연결 가능한 경우)
        existing_media = {}
        new_domains = []

        if spreadsheet:
            existing_media = load_media_directory(spreadsheet)
            new_domains = [d for d in unique_domains if d not in existing_media]
        else:
            new_domains = list(unique_domains)

        # 신규 도메인 분류 (OpenAI)
        if new_domains and openai_key:
            new_media = classify_media_outlets_batch(new_domains, openai_key)
            existing_media.update(new_media)

            # media_directory 업데이트 (Sheets)
            if spreadsheet:
                update_media_directory(spreadsheet, new_media)

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
