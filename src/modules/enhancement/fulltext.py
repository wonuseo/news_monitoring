"""
fulltext.py - Full-Text Article Extraction Module
위험도가 높은 기사에 대해 전문을 스크래핑하는 모듈
"""

import time
import requests
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
from bs4 import BeautifulSoup


def fetch_full_text(url: str, timeout: int = 10) -> Dict[str, str]:
    """
    단일 기사의 전문을 스크래핑

    Args:
        url: 기사 URL (Naver 또는 원본 링크)
        timeout: 요청 타임아웃 (초)

    Returns:
        {
            "full_text": 기사 전문 (또는 빈 문자열),
            "status": "success" | "paywall" | "404" | "timeout" | "error"
        }
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }

        response = requests.get(url, headers=headers, timeout=timeout)

        # 404/403 처리
        if response.status_code == 404:
            return {"full_text": "", "status": "404"}
        elif response.status_code == 403:
            return {"full_text": "", "status": "paywall"}
        elif response.status_code >= 400:
            return {"full_text": "", "status": f"http_{response.status_code}"}

        response.raise_for_status()

        # BeautifulSoup으로 파싱
        soup = BeautifulSoup(response.content, "lxml")

        # Naver 뉴스 기사 본문 추출
        article_body = None

        # 1. 네이버 뉴스 본문 (id="articleBodyContents")
        article_body = soup.find("div", id="articleBodyContents")

        # 2. 또는 일반 article 태그
        if not article_body:
            article_body = soup.find("article")

        # 3. 또는 div class가 content인 경우
        if not article_body:
            article_body = soup.find("div", class_=lambda x: x and "content" in x.lower())

        if not article_body:
            return {"full_text": "", "status": "no_content"}

        # HTML 태그 제거 및 텍스트 추출
        full_text = article_body.get_text(separator="\n", strip=True)

        # 공백 정리
        full_text = "\n".join(line.strip() for line in full_text.split("\n") if line.strip())

        if not full_text or len(full_text) < 50:
            return {"full_text": "", "status": "too_short"}

        return {"full_text": full_text[:5000], "status": "success"}  # 최대 5000자 제한

    except requests.exceptions.Timeout:
        return {"full_text": "", "status": "timeout"}
    except requests.exceptions.RequestException as e:
        return {"full_text": "", "status": "request_error"}
    except Exception as e:
        return {"full_text": "", "status": "error"}


def batch_fetch_full_text(df: pd.DataFrame, risk_levels: List[str] = None,
                         max_articles: Optional[int] = None) -> pd.DataFrame:
    """
    DataFrame의 기사들에 대해 일괄 전문 스크래핑

    Args:
        df: 분류된 DataFrame (sentiment, risk_level 포함)
        risk_levels: 스크래핑할 위험도 목록 (기본: ['상', '중'])
        max_articles: 최대 스크래핑 기사 수 (None = 무제한)

    Returns:
        다음 컬럼이 추가된 DataFrame:
        - full_text: 기사 전문
        - full_text_status: 스크래핑 상태
        - full_text_scraped_at: 스크래핑 시간
    """
    df = df.copy()

    # 기본값
    if risk_levels is None:
        risk_levels = ["상", "중"]

    # 초기화
    df["full_text"] = ""
    df["full_text_status"] = "not_scraped"
    df["full_text_scraped_at"] = ""

    # 필터링: 지정한 위험도만
    if "risk_level" in df.columns:
        mask = df["risk_level"].isin(risk_levels)
        indices_to_scrape = df[mask].index.tolist()
    else:
        # risk_level 컬럼 없으면 모두 스크래핑
        indices_to_scrape = df.index.tolist()

    if max_articles:
        indices_to_scrape = indices_to_scrape[:max_articles]

    print(f"📄 전문 스크래핑 시작: {len(indices_to_scrape)}개 기사")

    for i, idx in enumerate(indices_to_scrape):
        try:
            # 링크 선택: originallink 우선, 없으면 link
            url = df.at[idx, "link"]
            if "originallink" in df.columns and df.at[idx, "originallink"]:
                url = df.at[idx, "originallink"]

            if not url or not isinstance(url, str):
                df.at[idx, "full_text_status"] = "no_url"
                continue

            # 기사 제목 (로깅용)
            title = df.at[idx, "title"] if "title" in df.columns else "Unknown"

            # 전문 스크래핑
            result = fetch_full_text(url)

            df.at[idx, "full_text"] = result["full_text"]
            df.at[idx, "full_text_status"] = result["status"]
            df.at[idx, "full_text_scraped_at"] = datetime.now().isoformat()

            # 진행상황 로깅
            status_icon = "✓" if result["status"] == "success" else "⚠"
            print(f"  [{i+1}/{len(indices_to_scrape)}] {status_icon} {title[:50]}... ({result['status']})")

            # Rate limiting
            time.sleep(1)

        except Exception as e:
            print(f"  [{i+1}/{len(indices_to_scrape)}] ❌ 오류: {str(e)[:50]}")
            df.at[idx, "full_text_status"] = "error"

    # 통계
    success_count = (df["full_text_status"] == "success").sum()
    paywall_count = (df["full_text_status"] == "paywall").sum()
    error_count = (df["full_text_status"].isin(["error", "404", "timeout", "no_content"])).sum()

    print(f"\n✅ 스크래핑 완료:")
    print(f"  - 성공: {success_count}개")
    print(f"  - 페이월: {paywall_count}개")
    print(f"  - 실패: {error_count}개")

    return df
