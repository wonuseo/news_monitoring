"""
collect.py - Naver News Collection Module
네이버 뉴스 API를 통해 브랜드 관련 기사 수집
"""

import time
import requests
import pandas as pd
from typing import List, Dict
from pathlib import Path


# 브랜드 설정 (config/brands.yaml에서 로드, 없으면 하드코딩 fallback)
def _load_brands():
    try:
        from src.utils.config import load_config
        cfg = load_config("brands")
        return cfg.get("our_brands", []), cfg.get("competitors", [])
    except Exception:
        return (
            ["롯데호텔", "호텔롯데", "L7", "시그니엘"],
            ["신라호텔", "조선호텔"],
        )

OUR_BRANDS, COMPETITORS = _load_brands()

NAVER_API_URL = "https://openapi.naver.com/v1/search/news.json"


def fetch_naver(query: str, display: int, start: int, sort: str,
                naver_id: str, naver_secret: str) -> List[Dict]:
    """
    네이버 뉴스 검색 API 호출

    Args:
        query: 검색어
        display: 가져올 개수
        start: 시작 인덱스
        sort: 정렬 방식 (date/sim)
        naver_id: 네이버 클라이언트 ID
        naver_secret: 네이버 클라이언트 Secret

    Returns:
        기사 목록 (dict 리스트)
    """
    headers = {
        "X-Naver-Client-Id": naver_id,
        "X-Naver-Client-Secret": naver_secret
    }
    params = {
        "query": query,
        "display": display,
        "start": start,
        "sort": sort
    }

    try:
        response = requests.get(NAVER_API_URL, headers=headers, params=params, timeout=10)

        if response.status_code == 401:
            raise RuntimeError("네이버 API 인증 실패 (401). 클라이언트 ID/Secret을 확인하세요.")
        elif response.status_code == 429:
            raise RuntimeError("네이버 API 요청 한도 초과 (429). 잠시 후 다시 시도하세요.")
        elif response.status_code >= 500:
            raise RuntimeError(f"네이버 API 서버 오류 ({response.status_code})")

        response.raise_for_status()
        data = response.json()
        return data.get("items", [])

    except requests.exceptions.RequestException as e:
        print(f"⚠️  '{query}' 검색 중 오류 발생: {e}")
        return []


def fetch_naver_paginated(query: str, display: int, max_pages: int, sort: str,
                          naver_id: str, naver_secret: str, existing_links: set = None) -> List[Dict]:
    """
    페이지네이션을 통해 여러 페이지의 기사 수집

    중복 링크는 skip하고 페이지 끝까지 검사
    페이지네이션 중단: (A) 해당 페이지 새 기사 0개 또는 (B) 연속 N페이지 새 기사 0개

    Args:
        query: 검색어
        display: 한 페이지당 가져올 개수 (기본: 100)
        max_pages: 최대 페이지 수 (권장: 9 for 90% 쿼터 안전 마진)
        sort: 정렬 방식 (date/sim)
        naver_id: 네이버 클라이언트 ID
        naver_secret: 네이버 클라이언트 Secret
        existing_links: 기존 링크 set (중복 체크용)

    Returns:
        모든 페이지의 기사 목록 (dict 리스트)
    """
    all_items = []
    existing_links = existing_links or set()
    consecutive_empty_pages = 0  # 연속 빈 페이지 카운터
    max_consecutive_empty = 3  # 연속 빈 페이지 허용 한계

    for page in range(1, max_pages + 1):
        start = (page - 1) * display + 1
        print(f"    Page {page}/{max_pages}...", end="", flush=True)

        items = fetch_naver(query, display, start, sort, naver_id, naver_secret)

        if not items:
            print(" (no more articles)")
            break

        # 중복 체크: 중복은 skip, 새 기사만 수집
        new_items = []
        dup_count = 0
        for item in items:
            item_link = item.get("link", "")
            if item_link in existing_links:
                dup_count += 1
                continue  # 중복은 skip, 다음 기사 검사
            new_items.append(item)
            existing_links.add(item_link)

        all_items.extend(new_items)
        new_count = len(new_items)

        # 로그 출력
        print(f" {new_count} new, {dup_count} dup", end="")

        # 중단 조건 A: 해당 페이지에서 새 기사 0개
        if new_count == 0:
            consecutive_empty_pages += 1
            print(f" (새 기사 0개, 연속 {consecutive_empty_pages}/{max_consecutive_empty})")

            # 중단 조건 B: 연속 N페이지 새 기사 0개
            if consecutive_empty_pages >= max_consecutive_empty:
                print(f"    ⚠️  연속 {consecutive_empty_pages}페이지 새 기사 없음 → 수집 중단")
                break
        else:
            consecutive_empty_pages = 0  # 새 기사 있으면 카운터 리셋
            print()

        # 페이지 간 요청 사이에 딜레이 (Rate limiting)
        if page < max_pages:
            time.sleep(0.2)

    return all_items


def collect_all_news(brands: List[str], competitors: List[str],
                     display: int, max_pages: int, sort: str,
                     naver_id: str, naver_secret: str,
                     existing_links: set = None,
                     spreadsheet = None) -> pd.DataFrame:
    """
    모든 브랜드와 경쟁사 뉴스 수집 (각 브랜드마다 즉시 Sheets 동기화)

    Args:
        existing_links: 기존 링크 set (중복 체크용, Sheets에서 미리 로드)
        spreadsheet: Google Sheets 객체 (즉시 동기화용, 선택사항)

    Returns:
        DataFrame with columns: query, group, title, description, pubDate, originallink, link
    """
    all_rows = []

    existing_links = existing_links or set()

    # 우리 브랜드 수집
    print(f"📰 우리 브랜드 뉴스 수집 중 (최대 {max_pages} 페이지)...")
    for query in brands:
        print(f"  → {query}")
        items = fetch_naver_paginated(query, display, max_pages, sort, naver_id, naver_secret, existing_links)
        brand_rows = []
        for item in items:
            brand_rows.append({
                "query": query,
                "group": "OUR",
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "pubDate": item.get("pubDate", ""),
                "originallink": item.get("originallink", ""),
                "link": item.get("link", "")
            })
        all_rows.extend(brand_rows)

        # 각 브랜드 수집 완료 시 즉시 Sheets 동기화
        if brand_rows:
            _save_immediately(brand_rows, spreadsheet)

        time.sleep(0.1)  # Rate limit 방지

    # 경쟁사 수집
    print(f"\n📰 경쟁사 뉴스 수집 중 (최대 {max_pages} 페이지)...")
    for query in competitors:
        print(f"  → {query}")
        items = fetch_naver_paginated(query, display, max_pages, sort, naver_id, naver_secret, existing_links)
        competitor_rows = []
        for item in items:
            competitor_rows.append({
                "query": query,
                "group": "COMPETITOR",
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "pubDate": item.get("pubDate", ""),
                "originallink": item.get("originallink", ""),
                "link": item.get("link", "")
            })
        all_rows.extend(competitor_rows)

        # 각 경쟁사 수집 완료 시 즉시 Sheets 동기화
        if competitor_rows:
            _save_immediately(competitor_rows, spreadsheet)

        time.sleep(0.1)

    df = pd.DataFrame(all_rows)
    print(f"\n✅ 총 {len(df)}개 기사 수집 완료")
    return df


def _save_immediately(rows: List[Dict], spreadsheet = None):
    """
    수집한 기사를 즉시 Google Sheets에 동기화

    Args:
        rows: 저장할 기사 리스트
        spreadsheet: Google Sheets 객체
    """
    if not rows:
        return

    df_new = pd.DataFrame(rows)

    # Google Sheets 동기화 (df_new 직접 전달)
    if spreadsheet:
        try:
            from src.modules.export.sheets import sync_to_sheets
            sync_result = sync_to_sheets(df_new, spreadsheet, "raw_data")
            if sync_result['added'] > 0:
                print(f"    ☁️  Sheets 동기화: {sync_result['added']}개 추가")
        except Exception as e:
            print(f"    ⚠️  Sheets 동기화 실패: {e}")
