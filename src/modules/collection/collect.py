"""
collect.py - Naver News Collection Module
네이버 뉴스 API를 통해 브랜드 관련 기사 수집
"""

import time
import requests
import pandas as pd
from typing import List, Dict
from pathlib import Path


# 브랜드 설정
OUR_BRANDS = ["롯데호텔", "호텔롯데", "L7", "시그니엘"]
COMPETITORS = ["신라호텔", "조선호텔"]

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

    for page in range(1, max_pages + 1):
        start = (page - 1) * display + 1
        print(f"    Page {page}/{max_pages}...", end="", flush=True)

        items = fetch_naver(query, display, start, sort, naver_id, naver_secret)

        if not items:
            print(" (no more articles)")
            break

        # 중복 체크: 기존 링크와 중복되는 article이 발견되면 조기 종료
        duplicate_found = False
        new_items = []
        for item in items:
            item_link = item.get("link", "")
            if item_link in existing_links:
                duplicate_found = True
                break
            new_items.append(item)
            existing_links.add(item_link)

        all_items.extend(new_items)
        print(f" {len(new_items)} articles", end="")

        if duplicate_found:
            print(" (중복 발견, 수집 중단)")
            break

        print()

        # 페이지 간 요청 사이에 딜레이 (Rate limiting)
        if page < max_pages:
            time.sleep(0.2)

    return all_items


def collect_all_news(brands: List[str], competitors: List[str],
                     display: int, max_pages: int, sort: str,
                     naver_id: str, naver_secret: str,
                     raw_csv_path: str = None) -> pd.DataFrame:
    """
    모든 브랜드와 경쟁사 뉴스 수집

    Args:
        raw_csv_path: raw.csv 파일 경로 (중복 체크용, 선택사항)

    Returns:
        DataFrame with columns: query, group, title, description, pubDate, originallink, link
    """
    all_rows = []

    # 기존 링크 로드 (raw.csv에서)
    existing_links = set()
    if raw_csv_path and Path(raw_csv_path).exists():
        try:
            df_existing = pd.read_csv(raw_csv_path, encoding='utf-8-sig')
            if 'link' in df_existing.columns:
                existing_links = set(df_existing['link'].dropna().tolist())
                print(f"📂 기존 raw.csv에서 {len(existing_links)}개 링크 로드 (중복 체크용)\n")
        except Exception as e:
            print(f"⚠️  raw.csv 로드 실패: {e}\n")

    # 우리 브랜드 수집
    print(f"📰 우리 브랜드 뉴스 수집 중 (최대 {max_pages} 페이지)...")
    for query in brands:
        print(f"  → {query}")
        items = fetch_naver_paginated(query, display, max_pages, sort, naver_id, naver_secret, existing_links)
        for item in items:
            all_rows.append({
                "query": query,
                "group": "OUR",
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "pubDate": item.get("pubDate", ""),
                "originallink": item.get("originallink", ""),
                "link": item.get("link", "")
            })
        time.sleep(0.1)  # Rate limit 방지

    # 경쟁사 수집
    print(f"\n📰 경쟁사 뉴스 수집 중 (최대 {max_pages} 페이지)...")
    for query in competitors:
        print(f"  → {query}")
        items = fetch_naver_paginated(query, display, max_pages, sort, naver_id, naver_secret, existing_links)
        for item in items:
            all_rows.append({
                "query": query,
                "group": "COMPETITOR",
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "pubDate": item.get("pubDate", ""),
                "originallink": item.get("originallink", ""),
                "link": item.get("link", "")
            })
        time.sleep(0.1)

    df = pd.DataFrame(all_rows)
    print(f"\n✅ 총 {len(df)}개 기사 수집 완료")
    return df
