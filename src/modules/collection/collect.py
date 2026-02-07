"""
collect.py - Naver News Collection Module
네이버 뉴스 API를 통해 브랜드 관련 기사 수집
"""

import time
import requests
import pandas as pd
from typing import List, Dict


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


def collect_all_news(brands: List[str], competitors: List[str], 
                     display: int, start: int, sort: str,
                     naver_id: str, naver_secret: str) -> pd.DataFrame:
    """
    모든 브랜드와 경쟁사 뉴스 수집
    
    Returns:
        DataFrame with columns: query, group, title, description, pubDate, originallink, link
    """
    all_rows = []
    
    # 우리 브랜드 수집
    print(f"📰 우리 브랜드 뉴스 수집 중...")
    for query in brands:
        print(f"  → {query}")
        items = fetch_naver(query, display, start, sort, naver_id, naver_secret)
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
    print(f"\n📰 경쟁사 뉴스 수집 중...")
    for query in competitors:
        print(f"  → {query}")
        items = fetch_naver(query, display, start, sort, naver_id, naver_secret)
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
