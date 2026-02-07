"""
scrape.py - Naver News Scraping Module
네이버 뉴스를 BeautifulSoup로 스크래핑하여 날짜 범위 기반 기사 수집
"""

import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
from html import unescape
import re


def scrape_naver_news_by_date(query: str, start_date: str, end_date: str,
                              max_pages: int = 10) -> List[Dict]:
    """
    네이버 뉴스를 날짜 범위로 스크래핑

    Args:
        query: 검색어 (브랜드명)
        start_date: 시작 날짜 (YYYY-MM-DD 형식)
        end_date: 종료 날짜 (YYYY-MM-DD 형식)
        max_pages: 최대 페이지 수 (기본값: 10)

    Returns:
        기사 목록 [{"title": ..., "description": ..., "link": ..., "pubDate": ...}, ...]
    """
    all_articles = []

    # 날짜를 YYYYMMDD 형식으로 변환
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        print(f"❌ 날짜 형식 오류: {e}")
        return []

    # 검색 URL
    base_url = "https://search.naver.com/search.naver"

    for page in range(1, max_pages + 1):
        try:
            # 네이버 검색: ds(시작일), de(종료일) 파라미터 사용
            params = {
                "where": "news",
                "query": query,
                "ds": start_date.replace("-", "."),
                "de": end_date.replace("-", "."),
                "start": (page - 1) * 10 + 1,
                "news_by_date": "1"
            }

            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            }

            response = requests.get(base_url, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "lxml")

            # 네이버 뉴스 검색 결과 파싱
            articles = soup.find_all("li", class_="bx")

            if not articles:
                print(f"  페이지 {page}: 결과 없음")
                break

            for article in articles:
                try:
                    # 제목
                    title_elem = article.find("a", class_="news_tit")
                    if not title_elem:
                        continue
                    title = title_elem.get_text(strip=True)

                    # 링크
                    link = title_elem.get("href", "")

                    # 설명
                    desc_elem = article.find("div", class_="dsc")
                    description = ""
                    if desc_elem:
                        description = desc_elem.get_text(strip=True)

                    # 출판 날짜
                    date_elem = article.find("span", class_="sub_time")
                    pubdate_str = ""
                    if date_elem:
                        pubdate_str = date_elem.get_text(strip=True)

                    # 출처 (원본 링크는 네이버 크롤링으로는 어려움)
                    source_elem = article.find("a", class_="press")
                    source = ""
                    if source_elem:
                        source = source_elem.get_text(strip=True)

                    if title and link:
                        all_articles.append({
                            "title": unescape(title),
                            "description": unescape(description),
                            "link": link,
                            "originallink": "",  # 스크래핑으로는 원본 링크 미포함
                            "pubDate": pubdate_str,
                            "source": source
                        })

                except Exception as e:
                    print(f"    기사 파싱 오류: {e}")
                    continue

            print(f"  페이지 {page}: {len(articles)}개 기사")
            time.sleep(0.5)  # Rate limiting

        except requests.exceptions.RequestException as e:
            print(f"  페이지 {page}: 요청 오류 - {e}")
            break
        except Exception as e:
            print(f"  페이지 {page}: 오류 - {e}")
            break

    return all_articles


def parse_naver_date(date_str: str) -> Optional[str]:
    """
    네이버 검색 결과의 상대 시간 문자열을 절대 날짜로 변환
    예: "2시간 전" → ISO 형식

    Note: 스크래핑 시점의 시간 정보가 필요하므로 현재 시간 기준으로 계산
    """
    if not date_str or not date_str.strip():
        return None

    try:
        now = datetime.now()

        # 상대 시간 파싱
        if "초 전" in date_str:
            secs = int(re.search(r'(\d+)', date_str).group(1))
            dt = now - timedelta(seconds=secs)
        elif "분 전" in date_str:
            mins = int(re.search(r'(\d+)', date_str).group(1))
            dt = now - timedelta(minutes=mins)
        elif "시간 전" in date_str:
            hours = int(re.search(r'(\d+)', date_str).group(1))
            dt = now - timedelta(hours=hours)
        elif "일 전" in date_str:
            days = int(re.search(r'(\d+)', date_str).group(1))
            dt = now - timedelta(days=days)
        else:
            # 절대 날짜 형식 시도 (예: "2026.02.07")
            dt = datetime.strptime(date_str.replace(".", "-"), "%Y-%m-%d")

        return dt.isoformat()

    except Exception as e:
        print(f"⚠️  날짜 파싱 실패 '{date_str}': {e}")
        return None


def merge_api_and_scrape(df_api: pd.DataFrame, df_scrape: pd.DataFrame) -> pd.DataFrame:
    """
    API 수집 데이터와 스크래핑 데이터 병합

    Args:
        df_api: API로 수집한 DataFrame
        df_scrape: 스크래핑으로 수집한 DataFrame

    Returns:
        병합된 DataFrame (API 데이터 우선)
    """
    print("🔀 API와 스크래핑 데이터 병합 중...")

    if len(df_api) == 0:
        print("  ⚠️  API 데이터 없음, 스크래핑 데이터만 사용")
        df_scrape["data_source"] = "scrape"
        df_scrape["scraped_at"] = datetime.now().isoformat()
        return df_scrape

    if len(df_scrape) == 0:
        print("  ⚠️  스크래핑 데이터 없음, API 데이터만 사용")
        df_api["data_source"] = "api"
        return df_api

    # 필수 컬럼 확인
    required_cols = ["title", "description", "link", "originallink", "pubDate"]

    # 데이터 소스 마킹
    df_api = df_api.copy()
    df_scrape = df_scrape.copy()

    df_api["data_source"] = "api"
    df_scrape["data_source"] = "scrape"
    df_scrape["scraped_at"] = datetime.now().isoformat()

    # originallink가 비어있을 수 있으므로 link로 정규화
    df_api["pk"] = df_api["originallink"].where(
        df_api["originallink"].notna() & (df_api["originallink"] != ""),
        df_api["link"]
    )
    df_scrape["pk"] = df_scrape["link"]

    # 스크래핑 데이터 중 API와 겹치는 것 제거
    api_links = set(df_api["pk"].dropna())
    df_scrape_new = df_scrape[~df_scrape["pk"].isin(api_links)]

    # 병합
    df_merged = pd.concat([df_api, df_scrape_new], ignore_index=True)
    df_merged = df_merged.drop(columns=["pk"])

    print(f"  ✅ 병합 완료: API {len(df_api)}개 + 스크래핑 {len(df_scrape_new)}개 = 총 {len(df_merged)}개")

    return df_merged


def collect_with_scraping(brands: List[str], competitors: List[str],
                         start_date: str, end_date: str,
                         max_pages: int = 10) -> pd.DataFrame:
    """
    API와 스크래핑을 결합한 종합 수집

    Args:
        brands: 우리 브랜드 목록
        competitors: 경쟁사 목록
        start_date: 시작 날짜 (YYYY-MM-DD)
        end_date: 종료 날짜 (YYYY-MM-DD)
        max_pages: 페이지당 최대 페이지 수

    Returns:
        병합된 DataFrame
    """
    all_rows = []

    # 우리 브랜드
    print(f"📰 우리 브랜드 뉴스 스크래핑 중...")
    for query in brands:
        print(f"  → {query} ({start_date} ~ {end_date})")
        articles = scrape_naver_news_by_date(query, start_date, end_date, max_pages)
        for article in articles:
            article["query"] = query
            article["group"] = "OUR"
        all_rows.extend(articles)

    # 경쟁사
    print(f"\n📰 경쟁사 뉴스 스크래핑 중...")
    for query in competitors:
        print(f"  → {query} ({start_date} ~ {end_date})")
        articles = scrape_naver_news_by_date(query, start_date, end_date, max_pages)
        for article in articles:
            article["query"] = query
            article["group"] = "COMPETITOR"
        all_rows.extend(articles)

    df = pd.DataFrame(all_rows)
    df["data_source"] = "scrape"
    df["scraped_at"] = datetime.now().isoformat()

    print(f"\n✅ 총 {len(df)}개 기사 스크래핑 완료")
    return df
