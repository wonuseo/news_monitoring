"""
scrape.py - Naver News Scraping Module with Playwright
Playwright를 사용하여 네이버 뉴스 검색 결과를 스크래핑 (JavaScript 렌더링 지원)
"""

import time
import pandas as pd
from datetime import datetime
from typing import List, Dict
from playwright.sync_api import sync_playwright


def scrape_naver_news_by_date(query: str, start_date: str, end_date: str,
                              max_pages: int = 10) -> List[Dict]:
    """
    Playwright를 사용하여 Naver 뉴스 검색 결과 스크래핑 (JavaScript 렌더링 지원)

    Args:
        query: 검색어 (브랜드명)
        start_date: 시작 날짜 (YYYY-MM-DD 형식)
        end_date: 종료 날짜 (YYYY-MM-DD 형식)
        max_pages: 최대 페이지 수 (기본값: 10)

    Returns:
        기사 목록 [{"title": ..., "description": ..., "link": ..., "pubDate": ...}, ...]
    """
    all_articles = []

    # 날짜 검증
    try:
        datetime.strptime(start_date, "%Y-%m-%d")
        datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        print(f"❌ 날짜 형식 오류: {e}")
        return []

    base_url = "https://search.naver.com/search.naver"

    try:
        with sync_playwright() as p:
            # 각 페이지 요청마다 새 브라우저 인스턴스 생성 (macOS 안정성)
            for page_num in range(1, max_pages + 1):
                browser = None
                try:
                    # 브라우저 실행 (페이지마다 새로 실행)
                    browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
                    page = browser.new_page()
                    page.set_viewport_size({"width": 1920, "height": 1080})

                    # URL 구성
                    start_idx = (page_num - 1) * 10 + 1
                    params = {
                        "where": "news",
                        "query": query,
                        "ds": start_date.replace("-", "."),
                        "de": end_date.replace("-", "."),
                        "start": start_idx,
                        "news_by_date": "1"
                    }

                    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
                    url = f"{base_url}?{query_string}"

                    print(f"  📄 페이지 {page_num} 로드 중...", end="", flush=True)
                    try:
                        page.goto(url, wait_until="networkidle", timeout=30000)
                    except Exception:
                        # networkidle 타임아웃 시 load로 폴백
                        page.goto(url, wait_until="load", timeout=10000)

                    # JavaScript 렌더링 대기
                    time.sleep(2)

                    # 기사 항목 추출
                    articles_html = page.query_selector_all("li.bx div.news_wrap")
                    page_articles = 0

                    for article_element in articles_html:
                        try:
                            # 제목 추출
                            title_elem = article_element.query_selector("a.news_tit")
                            if not title_elem:
                                continue
                            title = title_elem.get_attribute("title") or title_elem.text_content()
                            title = title.strip() if title else ""

                            # 링크 추출
                            link = title_elem.get_attribute("href") or ""

                            # 설명 추출
                            desc_elem = article_element.query_selector("div.dsc")
                            description = desc_elem.text_content().strip() if desc_elem else ""

                            # 날짜 추출
                            date_elem = article_element.query_selector("span.sub_time")
                            pub_date = ""
                            if date_elem:
                                pub_date = date_elem.text_content().strip()

                            # 출처 추출
                            press_elem = article_element.query_selector("a.press")
                            source = press_elem.text_content().strip() if press_elem else ""

                            if title and link:
                                all_articles.append({
                                    "title": title,
                                    "description": description,
                                    "link": link,
                                    "originallink": "",
                                    "pubDate": pub_date,
                                    "source": source
                                })
                                page_articles += 1

                        except Exception:
                            continue

                    print(f" ✓ {page_articles}개 기사")

                    # 더 이상 기사가 없으면 중단
                    if page_articles == 0:
                        print(f"  ⚠️  더 이상 기사가 없습니다.")
                        break

                    # Rate limiting
                    time.sleep(0.5)

                except Exception as e:
                    print(f" ❌ 오류: {e}")
                    if page_num == 1:
                        print(f"  첫 페이지 로드 실패. 스크래핑을 건너뜁니다.")
                        break
                finally:
                    # 각 페이지 요청 후 브라우저 정리
                    if browser:
                        try:
                            browser.close()
                        except Exception:
                            pass

    except Exception as e:
        print(f"❌ Playwright 오류: {e}")
        return []

    return all_articles


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
