"""
test_reprocess_checker.py - reprocess_checker 모듈 검증
더미 데이터로 모든 재처리 규칙 + clear_classified_at + 통계 출력 검증
Sheets-only 모드: MockSpreadsheet으로 Sheets 인터페이스 모킹
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules.processing.reprocess_checker import (
    REPROCESS_RULES,
    check_reprocess_targets,
    clear_classified_at_for_targets,
    print_reprocess_stats,
)

PASS = 0
FAIL = 0


def ok(cond, label):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label}")


# ──────────────────────────────────────────────
# Mock Sheets 인터페이스
# ──────────────────────────────────────────────

class MockWorksheet:
    def __init__(self, records):
        self._records = records

    def get_all_records(self):
        return self._records


class MockSpreadsheet:
    def __init__(self, worksheets_data: dict):
        self._data = worksheets_data

    def worksheet(self, name):
        if name not in self._data:
            raise Exception(f"Worksheet '{name}' not found")
        return MockWorksheet(self._data[name])


def make_mock_sheets(df_result):
    """DataFrame을 MockSpreadsheet으로 변환"""
    records = df_result.to_dict('records') if df_result is not None else []
    return MockSpreadsheet({"total_result": records})


# ──────────────────────────────────────────────
# 더미 데이터 생성
# ──────────────────────────────────────────────

def make_raw():
    """raw 10건: link_01 ~ link_10"""
    return pd.DataFrame({
        "link": [f"https://example.com/link_{i:02d}" for i in range(1, 11)],
        "title": [f"기사 제목 {i}" for i in range(1, 11)],
        "description": [f"기사 내용 {i}" for i in range(1, 11)],
        "pubDate": ["2026-02-10"] * 10,
        "query": ["롯데호텔"] * 10,
        "group": ["OUR"] * 10,
    })


def make_result_complete(n=8):
    """total_result 완전 처리 n건 (link_01 ~ link_0n)"""
    return pd.DataFrame({
        "link": [f"https://example.com/link_{i:02d}" for i in range(1, n + 1)],
        "brand_relevance": ["관련"] * n,
        "sentiment_stage": ["중립"] * n,
        "source": ["일반기사"] * n,
        "media_domain": ["example.com"] * n,
        "date_only": ["2026-02-10"] * n,
        "classified_at": ["2026-02-10T12:00:00"] * n,
    })


def make_result_with_missing_fields():
    """
    total_result 8건, 필드 누락 시나리오:
    - link_01: 정상
    - link_02: brand_relevance 빈칸
    - link_03: sentiment_stage BOM만 있음
    - link_04: source 빈칸
    - link_05: media_domain NaN
    - link_06: date_only 빈칸
    - link_07: brand_relevance + sentiment_stage 둘 다 빈칸
    - link_08: 정상
    """
    return pd.DataFrame({
        "link": [f"https://example.com/link_{i:02d}" for i in range(1, 9)],
        "brand_relevance":  ["관련", "",     "관련",    "관련", "관련", "관련", "",     "관련"],
        "sentiment_stage":  ["중립", "중립", "\ufeff",  "중립", "중립", "중립", "",     "부정 후보"],
        "source":           ["일반기사"] * 3 + [""] + ["일반기사"] * 4,
        "media_domain":     ["a.com", "b.com", "c.com", "d.com", None, "f.com", "g.com", "h.com"],
        "date_only":        ["2026-02-10"] * 5 + [""] + ["2026-02-10"] * 2,
        "classified_at":    ["2026-02-10T12:00:00"] * 8,
    })


# ──────────────────────────────────────────────
# 테스트 케이스
# ──────────────────────────────────────────────

def test_1_no_result():
    """total_result가 없을 때 → raw 전체가 재처리 대상"""
    print("\n[Test 1] total_result 없음 → raw 전체가 재처리 대상")
    df_raw = make_raw()
    # Sheets 미연결 (None) → total_result 없음 → raw 전체 재처리
    result = check_reprocess_targets(df_raw, spreadsheet=None)

    ok(len(result["df_to_reprocess"]) == 10, f"재처리 대상 10건 (실제: {len(result['df_to_reprocess'])})")
    ok(result["stats"]["missing_from_result"] == 10, f"missing_from_result=10 (실제: {result['stats']['missing_from_result']})")
    ok(result["stats"]["total_reprocess_targets"] == 10, f"total=10 (실제: {result['stats']['total_reprocess_targets']})")
    print_reprocess_stats(result["stats"])


def test_2_all_complete():
    """result가 raw와 동일하고 모든 필드 정상 → 재처리 0건"""
    print("\n[Test 2] 모두 완료 → 재처리 0건")
    df_raw = make_raw()
    mock_sheets = make_mock_sheets(make_result_complete(n=10))

    result = check_reprocess_targets(df_raw, spreadsheet=mock_sheets)
    ok(len(result["df_to_reprocess"]) == 0, f"재처리 대상 0건 (실제: {len(result['df_to_reprocess'])})")
    ok(result["stats"]["total_reprocess_targets"] == 0, f"total=0 (실제: {result['stats']['total_reprocess_targets']})")
    ok(result["stats"]["missing_from_result"] == 0, f"missing=0 (실제: {result['stats']['missing_from_result']})")
    print_reprocess_stats(result["stats"])


def test_3_missing_from_result():
    """raw 10건, result 8건 → 2건 미존재"""
    print("\n[Test 3] result에 없는 기사 2건 (Rule 1)")
    df_raw = make_raw()
    mock_sheets = make_mock_sheets(make_result_complete(n=8))

    result = check_reprocess_targets(df_raw, spreadsheet=mock_sheets)
    ok(result["stats"]["missing_from_result"] == 2, f"missing=2 (실제: {result['stats']['missing_from_result']})")
    reprocess_links = result["reprocess_links"]
    ok("https://example.com/link_09" in reprocess_links, "link_09 포함")
    ok("https://example.com/link_10" in reprocess_links, "link_10 포함")
    ok(result["stats"]["total_reprocess_targets"] == 2, f"total=2 (실제: {result['stats']['total_reprocess_targets']})")
    print_reprocess_stats(result["stats"])


def test_4_field_level_missing():
    """필드별 누락 감지 (BOM, 빈칸, NaN)"""
    print("\n[Test 4] 필드별 누락 감지 (Rules 2-6)")
    df_raw = make_raw()
    mock_sheets = make_mock_sheets(make_result_with_missing_fields())

    result = check_reprocess_targets(df_raw, spreadsheet=mock_sheets)
    s = result["stats"]
    fm = s["field_missing"]

    # link_02, link_07 → brand_relevance 빈칸
    ok(fm["brand_relevance"] == 2, f"brand_relevance 누락 2건 (실제: {fm['brand_relevance']})")
    # link_03 → BOM만, link_07 → 빈칸
    ok(fm["sentiment_stage"] == 2, f"sentiment_stage 누락 2건 (실제: {fm['sentiment_stage']})")
    # link_04
    ok(fm["source"] == 1, f"source 누락 1건 (실제: {fm['source']})")
    # link_05 → NaN
    ok(fm["media_domain"] == 1, f"media_domain 누락 1건 (실제: {fm['media_domain']})")
    # link_06
    ok(fm["date_only"] == 1, f"date_only 누락 1건 (실제: {fm['date_only']})")

    # raw에 없는 2건(link_09, link_10) + 필드 누락 6건(link_02~07) = 총 8건
    ok(s["missing_from_result"] == 2, f"missing_from_result=2 (실제: {s['missing_from_result']})")
    # link_02,03,04,05,06,07 + link_09,10 = 8
    ok(s["total_reprocess_targets"] == 8, f"total=8 (실제: {s['total_reprocess_targets']})")
    ok(len(result["df_to_reprocess"]) == 8, f"df 8건 (실제: {len(result['df_to_reprocess'])})")

    print_reprocess_stats(s)


def test_5_clear_classified_at():
    """classified_at 초기화 동작 확인"""
    print("\n[Test 5] clear_classified_at_for_targets()")
    df = pd.DataFrame({
        "link": ["https://a.com/1", "https://a.com/2", "https://a.com/3"],
        "classified_at": ["2026-02-10T12:00:00", "2026-02-10T12:00:00", "2026-02-10T12:00:00"],
    })
    target_links = {"https://a.com/1", "https://a.com/3"}

    result = clear_classified_at_for_targets(df, target_links)

    ok(result.loc[0, "classified_at"] == "", f"link_1 classified_at 초기화 (실제: '{result.loc[0, 'classified_at']}')")
    ok(result.loc[1, "classified_at"] == "2026-02-10T12:00:00", f"link_2 유지 (실제: '{result.loc[1, 'classified_at']}')")
    ok(result.loc[2, "classified_at"] == "", f"link_3 classified_at 초기화 (실제: '{result.loc[2, 'classified_at']}')")

    # 원본 변경 없음 확인
    ok(df.loc[0, "classified_at"] == "2026-02-10T12:00:00", "원본 DataFrame 불변")


def test_6_classified_at_not_in_columns():
    """classified_at 컬럼이 없는 DataFrame에도 에러 없이 동작"""
    print("\n[Test 6] classified_at 컬럼 없는 경우")
    df = pd.DataFrame({
        "link": ["https://a.com/1"],
        "title": ["test"],
    })
    result = clear_classified_at_for_targets(df, {"https://a.com/1"})
    ok("classified_at" not in result.columns, "classified_at 컬럼 없어도 에러 없음")


def test_7_combined_new_and_reprocess():
    """신규 수집 + 재처리 대상 병합 시뮬레이션 (main.py STEP 1.5 로직)"""
    print("\n[Test 7] 신규 + 재처리 병합 시뮬레이션")
    df_raw = make_raw()

    # 신규 수집 3건 (link_11, link_12, link_13)
    df_raw_new = pd.DataFrame({
        "link": [f"https://example.com/link_{i}" for i in range(11, 14)],
        "title": [f"신규 기사 {i}" for i in range(11, 14)],
        "pubDate": ["2026-02-12"] * 3,
    })

    mock_sheets = make_mock_sheets(make_result_complete(n=8))
    recheck = check_reprocess_targets(df_raw, spreadsheet=mock_sheets)
    df_reprocess = recheck["df_to_reprocess"]

    # link_09, link_10이 재처리 대상
    ok(len(df_reprocess) == 2, f"재처리 대상 2건 (실제: {len(df_reprocess)})")

    # main.py STEP 1.5 병합 로직 시뮬레이션
    if len(df_reprocess) > 0:
        df_reprocess = clear_classified_at_for_targets(df_reprocess, recheck["reprocess_links"])
        df_to_process = pd.concat([df_raw_new, df_reprocess], ignore_index=True)
        df_to_process = df_to_process.drop_duplicates(subset=["link"], keep="first")
    else:
        df_to_process = df_raw_new

    ok(len(df_to_process) == 5, f"처리 대상 5건 (신규3 + 재처리2) (실제: {len(df_to_process)})")

    # 중복 제거 확인
    unique_links = set(df_to_process["link"].tolist())
    ok(len(unique_links) == 5, f"중복 없음 (실제: {len(unique_links)})")

    print_reprocess_stats(recheck["stats"])


def test_8_empty_raw():
    """raw가 빈 경우"""
    print("\n[Test 8] raw가 빈 경우")
    df_raw = pd.DataFrame(columns=["link", "title"])
    result = check_reprocess_targets(df_raw, spreadsheet=None)
    ok(len(result["df_to_reprocess"]) == 0, f"재처리 0건 (실제: {len(result['df_to_reprocess'])})")
    ok(result["stats"]["total_reprocess_targets"] == 0, f"total=0 (실제: {result['stats']['total_reprocess_targets']})")


# ──────────────────────────────────────────────
# 실행
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("reprocess_checker.py 검증 테스트")
    print("=" * 60)

    test_1_no_result()
    test_2_all_complete()
    test_3_missing_from_result()
    test_4_field_level_missing()
    test_5_clear_classified_at()
    test_6_classified_at_not_in_columns()
    test_7_combined_new_and_reprocess()
    test_8_empty_raw()

    print("\n" + "=" * 60)
    print(f"결과: {PASS} PASS / {FAIL} FAIL (총 {PASS + FAIL}건)")
    print("=" * 60)

    sys.exit(1 if FAIL > 0 else 0)
