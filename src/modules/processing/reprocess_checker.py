"""
reprocess_checker.py - 재처리 대상 검사 모듈
Sheets/CSV의 total_result를 검사하여 누락/미완료 기사를 찾아내고
재처리 대상으로 반환한다.
"""

import pandas as pd
from typing import Dict, Any, Optional, Set

from src.utils.text_cleaning import is_empty_or_invisible

# ──────────────────────────────────────────────
# 재처리 규칙 설정
# field: 검사할 컬럼명
# label: 콘솔 출력용 한글 레이블
# ──────────────────────────────────────────────
REPROCESS_RULES = [
    {"field": "brand_relevance", "label": "브랜드 관련성"},
    {"field": "sentiment_stage", "label": "감성 분석"},
    {"field": "source",          "label": "기사 유형"},
    {"field": "media_domain",    "label": "언론사 도메인"},
    {"field": "date_only",       "label": "날짜"},
]


def check_reprocess_targets(
    df_raw: pd.DataFrame,
    spreadsheet=None,
) -> Dict[str, Any]:
    """
    total_result(Sheets 우선, fallback CSV)를 검사하여 재처리 대상을 반환.

    Args:
        df_raw: 전체 raw 데이터 DataFrame
        spreadsheet: gspread Spreadsheet 객체 (필수)

    Returns:
        {
            "df_to_reprocess": pd.DataFrame,
            "reprocess_links": set,
            "stats": { ... }
        }
    """
    stats = {
        "total_raw": len(df_raw),
        "total_result": 0,
        "missing_from_result": 0,
        "field_missing": {rule["field"]: 0 for rule in REPROCESS_RULES},
        "total_reprocess_targets": 0,
    }

    raw_links = set(df_raw["link"].dropna().tolist()) if "link" in df_raw.columns else set()

    # total_result 로드 (Sheets에서만)
    df_result = _load_total_result(spreadsheet)

    if df_result is None or len(df_result) == 0:
        # total_result 없음 → raw 전체가 재처리 대상
        stats["missing_from_result"] = len(raw_links)
        stats["total_reprocess_targets"] = len(raw_links)
        return {
            "df_to_reprocess": df_raw.copy(),
            "reprocess_links": raw_links,
            "stats": stats,
        }

    stats["total_result"] = len(df_result)
    result_links = set(df_result["link"].dropna().tolist()) if "link" in df_result.columns else set()

    # Rule 1: raw에 있고 total_result에 없는 link
    missing_links = raw_links - result_links
    stats["missing_from_result"] = len(missing_links)

    # Rules 2+: total_result에 있는 기사 중 필드별 누락
    field_missing_links: Set[str] = set()
    for rule in REPROCESS_RULES:
        field = rule["field"]
        if field not in df_result.columns:
            # 컬럼 자체가 없으면 전부 누락
            count = len(df_result)
            links = set(df_result["link"].dropna().tolist())
        else:
            mask = df_result[field].apply(is_empty_or_invisible)
            count = int(mask.sum())
            links = set(df_result.loc[mask, "link"].dropna().tolist())

        stats["field_missing"][field] = count
        field_missing_links |= links

    # 합집합
    all_reprocess_links = missing_links | field_missing_links
    stats["total_reprocess_targets"] = len(all_reprocess_links)

    # df_raw에서 재처리 대상만 필터링
    if all_reprocess_links:
        df_to_reprocess = df_raw[df_raw["link"].isin(all_reprocess_links)].copy()
    else:
        df_to_reprocess = pd.DataFrame(columns=df_raw.columns)

    return {
        "df_to_reprocess": df_to_reprocess,
        "reprocess_links": all_reprocess_links,
        "stats": stats,
    }


def load_raw_data_from_sheets(spreadsheet) -> pd.DataFrame:
    """
    Sheets raw_data 탭을 DataFrame으로 로드 (--recheck_only 용).

    Args:
        spreadsheet: gspread Spreadsheet 객체

    Returns:
        pd.DataFrame (빈 경우 empty DataFrame)
    """
    try:
        worksheet = spreadsheet.worksheet("raw_data")
        records = worksheet.get_all_records()
        if not records:
            print("  ℹ️  raw_data 시트가 비어있습니다.")
            return pd.DataFrame()
        df = pd.DataFrame(records)
        print(f"📂 Sheets raw_data에서 {len(df)}개 기사 로드")
        return df
    except Exception as e:
        print(f"⚠️  raw_data 시트 로드 실패: {e}")
        return pd.DataFrame()


def clear_classified_at_for_targets(
    df: pd.DataFrame, reprocess_links: set
) -> pd.DataFrame:
    """
    재처리 대상 기사의 classified_at 필드를 "" 으로 초기화하여
    classify_llm의 already_classified()가 재분류하도록 유도.

    Args:
        df: 처리 대상 DataFrame
        reprocess_links: 재처리 대상 link 집합

    Returns:
        classified_at이 초기화된 DataFrame
    """
    df = df.copy()
    if "classified_at" in df.columns:
        mask = df["link"].isin(reprocess_links)
        df.loc[mask, "classified_at"] = ""
    return df


def print_reprocess_stats(stats: Dict[str, Any]) -> None:
    """필드별 누락 통계를 콘솔에 출력."""
    print(f"\n📊 재처리 대상 검사 결과:")
    print(f"  - 전체 raw: {stats['total_raw']}개")
    print(f"  - 전체 total_result: {stats['total_result']}개")
    print(f"  - result 미존재: {stats['missing_from_result']}개")

    field_missing = stats.get("field_missing", {})
    if any(v > 0 for v in field_missing.values()):
        print(f"  - 필드별 누락:")
        for rule in REPROCESS_RULES:
            field = rule["field"]
            count = field_missing.get(field, 0)
            if count > 0:
                print(f"      {rule['label']}({field}): {count}개")

    print(f"  - 총 재처리 대상: {stats['total_reprocess_targets']}개")


# ──────────────────────────────────────────────
# 내부 헬퍼
# ──────────────────────────────────────────────

def _load_total_result(spreadsheet) -> Optional[pd.DataFrame]:
    """Sheets total_result에서 로드."""
    if spreadsheet:
        try:
            worksheet = spreadsheet.worksheet("total_result")
            records = worksheet.get_all_records()
            if records:
                df = pd.DataFrame(records)
                print(f"📂 Sheets total_result에서 {len(df)}개 기사 로드")
                return df
            else:
                print("  ℹ️  total_result 시트가 비어있습니다.")
        except Exception as e:
            print(f"  ⚠️  total_result 시트 로드 실패: {e}")

    return None
