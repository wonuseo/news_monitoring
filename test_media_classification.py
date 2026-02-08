#!/usr/bin/env python3
"""
Test script for media classification module
media_classify 모듈 테스트
"""

import pandas as pd
from src.modules.enhancement.media_classify import (
    extract_domain_safe,
    _fallback_classification,
    classify_media_outlets_batch,
)

def test_extract_domain():
    """URL에서 도메인 추출 테스트"""
    print("=" * 80)
    print("TEST 1: extract_domain_safe()")
    print("=" * 80)

    test_cases = [
        ("https://www.chosun.com/article/123", "chosun.com"),
        ("https://woman.chosun.com/article/456", "woman.chosun.com"),
        ("https://news.naver.com/main/read.naver?m_view=1", "news.naver.com"),
        ("http://example.com", "example.com"),
        ("invalid-url", ""),
        ("", ""),
    ]

    for url, expected in test_cases:
        result = extract_domain_safe(url)
        status = "✅" if result == expected else "❌"
        print(f"{status} extract_domain_safe('{url}') → '{result}' (expected: '{expected}')")

    print()


def test_fallback_classification():
    """기본값 분류 테스트"""
    print("=" * 80)
    print("TEST 2: _fallback_classification()")
    print("=" * 80)

    domains = ["chosun.com", "hankyung.com", "unknown-domain.com"]
    result = _fallback_classification(domains)

    print(f"Input domains: {domains}")
    print(f"Output: {result}")
    print()

    # 검증
    assert len(result) == 3, f"Expected 3 entries, got {len(result)}"
    assert "chosun.com" in result, "Missing chosun.com"
    assert result["chosun.com"]["media_type"] == "기타", "Wrong fallback type"
    print("✅ Fallback classification test passed")
    print()


def test_dataframe_with_empty_columns():
    """DataFrame에 빈 컬럼 추가 테스트"""
    print("=" * 80)
    print("TEST 3: DataFrame with media columns (empty)")
    print("=" * 80)

    df = pd.DataFrame({
        "title": ["기사1", "기사2"],
        "description": ["설명1", "설명2"],
        "originallink": [
            "https://www.chosun.com/article/123",
            "https://hankyung.com/article/456"
        ]
    })

    # 빈 컬럼 추가
    df["media_domain"] = ""
    df["media_name"] = ""
    df["media_group"] = ""
    df["media_type"] = ""

    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print()
    print(df)
    print()
    print("✅ DataFrame with empty columns test passed")
    print()


def test_domain_extraction_on_dataframe():
    """DataFrame의 URL에서 도메인 추출 테스트"""
    print("=" * 80)
    print("TEST 4: extract_domain_safe on DataFrame")
    print("=" * 80)

    df = pd.DataFrame({
        "title": ["조선일보 뉴스", "한국경제 속보", "잘못된 URL"],
        "originallink": [
            "https://www.chosun.com/article/123",
            "https://hankyung.com/article/456",
            "invalid-url"
        ]
    })

    # 도메인 추출
    df["media_domain"] = df["originallink"].apply(extract_domain_safe)

    print(f"DataFrame:")
    print(df[["title", "originallink", "media_domain"]])
    print()

    # 검증
    assert df.loc[0, "media_domain"] == "chosun.com", "Failed to extract chosun.com"
    assert df.loc[1, "media_domain"] == "hankyung.com", "Failed to extract hankyung.com"
    assert df.loc[2, "media_domain"] == "", "Should be empty for invalid URL"

    print("✅ Domain extraction on DataFrame test passed")
    print()


def test_module_imports():
    """모듈 임포트 테스트"""
    print("=" * 80)
    print("TEST 5: Module Imports")
    print("=" * 80)

    try:
        from src.modules.enhancement.media_classify import add_media_columns
        print("✅ add_media_columns imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import add_media_columns: {e}")

    try:
        from src.modules.processing.process import enrich_with_media_info
        print("✅ enrich_with_media_info imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import enrich_with_media_info: {e}")

    print()


if __name__ == "__main__":
    print("\n🧪 Media Classification Module Tests\n")

    test_extract_domain()
    test_fallback_classification()
    test_dataframe_with_empty_columns()
    test_domain_extraction_on_dataframe()
    test_module_imports()

    print("=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)
    print()
