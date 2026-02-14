"""
text_cleaning.py - Invisible Character Removal Utilities
BOM 및 불가시 문자 제거 공용 유틸리티
"""

import re
import pandas as pd

# 정규식으로 모든 invisible/제어 문자 일괄 제거
# BOM, Zero Width, 제어 문자, 방향 마크, non-breaking space 등
_INVISIBLE_RE = re.compile(
    r'[\ufeff\ufffe'                    # BOM
    r'\u200b-\u200f'                    # Zero Width + 방향 마크
    r'\u2028-\u202f'                    # 줄/단락 구분자 + 방향 포맷
    r'\u2060'                           # Word Joiner
    r'\u180e'                           # Mongolian Vowel Separator
    r'\u00a0'                           # Non-Breaking Space
    r'\u3000'                           # Ideographic Space (전각 공백)
    r'\u00ad'                           # Soft Hyphen
    r'\ufff9-\ufffc'                    # Interlinear Annotation
    r'\x00-\x08\x0b\x0c\x0e-\x1f'     # C0 제어 문자 (탭/개행 제외)
    r'\x7f-\x9f'                        # DEL + C1 제어 문자
    r']'
)


def strip_invisible(value) -> str:
    """
    BOM/불가시 문자를 제거하고 strip한 문자열을 반환.
    NaN/None이면 빈 문자열 반환.
    """
    if pd.isna(value) or value is None:
        return ""
    return _INVISIBLE_RE.sub('', str(value)).strip()


# sheets.py의 clean_bom()을 대체하는 호환 별칭
clean_bom = strip_invisible


def is_empty_or_invisible(value) -> bool:
    """빈 값이거나 불가시 문자만 포함된 값인지 체크"""
    return strip_invisible(value) == ""


def clean_dataframe_invisible(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame의 모든 문자열 컬럼에서 불가시 문자 제거"""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(
                lambda x: strip_invisible(x) if pd.notna(x) and x != '' else x
            )
    return df