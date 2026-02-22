"""
keyword_extractor.py - Category-specific Keyword Extraction
카테고리별 특징 키워드 추출 (Log-odds ratio 기반)
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


def extract_keywords_by_category(
    df: pd.DataFrame,
    category_column: str,
    text_columns: List[str] = ["title", "description"],
    top_k: int = 20,
    pos_tags: List[str] = ["NNG", "NNP", "VV", "VA"],
    min_count: int = 3,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    카테고리별로 특징적인 키워드를 추출 (Log-odds ratio + Laplace smoothing)

    Args:
        df: 분석 결과 DataFrame
        category_column: 카테고리 컬럼명 (예: "sentiment_stage", "danger_level", "issue_category")
        text_columns: 분석할 텍스트 컬럼 리스트 (기본: ["title", "description"])
        top_k: 카테고리별 상위 K개 키워드 (기본: 20)
        pos_tags: 추출할 형태소 태그 (기본: ["NNG", "NNP", "VV", "VA"] - 명사, 동사, 형용사)
        min_count: 최소 출현 횟수 (기본: 3)

    Returns:
        {category_name: [(keyword, log_odds_score), ...]} 형식의 딕셔너리
    """
    # Kiwi 로드 (한 번만 로드)
    try:
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
    except ImportError:
        print("  ❌ kiwipiepy가 설치되지 않았습니다: pip install kiwipiepy")
        return {}
    except Exception as e:
        print(f"  ❌ Kiwi 로드 실패: {e}")
        return {}

    # 카테고리 확인
    if category_column not in df.columns:
        print(f"  ❌ '{category_column}' 컬럼이 없습니다.")
        return {}

    # 빈 값 제거
    df_valid = df[df[category_column].notna() & (df[category_column] != "")].copy()
    if len(df_valid) == 0:
        print(f"  ⚠️  '{category_column}' 컬럼에 유효한 데이터가 없습니다.")
        return {}

    # 카테고리별로 텍스트 추출
    category_tokens = defaultdict(list)  # {category: [token1, token2, ...]}

    for category in df_valid[category_column].unique():
        category_df = df_valid[df_valid[category_column] == category]

        for _, row in category_df.iterrows():
            combined_text = " ".join([
                str(row.get(col, "")) for col in text_columns
            ])

            # Kiwi 형태소 분석
            try:
                result = kiwi.analyze(combined_text)
                if not result or not result[0]:
                    continue

                # 첫 번째 분석 결과만 사용
                tokens = result[0][0]
                for token in tokens:
                    # 형태소 태그 필터링
                    if token.tag in pos_tags and len(token.form) >= 2:
                        category_tokens[category].append(token.form)
            except Exception as e:
                continue

    # Log-odds ratio 계산
    category_keywords = {}

    # 전체 단어 빈도 계산
    all_tokens = []
    for tokens in category_tokens.values():
        all_tokens.extend(tokens)
    total_counter = Counter(all_tokens)

    # 카테고리별 Log-odds 계산
    for category, tokens in category_tokens.items():
        if len(tokens) == 0:
            continue

        category_counter = Counter(tokens)
        log_odds_scores = []

        for word, cat_count in category_counter.items():
            # 최소 출현 횟수 필터링
            if cat_count < min_count:
                continue

            # Log-odds ratio with Laplace smoothing
            # log( (cat_count + α) / (cat_total + α*V) ) - log( (other_count + α) / (other_total + α*V) )
            alpha = 0.01  # Laplace smoothing parameter
            vocab_size = len(total_counter)

            cat_total = len(tokens)
            other_count = total_counter[word] - cat_count
            other_total = len(all_tokens) - cat_total

            # Laplace smoothing 적용
            cat_prob = (cat_count + alpha) / (cat_total + alpha * vocab_size)
            other_prob = (other_count + alpha) / (other_total + alpha * vocab_size)

            # Log-odds ratio
            log_odds = np.log(cat_prob) - np.log(other_prob)
            log_odds_scores.append((word, log_odds))

        # 상위 K개 선택
        log_odds_scores.sort(key=lambda x: x[1], reverse=True)
        category_keywords[category] = log_odds_scores[:top_k]

    return category_keywords


def print_keywords(category_keywords: Dict[str, List[Tuple[str, float]]], max_display: int = 10):
    """
    추출된 키워드를 콘솔에 출력

    Args:
        category_keywords: extract_keywords_by_category() 결과
        max_display: 카테고리별 최대 출력 개수 (기본: 10)
    """
    print("\n" + "=" * 80)
    print("📋 카테고리별 특징 키워드")
    print("=" * 80)

    for category, keywords in category_keywords.items():
        print(f"\n[{category}]")
        for rank, (word, score) in enumerate(keywords[:max_display], 1):
            print(f"  {rank:2d}. {word:15s} (Log-odds: {score:+.4f})")

        if len(keywords) > max_display:
            print(f"  ... (+{len(keywords) - max_display}개 더)")

    print("\n" + "=" * 80)


def extract_all_categories(
    df: pd.DataFrame,
    top_k: int = 20,
    max_display: int = 10,
    spreadsheet=None
):
    """
    모든 주요 카테고리에 대해 키워드를 추출하고 Google Sheets로 저장

    Args:
        df: 분석 결과 DataFrame
        top_k: 카테고리별 상위 K개 키워드
        max_display: 콘솔 출력 시 카테고리별 최대 표시 개수
        spreadsheet: Google Sheets spreadsheet 객체 (선택사항)
    """
    # 추출할 카테고리 목록
    categories = [
        ("sentiment_stage", "감정 단계"),
        ("danger_level", "위험도"),
        ("issue_category", "이슈 카테고리"),
        ("news_category", "뉴스 카테고리"),
        ("brand_relevance", "브랜드 관련성")
    ]

    all_keywords_data = []  # Google Sheets 저장용

    for col_name, display_name in categories:
        if col_name not in df.columns:
            continue

        keywords = extract_keywords_by_category(
            df=df,
            category_column=col_name,
            top_k=top_k,
        )

        if keywords:
            # Google Sheets 저장용 데이터 수집
            for category, keyword_list in keywords.items():
                for rank, (word, score) in enumerate(keyword_list, 1):
                    all_keywords_data.append({
                        "category_type": display_name,
                        "category": category,
                        "rank": rank,
                        "keyword": word,
                        "log_odds_score": round(score, 4)
                    })

    # Google Sheets에 저장 (선택사항)
    if spreadsheet and len(all_keywords_data) > 0:
        try:
            df_keywords = pd.DataFrame(all_keywords_data)
            sheet_name = "keywords"

            # 기존 시트 삭제 후 새로 생성 (덮어쓰기)
            try:
                worksheet = spreadsheet.worksheet(sheet_name)
                spreadsheet.del_worksheet(worksheet)
            except:
                pass  # 시트가 없으면 스킵

            # 새 시트 생성 및 데이터 업로드
            worksheet = spreadsheet.add_worksheet(title=sheet_name, rows=len(df_keywords)+1, cols=len(df_keywords.columns))
            worksheet.update([df_keywords.columns.values.tolist()] + df_keywords.values.tolist())

            print(f"  키워드 추출 완료: {len(all_keywords_data)}개 → Sheets 'keywords' 탭")
        except Exception as e:
            print(f"  ⚠️  키워드 Sheets 업로드 실패: {e}")
