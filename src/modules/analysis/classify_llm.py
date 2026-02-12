"""
classify_llm.py - LLM-Only Classification Orchestrator
병렬 처리 지원, 실시간 CSV 저장
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import pandas as pd
from tqdm import tqdm

from .llm_engine import load_prompts, analyze_article_llm
from src.modules.export.sheets import clean_bom


# CSV 쓰기 Lock (멀티스레드 환경에서 파일 쓰기 경합 방지)
csv_write_lock = Lock()


def _process_single_article(
    idx: int,
    article: Dict,
    prompts_config: dict,
    openai_key: str
) -> Dict:
    """
    단일 기사 LLM 분석 (병렬 처리용 헬퍼 함수)

    Args:
        idx: DataFrame 인덱스
        article: {"title": ..., "description": ..., "query": ...}
        prompts_config: prompts.yaml 설정
        openai_key: OpenAI API 키

    Returns:
        {"idx": idx, "success": True/False, "result": {...}, "error": ..., "error_type": ...}
    """
    try:
        llm_result = analyze_article_llm(article, prompts_config, openai_key)
        return {
            "idx": idx,
            "success": True,
            "result": llm_result,
            "error": None,
            "error_type": None
        }
    except Exception as e:
        import traceback
        error_type = type(e).__name__
        error_msg = str(e)
        error_trace = traceback.format_exc()
        return {
            "idx": idx,
            "success": False,
            "result": None,
            "error": error_msg,
            "error_type": error_type,
            "error_trace": error_trace
        }


def classify_llm(
    df: pd.DataFrame,
    openai_key: str,
    chunk_size: int = 50,
    dry_run: bool = False,
    max_competitor_classify: int = 50,
    max_workers: int = 10,
    result_csv_path: Optional[str] = None,
    spreadsheet = None,
    raw_df: pd.DataFrame = None
) -> tuple[pd.DataFrame, Dict]:
    """
    LLM-only 분류 메인 함수 (병렬 처리 최적화)

    프로세스:
    1. 분류 대상 선택 (우리 브랜드 전체 + 경쟁사 상위 N개 또는 전체)
    2. LLM 분석 (ThreadPoolExecutor로 병렬 처리)
    3. 성공한 기사는 즉시 result.csv에 append
    4. 각 청크 완료 시 Google Sheets 동기화

    Args:
        df: 입력 DataFrame (title, description, query 필수)
        openai_key: OpenAI API 키
        chunk_size: 배치 크기 (청크당 기사 수)
        dry_run: True면 LLM 호출 생략
        max_competitor_classify: 경쟁사별 최대 분류 개수 (0이면 무제한)
        max_workers: 병렬 처리 워커 수 (기본값: 10)
        result_csv_path: 결과 저장 CSV 경로 (None이면 저장하지 않음)
        spreadsheet: Google Sheets 객체 (청크별 동기화용, 선택사항)
        raw_df: 원본 DataFrame (Sheets 동기화용, 선택사항)

    Returns:
        (분석 결과가 추가된 DataFrame, 메트릭 딕셔너리)
    """
    df = df.copy()

    # 메트릭 초기화
    metrics = {
        "articles_classified_llm": 0,
        "llm_api_calls": 0,
        "classification_errors": 0,
        "press_releases_skipped": 0,
    }

    # 초기화: 모든 결과 컬럼
    result_columns = [
        "brand_relevance",
        "brand_relevance_query_keywords",
        "sentiment_stage",
        "danger_level",
        "issue_category",
        "news_category",
        "news_keyword_summary"
    ]

    # 컬럼 초기화 (이미 존재하는 값은 보존)
    for col in result_columns:
        if col not in df.columns:
            df[col] = None

    if "classified_at" not in df.columns:
        df["classified_at"] = ""

    # Load prompts config
    print("\n🔧 LLM 분류 시작...")
    prompts_config = load_prompts()

    if dry_run:
        print("🔬 DRY RUN 모드: LLM 분석 생략")
        return df, metrics

    # ========================================
    # STEP 1: 분류 대상 선택
    # ========================================
    print("\n[1/2] 분류 대상 선택 중...")

    def already_classified(row) -> bool:
        value = clean_bom(row.get("classified_at", ""))
        return isinstance(value, str) and len(value.strip()) > 1

    def has_article_id(row) -> bool:
        value = clean_bom(row.get("article_id", ""))
        return isinstance(value, str) and len(value.strip()) >= 6

    indices_to_classify = []
    for idx, row in df.iterrows():
        if already_classified(row):
            continue
        if not has_article_id(row):
            continue
        indices_to_classify.append(idx)

    # 스킵된 기사 통계 (보도자료 전처리 등)
    skipped_count = sum(1 for _, row in df.iterrows() if already_classified(row))

    metrics["press_releases_skipped"] = skipped_count

    if len(indices_to_classify) == 0:
        if skipped_count > 0:
            print(f"ℹ️  전체 {len(df)}개 기사 중 {skipped_count}개는 이미 분류됨 (보도자료 등)")
            print("⚠️  LLM 분류 대상 없음")
        else:
            print("⚠️  분류 대상 없음")
        return df, metrics

    print(f"  선택된 기사: {len(indices_to_classify)}개")
    if skipped_count > 0:
        print(f"  - 스킵: {skipped_count}개 (이미 분류됨)")

    # ========================================
    # STEP 2: LLM 분석 (병렬 처리)
    # ========================================
    print(f"\n[2/2] LLM 분석 중 (청크 크기: {chunk_size}, 워커: {max_workers})...")

    timestamp = datetime.now().isoformat()
    total = len(indices_to_classify)
    total_chunks = (total + chunk_size - 1) // chunk_size

    # 전체 진행률 표시용 tqdm
    pbar = tqdm(total=total, desc="LLM 분석", unit="기사")

    for chunk_start in range(0, total, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total)
        chunk_indices = indices_to_classify[chunk_start:chunk_end]
        chunk_num = (chunk_start // chunk_size) + 1

        # 청크 내 기사 정보 준비
        articles_to_process = []
        for idx in chunk_indices:
            article = {
                "query": df.at[idx, "query"],
                "group": df.at[idx, "group"] if "group" in df.columns else "",
                "title": df.at[idx, "title"],
                "description": df.at[idx, "description"]
            }

            articles_to_process.append({
                "idx": idx,
                "article": article
            })

        # 병렬 처리 (ThreadPoolExecutor)
        success_count = 0
        fail_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(
                    _process_single_article,
                    item["idx"],
                    item["article"],
                    prompts_config,
                    openai_key
                ): item["idx"]
                for item in articles_to_process
            }

            # Process completed tasks
            for future in as_completed(future_to_idx):
                result = future.result()
                idx = result["idx"]

                if result["success"]:
                    # LLM 결과를 DataFrame에 병합
                    llm_result = result["result"]
                    for col in result_columns:
                        if col in llm_result:
                            value = llm_result[col]
                            # JSON serializable 타입으로 변환
                            if isinstance(value, (dict, list)):
                                value = json.dumps(value, ensure_ascii=False)
                            # 모든 BOM 및 invisible 문자 제거
                            if isinstance(value, str):
                                value = clean_bom(value)
                            df.at[idx, col] = value

                    df.at[idx, "classified_at"] = timestamp
                    success_count += 1

                    # 성공한 기사 즉시 CSV에 저장 (Lock 사용)
                    if result_csv_path:
                        try:
                            with csv_write_lock:
                                row_df = df.loc[[idx]].copy()
                                # 파일이 없으면 header 포함, 있으면 append만
                                file_exists = os.path.exists(result_csv_path)
                                row_df.to_csv(
                                    result_csv_path,
                                    mode='a' if file_exists else 'w',
                                    header=not file_exists,
                                    index=False,
                                    encoding='utf-8-sig' if not file_exists else 'utf-8'
                                )
                        except Exception as csv_err:
                            print(f"⚠️  CSV 저장 실패 [idx={idx}]: {csv_err}")
                else:
                    # Fallback: 빈 값 사용
                    for col in result_columns:
                        df.at[idx, col] = ""
                    df.at[idx, "classified_at"] = timestamp
                    fail_count += 1

                    # 실패 에러 메시지 출력
                    title = df.at[idx, 'title'] if pd.notna(df.at[idx, 'title']) else ""
                    print(f"\n❌ 분류 실패 [idx={idx}]:")
                    print(f"   제목: {title[:80]}...")
                    print(f"   에러 타입: {result.get('error_type', 'Unknown')}")
                    print(f"   에러 메시지: {result['error']}")
                    # Traceback은 너무 길어서 첫 실패만 출력
                    if fail_count == 1 and 'error_trace' in result:
                        print(f"   상세 Traceback (첫 실패만):\n{result['error_trace']}")

                # 진행률 업데이트
                pbar.update(1)

        pbar.set_postfix({"청크": f"{chunk_num}/{total_chunks}", "성공": success_count, "실패": fail_count})

        # 청크 완료 통계 출력
        chunk_total = success_count + fail_count
        success_rate = (success_count / chunk_total * 100) if chunk_total > 0 else 0
        print(f"\n  청크 {chunk_num}/{total_chunks} 완료: 성공 {success_count}/{chunk_total} ({success_rate:.1f}%)")

        # Google Sheets 즉시 동기화 (청크 완료 시마다)
        if spreadsheet and result_csv_path and raw_df is not None:
            try:
                # sync_raw_and_processed를 동적으로 임포트
                from src.modules.export.sheets import sync_raw_and_processed

                # result.csv 전체를 다시 읽어서 동기화 (중복 체크 자동, upsert 지원)
                if os.path.exists(result_csv_path):
                    df_result_current = pd.read_csv(result_csv_path, encoding='utf-8-sig')
                    sync_results = sync_raw_and_processed(raw_df, df_result_current, spreadsheet)
                    added_count = sum(r.get('added', 0) for r in sync_results.values())
                    updated_count = sum(r.get('updated', 0) for r in sync_results.values())
                    if added_count > 0 or updated_count > 0:
                        msg_parts = []
                        if added_count > 0:
                            msg_parts.append(f"{added_count}개 추가")
                        if updated_count > 0:
                            msg_parts.append(f"{updated_count}개 업데이트")
                        print(f"    ☁️  Sheets 동기화: {', '.join(msg_parts)}")
            except Exception as e:
                print(f"    ⚠️  Sheets 동기화 실패: {e}")

        # 청크 간 짧은 대기 (rate limiting)
        if chunk_num < total_chunks:
            time.sleep(0.5)

    pbar.close()

    # 전체 통계 출력
    total_processed = sum(1 for idx in indices_to_classify if pd.notna(df.at[idx, "classified_at"]))
    total_success = sum(1 for idx in indices_to_classify
                       if pd.notna(df.at[idx, "classified_at"])
                       and df.at[idx, "sentiment_stage"] != "")
    total_failed = total_processed - total_success
    overall_success_rate = (total_success / total_processed * 100) if total_processed > 0 else 0

    print(f"\n✅ LLM 분류 완료:")
    print(f"   총 처리: {total_processed}개 기사")
    print(f"   성공: {total_success}개 ({overall_success_rate:.1f}%)")
    print(f"   실패: {total_failed}개 ({100-overall_success_rate:.1f}%)")

    # 메트릭 업데이트
    metrics["articles_classified_llm"] = total_success
    metrics["llm_api_calls"] = total_success  # 1 article = 1 API call
    metrics["classification_errors"] = total_failed

    # 비용 추정 (gpt-5-nano: $0.00015/1K input tokens, $0.0006/1K output tokens)
    # 평균: ~1500 input tokens/article, ~300 output tokens/article
    avg_input_tokens_per_article = 1500
    avg_output_tokens_per_article = 300
    input_cost = (total_success * avg_input_tokens_per_article / 1000) * 0.00015
    output_cost = (total_success * avg_output_tokens_per_article / 1000) * 0.0006
    metrics["llm_cost_estimated"] = round(input_cost + output_cost, 4)

    return df, metrics


def get_classification_stats(df: pd.DataFrame) -> Dict:
    """
    분류 통계 생성

    Args:
        df: 분석 결과 DataFrame

    Returns:
        통계 딕셔너리
    """
    stats = {}

    # Sentiment distribution
    if "sentiment_stage" in df.columns:
        sentiment_counts = df["sentiment_stage"].value_counts().to_dict()
        stats["sentiment_stage"] = sentiment_counts

    # Danger distribution
    if "danger_level" in df.columns:
        danger_counts = df["danger_level"].value_counts().to_dict()
        stats["danger_level"] = danger_counts

    # Brand relevance distribution
    if "brand_relevance" in df.columns:
        relevance_counts = df["brand_relevance"].value_counts().to_dict()
        stats["brand_relevance"] = relevance_counts

    # Issue category distribution
    if "issue_category" in df.columns:
        category_counts = df["issue_category"].value_counts().to_dict()
        stats["issue_category"] = category_counts

    # News category distribution
    if "news_category" in df.columns:
        news_cat_counts = df["news_category"].value_counts().to_dict()
        stats["news_category"] = news_cat_counts

    return stats


def print_classification_stats(stats: Dict):
    """
    분류 통계 출력

    Args:
        stats: get_classification_stats() 결과
    """
    print("\n📊 분류 통계:")

    if "sentiment_stage" in stats:
        print("\n  감정 단계:")
        for sentiment, count in stats["sentiment_stage"].items():
            print(f"    - {sentiment}: {count}개")

    if "danger_level" in stats:
        print("\n  위험도:")
        for danger, count in stats["danger_level"].items():
            print(f"    - {danger}: {count}개")

    if "brand_relevance" in stats:
        print("\n  브랜드 관련성:")
        for relevance, count in stats["brand_relevance"].items():
            print(f"    - {relevance}: {count}개")

    if "issue_category" in stats:
        print("\n  이슈 카테고리 (상위 5개):")
        for category, count in sorted(stats["issue_category"].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"    - {category}: {count}개")

    if "news_category" in stats:
        print("\n  뉴스 카테고리 (상위 5개):")
        for category, count in sorted(stats["news_category"].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"    - {category}: {count}개")
