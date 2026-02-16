#!/usr/bin/env python3
"""
audit_topic_clusters.py - Topic 클러스터 자동 검증

모든 topic group (_t 클러스터)을 LLM으로 재검증하여 잘못 묶인 클러스터를 자동 탐지합니다.

사용법:
  python audit_topic_clusters.py                    # 검증만 (리포트 출력)
  python audit_topic_clusters.py --auto_fix          # 검증 + 자동 해체
  python audit_topic_clusters.py --min_confidence 0.7  # 신뢰도 임계값 조정
"""

import os
import argparse
import pandas as pd
from dotenv import load_dotenv
from typing import Dict, List, Tuple

from src.modules.export.sheets import connect_sheets, sync_to_sheets
from src.modules.analysis.llm_engine import (
    load_source_verifier_prompts,
    render_prompt,
    call_openai_structured,
)
from src.utils.openai_client import load_api_models


def _get_sv_model() -> str:
    """source_verification 모델 로드."""
    api_models = load_api_models()
    return api_models.get("source_verification", "gpt-4o-mini")


def verify_cluster_coherence(
    cluster_df: pd.DataFrame,
    cluster_id: str,
    openai_key: str,
) -> Dict:
    """
    LLM으로 클러스터 일관성 검증 (클러스터 내 모든 기사가 같은 주제인지).

    전략: 대표 기사(첫 번째)와 나머지 기사들을 각각 비교하여
    다른 주제로 판단되는 기사가 하나라도 있으면 클러스터를 해체 권장.

    Args:
        cluster_df: 클러스터 내 기사 DataFrame
        cluster_id: 클러스터 ID
        openai_key: OpenAI API 키

    Returns:
        {
            "cluster_id": str,
            "coherent": bool,  # 일관성 있는 클러스터인가
            "article_count": int,
            "mismatches": List[Dict],  # 다른 주제로 판단된 기사들
            "error": str or None,
        }
    """
    result = {
        "cluster_id": cluster_id,
        "coherent": True,
        "article_count": len(cluster_df),
        "mismatches": [],
        "error": None,
    }

    if len(cluster_df) < 2:
        result["coherent"] = True
        return result

    # 대표 기사 (첫 번째)
    representative = cluster_df.iloc[0]
    rep_summary = str(representative.get("news_keyword_summary", ""))
    rep_title = str(representative.get("title", ""))

    # 나머지 기사들과 비교
    prompts = load_source_verifier_prompts()
    ts_config = prompts.get("topic_similarity", {})

    system_prompt = ts_config.get("system", "")
    user_template = ts_config.get("user_prompt_template", "")
    response_schema = ts_config.get("response_schema", {})

    if not system_prompt or not user_template:
        result["error"] = "topic_similarity 프롬프트 없음"
        result["coherent"] = False
        return result

    model = _get_sv_model()

    for idx, row in cluster_df.iloc[1:].iterrows():
        article_summary = str(row.get("news_keyword_summary", ""))
        article_title = str(row.get("title", ""))

        if not article_summary or not rep_summary:
            continue

        # Context 구성
        context_a = f"제목: {rep_title}\n요약: {rep_summary}"
        context_b = f"제목: {article_title}\n요약: {article_summary}"

        context = {
            "context_a": context_a,
            "context_b": context_b,
        }

        user_prompt = render_prompt(user_template, context)

        try:
            llm_result = call_openai_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_schema=response_schema,
                openai_key=openai_key,
                model=model,
                label="클러스터검증",
                schema_name="topic_similarity_result",
            )

            if llm_result and "same_topic" in llm_result:
                same_topic = llm_result["same_topic"]
                reason = llm_result.get("reason", "")

                if not same_topic:
                    # 다른 주제로 판단됨
                    result["coherent"] = False
                    result["mismatches"].append({
                        "title": article_title,
                        "reason": reason,
                    })

        except Exception as e:
            result["error"] = f"LLM 검증 실패: {e}"
            result["coherent"] = False
            return result

    return result


def audit_all_topic_clusters(
    df: pd.DataFrame,
    openai_key: str,
    min_size: int = 2,
) -> Tuple[List[Dict], List[str]]:
    """
    모든 topic 클러스터를 검증하여 잘못 묶인 것을 탐지.

    Args:
        df: 전체 데이터 DataFrame
        openai_key: OpenAI API 키
        min_size: 검증 대상 최소 클러스터 크기 (기본 2)

    Returns:
        (audit_results, problematic_cluster_ids)
    """
    # _t로 시작하는 cluster_id만 필터링
    if "cluster_id" not in df.columns:
        print("⚠️  cluster_id 컬럼이 없습니다.")
        return [], []

    topic_mask = df["cluster_id"].str.contains("_t", na=False)
    df_topics = df[topic_mask].copy()

    if len(df_topics) == 0:
        print("ℹ️  검증할 topic 클러스터가 없습니다.")
        return [], []

    # cluster_id별 그룹핑
    cluster_groups = df_topics.groupby("cluster_id", dropna=False)
    print(f"\n📊 총 {len(cluster_groups)}개 topic 클러스터 검증 시작...")

    audit_results = []
    problematic_ids = []

    for cluster_id, cluster_df in cluster_groups:
        if len(cluster_df) < min_size:
            continue

        print(f"  🔍 검증 중: {cluster_id} ({len(cluster_df)}개 기사)...", end=" ")

        result = verify_cluster_coherence(cluster_df, cluster_id, openai_key)
        audit_results.append(result)

        if not result["coherent"]:
            problematic_ids.append(cluster_id)
            print(f"❌ 문제 발견 ({len(result['mismatches'])}개 불일치)")
        else:
            print("✅ 정상")

    return audit_results, problematic_ids


def generate_audit_report(audit_results: List[Dict], df: pd.DataFrame):
    """검증 결과 리포트 생성."""
    print("\n" + "=" * 100)
    print("📋 Topic 클러스터 검증 리포트")
    print("=" * 100)

    total = len(audit_results)
    coherent_count = sum(1 for r in audit_results if r["coherent"])
    problematic_count = total - coherent_count

    print(f"\n총 검증 클러스터: {total}개")
    print(f"  ✅ 정상: {coherent_count}개")
    print(f"  ❌ 문제 발견: {problematic_count}개")

    if problematic_count > 0:
        print(f"\n{'=' * 100}")
        print("❌ 문제가 있는 클러스터 상세:")
        print(f"{'=' * 100}")

        for result in audit_results:
            if not result["coherent"]:
                cluster_id = result["cluster_id"]
                cluster_df = df[df["cluster_id"] == cluster_id]

                print(f"\n[{cluster_id}] - {result['article_count']}개 기사")
                print(f"  불일치: {len(result['mismatches'])}개")

                # 기사 제목 출력
                print(f"\n  기사 목록:")
                for idx, row in cluster_df.iterrows():
                    print(f"    - {row['title'][:80]}")

                # 불일치 이유
                if result["mismatches"]:
                    print(f"\n  불일치 기사:")
                    for mismatch in result["mismatches"]:
                        print(f"    ❌ {mismatch['title'][:80]}")
                        print(f"       이유: {mismatch['reason']}")

                if result.get("error"):
                    print(f"  ⚠️  에러: {result['error']}")

    print(f"\n{'=' * 100}\n")


def fix_problematic_clusters(df: pd.DataFrame, cluster_ids: List[str]) -> pd.DataFrame:
    """
    문제가 있는 클러스터를 해체 (일반기사로 되돌림).

    Args:
        df: 전체 데이터 DataFrame
        cluster_ids: 해체할 cluster_id 리스트

    Returns:
        수정된 DataFrame
    """
    df = df.copy()

    for cluster_id in cluster_ids:
        mask = df["cluster_id"] == cluster_id
        affected_count = mask.sum()

        if affected_count > 0:
            print(f"  🔧 '{cluster_id}' 클러스터 해체 ({affected_count}개 기사)")
            df.loc[mask, "cluster_id"] = ""
            df.loc[mask, "source"] = "일반기사"
            df.loc[mask, "press_release_group"] = ""

    return df


def main():
    parser = argparse.ArgumentParser(description="Topic 클러스터 자동 검증")
    parser.add_argument("--auto_fix", action="store_true",
                        help="문제 클러스터 자동 해체 (없으면 리포트만 출력)")
    parser.add_argument("--min_size", type=int, default=2,
                        help="검증 대상 최소 클러스터 크기 (기본: 2)")
    parser.add_argument("--sheets_id", type=str, default=None,
                        help="Google Sheets ID (.env 대신 사용)")
    args = parser.parse_args()

    load_dotenv()

    # OpenAI API 키 확인
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        return

    # Google Sheets 연결
    creds_path = os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH", "service-account.json")
    sheet_id = args.sheets_id or os.getenv("GOOGLE_SHEET_ID")

    if not os.path.exists(creds_path) or not sheet_id:
        print("❌ Google Sheets 설정이 필요합니다.")
        print("  .env에 GOOGLE_SHEETS_CREDENTIALS_PATH, GOOGLE_SHEET_ID를 설정하세요.")
        return

    spreadsheet = connect_sheets(creds_path, sheet_id)
    if not spreadsheet:
        return

    # total_result 로드
    print("\n📂 총 결과 데이터 로드 중...")
    try:
        worksheet = spreadsheet.worksheet("total_result")
        records = worksheet.get_all_records()
    except Exception as e:
        print(f"❌ total_result 시트 로드 실패: {e}")
        return

    if not records:
        print("❌ total_result 시트가 비어있습니다.")
        return

    df = pd.DataFrame(records)
    print(f"✅ 로드 완료: {len(df)}개 기사")

    # 모든 topic 클러스터 검증
    audit_results, problematic_ids = audit_all_topic_clusters(
        df, openai_key, min_size=args.min_size
    )

    # 리포트 생성
    generate_audit_report(audit_results, df)

    # 자동 수정
    if args.auto_fix and problematic_ids:
        print(f"\n🔧 자동 수정 모드: {len(problematic_ids)}개 클러스터 해체 중...")
        df_fixed = fix_problematic_clusters(df, problematic_ids)

        print(f"\n변경 후 source 분포:")
        print(df_fixed["source"].value_counts().to_string())

        print("\n📤 Google Sheets 업데이트 중...")
        result = sync_to_sheets(
            df_fixed, spreadsheet, "total_result",
            force_update_existing=True,
        )
        print(f"✅ 완료: 추가 {result.get('added', 0)}개, "
              f"업데이트 {result.get('updated', 0)}개, "
              f"건너뜀 {result.get('skipped', 0)}개")

        # 로컬 CSV 백업도 업데이트
        csv_path = "../data/result.csv"
        if os.path.exists(csv_path):
            print(f"\n📁 로컬 CSV 백업 업데이트: {csv_path}")
            df_csv = pd.read_csv(csv_path, encoding="utf-8-sig")

            update_cols = ["link", "source", "cluster_id", "press_release_group"]
            df_updates = df_fixed[update_cols].copy()

            df_csv = df_csv.drop(columns=["source", "cluster_id", "press_release_group"], errors="ignore")
            df_csv = df_csv.merge(df_updates, on="link", how="left")
            df_csv.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"✅ CSV 업데이트 완료: {len(df_csv)}개 기사")

        print("\n✅ 자동 수정 완료!")

    elif problematic_ids:
        print(f"\n💡 수정 권장: {len(problematic_ids)}개 클러스터")
        print(f"\n자동 수정하려면 --auto_fix 플래그를 사용하세요:")
        print(f"  python audit_topic_clusters.py --auto_fix")

    else:
        print("\n✅ 모든 topic 클러스터가 정상입니다!")


if __name__ == "__main__":
    main()
