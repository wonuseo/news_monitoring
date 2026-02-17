"""
recluster_all.py - 전체 기사 재클러스터링 스크립트

Google Sheets의 total_result 전체를 대상으로 클러스터링을 처음부터 다시 실행.

처리 흐름:
  Step 1: Google Sheets total_result 로드
  Step 2: 기존 클러스터링 필드 초기화 (+ 일반기사 LLM 결과는 기본 유지)
  Step 3: detect_similar_articles(enable_llm_borderline=True)
  Step 4: summarize_clusters()
  Step 5: classify_press_releases() (--skip_llm_classify 시 스킵)
  Step 6: classify_llm() 일반기사 분류 (--skip_general_llm_classify 시 스킵)
  Step 7: verify_and_regroup_sources() (--skip_source_verify 시 스킵)
  Step 8: summarize_clusters() (새 토픽 그룹 요약)
  Step 9: 일관성 검증
  Step 10: Google Sheets 업데이트 + CSV 백업

Usage:
  python scripts/recluster_all.py                     # 전체 재클러스터링
  python scripts/recluster_all.py --dry_run            # Sheets 업데이트 없이 미리보기
  python scripts/recluster_all.py --skip_llm_classify  # 보도자료 LLM 분류 스킵
  python scripts/recluster_all.py --skip_general_llm_classify  # 일반기사 LLM 분류 스킵
  python scripts/recluster_all.py --force_general_llm_reclassify  # 일반기사 LLM 분류값 초기화 후 재분류
  python scripts/recluster_all.py --skip_source_verify # source 검증 스킵
  python scripts/recluster_all.py --sheets_id ID       # 특정 Sheets ID 사용
  python scripts/recluster_all.py --chunk_size 50      # LLM 분류 청크 크기
  python scripts/recluster_all.py --max_workers 3      # 병렬 워커 수
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.modules.export.sheets import connect_sheets, sync_to_sheets, filter_total_result_by_date
from src.modules.processing.press_release_detector import detect_similar_articles, summarize_clusters
from src.modules.analysis.classify_press_releases import classify_press_releases
from src.modules.analysis.classify_llm import classify_llm
from src.modules.analysis.source_verifier import verify_and_regroup_sources


def load_total_result_from_sheets(spreadsheet) -> pd.DataFrame:
    """Google Sheets total_result 탭을 DataFrame으로 로드."""
    try:
        worksheet = spreadsheet.worksheet("total_result")
        records = worksheet.get_all_records()
        if not records:
            print("  ℹ️  total_result 시트가 비어있습니다.")
            return pd.DataFrame()
        df = pd.DataFrame(records)
        print(f"📂 Sheets total_result에서 {len(df)}개 기사 로드")
        return df
    except Exception as e:
        print(f"❌ total_result 시트 로드 실패: {e}")
        return pd.DataFrame()


def reset_cluster_fields(df: pd.DataFrame, preserve_existing_llm: bool = True) -> pd.DataFrame:
    """클러스터링 필드 초기화. 필요 시 기존 일반기사 LLM 분류 결과는 유지."""
    df = df.copy()

    # 항상 재생성할 클러스터링 필드
    df["source"] = "일반기사"
    df["cluster_id"] = ""
    df["cluster_summary"] = ""

    llm_cols = [
        "brand_relevance",
        "brand_relevance_query_keywords",
        "sentiment_stage",
        "danger_level",
        "issue_category",
        "news_category",
        "news_keyword_summary",
        "classified_at",
    ]

    if preserve_existing_llm:
        # 없는 컬럼만 생성하고 기존 값은 유지 (classify_llm가 classified_at 기준으로 자동 스킵)
        for col in llm_cols:
            if col not in df.columns:
                df[col] = ""
        reused_count = int(df["classified_at"].astype(str).str.strip().ne("").sum()) if "classified_at" in df.columns else 0
        print(
            f"🔄 {len(df)}개 기사의 클러스터링 필드 초기화 완료 "
            f"(기존 LLM 분류 유지: {reused_count}개)"
        )
    else:
        for col in llm_cols:
            df[col] = ""
        print(f"🔄 {len(df)}개 기사의 클러스터링/소스 판정 필드 초기화 완료 (LLM 분류 강제 초기화)")

    return df


def validate_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """일관성 검증: 같은 cluster_id → 같은 cluster_summary, 같은 source."""
    df = df.copy()
    fixes = 0

    if "cluster_id" not in df.columns:
        return df

    clustered = df["cluster_id"].astype(str).str.strip() != ""

    # Rule 1: source=="일반기사" → cluster_id="", cluster_summary=""
    general_mask = df["source"] == "일반기사"
    r1 = (general_mask & clustered).sum()
    if r1 > 0:
        df.loc[general_mask, "cluster_id"] = ""
        df.loc[general_mask, "cluster_summary"] = ""
        fixes += r1
        print(f"  일관성 Rule 1: 일반기사 {r1}건의 cluster_id/cluster_summary 초기화")

    # Refresh clustered mask
    clustered = df["cluster_id"].astype(str).str.strip() != ""

    # Rule 2: 같은 cluster_id → 같은 source (첫 번째 값 전파)
    if clustered.any():
        for cid, cgroup in df[clustered].groupby("cluster_id"):
            sources = cgroup["source"].unique()
            if len(sources) > 1:
                target_source = cgroup["source"].iloc[0]
                mismatch = clustered & (df["cluster_id"] == cid) & (df["source"] != target_source)
                fix_count = mismatch.sum()
                if fix_count > 0:
                    df.loc[mismatch, "source"] = target_source
                    fixes += fix_count

    # Rule 3: 같은 cluster_id → 같은 cluster_summary (첫 비어있지 않은 값 전파)
    if "cluster_summary" in df.columns and clustered.any():
        for cid, cgroup in df[clustered].groupby("cluster_id"):
            summaries = cgroup["cluster_summary"].dropna().astype(str)
            summaries = summaries[summaries.str.strip() != ""]
            if len(summaries) > 0:
                first_val = summaries.iloc[0]
                mismatch = clustered & (df["cluster_id"] == cid) & (df["cluster_summary"] != first_val)
                fix_count = mismatch.sum()
                if fix_count > 0:
                    df.loc[mismatch, "cluster_summary"] = first_val
                    fixes += fix_count

    if fixes > 0:
        print(f"  ✅ 일관성 검증: 총 {fixes}건 수정")
    else:
        print(f"  ✅ 일관성 검증: 불일치 없음")

    return df


def print_summary(df: pd.DataFrame) -> None:
    """재클러스터링 결과 요약 출력."""
    print("\n" + "=" * 60)
    print("📊 재클러스터링 결과 요약")
    print("=" * 60)

    print(f"  전체 기사: {len(df)}개")

    if "source" in df.columns:
        source_dist = df["source"].value_counts().to_dict()
        print(f"  Source 분포:")
        for src, cnt in source_dist.items():
            print(f"    - {src}: {cnt}개")

    if "cluster_id" in df.columns:
        clustered = df[df["cluster_id"].astype(str).str.strip() != ""]
        n_clusters = clustered["cluster_id"].nunique()
        print(f"  클러스터 수: {n_clusters}개 ({len(clustered)}개 기사)")

        # 클러스터 크기 분포
        if n_clusters > 0:
            sizes = clustered.groupby("cluster_id").size()
            print(f"    평균 크기: {sizes.mean():.1f}, 최대: {sizes.max()}, 최소: {sizes.min()}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="전체 기사 재클러스터링")
    parser.add_argument("--dry_run", action="store_true", help="Sheets 업데이트 없이 미리보기")
    parser.add_argument("--skip_llm_classify", action="store_true", help="보도자료 LLM 분류 스킵")
    parser.add_argument("--skip_general_llm_classify", action="store_true", help="일반기사 LLM 분류 스킵")
    parser.add_argument("--force_general_llm_reclassify", action="store_true", help="일반기사 LLM 분류값 초기화 후 재분류")
    parser.add_argument("--skip_source_verify", action="store_true", help="source 검증 + 주제 그룹화 스킵")
    parser.add_argument("--sheets_id", type=str, default=None, help="특정 Google Sheets ID")
    parser.add_argument("--chunk_size", type=int, default=50, help="LLM 분류 청크 크기")
    parser.add_argument("--max_workers", type=int, default=3, help="병렬 워커 수")
    parser.add_argument("--outdir", type=str, default="data", help="CSV 백업 디렉토리")
    args = parser.parse_args()

    # .env 로드
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    openai_key = os.getenv("OPENAI_API_KEY")
    creds_path = os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH")
    sheet_id = args.sheets_id or os.getenv("GOOGLE_SHEET_ID")

    if not creds_path or not sheet_id:
        print("❌ GOOGLE_SHEETS_CREDENTIALS_PATH 또는 GOOGLE_SHEET_ID가 설정되지 않았습니다.")
        sys.exit(1)

    if not openai_key:
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        sys.exit(1)

    # 전체 진행률 바 (총 10개 Step)
    total_steps = 10
    print("\n" + "=" * 60)
    print("🔄 전체 기사 재클러스터링 시작")
    print("=" * 60)
    overall_pbar = tqdm(total=total_steps, desc="전체 진행", unit="step", position=0)

    # ─── Step 1: Google Sheets total_result 로드 ─────────────────────────
    print("\n📋 Step 1/10: Google Sheets total_result 로드")
    spreadsheet = connect_sheets(creds_path, sheet_id)
    if spreadsheet is None:
        print("❌ Google Sheets 연결 실패")
        sys.exit(1)

    df = load_total_result_from_sheets(spreadsheet)
    if df.empty:
        print("❌ total_result가 비어있습니다. 종료합니다.")
        sys.exit(1)

    # 필수 컬럼 확인
    required_cols = {"title", "description", "link", "query"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"❌ 필수 컬럼 누락: {missing}")
        sys.exit(1)
    overall_pbar.update(1)

    # ─── Step 2: 기존 클러스터링 필드 초기화 (+ LLM 재사용 옵션) ───────────
    if args.force_general_llm_reclassify:
        print("\n📋 Step 2/10: 기존 클러스터링/소스판정 필드 초기화 (일반기사 LLM 강제 재분류)")
    else:
        print("\n📋 Step 2/10: 기존 클러스터링 필드 초기화 (기존 일반기사 LLM 결과 재사용)")
    df = reset_cluster_fields(df, preserve_existing_llm=not args.force_general_llm_reclassify)
    overall_pbar.update(1)

    # ─── Step 3: TF-IDF + LLM 경계선 클러스터링 ─────────────────────────
    print("\n📋 Step 3/10: TF-IDF 유사도 + LLM 경계선 클러스터링")
    df = detect_similar_articles(
        df,
        spreadsheet=None,  # cumulative numbering 없이 1부터 시작
        enable_llm_borderline=True,
        openai_key=openai_key,
        group_our_brands=True,
    )
    overall_pbar.update(1)

    # ─── Step 4: 클러스터 요약 ───────────────────────────────────────────
    print("\n📋 Step 4/10: 클러스터 요약 생성")
    df = summarize_clusters(df, openai_key=openai_key)
    overall_pbar.update(1)

    # ─── Step 5: 보도자료 LLM 분류 ──────────────────────────────────────
    if not args.skip_llm_classify:
        print("\n📋 Step 5/10: 보도자료 대표 기사 LLM 분류")
        df, pr_stats = classify_press_releases(
            df,
            openai_key=openai_key,
            chunk_size=args.chunk_size,
            max_workers=args.max_workers,
        )
        print(f"  PR 분류 통계: {pr_stats}")
    else:
        print("\n📋 Step 5/10: 보도자료 LLM 분류 스킵 (--skip_llm_classify)")
    overall_pbar.update(1)

    # ─── Step 6: 일반기사 LLM 분류 ───────────────────────────────────────
    if not args.skip_general_llm_classify:
        print("\n📋 Step 6/10: 일반기사 LLM 분류")
        df, llm_stats = classify_llm(
            df,
            openai_key=openai_key,
            chunk_size=args.chunk_size,
            dry_run=False,  # recluster_all의 dry_run은 Sheets 업데이트만 제어
            max_competitor_classify=0,
            max_workers=args.max_workers,
        )
        print(f"  일반기사 분류 통계: {llm_stats}")
    else:
        print("\n📋 Step 6/10: 일반기사 LLM 분류 스킵 (--skip_general_llm_classify)")
    overall_pbar.update(1)

    # ─── Step 7: Source 검증 + 주제 그룹화 ───────────────────────────────
    if not args.skip_source_verify:
        print("\n📋 Step 7/10: Source 검증 + 주제 그룹화")
        df, sv_stats = verify_and_regroup_sources(df, openai_key=openai_key)
        print(f"  SV 통계: {sv_stats}")
    else:
        print("\n📋 Step 7/10: Source 검증 스킵 (--skip_source_verify)")
    overall_pbar.update(1)

    # ─── Step 8: 새 토픽 그룹 요약 ──────────────────────────────────────
    print("\n📋 Step 8/10: 새 토픽 그룹 요약 생성")
    df = summarize_clusters(df, openai_key=openai_key)
    overall_pbar.update(1)

    # ─── Step 9: 일관성 검증 ────────────────────────────────────────────
    print("\n📋 Step 9/10: 일관성 검증")
    df = validate_consistency(df)
    overall_pbar.update(1)

    # 결과 요약
    print_summary(df)

    # ─── Step 10: Google Sheets 업데이트 + CSV 백업 ──────────────────────
    if args.dry_run:
        print("\n🏃 Step 10/10: DRY RUN 모드 - Sheets 업데이트를 건너뜁니다.")
    else:
        print("\n📋 Step 10/10: Google Sheets 업데이트")
        df_upload = filter_total_result_by_date(df)
        sync_result = sync_to_sheets(
            df_upload,
            spreadsheet,
            "total_result",
            force_update_existing=True,
        )
        print(f"  Sheets 업데이트 결과: {sync_result}")
    overall_pbar.update(1)

    # CSV 백업
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = outdir / f"recluster_result_{timestamp}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n💾 CSV 백업 저장: {csv_path}")

    overall_pbar.close()
    print("\n✅ 재클러스터링 완료!")


if __name__ == "__main__":
    main()
