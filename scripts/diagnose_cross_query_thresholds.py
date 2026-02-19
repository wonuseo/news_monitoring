"""
diagnose_cross_query_thresholds.py

merge_cross_query_clusters 임계값 구간별 샘플 5개 추출.
총 5개 구간:
  1. auto_title    : t_cos >= 0.65 AND t_jac >= 0.15  (제목 기준 자동 병합)
  2. auto_desc     : d_cos >= 0.55 AND d_jac >= 0.08  (본문 기준 자동 병합, 제목 미달)
  3. border_title  : 0.58 <= t_cos < 0.65 AND t_jac >= 0.20  (경계선 - 제목)
  4. border_desc   : 0.48 <= d_cos < 0.55 AND d_jac >= 0.12  (경계선 - 본문)
  5. no_merge      : 위 모두 해당 없음 (병합 안 됨)

Usage:
    python scripts/diagnose_cross_query_thresholds.py
    python scripts/diagnose_cross_query_thresholds.py --sheet total_result --max_rows 2000
"""

import os
import re
import sys
import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.modules.export.sheets import connect_sheets

load_dotenv()

# ─── 임계값 (source_verifier.py 와 동일) ───────────────────────────────────
CROSS_TITLE_COS_THRESHOLD = 0.65
CROSS_TITLE_JAC_THRESHOLD = 0.15
CROSS_DESC_COS_THRESHOLD = 0.55
CROSS_DESC_JAC_THRESHOLD = 0.08
CROSS_TITLE_COS_BORDERLINE = (0.58, 0.65)
CROSS_TITLE_JAC_BORDERLINE_MIN = 0.20
CROSS_DESC_COS_BORDERLINE = (0.48, 0.55)
CROSS_DESC_JAC_BORDERLINE_MIN = 0.12

SAMPLE_PER_BUCKET = 5


def _clean_html(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("&quot;", " ").replace("&amp;", "&")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokenize_simple(s: str, min_token_len: int = 2) -> set:
    if not isinstance(s, str) or not s.strip():
        return set()
    s = s.lower()
    toks = re.findall(r"[가-힣a-z0-9]+", s)
    return {t for t in toks if len(t) >= min_token_len}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def classify_bucket(t_cos, t_jac, d_cos, d_jac) -> str:
    if t_cos >= CROSS_TITLE_COS_THRESHOLD and t_jac >= CROSS_TITLE_JAC_THRESHOLD:
        return "auto_title"
    if d_cos >= CROSS_DESC_COS_THRESHOLD and d_jac >= CROSS_DESC_JAC_THRESHOLD:
        return "auto_desc"
    title_border = (
        CROSS_TITLE_COS_BORDERLINE[0] <= t_cos < CROSS_TITLE_COS_BORDERLINE[1]
        and t_jac >= CROSS_TITLE_JAC_BORDERLINE_MIN
    )
    desc_border = (
        CROSS_DESC_COS_BORDERLINE[0] <= d_cos < CROSS_DESC_COS_BORDERLINE[1]
        and d_jac >= CROSS_DESC_JAC_BORDERLINE_MIN
    )
    if title_border:
        return "border_title"
    if desc_border:
        return "border_desc"
    return "no_merge"


def load_from_sheets(sheet_name: str, max_rows: int) -> pd.DataFrame:
    creds_path = os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH", "service-account.json")
    sheet_id = os.getenv("GOOGLE_SHEET_ID")
    if not os.path.exists(creds_path) or not sheet_id:
        raise RuntimeError("GOOGLE_SHEETS_CREDENTIALS_PATH 또는 GOOGLE_SHEET_ID 미설정")
    print(f"📊 Google Sheets 연결 중... ({sheet_name})")
    sp = connect_sheets(creds_path, sheet_id)
    ws = sp.worksheet(sheet_name)
    rows = ws.get_all_values()
    if not rows:
        raise RuntimeError("시트가 비어있음")
    headers = rows[0]
    data = rows[1:max_rows + 1] if max_rows else rows[1:]
    df = pd.DataFrame(data, columns=headers)
    print(f"  ✅ {len(df)}행 로드 완료")
    return df


def run(sheet_name: str, max_rows: int, output_path: str, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    df = load_from_sheets(sheet_name, max_rows)

    # 필수 컬럼 확인
    for col in ("title", "description", "query", "cluster_id"):
        if col not in df.columns:
            raise RuntimeError(f"필수 컬럼 없음: {col}")

    # 후보 수집 (merge_cross_query_clusters 방식 재현)
    candidates: List[Dict] = []
    clustered_mask = df["cluster_id"].astype(str).str.strip() != ""

    # 클러스터별 대표 기사
    for cid, cgroup in df[clustered_mask].groupby("cluster_id"):
        if "pub_datetime" in df.columns:
            repr_idx = cgroup["pub_datetime"].astype(str).idxmin()
        else:
            repr_idx = cgroup.index[0]
        source_val = str(cgroup["source"].iloc[0]) if "source" in cgroup.columns else ""
        candidates.append({
            "repr_idx": repr_idx,
            "cluster_id": str(cid),
            "query": str(cgroup["query"].iloc[0]),
            "source": source_val,
        })

    # 미클러스터 일반기사
    unclustered_mask = (~clustered_mask)
    if "source" in df.columns:
        unclustered_mask = unclustered_mask & (df["source"] == "일반기사")
    if "news_category" in df.columns:
        unclustered_mask = unclustered_mask & (df["news_category"] != "비관련")
    for idx in df[unclustered_mask].index:
        candidates.append({
            "repr_idx": idx,
            "cluster_id": "",
            "query": str(df.at[idx, "query"]) if "query" in df.columns else "",
            "source": "일반기사",
        })

    n = len(candidates)
    print(f"  후보 {n}개 (클러스터 대표 + 미클러스터 일반기사)")
    if n < 2:
        print("❌ 후보가 2개 미만 — 비교 불가")
        return

    # TF-IDF 벡터화
    repr_indices = [c["repr_idx"] for c in candidates]
    title_texts = [_clean_html(str(df.at[idx, "title"])) for idx in repr_indices]
    desc_texts = [_clean_html(str(df.at[idx, "description"])) for idx in repr_indices]
    title_toksets = [_tokenize_simple(t) for t in title_texts]
    desc_toksets = [_tokenize_simple(d) for d in desc_texts]

    title_corpus = [t if t.strip() else " " for t in title_texts]
    desc_corpus = [d if d.strip() else " " for d in desc_texts]

    print("  TF-IDF 벡터화 중...")
    vec_t = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1)
    vec_d = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1)
    X_t = vec_t.fit_transform(title_corpus)
    X_d = vec_d.fit_transform(desc_corpus)
    sim_t = cosine_similarity(X_t)
    sim_d = cosine_similarity(X_d)
    print("  ✅ 벡터화 완료")

    # 버킷별 샘플 수집
    buckets: Dict[str, List[Dict]] = {
        "auto_title": [],
        "auto_desc": [],
        "border_title": [],
        "border_desc": [],
        "no_merge": [],
    }

    total_pairs = n * (n - 1) // 2
    print(f"  전체 {total_pairs}쌍 분류 중...")

    pair_list = []
    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = candidates[i], candidates[j]
            # skip 조건
            if ci["cluster_id"] and ci["cluster_id"] == cj["cluster_id"]:
                continue
            if ci["query"] == cj["query"] and not ci["cluster_id"] and not cj["cluster_id"]:
                continue
            pair_list.append((i, j))

    # 랜덤 셔플로 no_merge 샘플 다양하게
    random.shuffle(pair_list)

    for i, j in pair_list:
        # 모든 버킷이 채워지면 조기 종료
        if all(len(v) >= SAMPLE_PER_BUCKET for v in buckets.values()):
            break

        t_cos = float(sim_t[i, j])
        d_cos = float(sim_d[i, j])
        t_jac = _jaccard(title_toksets[i], title_toksets[j])
        d_jac = _jaccard(desc_toksets[i], desc_toksets[j])

        bucket = classify_bucket(t_cos, t_jac, d_cos, d_jac)
        if len(buckets[bucket]) >= SAMPLE_PER_BUCKET:
            continue

        ci, cj = candidates[i], candidates[j]
        idx_i, idx_j = ci["repr_idx"], cj["repr_idx"]
        buckets[bucket].append({
            "bucket": bucket,
            "title_a": str(df.at[idx_i, "title"])[:60],
            "title_b": str(df.at[idx_j, "title"])[:60],
            "query_a": ci["query"],
            "query_b": cj["query"],
            "cluster_id_a": ci["cluster_id"] or "(미클러스터)",
            "cluster_id_b": cj["cluster_id"] or "(미클러스터)",
            "source_a": ci["source"],
            "source_b": cj["source"],
            "t_cos": round(t_cos, 4),
            "t_jac": round(t_jac, 4),
            "d_cos": round(d_cos, 4),
            "d_jac": round(d_jac, 4),
        })

    # 결과 출력
    all_rows = []
    for bucket_name in ["auto_title", "auto_desc", "border_title", "border_desc", "no_merge"]:
        rows = buckets[bucket_name]
        all_rows.extend(rows)

    df_out = pd.DataFrame(all_rows)

    print("\n" + "=" * 100)
    print("임계값 구간별 샘플 결과")
    print("=" * 100)

    bucket_labels = {
        "auto_title": f"[자동병합-제목] t_cos>={CROSS_TITLE_COS_THRESHOLD}, t_jac>={CROSS_TITLE_JAC_THRESHOLD}",
        "auto_desc": f"[자동병합-본문] d_cos>={CROSS_DESC_COS_THRESHOLD}, d_jac>={CROSS_DESC_JAC_THRESHOLD} (제목 미달)",
        "border_title": f"[경계선-제목] {CROSS_TITLE_COS_BORDERLINE[0]}<=t_cos<{CROSS_TITLE_COS_BORDERLINE[1]}, t_jac>={CROSS_TITLE_JAC_BORDERLINE_MIN}",
        "border_desc": f"[경계선-본문] {CROSS_DESC_COS_BORDERLINE[0]}<=d_cos<{CROSS_DESC_COS_BORDERLINE[1]}, d_jac>={CROSS_DESC_JAC_BORDERLINE_MIN}",
        "no_merge": "[병합안됨] 위 모두 해당 없음",
    }

    for bucket_name, label in bucket_labels.items():
        rows = buckets[bucket_name]
        print(f"\n{'─'*100}")
        print(f"  {label}  ({len(rows)}/{SAMPLE_PER_BUCKET}개)")
        print(f"{'─'*100}")
        if not rows:
            print("  (샘플 없음)")
            continue
        for k, r in enumerate(rows, 1):
            print(f"  [{k}] t_cos={r['t_cos']}, t_jac={r['t_jac']}, d_cos={r['d_cos']}, d_jac={r['d_jac']}")
            print(f"      A ({r['query_a']}, {r['cluster_id_a']}): {r['title_a']}")
            print(f"      B ({r['query_b']}, {r['cluster_id_b']}): {r['title_b']}")

    # CSV 저장
    df_out.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ CSV 저장: {output_path} ({len(df_out)}행)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-query 임계값 구간별 샘플 추출")
    parser.add_argument("--sheet", default="total_result", help="Sheets 탭 이름 (default: total_result)")
    parser.add_argument("--max_rows", type=int, default=3000, help="최대 로드 행 수 (default: 3000)")
    parser.add_argument("--output", default="scripts/cross_query_threshold_samples.csv", help="출력 CSV 경로")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    args = parser.parse_args()

    run(
        sheet_name=args.sheet,
        max_rows=args.max_rows,
        output_path=args.output,
        seed=args.seed,
    )
