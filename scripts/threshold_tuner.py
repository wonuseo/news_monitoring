"""
threshold_tuner.py - Cross-query 임계값 튜닝 도구

2단계 워크플로우:
  Step 1 (느림, 1회): Sheets 로드 + TF-IDF 벡터화 → 페어 점수 캐시
  Step 2 (빠름, 반복): 캐시 로드 → 임계값 적용 → 통계/샘플 출력

사용법:
  # Step 1: 페어 점수 사전 계산 (Sheets 연결 필요, ~1-2분)
  python scripts/threshold_tuner.py precompute

  # Step 2: 현재 임계값 기준 통계 + 샘플
  python scripts/threshold_tuner.py test

  # Step 2: 특정 임계값 테스트
  python scripts/threshold_tuner.py test --t_cos 0.60 --t_jac 0.10

  # Step 2: 여러 임계값 조합 스윕 (summary 테이블)
  python scripts/threshold_tuner.py sweep

  # Step 2: 스윕 범위 커스텀
  python scripts/threshold_tuner.py sweep --t_cos_range 0.55,0.60,0.65,0.70 --t_jac_range 0.10,0.15,0.20
"""

import os
import re
import sys
import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple
from itertools import product

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

# ─── 기본값 (source_verifier.py 와 동일) ───────────────────────────────────
DEFAULT = {
    "t_cos": 0.65,
    "t_jac": 0.15,
    "d_cos": 0.55,
    "d_jac": 0.08,
    "border_t_cos_lo": 0.58,
    "border_t_cos_hi": 0.65,
    "border_t_jac_min": 0.20,
    "border_d_cos_lo": 0.48,
    "border_d_cos_hi": 0.55,
    "border_d_jac_min": 0.12,
}

CACHE_PATH = Path("scripts/.pair_cache.csv")
SAMPLE_N = 5


# ────────────────────────────────────────────────────────────────────────────
# 유틸
# ────────────────────────────────────────────────────────────────────────────

def _clean_html(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("&quot;", " ").replace("&amp;", "&")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokenize(s: str, min_len: int = 2) -> set:
    if not isinstance(s, str) or not s.strip():
        return set()
    toks = re.findall(r"[가-힣a-z0-9]+", s.lower())
    return {t for t in toks if len(t) >= min_len}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def classify(row, cfg: dict) -> str:
    t_cos, t_jac = row["t_cos"], row["t_jac"]
    d_cos, d_jac = row["d_cos"], row["d_jac"]

    if t_cos >= cfg["t_cos"] and t_jac >= cfg["t_jac"]:
        return "auto_title"
    if d_cos >= cfg["d_cos"] and d_jac >= cfg["d_jac"]:
        return "auto_desc"

    t_border = cfg["border_t_cos_lo"] <= t_cos < cfg["border_t_cos_hi"] and t_jac >= cfg["border_t_jac_min"]
    d_border = cfg["border_d_cos_lo"] <= d_cos < cfg["border_d_cos_hi"] and d_jac >= cfg["border_d_jac_min"]
    if t_border:
        return "border_title"
    if d_border:
        return "border_desc"
    return "no_merge"


# ────────────────────────────────────────────────────────────────────────────
# Step 1: precompute
# ────────────────────────────────────────────────────────────────────────────

def cmd_precompute(args):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from src.modules.export.sheets import connect_sheets

    creds_path = os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH", "service-account.json")
    sheet_id = os.getenv("GOOGLE_SHEET_ID")
    if not os.path.exists(creds_path) or not sheet_id:
        print("❌ GOOGLE_SHEETS_CREDENTIALS_PATH 또는 GOOGLE_SHEET_ID 미설정")
        return

    print(f"📊 Google Sheets 연결 중...")
    sp = connect_sheets(creds_path, sheet_id)
    ws = sp.worksheet(args.sheet)
    rows = ws.get_all_values()
    if not rows:
        print("❌ 시트가 비어있음")
        return
    headers = rows[0]
    data = rows[1:args.max_rows + 1] if args.max_rows else rows[1:]
    df = pd.DataFrame(data, columns=headers)
    print(f"  ✅ {len(df)}행 로드 완료")

    # 후보 수집
    candidates: List[Dict] = []
    clustered_mask = df["cluster_id"].astype(str).str.strip() != ""

    for cid, cgroup in df[clustered_mask].groupby("cluster_id"):
        repr_idx = cgroup["pub_datetime"].astype(str).idxmin() if "pub_datetime" in df.columns else cgroup.index[0]
        candidates.append({
            "repr_idx": repr_idx,
            "cluster_id": str(cid),
            "query": str(cgroup["query"].iloc[0]) if "query" in cgroup.columns else "",
            "source": str(cgroup["source"].iloc[0]) if "source" in cgroup.columns else "",
        })

    unclustered_mask = ~clustered_mask
    if "source" in df.columns:
        unclustered_mask &= df["source"] == "일반기사"
    if "news_category" in df.columns:
        unclustered_mask &= df["news_category"] != "비관련"
    for idx in df[unclustered_mask].index:
        candidates.append({
            "repr_idx": idx,
            "cluster_id": "",
            "query": str(df.at[idx, "query"]) if "query" in df.columns else "",
            "source": "일반기사",
        })

    n = len(candidates)
    print(f"  후보 {n}개 → {n*(n-1)//2}쌍 계산 중...")

    repr_indices = [c["repr_idx"] for c in candidates]
    title_texts = [_clean_html(str(df.at[idx, "title"])) for idx in repr_indices]
    desc_texts = [_clean_html(str(df.at[idx, "description"])) for idx in repr_indices]
    title_tok = [_tokenize(t) for t in title_texts]
    desc_tok = [_tokenize(d) for d in desc_texts]

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

    print("  페어 레코드 생성 중...")
    records = []
    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = candidates[i], candidates[j]
            if ci["cluster_id"] and ci["cluster_id"] == cj["cluster_id"]:
                continue
            if ci["query"] == cj["query"] and not ci["cluster_id"] and not cj["cluster_id"]:
                continue
            records.append({
                "i": i, "j": j,
                "title_a": str(df.at[ci["repr_idx"], "title"])[:80],
                "title_b": str(df.at[cj["repr_idx"], "title"])[:80],
                "query_a": ci["query"], "query_b": cj["query"],
                "cluster_id_a": ci["cluster_id"] or "(미클러스터)",
                "cluster_id_b": cj["cluster_id"] or "(미클러스터)",
                "source_a": ci["source"], "source_b": cj["source"],
                "t_cos": round(float(sim_t[i, j]), 4),
                "t_jac": round(_jaccard(title_tok[i], title_tok[j]), 4),
                "d_cos": round(float(sim_d[i, j]), 4),
                "d_jac": round(_jaccard(desc_tok[i], desc_tok[j]), 4),
            })

    df_pairs = pd.DataFrame(records)
    CACHE_PATH.parent.mkdir(exist_ok=True)
    df_pairs.to_csv(CACHE_PATH, index=False, encoding="utf-8-sig")
    print(f"  ✅ {len(df_pairs)}쌍 캐시 저장: {CACHE_PATH}")


# ────────────────────────────────────────────────────────────────────────────
# Step 2: test
# ────────────────────────────────────────────────────────────────────────────

def _print_bucket(name: str, label: str, rows: pd.DataFrame, n: int = SAMPLE_N):
    print(f"\n{'─'*100}")
    print(f"  {label}")
    print(f"  총 {len(rows)}쌍")
    print(f"{'─'*100}")
    if rows.empty:
        print("  (없음)")
        return
    sample = rows.sample(min(n, len(rows)), random_state=42)
    for k, (_, r) in enumerate(sample.iterrows(), 1):
        print(f"  [{k}] t_cos={r.t_cos:.4f}  t_jac={r.t_jac:.4f}  d_cos={r.d_cos:.4f}  d_jac={r.d_jac:.4f}")
        print(f"      A ({r.query_a} / {r.cluster_id_a}): {r.title_a}")
        print(f"      B ({r.query_b} / {r.cluster_id_b}): {r.title_b}")


def cmd_test(args):
    if not CACHE_PATH.exists():
        print(f"❌ 캐시 없음. 먼저 실행: python scripts/threshold_tuner.py precompute")
        return

    df = pd.read_csv(CACHE_PATH, encoding="utf-8-sig")
    print(f"캐시 로드: {len(df)}쌍\n")

    cfg = {**DEFAULT}
    if args.t_cos is not None: cfg["t_cos"] = args.t_cos
    if args.t_jac is not None: cfg["t_jac"] = args.t_jac
    if args.d_cos is not None: cfg["d_cos"] = args.d_cos
    if args.d_jac is not None: cfg["d_jac"] = args.d_jac
    if args.border_t_cos_lo is not None: cfg["border_t_cos_lo"] = args.border_t_cos_lo
    if args.border_t_cos_hi is not None: cfg["border_t_cos_hi"] = args.border_t_cos_hi
    if args.border_t_jac_min is not None: cfg["border_t_jac_min"] = args.border_t_jac_min
    if args.border_d_cos_lo is not None: cfg["border_d_cos_lo"] = args.border_d_cos_lo
    if args.border_d_cos_hi is not None: cfg["border_d_cos_hi"] = args.border_d_cos_hi
    if args.border_d_jac_min is not None: cfg["border_d_jac_min"] = args.border_d_jac_min

    # borderline 범위는 auto 임계값에 연동 (명시 안 했을 때)
    if args.t_cos is not None and args.border_t_cos_hi is None:
        cfg["border_t_cos_hi"] = cfg["t_cos"]
    if args.d_cos is not None and args.border_d_cos_hi is None:
        cfg["border_d_cos_hi"] = cfg["d_cos"]

    print("적용 임계값:")
    print(f"  auto_title  : t_cos>={cfg['t_cos']}  t_jac>={cfg['t_jac']}")
    print(f"  auto_desc   : d_cos>={cfg['d_cos']}  d_jac>={cfg['d_jac']}  (제목 미달 시)")
    print(f"  border_title: {cfg['border_t_cos_lo']}<=t_cos<{cfg['border_t_cos_hi']}  t_jac>={cfg['border_t_jac_min']}")
    print(f"  border_desc : {cfg['border_d_cos_lo']}<=d_cos<{cfg['border_d_cos_hi']}  d_jac>={cfg['border_d_jac_min']}")

    df["bucket"] = df.apply(lambda r: classify(r, cfg), axis=1)

    buckets_def = [
        ("auto_title",   f"[자동병합-제목]   t_cos>={cfg['t_cos']}  t_jac>={cfg['t_jac']}"),
        ("auto_desc",    f"[자동병합-본문]   d_cos>={cfg['d_cos']}  d_jac>={cfg['d_jac']}  (제목 미달)"),
        ("border_title", f"[경계선-제목]     {cfg['border_t_cos_lo']}<=t_cos<{cfg['border_t_cos_hi']}  t_jac>={cfg['border_t_jac_min']}"),
        ("border_desc",  f"[경계선-본문]     {cfg['border_d_cos_lo']}<=d_cos<{cfg['border_d_cos_hi']}  d_jac>={cfg['border_d_jac_min']}"),
        ("no_merge",     "[병합안됨]        위 모두 해당 없음"),
    ]
    print("\n" + "="*100)
    for bname, label in buckets_def:
        _print_bucket(bname, label, df[df["bucket"] == bname])
    print("\n" + "="*100)
    print("전체 요약:")
    counts = df["bucket"].value_counts().reindex([b for b,_ in buckets_def], fill_value=0)
    for bname, label in buckets_def:
        print(f"  {bname:<14} {counts[bname]:>6}쌍   ({counts[bname]/len(df)*100:.1f}%)")
    print(f"  {'합계':<14} {len(df):>6}쌍")

    # 현재 임계값과 비교 (변경 사항만)
    if any(v is not None for v in [args.t_cos, args.t_jac, args.d_cos, args.d_jac]):
        base_cfg = {**DEFAULT}
        df["bucket_base"] = df.apply(lambda r: classify(r, base_cfg), axis=1)
        changed = df[df["bucket"] != df["bucket_base"]]
        if not changed.empty:
            print(f"\n▶ 기본값 대비 버킷 변경: {len(changed)}쌍")
            for (b_from, b_to), grp in changed.groupby(["bucket_base", "bucket"]):
                print(f"  {b_from} → {b_to}: {len(grp)}쌍")
                sample = grp.head(3)
                for _, r in sample.iterrows():
                    print(f"    t_cos={r.t_cos:.4f} t_jac={r.t_jac:.4f} d_cos={r.d_cos:.4f} d_jac={r.d_jac:.4f}")
                    print(f"    A: {r.title_a[:70]}")
                    print(f"    B: {r.title_b[:70]}")

    # CSV 저장
    out = Path("scripts/threshold_test_result.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n✅ 결과 저장: {out}")


# ────────────────────────────────────────────────────────────────────────────
# Step 2: sweep
# ────────────────────────────────────────────────────────────────────────────

def cmd_sweep(args):
    if not CACHE_PATH.exists():
        print(f"❌ 캐시 없음. 먼저 실행: python scripts/threshold_tuner.py precompute")
        return

    df = pd.read_csv(CACHE_PATH, encoding="utf-8-sig")
    total = len(df)
    print(f"캐시 로드: {total}쌍\n")

    t_cos_range = [float(x) for x in args.t_cos_range.split(",")]
    t_jac_range = [float(x) for x in args.t_jac_range.split(",")]
    d_cos_range = [float(x) for x in args.d_cos_range.split(",")]
    d_jac_range = [float(x) for x in args.d_jac_range.split(",")]

    combos = list(product(t_cos_range, t_jac_range, d_cos_range, d_jac_range))
    print(f"총 {len(combos)}개 조합 테스트\n")

    header = f"{'t_cos':>6} {'t_jac':>6} {'d_cos':>6} {'d_jac':>6}  |  {'auto_t':>7} {'auto_d':>7} {'brd_t':>7} {'brd_d':>7} {'no_mrg':>8}  |  {'병합%':>6}"
    print(header)
    print("─" * len(header))

    rows_out = []
    for t_cos, t_jac, d_cos, d_jac in combos:
        cfg = {
            **DEFAULT,
            "t_cos": t_cos, "t_jac": t_jac,
            "d_cos": d_cos, "d_jac": d_jac,
            "border_t_cos_hi": t_cos,
            "border_d_cos_hi": d_cos,
        }
        buckets = df.apply(lambda r: classify(r, cfg), axis=1)
        counts = buckets.value_counts()
        auto_t = counts.get("auto_title", 0)
        auto_d = counts.get("auto_desc", 0)
        brd_t  = counts.get("border_title", 0)
        brd_d  = counts.get("border_desc", 0)
        no_m   = counts.get("no_merge", 0)
        merge_pct = (auto_t + auto_d) / total * 100

        mark = " ◀ 현재" if (t_cos == DEFAULT["t_cos"] and t_jac == DEFAULT["t_jac"]
                              and d_cos == DEFAULT["d_cos"] and d_jac == DEFAULT["d_jac"]) else ""
        print(f"{t_cos:>6.2f} {t_jac:>6.2f} {d_cos:>6.2f} {d_jac:>6.2f}  |  "
              f"{auto_t:>7} {auto_d:>7} {brd_t:>7} {brd_d:>7} {no_m:>8}  |  {merge_pct:>5.1f}%{mark}")

        rows_out.append({
            "t_cos": t_cos, "t_jac": t_jac, "d_cos": d_cos, "d_jac": d_jac,
            "auto_title": auto_t, "auto_desc": auto_d,
            "border_title": brd_t, "border_desc": brd_d, "no_merge": no_m,
            "merge_pct": round(merge_pct, 2),
        })

    out = Path("scripts/threshold_sweep_result.csv")
    pd.DataFrame(rows_out).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n✅ 스윕 결과 저장: {out}")


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cross-query 임계값 튜닝 도구")
    sub = parser.add_subparsers(dest="cmd")

    # precompute
    p_pre = sub.add_parser("precompute", help="Sheets 로드 + 페어 점수 캐시 생성")
    p_pre.add_argument("--sheet", default="total_result")
    p_pre.add_argument("--max_rows", type=int, default=3000)

    # test
    p_test = sub.add_parser("test", help="임계값 테스트 (샘플 + 통계)")
    p_test.add_argument("--t_cos", type=float, default=None)
    p_test.add_argument("--t_jac", type=float, default=None)
    p_test.add_argument("--d_cos", type=float, default=None)
    p_test.add_argument("--d_jac", type=float, default=None)
    p_test.add_argument("--border_t_cos_lo", type=float, default=None)
    p_test.add_argument("--border_t_cos_hi", type=float, default=None)
    p_test.add_argument("--border_t_jac_min", type=float, default=None)
    p_test.add_argument("--border_d_cos_lo", type=float, default=None)
    p_test.add_argument("--border_d_cos_hi", type=float, default=None)
    p_test.add_argument("--border_d_jac_min", type=float, default=None)

    # sweep
    p_sw = sub.add_parser("sweep", help="여러 임계값 조합 요약 테이블")
    p_sw.add_argument("--t_cos_range", default="0.55,0.60,0.65,0.70")
    p_sw.add_argument("--t_jac_range", default="0.10,0.15,0.20")
    p_sw.add_argument("--d_cos_range", default="0.50,0.55,0.60")
    p_sw.add_argument("--d_jac_range", default="0.08,0.12")

    args = parser.parse_args()

    if args.cmd == "precompute":
        cmd_precompute(args)
    elif args.cmd == "test":
        cmd_test(args)
    elif args.cmd == "sweep":
        cmd_sweep(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
