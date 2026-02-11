"""
test_llm_quality.py - LLM 분류 품질 검증 테스트
brand_relevance, sentiment_stage, danger_level, issue_category, news_category 전체 검증

실행: python tests/test_llm_quality.py
필요: OPENAI_API_KEY 환경변수 또는 .env 파일
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 PYTHONPATH에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src.modules.analysis.llm_engine import (
    load_prompts,
    analyze_article_llm,
    _post_process_result,
)


# ============================================================
# 테스트 케이스 정의
# expected 필드: 반드시 일치해야 하는 값
# acceptable 필드: 여러 값 중 하나면 통과 (유연 검증)
# None = 해당 필드 검증 생략
# ============================================================
TEST_CASES = [
    # ──────────────────────────────────────────────
    # A. brand_relevance 테스트 (기존 4건)
    # ──────────────────────────────────────────────
    {
        "id": 1,
        "category": "brand_relevance",
        "description": "무관 - 장소 지명 (TS손보 캠페인)",
        "article": {
            "query": "롯데호텔",
            "group": "OUR",
            "title": "TS손보, 교통사고 예방 캠페인...울산 롯데호텔 앞 교차로",
            "description": "TS손해보험이 교통사고 예방을 위한 캠페인을 울산 롯데호텔 앞 교차로에서 진행했다. 보행자 안전 의식 제고를 위해 전국 주요 교차로에서 캠페인을 확대할 예정이다.",
        },
        "expected": {
            "brand_relevance": "무관",
            "sentiment_stage": "중립",     # 후처리 규칙: 무관→중립
            "danger_level": None,          # 후처리 규칙: 무관→null
            "issue_category": None,        # 후처리 규칙: 무관→null
        },
    },
    {
        "id": 2,
        "category": "brand_relevance",
        "description": "언급 - 계열사 나열 (롯데하이마트 재단장)",
        "article": {
            "query": "롯데호텔",
            "group": "OUR",
            "title": "롯데하이마트 잠실점 재단장...롯데호텔, 롯데백화점 등 롯데그룹 계열사 시너지",
            "description": "롯데하이마트가 잠실점을 대대적으로 재단장하며 롯데호텔, 롯데백화점 등 계열사와 시너지를 모색한다. 복합쇼핑공간으로 탈바꿈하여 고객 체류시간을 늘리겠다는 전략이다.",
        },
        "expected": {
            "brand_relevance": "언급",
        },
    },
    {
        "id": 3,
        "category": "brand_relevance",
        "description": "무관 - 행사장소 (동아대 성과교류회)",
        "article": {
            "query": "롯데호텔",
            "group": "OUR",
            "title": "동아대 RISE 성과교류회 개최...롯데호텔에서 열린",
            "description": "동아대학교가 RISE 사업 성과교류회를 부산 롯데호텔에서 개최했다. 산학협력 성과를 공유하고 향후 발전 방향을 논의했다.",
        },
        "expected": {
            "brand_relevance": "무관",
            "sentiment_stage": "중립",
            "danger_level": None,
            "issue_category": None,
        },
    },
    {
        "id": 4,
        "category": "brand_relevance",
        "description": "언급 - 종목 나열 (코스피 순환매)",
        "article": {
            "query": "호텔롯데",
            "group": "OUR",
            "title": "코스피 저평가 순환매...호텔신라(4.39%), 호텔롯데(2.1%)",
            "description": "코스피 저평가 종목에 대한 순환매가 이어지며 호텔신라가 4.39%, 호텔롯데가 2.1% 상승했다. 관광주 회복세가 두드러진다.",
        },
        "expected": {
            "brand_relevance": "언급",
        },
    },

    # ──────────────────────────────────────────────
    # B. sentiment_stage 테스트
    # ──────────────────────────────────────────────
    {
        "id": 5,
        "category": "sentiment",
        "description": "긍정 - 브랜드 수상/성과",
        "article": {
            "query": "롯데호텔",
            "group": "OUR",
            "title": "롯데호텔, 아시아 최고 비즈니스 호텔 3년 연속 수상",
            "description": "롯데호텔이 글로벌 여행 전문지 선정 '아시아 최고 비즈니스 호텔'에 3년 연속 선정됐다. 서비스 품질과 접근성 등에서 높은 평가를 받았다.",
        },
        "expected": {
            "brand_relevance": "관련",
            "sentiment_stage": "긍정",
            "danger_level": None,    # 긍정이면 danger=null
            "issue_category": None,
        },
        "acceptable": {
            "news_category": ["브랜드/마케팅", "사업/실적"],
        },
    },
    {
        "id": 6,
        "category": "sentiment",
        "description": "중립/긍정 - 사실 보도 (인사 발령)",
        "article": {
            "query": "신라호텔",
            "group": "COMPETITOR",
            "title": "신라호텔, 신임 총지배인에 김철수 부사장 선임",
            "description": "호텔신라는 신라호텔 신임 총지배인에 김철수 부사장을 선임했다고 밝혔다. 김 부사장은 30년간 호텔업계에서 근무한 베테랑이다.",
        },
        "expected": {
            "brand_relevance": "관련",
            "danger_level": None,
            "issue_category": None,
            "news_category": "인사/조직",
        },
        "acceptable": {
            "sentiment_stage": ["중립", "긍정"],  # 인사발령은 중립/긍정 경계선
        },
    },
    {
        "id": 7,
        "category": "sentiment",
        "description": "부정 확정 - 명확한 고객 피해 + SNS 확산",
        "article": {
            "query": "롯데호텔",
            "group": "OUR",
            "title": "롯데호텔 투숙객 '객실서 바퀴벌레 발견'...SNS 확산",
            "description": "롯데호텔 서울 투숙객이 객실에서 바퀴벌레를 발견했다며 SNS에 사진을 올려 논란이 확산되고 있다. 해당 투숙객은 환불을 요구했으나 호텔 측은 일부 보상만 제안한 것으로 알려졌다.",
        },
        "expected": {
            "brand_relevance": "관련",
            "sentiment_stage": "부정 확정",
        },
        "acceptable": {
            "danger_level": ["상", "중"],  # SNS 확산 시그널로 상도 합리적
            "issue_category": ["위생/식음", "서비스 품질/운영", "고객 분쟁"],
            "news_category": ["리스크/위기", "고객 경험"],
        },
    },
    {
        "id": 8,
        "category": "sentiment",
        "description": "부정 후보 - 불확실한 부정 (소문/미확인)",
        "article": {
            "query": "조선호텔",
            "group": "COMPETITOR",
            "title": "조선호텔 '노조 설립 추진' 소문...회사 측 '확인 중'",
            "description": "조선호텔 직원들 사이에서 노조 설립 추진 움직임이 있다는 소문이 돌고 있다. 회사 측은 '사실 확인 중'이라고만 밝혔다. 구체적인 진행 상황은 아직 알려지지 않았다.",
        },
        "expected": {
            "brand_relevance": "관련",
        },
        "acceptable": {
            "sentiment_stage": ["부정 후보", "중립"],
            "news_category": ["인사/조직", "리스크/위기"],
        },
    },

    # ──────────────────────────────────────────────
    # C. danger_level 테스트
    # ──────────────────────────────────────────────
    {
        "id": 9,
        "category": "danger_level",
        "description": "위험도 상 - 대형 화재 사고",
        "article": {
            "query": "롯데호텔",
            "group": "OUR",
            "title": "롯데호텔 부산점 지하주차장 화재...투숙객 200명 긴급 대피",
            "description": "롯데호텔 부산점 지하주차장에서 대형 화재가 발생해 투숙객 200여명이 긴급 대피했다. 소방당국이 출동해 진화 작업을 벌이고 있으며, 부상자 5명이 인근 병원으로 이송됐다.",
        },
        "expected": {
            "brand_relevance": "관련",
            "sentiment_stage": "부정 확정",
            "danger_level": "상",
            "issue_category": "안전/사고",
            "news_category": "리스크/위기",
        },
    },
    {
        "id": 10,
        "category": "danger_level",
        "description": "위험도 상 - 개인정보 유출",
        "article": {
            "query": "신라호텔",
            "group": "COMPETITOR",
            "title": "신라호텔 고객 개인정보 10만건 유출...해킹 공격 확인",
            "description": "신라호텔이 해킹 공격을 받아 고객 개인정보 약 10만건이 유출된 것으로 확인됐다. 유출 정보에는 이름, 연락처, 예약내역이 포함됐으며 개인정보보호위원회가 조사에 착수했다.",
        },
        "expected": {
            "brand_relevance": "관련",
            "sentiment_stage": "부정 확정",
            "danger_level": "상",
            "issue_category": "보안/개인정보/IT",
            "news_category": "리스크/위기",
        },
    },
    {
        "id": 11,
        "category": "danger_level",
        "description": "위험도 중 - 고객 환불 분쟁 확산 조짐",
        "article": {
            "query": "시그니엘",
            "group": "OUR",
            "title": "시그니엘 부산 '예약 취소 수수료 과다' 고객 불만 잇따라",
            "description": "시그니엘 부산의 예약 취소 수수료 정책에 대한 고객 불만이 온라인 커뮤니티를 중심으로 확산되고 있다. 여러 고객이 숙박 7일 전 취소에도 50% 수수료가 부과됐다고 주장했다.",
        },
        "expected": {
            "brand_relevance": "관련",
            "sentiment_stage": "부정 확정",
        },
        "acceptable": {
            "danger_level": ["상", "중", "하"],  # "확산" 표현으로 상향 가능
            "issue_category": ["고객 분쟁", "가격/상업"],
            "news_category": ["리스크/위기", "고객 경험"],
        },
    },
    {
        "id": 12,
        "category": "danger_level",
        "description": "위험도 하 - 단발성 경미 불만",
        "article": {
            "query": "L7",
            "group": "OUR",
            "title": "L7 홍대 투숙 후기 '조식 기대 이하'...블로거 리뷰",
            "description": "여행 블로거가 L7 홍대 투숙 후기를 올리며 조식 메뉴가 기대에 미치지 못했다고 평가했다. 다만 객실 인테리어와 위치는 매우 만족스러웠다고 덧붙였다.",
        },
        "expected": {
            "brand_relevance": "관련",
        },
        "acceptable": {
            "sentiment_stage": ["중립", "부정 후보"],
            "news_category": ["고객 경험", "상품/오퍼링"],
        },
    },

    # ──────────────────────────────────────────────
    # D. news_category 테스트
    # ──────────────────────────────────────────────
    {
        "id": 13,
        "category": "news_category",
        "description": "사업/실적 - 분기 실적 발표",
        "article": {
            "query": "호텔롯데",
            "group": "OUR",
            "title": "호텔롯데, 2분기 영업이익 1,200억...전년比 15% 성장",
            "description": "호텔롯데가 2분기 연결 기준 영업이익 1,200억원을 기록하며 전년 동기 대비 15% 성장했다. 해외 호텔 부문과 면세점 실적 회복이 실적 개선을 이끌었다.",
        },
        "expected": {
            "brand_relevance": "관련",
            "sentiment_stage": "긍정",
            "news_category": "사업/실적",
            "danger_level": None,
            "issue_category": None,
        },
    },
    {
        "id": 14,
        "category": "news_category",
        "description": "상품/오퍼링 - 신규 패키지 출시",
        "article": {
            "query": "시그니엘",
            "group": "OUR",
            "title": "시그니엘 서울, 봄맞이 '체리블라썸 스테이' 패키지 출시",
            "description": "시그니엘 서울이 봄 시즌을 맞아 석촌호수 벚꽃 뷰를 즐길 수 있는 '체리블라썸 스테이' 패키지를 출시했다. 객실 업그레이드와 애프터눈티 세트가 포함된다.",
        },
        "expected": {
            "brand_relevance": "관련",
            "sentiment_stage": "긍정",
            "news_category": "상품/오퍼링",
            "danger_level": None,
            "issue_category": None,
        },
    },
    {
        "id": 15,
        "category": "news_category",
        "description": "ESG/사회 - 사회공헌 활동",
        "article": {
            "query": "롯데호텔",
            "group": "OUR",
            "title": "롯데호텔, 지역아동센터에 크리스마스 케이크 1,000개 기부",
            "description": "롯데호텔이 연말을 맞아 전국 지역아동센터에 베이커리에서 만든 크리스마스 케이크 1,000개를 기부했다. 10년째 이어온 사회공헌 활동이다.",
        },
        "expected": {
            "brand_relevance": "관련",
            "sentiment_stage": "긍정",
            "news_category": "ESG/사회",
            "danger_level": None,
            "issue_category": None,
        },
    },
    {
        "id": 16,
        "category": "news_category",
        "description": "법무/규제 - 공정위 과징금",
        "article": {
            "query": "롯데호텔",
            "group": "OUR",
            "title": "공정위, 롯데호텔에 과징금 5억원 부과...가격 담합 혐의",
            "description": "공정거래위원회가 롯데호텔을 포함한 주요 호텔 3사에 가격 담합 혐의로 과징금을 부과했다. 롯데호텔에는 5억원의 과징금이 부과됐으며, 호텔 측은 불복 의사를 밝혔다.",
        },
        "expected": {
            "brand_relevance": "관련",
            "sentiment_stage": "부정 확정",
            "issue_category": "법무/규제",
            "news_category": "리스크/위기",
        },
        "acceptable": {
            "danger_level": ["상", "중"],
        },
    },

    # ──────────────────────────────────────────────
    # E. 경쟁사 + 복합 판단 테스트
    # ──────────────────────────────────────────────
    {
        "id": 17,
        "category": "complex",
        "description": "경쟁사 관련 - 신라호텔 신규 오픈",
        "article": {
            "query": "신라호텔",
            "group": "COMPETITOR",
            "title": "신라호텔, 제주 신라 '더 파크' 리조트 그랜드 오픈",
            "description": "신라호텔이 제주도에 새로운 콘셉트의 '더 파크' 리조트를 정식 오픈했다. 자연 친화적 설계와 프리미엄 다이닝 시설을 갖췄으며, 글로벌 고객 유치에 나선다.",
        },
        "expected": {
            "brand_relevance": "관련",
            "sentiment_stage": "긍정",
            "danger_level": None,
            "issue_category": None,
        },
        "acceptable": {
            "news_category": ["사업/실적", "상품/오퍼링", "브랜드/마케팅"],
        },
    },
    {
        "id": 18,
        "category": "complex",
        "description": "경계선 - 브랜드 인근 개발 (수혜 가능)",
        "article": {
            "query": "롯데호텔",
            "group": "OUR",
            "title": "서울시, 명동 관광특구 활성화 프로젝트 발표...롯데호텔 인근 보행로 확장",
            "description": "서울시가 명동 관광특구 활성화를 위해 롯데호텔 인근 보행로를 확장하고 야간 조명을 설치한다고 밝혔다. 외국인 관광객 유치 확대가 목표다.",
        },
        "expected": {
            "danger_level": None,
            "issue_category": None,
        },
        "acceptable": {
            "brand_relevance": ["무관", "언급", "관련"],  # 수혜 해석에 따라 다양
            "sentiment_stage": ["중립", "긍정"],
        },
    },
]


def _check_field(result, field, expected_val, acceptable_vals):
    """
    단일 필드 검증

    Returns:
        (pass: bool, message: str)
    """
    actual = result.get(field)

    # expected가 설정되어 있으면 정확히 일치해야 함
    if expected_val is not None:
        # None과 None 비교
        if expected_val is None and actual is None:
            return True, f"{field}=None [PASS]"
        if actual == expected_val:
            return True, f"{field}={actual} [PASS]"
        return False, f"{field}={actual} (기대: {expected_val}) [FAIL]"

    # acceptable만 있으면 그 중 하나면 통과
    if acceptable_vals is not None:
        if actual in acceptable_vals:
            return True, f"{field}={actual} [PASS] (허용: {acceptable_vals})"
        return False, f"{field}={actual} (허용: {acceptable_vals}) [FAIL]"

    # 둘 다 없으면 검증 생략
    return True, None


def run_tests():
    """전체 테스트 실행"""
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        sys.exit(1)

    prompts_config = load_prompts()

    print("=" * 70)
    print("LLM 분류 품질 테스트 (전체 필드 검증)")
    print(f"모델: {prompts_config.get('model', 'api_models.yaml 참조')}")
    print(f"테스트 케이스: {len(TEST_CASES)}건")
    print("=" * 70)

    # 검증 대상 필드
    CHECK_FIELDS = [
        "brand_relevance",
        "sentiment_stage",
        "danger_level",
        "issue_category",
        "news_category",
    ]

    total_checks = 0
    passed_checks = 0
    failed_checks = 0
    test_passed = 0
    test_failed = 0
    all_results = []

    for tc in TEST_CASES:
        print(f"\n{'─' * 60}")
        print(f"[#{tc['id']}] {tc['category'].upper()} | {tc['description']}")
        print(f"  query={tc['article']['query']}  group={tc['article']['group']}")
        print(f"  title: {tc['article']['title'][:70]}")

        try:
            result = analyze_article_llm(tc["article"], prompts_config, openai_key)
        except Exception as e:
            print(f"  ERROR: {e}")
            test_failed += 1
            all_results.append({"id": tc["id"], "status": "ERROR", "error": str(e)})
            continue

        if result is None:
            print(f"  ERROR: LLM 응답 없음")
            test_failed += 1
            all_results.append({"id": tc["id"], "status": "ERROR", "error": "null"})
            continue

        # 전체 결과 출력
        print(f"  결과: relevance={result.get('brand_relevance')} | "
              f"sentiment={result.get('sentiment_stage')} | "
              f"danger={result.get('danger_level')} | "
              f"issue={result.get('issue_category')}")
        print(f"        news_cat={result.get('news_category')} | "
              f"keywords={result.get('news_keyword_summary')}")

        # 필드별 검증
        expected = tc.get("expected", {})
        acceptable = tc.get("acceptable", {})
        tc_pass = True
        tc_checks = []

        for field in CHECK_FIELDS:
            exp_val = expected.get(field)
            acc_val = acceptable.get(field)

            # expected에 필드가 있거나 acceptable에 있을 때만 검증
            if field in expected or field in acceptable:
                ok, msg = _check_field(result, field, exp_val, acc_val)
                total_checks += 1
                if ok:
                    passed_checks += 1
                else:
                    failed_checks += 1
                    tc_pass = False
                if msg:
                    tc_checks.append(("  " + ("  " if ok else "  ") + msg))

        for msg in tc_checks:
            print(msg)

        if tc_pass:
            test_passed += 1
            print(f"  => PASS")
        else:
            test_failed += 1
            print(f"  => FAIL")

        all_results.append({
            "id": tc["id"],
            "status": "PASS" if tc_pass else "FAIL",
            "result": result,
        })

    # ──────────────────────────────────────────────
    # 최종 요약
    # ──────────────────────────────────────────────
    total_tests = test_passed + test_failed
    print(f"\n{'=' * 70}")
    print(f"테스트 결과 요약")
    print(f"{'=' * 70}")
    print(f"  테스트 케이스: {test_passed}/{total_tests} 통과"
          f" ({test_passed/total_tests*100:.0f}%)" if total_tests > 0 else "")
    print(f"  필드 검증:     {passed_checks}/{total_checks} 통과"
          f" ({passed_checks/total_checks*100:.0f}%)" if total_checks > 0 else "")

    if test_failed > 0:
        print(f"\n  실패한 테스트:")
        for r in all_results:
            if r["status"] in ("FAIL", "ERROR"):
                print(f"    - #{r['id']}: {r['status']}")

    # 카테고리별 통과율
    categories = {}
    for tc, r in zip(TEST_CASES, all_results):
        cat = tc["category"]
        if cat not in categories:
            categories[cat] = {"pass": 0, "fail": 0}
        if r["status"] == "PASS":
            categories[cat]["pass"] += 1
        else:
            categories[cat]["fail"] += 1

    print(f"\n  카테고리별:")
    for cat, counts in categories.items():
        total = counts["pass"] + counts["fail"]
        print(f"    {cat}: {counts['pass']}/{total}")

    print(f"{'=' * 70}")
    return test_failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
