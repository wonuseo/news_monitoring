# 뉴스 모니터링 시스템

네이버 뉴스 API로 호텔 브랜드 관련 기사를 수집하고, AI로 감정·카테고리·위험도를 분석하여 Excel + Word 리포트를 생성합니다.

![architecture](./news_monitoring_architecture.png)

## ✨ 주요 기능

- 📰 **자동 수집**: 네이버 뉴스 API로 브랜드 검색 (API 페이지네이션으로 9배 더 많은 기사)
- 🏢 **언론사 분류**: OpenAI로 자동 분류 (도메인 → 언론사명/그룹/유형)
- 🤖 **하이브리드 분석 시스템**:
  - **Rule-Based**: 정규식 패턴 매칭 (즉시, 전체 기사)
  - **LLM**: OpenAI GPT-4o-mini (선택적, 우리 브랜드 + 경쟁사 상위 N개)
  - **3단계 조정**: RB → LLM → Final (의사결정 투명성 확보)
  - **4개 차원 분석**:
    - Sentiment (4단계): POSITIVE / NEUTRAL / NEGATIVE_CANDIDATE / NEGATIVE_CONFIRMED
    - Danger (3등급): D1 / D2 / D3
    - Issue Category (11개): Safety, Legal, Security, Customer Dispute, etc.
    - Coverage Themes (최대 2개): Business, Risk/Crisis, Marketing, etc.
- 📊 **CSV 출력**: raw.csv, result.csv (UTF-8 BOM, Looker Studio 호환)
- 📄 **Word 리포트**: 위험도별 구조화된 문서
- ☁️ **Google Sheets 동기화**: 증분 업로드 (중복 제거)

## 🗂️ 프로젝트 구조

```
├── main.py                      # 메인 실행 파일
├── src/modules/
│   ├── collection/              # 1. 뉴스 수집
│   │   ├── collect.py           #    - 네이버 API 페이지네이션
│   │   └── scrape.py            #    - 브라우저 스크래핑 (선택)
│   ├── processing/              # 2. 데이터 처리
│   │   ├── process.py           #    - 정규화, 중복제거, TF-IDF 유사도
│   │   ├── media_classify.py    #    - 언론사 분류 (OpenAI)
│   │   ├── fulltext.py          #    - 전문 스크래핑 (선택)
│   │   └── looker_prep.py       #    - Looker Studio 컬럼
│   ├── analysis/                # 3. 하이브리드 분석
│   │   ├── hybrid.py            #    - 오케스트레이터
│   │   ├── rule_engine.py       #    - 정규식 패턴 엔진
│   │   ├── llm_engine.py        #    - OpenAI 엔진
│   │   ├── rules.yaml           #    - Rule-Based 설정
│   │   └── prompts.yaml         #    - LLM 프롬프트 설정
│   └── export/                  # 4. 리포트 생성
│       ├── report.py            #    - CSV + Word
│       └── sheets.py            #    - Google Sheets 동기화
├── .env                         # API 키 설정
└── data/                        # 출력 디렉토리
    ├── raw.csv
    ├── result.csv
    ├── media_directory.csv
    └── report.docx
```

## 🚀 빠른 시작

### 1. 의존성 설치

```bash
pip install requests python-dotenv pandas openpyxl python-docx
```

### 2. API 키 설정

`.env` 파일 생성:
```bash
NAVER_CLIENT_ID=your_naver_client_id
NAVER_CLIENT_SECRET=your_naver_client_secret
OPENAI_API_KEY=sk-your_openai_api_key
```

**API 키 발급:**
- 네이버: https://developers.naver.com/apps/#/register
- OpenAI: https://platform.openai.com/api-keys

### 3. 실행

```bash
python main.py
```

## 📋 사용 예시

### 기본 실행
```bash
python main.py
```

### 더 많은 기사 수집
```bash
python main.py --display 200
```

### 경쟁사 분석 강화
```bash
python main.py --max_competitor_classify 50
```

### 타임아웃 방지 (청크 크기 조정)
```bash
python main.py --chunk_size 50
```

### 테스트 실행 (AI 분류 생략)
```bash
python main.py --dry_run
```

### 모든 옵션
```bash
python main.py \
  --display 100 \
  --sort date \
  --outdir reports \
  --max_competitor_classify 20 \
  --chunk_size 100
```

## 🎯 하이브리드 분석 프로세스

### 분석 전략

**Rule-Based + LLM 하이브리드 접근:**
- **Rule-Based (RB)**: 정규식 패턴 매칭 → 전체 기사, 즉시 (0.1초/기사)
- **LLM**: OpenAI GPT-4o-mini → 선택적 (우리 브랜드 전체 + 경쟁사 상위 N개)
- **Final**: RB vs LLM 조정 → 의사결정 규칙 기록 (투명성)

### 4개 차원 분석

#### 1. Sentiment (감정)
**4단계 분류:**
- **POSITIVE**: 수상, 극찬, 1위, 선정, 추천
- **NEUTRAL**: 일반 뉴스, 사실 전달
- **NEGATIVE_CANDIDATE**: 의혹, 논란 제기, 조사 착수, 예약 오류
- **NEGATIVE_CONFIRMED**: 사고, 화재, 기소, 개인정보 유출, 식중독

**3단계 프로세스:**
1. `sentiment_rb` - Rule-Based 판단 (정규식 우선순위)
2. `sentiment_llm` - LLM 독립 판단 (confidence, evidence, rationale)
3. `sentiment_final` - 최종 조정 (decision_rule: KEEP_RB / KEEP_LLM / RECALL_UPGRADE)

#### 2. Danger (위험도)
**3등급 분류 (BRAND_TARGETED + NEGATIVE만):**
- **D3 (🔴)**: 대중 성명 필요 (사망, 대형화재, 기소, 랜섬웨어, 영업정지)
- **D2 (🟡)**: 지속 모니터링 필요 (시스템 장애, 환불 분쟁, 논란 확산)
- **D1 (🟢)**: 경미한 부정 이슈 (단일 불만, 확산 없음)

**3단계 프로세스:**
1. `danger_rb` - Rule-Based 점수 계산 (hard_trigger, high_risk_category, attribution, amplification)
2. `danger_llm` - LLM 독립 판단 (severity, attribution, momentum)
3. `danger_final` - 최종 조정 (hard_trigger override for D3)

#### 3. Issue Category (이슈 카테고리)
**11개 카테고리 (1개 선택):**
- **Safety / Incident**: 사고, 화재, 부상, 사망
- **Hygiene / Food**: 위생, 식중독, 이물질
- **Security / Privacy / IT**: 개인정보 유출, 해킹, 시스템 장애
- **Legal / Regulation**: 수사, 기소, 소송, 제재
- **Customer Dispute**: 환불, 보상, 민원, 불만
- **Service Quality / Operations**: 서비스, 운영 미흡
- **Pricing / Commercial**: 요금, 바가지, 가격 논란
- **Labor / HR**: 노조, 파업, 갑질
- **Governance / Ethics**: 비리, 횡령, 은폐
- **Reputation / PR**: 논란, 여론, 불매
- **OTHER**: 위 카테고리에 해당 없음

**3단계 프로세스:**
1. `issue_category_rb` - Rule-Based 점수 기반 top 1
2. `issue_category_llm` - LLM 독립 판단 (playbook-driven)
3. `issue_category_final` - 최종 조정 (PLAYBOOK_TIE_BREAK)

#### 4. Coverage Themes (커버리지 테마)
**8개 테마 (최대 2개 선택):**
- **Business / Performance**: 실적, 매출, 투자, 확장
- **Brand / Marketing**: 캠페인, 광고, 수상, 랭킹
- **Product / Offering**: 패키지, 신메뉴, 객실, 시설
- **Customer Experience**: 만족, 리뷰, 후기
- **Operations / Technology**: AI, 디지털, 시스템, 운영
- **People / Organization**: 인사, 조직개편, 채용
- **Risk / Crisis**: 사고, 수사, 논란, 파장
- **ESG / Social**: 환경, 기부, 사회공헌
- **OTHER**: 위 테마에 해당 없음

**3단계 프로세스:**
1. `coverage_themes_rb` - Rule-Based 점수 기반 top 2
2. `coverage_themes_llm` - LLM 독립 판단 (max 2)
3. `coverage_themes_final` - 최종 조정 (max 2)

### 비용 효율

**선택적 LLM 분석:**
- 우리 브랜드: 전체 기사 LLM 분석
- 경쟁사: 최신 N개만 LLM 분석 (기본값: 50개)
- Rule-Based는 항상 전체 기사 분석 (비용 없음)

**비용 절감:**
- Rule-Based로 기본 분류 → LLM으로 정밀 조정
- 청크 단위 처리 (기본 100개) → API 호출 97% 감소

## 📊 출력 파일

### CSV 파일

1. **raw.csv**: 네이버 API에서 수집한 원본 데이터 (UTF-8 BOM)
2. **result.csv**: 하이브리드 분석 결과 (UTF-8 BOM, Looker Studio 호환)
   - Rule-Based 컬럼: `*_rb` (brand_scope_rb, sentiment_rb, danger_rb, etc.)
   - LLM 컬럼: `*_llm` (sentiment_llm, danger_llm, issue_category_llm, etc.)
   - Final 컬럼: `*_final` (sentiment_final, danger_final, issue_category_final, etc.)
   - 메타데이터: confidence, decision_rule, evidence, rationale
3. **media_directory.csv**: 언론사 디렉토리 (자동 업데이트, 지속)

### Google Sheets (선택)

`--sheets` 플래그 사용 시:
- **raw_data 탭**: 원본 데이터
- **result 탭**: 분류 결과 (모든 하이브리드 분석 컬럼 포함)
- 증분 업로드: 중복 자동 제거 (link 기준)

### Word 문서 (report.docx)

5개 섹션으로 구성:
1. **긴급 대응 필요 (Danger: D3)** - 🔴
2. **모니터링 필요 (Danger: D2)** - 🟡
3. **경미한 이슈 (Danger: D1)** - 🟢
4. **긍정 뉴스 (POSITIVE)** - 😊
5. **경쟁사 동향**

각 기사별로:
- 브랜드명, 최종 카테고리 (issue_category_final)
- 제목, 최종 설명 (sentiment_final_rationale, danger_final_rationale)
- 날짜, 링크

## ⚙️ 설정 변경

### 브랜드 수정

`src/modules/collection/collect.py` 파일에서:
```python
OUR_BRANDS = ["롯데호텔", "호텔롯데", "L7", "시그니엘"]
COMPETITORS = ["신라호텔", "조선호텔"]
```

### 하이브리드 분석 설정

**Rule-Based 설정** (`src/modules/analysis/rules.yaml`):
```yaml
# 브랜드 정의
brands:
  our: ["롯데호텔", "호텔롯데", "시그니엘", "L7"]
  competitors: ["신라호텔", "조선호텔"]

# Sentiment 정규식 패턴
sentiment:
  positive_triggers_regex:
    - "(?i)수상|선정|1위|최고|호평"
  negative_confirmed_triggers:
    incident_regex: ["(?i)사고|화재|붕괴|대피"]
    legal_reg_regex: ["(?i)수사|기소|고소|고발"]
    # ...

# Danger 점수 계산
danger:
  thresholds:
    D3: { score_min: 50, hard_trigger_override: true }
    D2: { score_min: 20 }
    D1: { score_min: 0 }
  score_components:
    hard_trigger: { points: 50, regex: [...] }
    high_risk_category: { points: 20 }
    # ...

# Issue Category / Coverage Themes
categorization:
  issue_category:
    categories:
      "Safety / Incident": { score: 30, regex: [...] }
      "Legal / Regulation": { score: 30, regex: [...] }
      # ...
```

**LLM 프롬프트 설정** (`src/modules/analysis/prompts.yaml`):
```yaml
# 정책 텍스트
policy_text:
  sentiment: |
    POSITIVE / NEUTRAL / NEGATIVE_CANDIDATE / NEGATIVE_CONFIRMED.
    Recall-first: false positives acceptable; avoid false negatives.
  danger: |
    Danger means response necessity.
    D3: public statement likely needed
    D2: continuous monitoring required
    D1: minor negative issue
  category: |
    ONE Issue Category + up to TWO Coverage Themes.
    Prefer most operationally actionable playbook.

# 프롬프트 템플릿
prompts:
  sentiment_llm:
    system: "You are a risk monitoring analyst..."
    user: "POLICY: {{policy_sentiment}}\nARTICLE: {{title}}..."
  sentiment_final:
    system: "You are the final arbiter..."
    # ...
```

**재학습 불필요**: YAML 파일 수정만으로 로직 변경 가능

## 💰 비용 효율

### 하이브리드 전략
- **Rule-Based**: 전체 기사, 무료, 즉시
- **LLM**: 선택적 (우리 브랜드 + 경쟁사 상위 N개)
- **예시**: 5400개 수집 → 우리 브랜드 900개 + 경쟁사 100개 = 1000개만 LLM 분석

### LLM 분석 단계
각 기사당 최대 6번 API 호출:
1. sentiment_llm (1 call)
2. sentiment_final (1 call)
3. danger_llm (1 call, 조건부)
4. danger_final (1 call, 조건부)
5. category_llm (1 call)
6. category_final (1 call)

**실제**: 대부분 4-5번 (danger는 BRAND_TARGETED + NEGATIVE만)

### 청크 처리
- 기본 청크 크기: 100개
- 1000개 기사 → 10개 청크
- Rate limiting: 0.5초/기사
- **총 소요 시간**: ~8-10분 (1000개 기사 LLM 분석)

### 비용 절감
- Rule-Based로 기본 필터링 → LLM은 고가치 기사만
- 경쟁사는 최신 N개만 분석 (기본 50개)
- **~80% 비용 절감** (vs 전체 LLM 분석)

## 🔧 문제 해결

### 인증 오류
- **401 (네이버)**: `NAVER_CLIENT_ID`와 `NAVER_CLIENT_SECRET` 확인
- **401 (OpenAI)**: `OPENAI_API_KEY` 확인

### 타임아웃 오류
```bash
# 청크 크기 줄이기
python main.py --chunk_size 50

# 또는 더 작게
python main.py --chunk_size 30
```

### Rate Limit 오류
- **429 (네이버)**: 0.1초 대기가 내장되어 있음
- **429 (OpenAI)**: 5초 대기 후 재시도, 필요시 `--chunk_size` 줄이기

### 결과가 없을 때
- 검색어 철자 확인 (한글 표기)
- `--display 200`으로 더 많은 기사 수집
- 네이버 API 할당량 확인

## 📈 성능

- **수집**: ~6초 (6개 브랜드 × 100개)
- **처리**: ~1초
- **AI 분류**: ~30초 (365개 기사, 청크 크기 100)
- **리포트**: ~2초
- **총 소요 시간**: ~40초

## 📝 라이센스

MIT

## 🤝 기여

이슈와 풀 리퀘스트를 환영합니다!
