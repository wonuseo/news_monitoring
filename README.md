# 뉴스 모니터링 시스템

네이버 뉴스 API로 호텔 브랜드 관련 기사를 수집하고, AI로 감정·카테고리·위험도를 분석하여 Google Sheets + CSV + Word 리포트를 생성합니다.

![architecture](resources/news_monitoring_architecture.png)

## ✨ 주요 기능

- 📰 **자동 수집**: 네이버 뉴스 API로 브랜드 검색 (API 페이지네이션으로 9배 더 많은 기사)
- 🏢 **언론사 분류**: OpenAI로 자동 분류 (도메인 → 언론사명/그룹/유형)
- 🤖 **LLM 기반 분석 시스템**:
  - **OpenAI GPT-4o-mini**: 구조화된 출력 (Structured Output)
  - **선택적 분석**: 우리 브랜드 전체 + 경쟁사 전체 (기본값)
  - **설정 기반**: prompts.yaml 수정으로 로직 변경 (재학습 불필요)
  - **6개 차원 분석**:
    - Brand Relevance (4단계): 관련 / 언급 / 무관 / 판단 필요
    - Sentiment (4단계): 긍정 / 중립 / 부정 후보 / 부정 확정
    - Danger (3등급): 상 / 중 / 하
    - Issue Category (11개): 안전/사고, 법무/규제, 보안/개인정보 등
    - News Category (9개): 사업/실적, 브랜드/마케팅, 제품/서비스 등
    - Keyword Summary: 5단어 한글 요약
- ☁️ **Google Sheets (주 저장소)**: 자동 동기화, 증분 업로드 (중복 제거)
- 📊 **CSV 백업**: raw.csv, result.csv (troubleshooting 용도, UTF-8 BOM)
- 📄 **Word 리포트**: 위험도별 구조화된 문서

## 🗂️ 프로젝트 구조

```
├── main.py                      # 메인 실행 파일
├── src/modules/
│   ├── collection/              # 1. 뉴스 수집
│   │   ├── collect.py           #    - 네이버 API 페이지네이션
│   │   └── scrape.py            #    - 브라우저 스크래핑 (선택)
│   ├── processing/              # 2. 데이터 처리
│   │   ├── process.py           #    - 정규화, 중복제거, CSV I/O
│   │   ├── press_release_detector.py # - 보도자료 탐지 및 요약
│   │   ├── media_classify.py    #    - 언론사 분류 (OpenAI)
│   │   ├── fulltext.py          #    - 전문 스크래핑 (선택)
│   │   └── looker_prep.py       #    - 시계열 컬럼 (선택)
│   ├── analysis/                # 3. LLM 분석
│   │   ├── classify_llm.py      #    - LLM 분류기 (메인)
│   │   ├── llm_engine.py        #    - OpenAI Structured Output 엔진
│   │   ├── preset_pr.py         #    - 보도자료 사전 설정 (비용 절감)
│   │   ├── keyword_extractor.py #    - 키워드 추출 (kiwipiepy + Log-odds)
│   │   └── prompts.yaml         #    - LLM 프롬프트 및 스키마
│   ├── monitoring/              # 4. 모니터링
│   │   └── logger.py            #    - 실행 메트릭 로깅 (CSV + Sheets)
│   └── export/                  # 5. 리포트 생성
│       ├── report.py            #    - CSV + Word
│       └── sheets.py            #    - Google Sheets 동기화
├── .env                         # API 키 설정
└── data/                        # 출력 디렉토리
    ├── raw.csv
    ├── result.csv
    ├── media_directory.csv
    ├── keywords/                # 카테고리별 키워드 (자동)
    ├── logs/run_history.csv     # 실행 히스토리 (자동)
    └── report.docx
```

## 🚀 빠른 시작

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. API 키 및 Google Sheets 설정

`.env` 파일 생성:
```bash
# 네이버 API (필수)
NAVER_CLIENT_ID=your_naver_client_id
NAVER_CLIENT_SECRET=your_naver_client_secret

# OpenAI API (필수)
OPENAI_API_KEY=sk-your_openai_api_key

# Google Sheets (권장 - 주 저장소)
GOOGLE_SHEETS_CREDENTIALS_PATH=.secrets/your-service-account.json
GOOGLE_SHEET_ID=your_google_sheet_id
```

**API 키 발급:**
- 네이버: https://developers.naver.com/apps/#/register
- OpenAI: https://platform.openai.com/api-keys
- Google Sheets: [Service Account 생성](https://console.cloud.google.com/iam-admin/serviceaccounts)

**Google Sheets 설정 (권장):**
1. Google Cloud Console에서 Service Account 생성
2. JSON 키 다운로드 → `.secrets/` 폴더에 저장
3. Google Sheet 생성 후 Service Account 이메일에 편집 권한 부여
4. Sheet ID를 `.env`에 추가

### 3. 실행

```bash
python main.py
```

**참고:** Google Sheets 설정이 없으면 CSV 파일만 생성됩니다.

## 📋 사용 예시

### 기본 실행 (100개 기사/브랜드)
```bash
python main.py
```

### API 페이지네이션 (권장, 900개 기사/브랜드)
```bash
python main.py --max_api_pages 9
```

### 경쟁사 분석 제한 (비용 절감)
```bash
python main.py --max_competitor_classify 20
```

### 타임아웃 방지 (청크 크기 조정)
```bash
python main.py --chunk_size 50
```

### 테스트 실행 (AI 분류 생략, 10개만 표시)
```bash
python main.py --dry_run --display 10
```

### 수집만 하기 (분류 생략, Sheets 자동 동기화)
```bash
python main.py --raw_only
```

### 전처리까지만 (분류 생략, Sheets 동기화)
```bash
python main.py --preprocess_only
```

### 키워드 추출 개수 조정 (기본: 20개, 자동 실행)
```bash
python main.py --keyword_top_k 30
```

### 브라우저 스크래핑 (날짜 범위 지정)
```bash
python main.py --scrape --start_date 2026-01-01 --end_date 2026-02-08
```

### 전문 추출 (고/중 위험도만)
```bash
python main.py --fulltext --fulltext_risk_levels 상,중
```

### 모든 옵션
```bash
python main.py \
  --max_api_pages 9 \
  --max_competitor_classify 20 \
  --chunk_size 100 \
  --extract_keywords \
  --outdir reports
```

## 🎯 LLM 분석 프로세스

### 분석 전략

**LLM 기반 구조화된 출력:**
- **모델**: OpenAI GPT-4o-mini (prompts.yaml 설정)
- **전체 분석**: 우리 브랜드 전체 + 경쟁사 전체 (기본값)
  - `--max_competitor_classify N`으로 경쟁사 제한 가능
- **보도자료**: 사전 설정값 적용 (LLM 생략, 비용 절감)
- **설정 기반**: prompts.yaml만 수정하면 로직 변경 (재학습 불필요)

### 6개 차원 분석

#### 1. Brand Relevance (브랜드 관련성)
**4단계 분류:**
- **관련**: 우리 브랜드가 기사의 핵심 주제
- **언급**: 우리 브랜드가 부분적으로 언급됨
- **무관**: 우리 브랜드와 무관
- **판단 필요**: 애매한 경우

**출력 컬럼:**
- `brand_relevance`: "관련" / "언급" / "무관" / "판단 필요"
- `brand_relevance_query_keywords`: 관련 키워드 배열

#### 2. Sentiment (감정)
**4단계 분류:**
- **긍정**: 수상, 극찬, 1위, 선정, 추천
- **중립**: 일반 뉴스, 사실 전달
- **부정 후보**: 의혹, 논란 제기, 조사 착수, 예약 오류
- **부정 확정**: 사고, 화재, 기소, 개인정보 유출, 식중독

**출력 컬럼:**
- `sentiment_stage`: "긍정" / "중립" / "부정 후보" / "부정 확정"

#### 3. Danger (위험도)
**3등급 분류 (브랜드 관련 + 부정일 때만):**
- **상 (🔴)**: 대중 성명 필요 (사망, 대형화재, 기소, 랜섬웨어, 영업정지)
- **중 (🟡)**: 지속 모니터링 필요 (시스템 장애, 환불 분쟁, 논란 확산)
- **하 (🟢)**: 경미한 부정 이슈 (단일 불만, 확산 없음)

**출력 컬럼:**
- `danger_level`: "상" / "중" / "하" / null (관련 + 부정일 때만)

#### 4. Issue Category (이슈 카테고리)
**11개 카테고리 (1개 선택):**
- **안전/사고**: 화재, 부상, 사망
- **위생/식품**: 식중독, 이물질
- **보안/개인정보/IT**: 해킹, 유출, 시스템 장애
- **법무/규제**: 수사, 기소, 소송
- **고객 분쟁**: 환불, 보상, 민원
- **서비스품질/운영**: 서비스 미흡
- **가격/상업**: 요금 논란
- **노동/인사**: 노조, 파업
- **지배구조/윤리**: 비리, 횡령
- **평판/PR**: 논란, 여론
- **기타**: 위 카테고리 해당 없음

**출력 컬럼:**
- `issue_category`: 11개 중 1개 (부정일 때만) 또는 null

#### 5. News Category (뉴스 카테고리)
**9개 카테고리 (1개 선택):**
- **사업/실적**: 매출, 투자, 확장
- **브랜드/마케팅**: 캠페인, 광고, 수상
- **제품/서비스**: 패키지, 신메뉴, 객실
- **고객경험**: 만족, 리뷰
- **운영/기술**: AI, 디지털, 시스템
- **인사/조직**: 조직개편, 채용
- **리스크/위기**: 사고, 수사, 논란
- **ESG/사회**: 환경, 기부
- **기타**: 위 카테고리 해당 없음

**출력 컬럼:**
- `news_category`: 9개 중 1개

#### 6. Keyword Summary (키워드 요약)
**5단어 한글 요약:**
- LLM이 기사 내용을 5개 단어로 요약
- 예: "롯데호텔 제주 리조트 오픈"

**출력 컬럼:**
- `news_keyword_summary`: 5단어 한글 요약

### 비용 효율

**선택적 LLM 분석:**
- 우리 브랜드: 전체 기사 분석
- 경쟁사: 전체 기사 분석 (기본값)
  - `--max_competitor_classify N`으로 제한 가능
- 보도자료: 사전 설정값 적용 (LLM 생략)

**비용 절감:**
- 보도자료 사전 설정 → LLM 비용 절감
- 청크 단위 처리 (기본 100개) → API 호출 최적화
- 경쟁사 제한 옵션 → 추가 비용 절감

## 📊 출력 파일

### Google Sheets (주 저장소)

**.env 설정 시 자동 동기화:**
- **raw_data 탭**: 원본 데이터 (네이버 API 수집 결과)
- **result 탭**: LLM 분류 결과 (모든 분석 컬럼 포함)
- **logs 탭**: 실행 히스토리 (메트릭 추적)
- **keywords 탭**: 카테고리별 키워드 (자동)
- **증분 업로드**: 중복 자동 제거 (link 기준)
- **실시간 협업**: 팀원과 공유 및 실시간 업데이트

### CSV 파일 (백업/Troubleshooting)

1. **raw.csv**: 네이버 API에서 수집한 원본 데이터 (UTF-8 BOM)
   - 컬럼: title, description, link, originallink, pubDate, query, group
2. **result.csv**: LLM 분류 결과 (UTF-8 BOM)
   - 수집 컬럼: title, description, link, pub_datetime, query, group
   - 처리 컬럼: article_id, article_no, source, cluster_id, cluster_summary
   - 언론사 컬럼: media_domain, media_name, media_group, media_type
   - LLM 컬럼: brand_relevance, brand_relevance_query_keywords, sentiment_stage, danger_level, issue_category, news_category, news_keyword_summary, classified_at
3. **media_directory.csv**: 언론사 디렉토리 (자동 업데이트, 지속)
4. **logs/run_history.csv**: 실행 히스토리 (자동, 34개 메트릭)
5. **keywords/*.csv**: 카테고리별 키워드 (자동)

### Word 문서 (report.docx)

5개 섹션으로 구성:
1. **긴급 대응 필요 (위험도: 상)** - 🔴
2. **모니터링 필요 (위험도: 중)** - 🟡
3. **경미한 이슈 (위험도: 하)** - 🟢
4. **긍정 뉴스** - 😊
5. **경쟁사 동향**

각 기사별로:
- 브랜드명, 이슈 카테고리 (issue_category)
- 제목, 키워드 요약 (news_keyword_summary)
- 감정 단계 (sentiment_stage), 위험도 (danger_level)
- 날짜, 링크

## ⚙️ 설정 변경

### 브랜드 수정

`src/modules/collection/collect.py` 파일에서:
```python
OUR_BRANDS = ["롯데호텔", "호텔롯데", "L7", "시그니엘"]
COMPETITORS = ["신라호텔", "조선호텔"]
```

### LLM 분석 설정

**프롬프트 및 스키마** (`src/modules/analysis/prompts.yaml`):
```yaml
# 모델 설정
model: "gpt-4o-mini"

# 시스템 프롬프트
system_prompt: |
  당신은 호텔 브랜드 뉴스 모니터링 전문가입니다.
  기사를 분석하여 브랜드 관련성, 감정, 위험도, 카테고리를 판단합니다.

# 출력 스키마 (JSON)
output_schema:
  type: "object"
  properties:
    brand_relevance: { type: "string", enum: ["관련", "언급", "무관", "판단 필요"] }
    sentiment_stage: { type: "string", enum: ["긍정", "중립", "부정 후보", "부정 확정"] }
    danger_level: { type: ["string", "null"], enum: ["상", "중", "하", null] }
    issue_category: { type: ["string", "null"] }
    news_category: { type: "string" }
    news_keyword_summary: { type: "string" }

# 판단 기준
decision_rules:
  sentiment: |
    긍정: 수상, 극찬, 1위, 선정
    중립: 일반 뉴스, 사실 전달
    부정 후보: 의혹, 논란 제기
    부정 확정: 사고, 화재, 기소
  danger: |
    상: 대중 성명 필요 (사망, 대형화재, 기소)
    중: 지속 모니터링 필요 (시스템 장애, 논란)
    하: 경미한 부정 이슈
  # ...
```

**재학습 불필요**: `prompts.yaml` 수정만으로 로직 변경 가능

## 💰 비용 효율

### LLM 분석 전략
- **모델**: OpenAI GPT-4o-mini (저렴, 빠름)
- **전체 분석**: 우리 브랜드 전체 + 경쟁사 전체 (기본값)
- **경쟁사 제한**: `--max_competitor_classify N`으로 비용 절감 가능
- **보도자료 생략**: 사전 설정값 적용 (LLM 호출 없음)

### LLM API 호출
각 기사당 1번 API 호출:
- 구조화된 출력 (Structured Output)으로 단일 호출
- 6개 차원 동시 분석 (brand_relevance, sentiment_stage, danger_level, issue_category, news_category, news_keyword_summary)
- 보도자료는 LLM 생략 (사전 설정값 적용)

### 청크 처리
- 기본 청크 크기: 100개
- 병렬 처리: 기본 10개 워커
- **예시**: 1000개 기사 → 10개 청크 → 병렬 처리
- **총 소요 시간**: ~5-7분 (1000개 기사)

### 비용 절감 팁
- 보도자료 사전 설정 → LLM 비용 절감 (약 30-40%)
- 경쟁사 제한 옵션 → 추가 절감
- **예시**: 5400개 수집 → 보도자료 1800개 제외 → 3600개 분석
  - 경쟁사 제한 없음: 3600개 LLM 분석
  - 경쟁사 20개 제한: 우리 브랜드 900개 + 경쟁사 20개 = 920개 LLM 분석 (75% 절감)

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

### Google Sheets 연결 실패
- `.env` 파일에 `GOOGLE_SHEETS_CREDENTIALS_PATH` 및 `GOOGLE_SHEET_ID` 확인
- Service Account JSON 파일 경로가 올바른지 확인
- Google Sheet에 Service Account 이메일 편집 권한 부여 확인
- 연결 실패 시에도 CSV 파일은 정상 생성됨

## 📈 성능

**기본 실행 (100개 기사/브랜드):**
- **수집**: ~6초 (6개 브랜드 × 100개 = 600개)
- **처리**: ~2초 (정규화, 중복제거, 보도자료 탐지, 언론사 분류)
- **LLM 분류**: ~3-5분 (보도자료 제외, 병렬 처리)
- **리포트**: ~2초 (CSV + Word)
- **Sheets 동기화**: ~5초
- **총 소요 시간**: ~4-6분

**API 페이지네이션 (900개 기사/브랜드):**
- **수집**: ~54초 (6개 브랜드 × 9페이지 = 5400개)
- **처리**: ~10초
- **LLM 분류**: ~20-30분 (보도자료 제외, 병렬 처리)
- **총 소요 시간**: ~25-35분

**성능 최적화:**
- 병렬 처리: 기본 10개 워커 (`--max_workers 20`으로 증가 가능)
- 청크 크기: 기본 100개 (타임아웃 시 `--chunk_size 50`으로 감소)
- 경쟁사 제한: `--max_competitor_classify 20`으로 시간/비용 절감

## 📝 라이센스

MIT

## 🤝 기여

이슈와 풀 리퀘스트를 환영합니다!
