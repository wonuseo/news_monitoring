# 뉴스 모니터링 시스템

네이버 뉴스 API로 호텔 브랜드 관련 기사를 수집하고, AI로 감정·카테고리·위험도를 분석하여 Excel + Word 리포트를 생성합니다.

![Architecture](assets/news_monitoring_architecture.png)
## ✨ 주요 기능

- 📰 **자동 수집**: 네이버 뉴스 API로 브랜드 검색
- 🤖 **3단계 AI 분석**:
  1. 감정 분석 (긍정/중립/부정)
  2. 카테고리 분류 (한국어 7개 카테고리)
  3. 위험도 평가 (부정 기사만: 상/중/하)
- 📊 **Excel 출력**: 원본/처리/결과 데이터
- 📄 **Word 리포트**: 위험도별 구조화된 문서

## 🗂️ 프로젝트 구조

```
├── main.py           # 메인 실행 파일
├── collect.py        # 1. 네이버 뉴스 수집
├── process.py        # 2. 데이터 정리 + 중복제거
├── classify.py       # 3. AI 카테고리화 + 위험도 판단
├── report.py         # 4. 콘솔 + Word 리포트 생성
├── .env              # API 키 설정
└── data/             # 출력 디렉토리
    ├── raw.xlsx
    ├── processed.xlsx
    ├── result.xlsx
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

## 🎯 AI 분석 프로세스

### 1단계: 감정 분석
모든 기사를 분석하여 분류:
- **긍정**: 수상, 성장, 좋은 소식
- **중립**: 일반 뉴스, 사실 전달
- **부정**: 사고, 논란, 손실

### 2단계: 카테고리 분류
모든 기사를 7개 카테고리로 분류:
- **법률/규제**: 소송, 조사, 규제 이슈
- **보안/데이터**: 해킹, 정보 유출
- **안전/사고**: 화재, 사망, 부상
- **재무/실적**: 실적, 주가, 투자
- **제품/서비스**: 신규 오픈, 리뉴얼
- **평판/SNS**: 여론, 불매운동
- **운영/기타**: 일반 운영, 인사

### 3단계: 위험도 평가
**부정 기사만** 위험도 평가:
- **상 (🔴)**: 즉각 대응 필요 (화재, 사망, 영업정지 등)
- **중 (🟡)**: 모니터링 필요 (고객 불만, 조사 등)
- **하 (🟢)**: 경미한 영향 (소규모 논란 등)

긍정/중립 기사는 위험도 평가 없음 (비용 절감)

## 📊 출력 파일

### Excel 파일

1. **raw.xlsx**: 네이버 API에서 수집한 원본 데이터
2. **processed.xlsx**: HTML 제거, 중복 제거된 정제 데이터
3. **result.xlsx**: AI 분류 결과 (여러 시트)
   - `전체데이터`: 모든 기사
   - `우리_부정`: 우리 브랜드 부정 기사
   - `우리_긍정`: 우리 브랜드 긍정 기사
   - `경쟁사`: 경쟁사 기사

### Word 문서 (report.docx)

5개 섹션으로 구성:
1. **긴급 대응 필요 (위험도: 상)** - 🔴
2. **모니터링 필요 (위험도: 중)** - 🟡
3. **경미한 이슈 (위험도: 하)** - 🟢
4. **긍정 뉴스** - 😊
5. **경쟁사 동향**

각 기사별로:
- 브랜드명, 카테고리
- 제목, 이유, 날짜, 링크

## ⚙️ 설정 변경

### 브랜드 수정

`collect.py` 파일에서:
```python
OUR_BRANDS = ["롯데호텔", "호텔롯데", "L7", "시그니엘"]
COMPETITORS = ["신라호텔", "조선호텔"]
```

### 카테고리 수정

`classify.py` 파일에서:
```python
CATEGORIES = [
    "법률/규제",
    "보안/데이터",
    "안전/사고",
    "재무/실적",
    "제품/서비스",
    "평판/SNS",
    "운영/기타",
]
```

## 💰 비용 효율

### 청킹 전략
- **365개 기사** → **4개 청크** (100개씩)
- 청크당 3번 API 호출 (감정 + 카테고리 + 위험도)
- **총 12번 API 호출** (vs 개별 처리 시 365번)
- **97% 비용 절감!**

### 부정 기사만 위험도 평가
- 긍정/중립 기사는 위험도 평가 생략
- 추가 비용 절감

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
