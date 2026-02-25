# USAS Korean Hate Speech Tagger

한국어 혐오 표현(Hate Speech) 텍스트에 USAS 의미 태그를 부착하는 도구.

PyMUSAS Neural Multilingual Tagger (`ucrelnlp/PyMUSAS-Neural-Multilingual-Base-BEM`) 기반.

---

## 📁 파일 구조

```
usas_korean/
├── run.py                # 통합 실행 파일 (진입점)
├── method1_wsd_torch.py  # 방법 1: PyTorch 직접 방식
├── method2_spacy.py      # 방법 2: spaCy 파이프라인 (Windows에서는 실행 불가)
├── method3_konlpy.py     # 방법 3: KoNLPy + PyTorch (한국어 최적화)
├── requirements.txt      # 의존성 목록
├── sample_texts.txt      # 샘플 입력 파일
└── README.md             # 이 파일
```

---

## ⚙️ 설치

### 1단계 — Python 패키지

```bash
# 기본 패키지 (방법 1, 3 공통)
pip install wsd-torch-models transformers torch

# 방법 2 추가
pip install pymusas spacy

# 방법 2 — spaCy 모델 설치 (Base: 1GB / Small: 500MB)
## Base 모델 (정확도 높음, 307M 파라미터)
pip install https://github.com/UCREL/pymusas-models/releases/download/xx_none_none_none_multilingualbasebem-0.4.0/xx_none_none_none_multilingualbasebem-0.4.0-py3-none-any.whl

## Small 모델 (빠름, 140M 파라미터)
pip install https://github.com/UCREL/pymusas-models/releases/download/xx_none_none_none_multilingualsmallbem-0.4.0/xx_none_none_none_multilingualsmallbem-0.4.0-py3-none-any.whl

# 방법 3 추가 (Java 8 이상 필요)
pip install konlpy
```

### 2단계 — Java 설치 (방법 3 전용)

- Java 8 이상 필요: https://www.java.com/ko/download/
- 설치 후 환경변수 `JAVA_HOME` 설정 필요 (Windows)

---

## 🚀 실행 방법

### 대화형 모드 (방법 선택 메뉴)

```bash
python run.py
```

### 방법 직접 지정

```bash
# 방법 1 — PyTorch 직접
python run.py --method 1

# 방법 2 — spaCy (Small 모델)
python run.py --method 2 --small

# 방법 3 — KoNLPy + PyTorch
python run.py --method 3 --tagger okt
```

### 배치 처리 + CSV 저장

```bash
# sample_texts.txt 를 배치 처리하여 result.csv로 저장
python run.py --method 3 --batch sample_texts.txt --output result.csv

# 혐오 관련 형태소만 출력
python run.py --method 3 --batch sample_texts.txt --hate-only
```

---

## 📊 방법별 비교

| | 방법 1 | 방법 2 | 방법 3 |
|---|---|---|---|
| **토크나이저** | 공백 분리 | spaCy ko | KoNLPy 형태소 |
| **한국어 적합성** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **설치 난이도** | 쉬움 | 보통 | Java 필요 |
| **혐오 분석 기능** | 없음 | 없음 | 있음 (통계 포함) |
| **배치 처리** | 가능 | 빠름 | 가능 |
| **GPU 지원** | ✅ | ❌ | ✅ |

---

## ⚠️ 주의 사항

1. **한국어 공식 평가 없음**: 모델이 영어 데이터로 학습됨. 한국어 결과는 참고용이며 직접 검증 필요.
2. **USAS 태그셋**: USAS는 영어 기반 의미 분류 체계로, 한국어에 1:1 대응 안 될 수 있음.
3. **혐오 관련 주요 태그**: `S1.2` (부정 평가), `E4.2` (분노), `S7.x` (사회 집단), `E5` (혐오)
4. **첫 실행**: HuggingFace에서 모델 자동 다운로드 (~1GB). 인터넷 연결 필요.

---

## 📚 참고 문헌

- Moore et al. (2026). *Creating a Hybrid Rule and Neural Network Based Semantic Tagger using Silver Standard Data*. arXiv:2601.09648
- HuggingFace 모델: https://huggingface.co/collections/ucrelnlp/usas-neural-taggers-10
- PyMUSAS 공식 문서: https://ucrel.github.io/pymusas/
