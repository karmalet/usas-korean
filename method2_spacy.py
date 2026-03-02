"""
방법 2: spaCy 파이프라인 방식
- 설치:
    pip install pymusas spacy
    pip install https://github.com/UCREL/pymusas-models/releases/download/xx_none_none_none_multilingualbasebem-0.4.0/xx_none_none_none_multilingualbasebem-0.4.0-py3-none-any.whl
- spaCy 한국어 토크나이저 사용 (blank "ko")
- 배치 처리에 유리
- Small 모델 사용 시 설치 URL을 SMALL_MODEL_URL로 변경
"""

from __future__ import annotations
import spacy
from spacy.language import Language

BASE_MODEL_NAME  = r"xx_none_none_none_multilingualbasebem"
SMALL_MODEL_NAME = r"xx_none_none_none_multilingualsmallbem"

BASE_MODEL_URL  = (
    "https://github.com/UCREL/pymusas-models/releases/download/"
    "xx_none_none_none_multilingualbasebem-0.4.0/"
    "xx_none_none_none_multilingualbasebem-0.4.0-py3-none-any.whl"
)
SMALL_MODEL_URL = (
    "https://github.com/UCREL/pymusas-models/releases/download/"
    "xx_none_none_none_multilingualsmallbem-0.4.0/"
    "xx_none_none_none_multilingualsmallbem-0.4.0-py3-none-any.whl"
)


def load_pipeline(use_small: bool = False) -> Language:
    """
    한국어 spaCy 파이프라인에 PyMUSAS 뉴럴 태거를 추가합니다.

    Args:
        use_small: True면 Small 모델(140M), False면 Base 모델(307M) 사용
    """
    model_name = SMALL_MODEL_NAME if use_small else BASE_MODEL_NAME
    size_label = "Small(140M)" if use_small else "Base(307M)"
    print(f"[방법 2] spaCy 파이프라인 로딩 중... ({size_label})")

    try:
        tagger_pipeline = spacy.load(model_name)
    except OSError:
        print(f"\n[오류] '{model_name}' 모델이 설치되어 있지 않습니다.")
        url = SMALL_MODEL_URL if use_small else BASE_MODEL_URL
        print(f"아래 명령어로 설치하세요:\n  pip install {url}\n")
        raise

    # 한국어 토크나이저
    nlp = spacy.blank("en")
    nlp.add_pipe("pymusas_neural_tagger", source=tagger_pipeline)
    print("[방법 2] 파이프라인 로드 완료")
    return nlp


def tag_text(text: str, nlp: Language) -> list[dict]:
    """
    단일 텍스트를 USAS 태깅합니다.

    Returns:
        [{"token": str, "tags": list[str], "definitions": list[str]}, ...]
    """
    doc = nlp(text)

    # 태그 설명 맵 가져오기
    try:
        tagger = nlp.get_pipe("pymusas_neural_tagger")
        label_map = tagger.model.label_to_definition
    except (KeyError, AttributeError):
        label_map = {}

    results = []
    for token in doc:
        tags = list(token._.pymusas_tags) if token._.pymusas_tags else []
        definitions = [label_map.get(t, "") for t in tags]
        results.append({"token": token.text, "tags": tags, "definitions": definitions})
    return results


def tag_batch(texts: list[str], nlp: Language, batch_size: int = 32) -> list[list[dict]]:
    """
    여러 텍스트를 배치로 USAS 태깅합니다.

    Args:
        texts: 텍스트 리스트
        nlp: 로드된 spaCy 파이프라인
        batch_size: 배치 크기
    """
    # 태그 설명 맵 가져오기
    try:
        tagger = nlp.get_pipe("pymusas_neural_tagger")
        label_map = tagger.model.label_to_definition
    except (KeyError, AttributeError):
        label_map = {}

    all_results = []
    for i, doc in enumerate(nlp.pipe(texts, batch_size=batch_size)):
        results = []
        for token in doc:
            tags = list(token._.pymusas_tags) if token._.pymusas_tags else []
            definitions = [label_map.get(t, "") for t in tags]
            results.append({"token": token.text, "tags": tags, "definitions": definitions})
        all_results.append(results)
        if (i + 1) % 10 == 0:
            print(f"  처리 완료: {i + 1}/{len(texts)}")
    return all_results


def print_results(results: list[dict]) -> None:
    """결과를 보기 좋게 출력합니다."""
    print(f"\n{'토큰':<15} {'1위 태그':<10} {'설명'}")
    print("-" * 60)
    for r in results:
        top_tag = r["tags"][0] if r["tags"] else "N/A"
        top_def = r["definitions"][0] if r["definitions"] else ""
        all_tags = ", ".join(r["tags"])
        print(f"{r['token']:<15} {top_tag:<10} {top_def}")
        print(f"  └ 전체 태그: [{all_tags}]")


def run_interactive(use_small: bool = False) -> None:
    """대화형 실행 모드"""
    nlp = load_pipeline(use_small)

    size_label = "Small" if use_small else "Base"
    print(f"\n=== 방법 2: spaCy 파이프라인 ({size_label} 모델) ===")
    print("텍스트를 입력하면 USAS 태깅 결과를 보여줍니다.")
    print("종료하려면 'q' 또는 빈 줄 입력\n")

    while True:
        text = input("입력 텍스트: ").strip()
        if not text or text.lower() == "q":
            break
        results = tag_text(text, nlp)
        print_results(results)
        print()


if __name__ == "__main__":
    run_interactive()
