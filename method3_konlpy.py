"""
방법 3: KoNLPy 형태소 분석 + wsd-torch-models (한국어 최적화)
- 설치: pip install konlpy wsd-torch-models transformers torch
- KoNLPy로 형태소 분리 후 USAS 태깅
- 한국어 혐오 표현 분석에 가장 적합한 방식
- 사용 가능한 형태소 분석기: Okt (기본), Kkma, Komoran, Hannanum
  ※ Mecab는 별도 설치 필요 (속도 최고이나 Windows 설치 복잡)
"""

from __future__ import annotations
import torch
from transformers import AutoTokenizer
from wsd_torch_models.bem import BEM

# KoNLPy import (설치 확인)
try:
    from konlpy.tag import Okt, Kkma, Komoran
    KONLPY_AVAILABLE = True
except ImportError:
    KONLPY_AVAILABLE = False


MODEL_NAME = "ucrelnlp/PyMUSAS-Neural-Multilingual-Base-BEM"

# USAS 태그 중 혐오 표현과 관련성 높은 카테고리
HATE_RELEVANT_TAGS = {
    "S1.2": "부정적 평가 (Negative evaluation)",
    "E4.2": "분노/폭력적 감정 (Violence/Angry)",
    "S7.1": "사회적 집단 (Social groups)",
    "S7.2": "집단 구성원 (Group membership)",
    "X5.1": "인종/민족 편견",
    "S1.1": "긍정적 평가",
    "E1": "감정 일반",
    "E3": "슬픔",
    "E4": "두려움/분노",
    "E5": "혐오",
    "E6": "수치/당혹",
    "S6": "도덕/윤리",
    "X9.2": "욕설/비속어",
}


def load_model(device: str = "cuda") -> tuple:
    """모델과 토크나이저를 로드합니다."""
    print(f"[방법 3] 모델 로딩 중: {MODEL_NAME}")
    model = BEM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
    model.eval()
    model.to(device)
    param_device = next(model.parameters()).device
    print(f"[방법 3] 모델 로드 완료 (설정: {device}, 실제: {param_device})")
    return model, tokenizer


def get_tagger(tagger_name: str = "okt"):
    """
    KoNLPy 형태소 분석기를 반환합니다.

    Args:
        tagger_name: "okt" | "kkma" | "komoran"
    """
    if not KONLPY_AVAILABLE:
        raise ImportError(
            "KoNLPy가 설치되지 않았습니다. 'pip install konlpy' 를 실행하세요.\n"
            "또한 Java 8 이상이 필요합니다: https://www.java.com/ko/download/"
        )

    tagger_name = tagger_name.lower()
    print(f"[방법 3] 형태소 분석기 초기화 중: {tagger_name.upper()}")
    if tagger_name == "okt":
        tagger = Okt()
    elif tagger_name == "kkma":
        tagger = Kkma()
    elif tagger_name == "komoran":
        tagger = Komoran()
    else:
        print(f"알 수 없는 분석기 '{tagger_name}', Okt로 대체합니다.")
        tagger = Okt()
    print(f"[방법 3] {tagger.__class__.__name__} 초기화 완료")
    return tagger


def tokenize_korean(text: str, tagger) -> list[str]:
    """
    한국어 텍스트를 형태소 단위로 분리합니다.

    Returns:
        형태소 리스트
    """
    return tagger.morphs(text)


def tag_text(
    text: str,
    model: BEM,
    tokenizer,
    tagger,
    top_n: int = 5,
) -> list[dict]:
    """
    KoNLPy 형태소 분석 후 USAS 태깅합니다.

    Returns:
        [{"token": str, "pos": str, "tags": list[str],
          "top_tag": str, "definition": str,
          "hate_relevant": bool}, ...]
    """
    # 형태소 분리 + 품사 태깅
    pos_tagged = tagger.pos(text)  # [("어절", "품사"), ...]
    morphs = [morph for morph, _ in pos_tagged]

    if not morphs:
        return []

    with torch.inference_mode():
        predictions = model.predict(morphs, sub_word_tokenizer=tokenizer, top_n=top_n)

    results = []
    for (morph, pos), tags in zip(pos_tagged, predictions):
        tag_list = list(tags)
        top_tag = tag_list[0] if tag_list else "N/A"
        definition = model.label_to_definition.get(top_tag, "")
        # 혐오 표현 관련 태그 여부 확인
        hate_flag = any(
            t.startswith(prefix) for t in tag_list
            for prefix in HATE_RELEVANT_TAGS
        )
        results.append({
            "token": morph,
            "pos": pos,
            "tags": tag_list,
            "top_tag": top_tag,
            "definition": definition,
            "hate_relevant": hate_flag,
        })
    return results


def print_results(results: list[dict], show_hate_only: bool = False) -> None:
    """결과를 보기 좋게 출력합니다."""
    print(f"\n{'형태소':<12} {'품사':<8} {'USAS 1위':<10} {'설명':<35} {'혐오관련'}")
    print("-" * 80)
    for r in results:
        if show_hate_only and not r["hate_relevant"]:
            continue
        flag = "🔴" if r["hate_relevant"] else ""
        print(
            f"{r['token']:<12} {r['pos']:<8} "
            f"{r['top_tag']:<10} {r['definition']:<35} {flag}"
        )


def analyze_hate_speech(results: list[dict]) -> dict:
    """
    혐오 표현 관련 태그 통계를 분석합니다.

    Returns:
        {"hate_token_ratio": float, "top_hate_tags": list, "flagged_tokens": list}
    """
    total = len(results)
    if total == 0:
        return {}

    hate_tokens = [r for r in results if r["hate_relevant"]]
    tag_counts: dict[str, int] = {}
    for r in hate_tokens:
        for t in r["tags"]:
            for prefix, label in HATE_RELEVANT_TAGS.items():
                if t.startswith(prefix):
                    key = f"{t} ({label})"
                    tag_counts[key] = tag_counts.get(key, 0) + 1

    top_hate_tags = sorted(tag_counts.items(), key=lambda x: -x[1])[:5]

    return {
        "total_tokens": total,
        "hate_token_count": len(hate_tokens),
        "hate_token_ratio": len(hate_tokens) / total,
        "top_hate_tags": top_hate_tags,
        "flagged_tokens": [r["token"] for r in hate_tokens],
    }


def run_interactive(tagger_name: str = "okt", device: str = "gpu") -> None:
    """대화형 실행 모드"""
    model, tokenizer = load_model(device)
    tagger = get_tagger(tagger_name)

    print(f"\n=== 방법 3: KoNLPy({tagger_name.upper()}) + PyMUSAS ===")
    print("텍스트를 입력하면 형태소 분석 + USAS 태깅 결과를 보여줍니다.")
    print("종료하려면 'q' 또는 빈 줄 입력\n")

    while True:
        text = input("입력 텍스트: ").strip()
        if not text or text.lower() == "q":
            break

        results = tag_text(text, model, tokenizer, tagger)
        print_results(results)

        stats = analyze_hate_speech(results)
        if stats:
            ratio = stats["hate_token_ratio"] * 100
            print(f"\n📊 혐오 관련 형태소 비율: {ratio:.1f}% ({stats['hate_token_count']}/{stats['total_tokens']})")
            if stats["top_hate_tags"]:
                print("  주요 혐오 관련 태그:")
                for tag, cnt in stats["top_hate_tags"]:
                    print(f"    - {tag}: {cnt}회")
        print()


if __name__ == "__main__":
    run_interactive()
