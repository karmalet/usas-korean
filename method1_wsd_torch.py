"""
방법 1: wsd-torch-models 직접 사용 (PyTorch 방식)
- 설치: pip install wsd-torch-models transformers torch
- 한국어 공백 분리 기반 (형태소 분석 없음)
- 가장 단순하고 직접적인 방식
"""

from __future__ import annotations
import torch
from transformers import AutoTokenizer
from wsd_torch_models.bem import BEM


MODEL_NAME = "ucrelnlp/PyMUSAS-Neural-Multilingual-Base-BEM"


def load_model(device: str = "gpu") -> tuple:
    """모델과 토크나이저를 로드합니다."""
    print(f"[방법 1] 모델 로딩 중: {MODEL_NAME}")
    model = BEM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
    model.eval()
    model.to(device)
    print(f"[방법 1] 모델 로드 완료 (device={device})")
    return model, tokenizer


def tag_text(
    text: str,
    model: BEM,
    tokenizer,
    top_n: int = 5,
) -> list[dict]:
    """
    텍스트를 공백 분리하여 USAS 태깅합니다.

    Returns:
        [{"token": str, "tags": list[str], "definitions": list[str]}, ...]
    """
    tokens = text.split()
    if not tokens:
        return []

    with torch.inference_mode():
        predictions = model.predict(tokens, sub_word_tokenizer=tokenizer, top_n=top_n)

    results = []
    for token, tags in zip(tokens, predictions):
        definitions = [model.label_to_definition.get(t, "") for t in tags]
        results.append({"token": token, "tags": list(tags), "definitions": definitions})
    return results


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


def run_interactive(device: str = "gpu") -> None:
    """대화형 실행 모드"""
    model, tokenizer = load_model(device)

    print("\n=== 방법 1: PyTorch 직접 방식 ===")
    print("텍스트를 입력하면 USAS 태깅 결과를 보여줍니다.")
    print("종료하려면 'q' 또는 빈 줄 입력\n")

    while True:
        text = input("입력 텍스트: ").strip()
        if not text or text.lower() == "q":
            break
        results = tag_text(text, model, tokenizer)
        print_results(results)
        print()


def run_batch(texts: list[str], device: str = "gpu") -> list[list[dict]]:
    """배치 처리 모드"""
    model, tokenizer = load_model(device)
    all_results = []
    for text in texts:
        results = tag_text(text, model, tokenizer)
        all_results.append(results)
    return all_results


if __name__ == "__main__":
    run_interactive()
