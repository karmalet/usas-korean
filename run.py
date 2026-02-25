"""
USAS Korean Hate Speech Tagger — 통합 실행 파일
================================================
방법 선택:
  python run.py --method 1              # PyTorch 직접 방식
  python run.py --method 2              # spaCy 파이프라인
  python run.py --method 3              # KoNLPy + PyTorch (한국어 최적화)

추가 옵션:
  --device cuda                         # GPU 사용 (방법 1, 3)
  --small                               # Small 모델 사용 (방법 2)
  --tagger kkma                         # 형태소 분석기 선택: okt|kkma|komoran (방법 3)
  --batch "파일경로.txt"                # 텍스트 파일 배치 처리
  --output "결과.csv"                   # 결과 CSV 저장
  --top-n 3                             # 출력할 태그 수 (기본 5)
  --hate-only                           # 혐오 관련 형태소만 표시 (방법 3)

예시:
  python run.py --method 1
  python run.py --method 2 --small
  python run.py --method 3 --tagger okt
  python run.py --method 3 --batch hate_speech.txt --output result.csv
"""

from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="USAS Korean Hate Speech Semantic Tagger",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--method", "-m",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="실행할 방법 선택 (1, 2, 3)",
    )
    parser.add_argument(
        "--device", "-d",
        default="cpu",
        choices=["cpu", "cuda"],
        help="연산 장치 (방법 1, 3에서 사용). 기본값: cpu",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Small 모델 사용 (방법 2 전용). 기본값: Base 모델",
    )
    parser.add_argument(
        "--tagger", "-t",
        default="okt",
        choices=["okt", "kkma", "komoran"],
        help="KoNLPy 형태소 분석기 (방법 3 전용). 기본값: okt",
    )
    parser.add_argument(
        "--batch", "-b",
        type=str,
        default=None,
        metavar="FILE",
        help="배치 처리할 텍스트 파일 경로 (한 줄 = 하나의 텍스트)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        metavar="FILE",
        help="결과를 저장할 CSV 파일 경로",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="출력할 태그 수. 기본값: 5",
    )
    parser.add_argument(
        "--hate-only",
        action="store_true",
        help="혐오 관련 형태소만 표시 (방법 3 전용)",
    )
    return parser.parse_args()


def select_method_interactive() -> int:
    """방법을 대화형으로 선택합니다."""
    print("=" * 60)
    print("  USAS Korean Hate Speech Semantic Tagger")
    print("=" * 60)
    print()
    print("  [1] 방법 1 — PyTorch 직접 방식 (wsd-torch-models)")
    print("        · 공백 기반 토큰화 | 가장 단순")
    print("        · 필요: pip install wsd-torch-models transformers torch")
    print()
    print("  [2] 방법 2 — spaCy 파이프라인")
    print("        · spaCy ko 토크나이저 | 배치 처리에 강함")
    print("        · 필요: pip install pymusas spacy + 모델 whl 설치")
    print()
    print("  [3] 방법 3 — KoNLPy + PyTorch (한국어 최적화)")
    print("        · 형태소 분석 + 혐오 표현 통계 분석 포함")
    print("        · 필요: pip install konlpy wsd-torch-models transformers torch")
    print("        · Java 8 이상 필요 (https://www.java.com)")
    print()
    print("  [q] 종료")
    print()

    while True:
        choice = input("방법 선택 (1/2/3/q): ").strip()
        if choice in ("1", "2", "3"):
            return int(choice)
        if choice.lower() == "q":
            print("종료합니다.")
            sys.exit(0)
        print("  1, 2, 3 중 하나를 입력하세요.")


def load_batch_file(filepath: str) -> list[str]:
    """텍스트 파일에서 줄 단위로 텍스트를 읽어옵니다."""
    path = Path(filepath)
    if not path.exists():
        print(f"[오류] 파일을 찾을 수 없습니다: {filepath}")
        sys.exit(1)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    texts = [line for line in lines if line]
    print(f"[배치] {len(texts)}개 텍스트 로드 완료: {filepath}")
    return texts


def save_results_csv(
    texts: list[str],
    all_results: list[list[dict]],
    filepath: str,
    method: int,
) -> None:
    """결과를 CSV로 저장합니다."""
    path = Path(filepath)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)

        if method in (1, 2):
            writer.writerow(["원문", "토큰", "태그_1위", "태그_전체"])
            for text, results in zip(texts, all_results):
                for r in results:
                    top = r["tags"][0] if r["tags"] else ""
                    all_tags = "|".join(r["tags"])
                    writer.writerow([text, r["token"], top, all_tags])

        elif method == 3:
            writer.writerow([
                "원문", "형태소", "품사", "태그_1위",
                "설명", "혐오관련", "태그_전체"
            ])
            for text, results in zip(texts, all_results):
                for r in results:
                    all_tags = "|".join(r["tags"])
                    writer.writerow([
                        text,
                        r["token"],
                        r.get("pos", ""),
                        r["top_tag"],
                        r["definition"],
                        "Y" if r.get("hate_relevant") else "N",
                        all_tags,
                    ])

    print(f"[저장] 결과 저장 완료: {filepath}")


# ─────────────────────────────────────────────────────────────
# 방법별 실행 로직
# ─────────────────────────────────────────────────────────────

def run_method1(args: argparse.Namespace) -> None:
    from method1_wsd_torch import load_model, tag_text, print_results, run_interactive

    if args.batch:
        texts = load_batch_file(args.batch)
        model, tokenizer = load_model(args.device)
        all_results = []
        for i, text in enumerate(texts):
            r = tag_text(text, model, tokenizer, top_n=args.top_n)
            all_results.append(r)
            print(f"\n[{i+1}/{len(texts)}] {text}")
            print_results(r)
        if args.output:
            save_results_csv(texts, all_results, args.output, method=1)
    else:
        run_interactive(args.device)


def run_method2(args: argparse.Namespace) -> None:
    from method2_spacy import load_pipeline, tag_text, tag_batch, print_results, run_interactive

    if args.batch:
        texts = load_batch_file(args.batch)
        nlp = load_pipeline(use_small=args.small)
        print(f"\n[배치] {len(texts)}개 처리 중...")
        all_results = tag_batch(texts, nlp)
        for i, (text, results) in enumerate(zip(texts, all_results)):
            print(f"\n[{i+1}] {text}")
            print_results(results)
        if args.output:
            save_results_csv(texts, all_results, args.output, method=2)
    else:
        run_interactive(use_small=args.small)


def run_method3(args: argparse.Namespace) -> None:
    from method3_konlpy import (
        load_model, get_tagger, tag_text, print_results,
        analyze_hate_speech, run_interactive,
    )

    if args.batch:
        texts = load_batch_file(args.batch)
        model, tokenizer = load_model(args.device)
        tagger = get_tagger(args.tagger)
        all_results = []
        print(f"\n[배치] {len(texts)}개 처리 중...")
        for i, text in enumerate(texts):
            r = tag_text(text, model, tokenizer, tagger, top_n=args.top_n)
            all_results.append(r)
            print(f"\n[{i+1}/{len(texts)}] {text}")
            print_results(r, show_hate_only=args.hate_only)
            stats = analyze_hate_speech(r)
            if stats:
                ratio = stats["hate_token_ratio"] * 100
                print(f"  혐오 관련 비율: {ratio:.1f}% | 플래그 토큰: {stats['flagged_tokens']}")
        if args.output:
            save_results_csv(texts, all_results, args.output, method=3)
    else:
        run_interactive(tagger_name=args.tagger, device=args.device)


# ─────────────────────────────────────────────────────────────
# 메인 진입점
# ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # 방법이 지정되지 않으면 대화형 선택
    if args.method is None:
        args.method = select_method_interactive()
        print()

    method_labels = {
        1: "방법 1: PyTorch 직접 방식",
        2: "방법 2: spaCy 파이프라인",
        3: "방법 3: KoNLPy + PyTorch",
    }
    print(f"▶ {method_labels[args.method]} 시작\n")

    if args.method == 1:
        run_method1(args)
    elif args.method == 2:
        run_method2(args)
    elif args.method == 3:
        run_method3(args)


if __name__ == "__main__":
    main()
