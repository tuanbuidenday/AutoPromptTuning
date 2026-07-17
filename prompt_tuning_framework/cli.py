"""Giao diện dòng lệnh của framework.

    prompt-tune plugins
    prompt-tune run --dataset data.csv --prompt "..." --task "..." --labels Yes No
    prompt-tune run --config my_config.yml --dataset data.csv --prompt "..."
"""
import argparse
import logging
import sys
from typing import List, Optional

from . import components  # noqa: F401  (nạp để plugin tự đăng ký)
from .config import tuner_from_yaml
from .core.interfaces import BaseCallback
from .core.registry import KINDS, available
from .core.tuner import PromptTuner
from .core.types import EvalResult, PromptVersion
from .data import load_samples_csv

BAR = "=" * 64


class TerminalProgress(BaseCallback):
    """In tiến trình từng vòng ra terminal."""

    def on_iteration_end(self, iteration: int, version: PromptVersion,
                         result: EvalResult) -> None:
        n = len(result.results)
        ok = result.num_correct
        print(f"  vòng {iteration}: {result.score:5.1f}/100  "
              f"({ok}/{n} đúng, {len(result.errors)} sai)")


def _print_report(tuner: PromptTuner, best: Optional[PromptVersion]) -> None:
    history = tuner.store.history()
    print("\n" + BAR)
    print("CÁC PHIÊN BẢN PROMPT")
    print(BAR)
    for v in history:
        star = "  <-- TỐT NHẤT" if best and v.version == best.version else ""
        score = f"{v.score}/100" if v.score is not None else "chưa chấm"
        print(f"\n[v{v.version}] điểm: {score}{star}\n{v.text}")

    if best and history:
        print("\n" + BAR)
        print(f"TRƯỚC: ({history[0].score}/100) {history[0].text}")
        print(f"SAU  : ({best.score}/100) {best.text}")
        print(BAR)


def cmd_plugins(args: argparse.Namespace) -> int:
    print("Plugin đã đăng ký:")
    for kind in KINDS:
        print(f"  {kind:10}: {', '.join(available(kind)) or '(trống)'}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    samples = load_samples_csv(args.dataset)
    labeled = [s for s in samples if s.label]
    print(f"Nạp {len(samples)} ca test ({len(labeled)} ca có nhãn) từ {args.dataset}")

    if args.config:
        tuner = tuner_from_yaml(args.config)
        if args.task:
            tuner.task_description = args.task
        if args.max_iters:
            tuner.max_iters = args.max_iters
    else:
        from .components import (AccuracyEvaluator, LLMExecutor,
                                 LLMRewriteOptimizer)
        from .llm import default_model
        labels: List[str] = args.labels or sorted({s.label for s in labeled})
        # Model mặc định bám theo --provider, nếu không đổi provider sang openai
        # mà quên --model thì sẽ gửi tên model Gemini sang OpenAI.
        exec_model = args.model or default_model(args.provider, "executor")
        opt_model = args.optimizer_model or default_model(args.provider, "optimizer")
        print(f"Provider: {args.provider} | chạy: {exec_model} | tối ưu: {opt_model}")
        tuner = PromptTuner(
            executor=LLMExecutor(provider=args.provider, model=exec_model,
                                 labels=labels),
            evaluator=AccuracyEvaluator(),
            optimizer=LLMRewriteOptimizer(provider=args.provider,
                                          model=opt_model, labels=labels),
            task_description=args.task or "",
            max_iters=args.max_iters or 3,
            target_score=args.target_score,
        )

    tuner.callbacks = list(tuner.callbacks) + [TerminalProgress()]

    print(f"Chạy tối ưu (tối đa {tuner.max_iters} vòng)...")
    best = tuner.run(args.prompt, samples)
    _print_report(tuner, best)
    return 0 if best else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="prompt-tune",
        description="Framework tối ưu prompt tự động — chạy trên terminal.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("plugins", help="Liệt kê plugin đã đăng ký").set_defaults(func=cmd_plugins)

    r = sub.add_parser("run", help="Chạy vòng lặp tối ưu prompt")
    r.add_argument("--dataset", required=True, help="CSV có cột text,label")
    r.add_argument("--prompt", required=True, help="Prompt ban đầu cần tối ưu")
    r.add_argument("--task", default="", help="Mô tả task")
    r.add_argument("--config", help="File YAML cấu hình (bỏ qua các cờ model bên dưới)")
    r.add_argument("--labels", nargs="+", help="Nhãn hợp lệ, vd: --labels Yes No")
    r.add_argument("--provider", default="google", choices=["google", "openai"],
                   help="API key lấy từ biến môi trường GOOGLE_API_KEY / OPENAI_API_KEY")
    r.add_argument("--model", help="Model chạy prompt (mặc định: model rẻ của --provider)")
    r.add_argument("--optimizer-model",
                   help="Model viết prompt mới (mặc định: model rẻ của --provider)")
    r.add_argument("--max-iters", type=int, help="Số vòng tối đa")
    r.add_argument("--target-score", type=float, default=100.0, help="Đạt điểm này thì dừng")
    r.set_defaults(func=cmd_run)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"Lỗi: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
