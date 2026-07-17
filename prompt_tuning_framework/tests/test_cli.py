"""Test giao diện dòng lệnh — chạy offline bằng component giả."""
import pytest

from prompt_tuning_framework import BaseExecutor, BaseOptimizer, Prediction, register
from prompt_tuning_framework.cli import main

CSV = ("text,label\n"
       "It barks at strangers.,Yes\n"
       "It purrs on my lap.,No\n"
       "We walk it on a leash.,Yes\n"
       "It climbs the shelf.,No\n")


@register("executor", "cli_fake")
class CliFakeExecutor(BaseExecutor):
    def execute(self, prompt, samples):
        informed = "leash" in prompt.lower()
        return [
            Prediction(
                sample_id=s.id,
                output=("Yes" if any(w in s.text.lower() for w in ("bark", "leash"))
                        else "No") if informed else "Yes",
            )
            for s in samples
        ]


@register("optimizer", "cli_fake")
class CliFakeOptimizer(BaseOptimizer):
    def propose(self, prompt, result, task_description="", history=None):
        return "A dog barks or walks on a leash; a cat does not. Answer Yes or No."


@pytest.fixture
def dataset(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text(CSV, encoding="utf-8")
    return str(p)


@pytest.fixture
def config(tmp_path):
    p = tmp_path / "c.yml"
    p.write_text(
        "task_description: phan loai cho/meo\n"
        "max_iters: 3\n"
        "target_score: 100\n"
        "executor: {name: cli_fake}\n"
        "evaluator: {name: accuracy}\n"
        "optimizer: {name: cli_fake}\n",
        encoding="utf-8",
    )
    return str(p)


def test_plugins_command(capsys):
    assert main(["plugins"]) == 0
    out = capsys.readouterr().out
    assert "optimizer" in out and "llm_rewrite" in out


def test_run_command_optimizes_and_prints_report(dataset, config, capsys):
    rc = main(["run", "--dataset", dataset, "--prompt", "Is this a dog? Yes or No",
               "--config", config])
    out = capsys.readouterr().out

    assert rc == 0
    assert "Nạp 4 ca test" in out
    assert "vòng 0:" in out              # tiến trình từng vòng
    assert "CÁC PHIÊN BẢN PROMPT" in out  # báo cáo
    assert "TỐT NHẤT" in out
    assert "TRƯỚC:" in out and "SAU  :" in out
    assert "100.0/100" in out             # tối ưu thành công


def test_run_overrides_max_iters_from_cli(dataset, config, capsys):
    main(["run", "--dataset", dataset, "--prompt", "vague prompt",
          "--config", config, "--max-iters", "1"])
    out = capsys.readouterr().out
    assert "tối đa 1 vòng" in out


def test_run_missing_dataset_errors(config, capsys):
    rc = main(["run", "--dataset", "/khong/co.csv", "--prompt", "p", "--config", config])
    assert rc == 2
    assert "Lỗi:" in capsys.readouterr().err


def test_missing_required_arg_exits(dataset):
    with pytest.raises(SystemExit):
        main(["run", "--dataset", dataset])  # thiếu --prompt
