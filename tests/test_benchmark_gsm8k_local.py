from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def load_benchmark_module() -> ModuleType:
    script_path = Path(__file__).parents[1] / "scripts" / "benchmark_gsm8k_local.py"
    spec = importlib.util.spec_from_file_location("benchmark_gsm8k_local", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_extract_gsm8k_metrics_prefers_strict_match(tmp_path: Path) -> None:
    module = load_benchmark_module()
    result_dir = tmp_path / "qwen"
    result_dir.mkdir()
    (result_dir / "samples.json").write_text(
        json.dumps([{"doc_id": 0, "target": "6"}]),
        encoding="utf-8",
    )
    (result_dir / "results.json").write_text(
        json.dumps(
            {
                "results": {
                    "gsm8k_cot": {
                        "exact_match,strict-match": 0.125,
                        "exact_match_stderr,strict-match": 0.01,
                        "exact_match,flexible-extract": 0.25,
                        "exact_match_stderr,flexible-extract": 0.02,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    metrics = module.extract_gsm8k_metrics(result_dir)

    assert metrics["exact_match,strict-match"] == 0.125
    assert metrics["exact_match,flexible-extract"] == 0.25


def test_extract_gsm8k_metrics_requires_strict_match(tmp_path: Path) -> None:
    module = load_benchmark_module()
    result_dir = tmp_path / "qwen"
    result_dir.mkdir()
    (result_dir / "results.json").write_text(
        json.dumps({"results": {"gsm8k_cot": {"exact_match,flexible-extract": 0.25}}}),
        encoding="utf-8",
    )

    try:
        module.extract_gsm8k_metrics(result_dir)
    except KeyError as exc:
        assert "exact_match,strict-match" in str(exc)
    else:
        raise AssertionError("strict-match absence should fail")


def test_lm_eval_command_uses_gateway_chat_completions(tmp_path: Path) -> None:
    module = load_benchmark_module()
    args = SimpleNamespace(
        task="gsm8k_cot",
        num_fewshot=8,
        limit=128,
        batch_size="1",
        max_gen_toks=256,
        temperature=0,
        num_concurrent=1,
        log_samples=True,
    )

    command = module._lm_eval_command(
        module.BenchmarkTarget(name="fusion", model="openrouter/fusion"),
        "http://127.0.0.1:8080/v1",
        tmp_path,
        args,
    )
    command_text = " ".join(command)

    assert "--model local-chat-completions" in command_text
    assert "model=openrouter/fusion" in command_text
    assert "base_url=http://127.0.0.1:8080/v1/chat/completions" in command_text
    assert "--tasks gsm8k_cot" in command_text
    assert "--num_fewshot 8" in command_text
    assert "--log_samples" in command


def test_dry_run_prints_four_targets(tmp_path: Path, monkeypatch, capsys) -> None:
    module = load_benchmark_module()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
fusion:
  default_analysis_models:
    - "ollama-qwen-tiny"
    - "ollama-smollm-micro"
    - "ollama-gemma-micro"
  default_judge_model: "ollama-qwen-tiny"
models:
  ollama-qwen-tiny:
    base_url: "http://127.0.0.1:11434/v1"
    api_key: "ollama"
    model: "qwen2.5:0.5b"
  ollama-smollm-micro:
    base_url: "http://127.0.0.1:11434/v1"
    api_key: "ollama"
    model: "smollm2:135m"
  ollama-gemma-micro:
    base_url: "http://127.0.0.1:11434/v1"
    api_key: "ollama"
    model: "gemma3:270m"
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_gsm8k_local.py",
            "--config",
            str(config_path),
            "--output-root",
            str(tmp_path / "results"),
            "--run-name",
            "dry-run",
            "--dry-run",
        ],
    )

    assert module.main() == 0

    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 4
    assert any("model=openrouter/fusion" in line for line in lines)
    assert any("model=ollama-qwen-tiny" in line for line in lines)
