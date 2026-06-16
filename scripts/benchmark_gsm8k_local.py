from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from local_fusion_gateway.config import GatewayConfig, load_config

DEFAULT_CONFIG = "config.benchmark.local-tiny.example.yaml"
DEFAULT_OUTPUT_ROOT = "benchmark-results/gsm8k-local-tiny"
DEFAULT_TASK = "gsm8k_cot"
DEFAULT_LIMIT = 128
DEFAULT_MAX_GEN_TOKS = 256
DEFAULT_TEMPERATURE = 0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 1200
STRICT_METRIC_KEY = "exact_match,strict-match"
STRICT_STDERR_KEY = "exact_match_stderr,strict-match"
FLEXIBLE_METRIC_KEY = "exact_match,flexible-extract"
FLEXIBLE_STDERR_KEY = "exact_match_stderr,flexible-extract"


@dataclass(frozen=True, slots=True)
class BenchmarkTarget:
    name: str
    model: str


@dataclass(frozen=True, slots=True)
class RunSummary:
    target: str
    model: str
    status: str
    elapsed_seconds: float
    output_path: str
    strict_match: float | None = None
    strict_match_stderr: float | str | None = None
    flexible_extract: float | None = None
    flexible_extract_stderr: float | str | None = None
    error: str | None = None


def main() -> int:
    args = _parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    targets = _selected_targets(config, args.targets)
    output_dir = _make_output_dir(Path(args.output_root), args.run_name)
    port = args.port or _free_port()
    gateway_base_url = f"http://127.0.0.1:{port}/v1"

    if args.dry_run:
        for target in targets:
            print(" ".join(_lm_eval_command(target, gateway_base_url, output_dir, args)))
        return 0

    if args.pull_missing:
        _pull_missing_models(config)
    else:
        _assert_ollama_models_present(config)

    summaries: list[RunSummary] = []
    with _gateway_process(config_path, port, args.request_timeout_seconds):
        _wait_for_gateway(port)
        for target in targets:
            summaries.append(_run_lm_eval(target, gateway_base_url, output_dir, args))

    _write_summary(output_dir, args, summaries)
    return 0 if all(summary.status == "ok" for summary in summaries) else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run GSM8K CoT 8-shot strict-match benchmarks against local tiny "
            "Ollama models and Local Fusion Gateway."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Gateway YAML config.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="GSM8K sample limit.")
    parser.add_argument("--task", default=DEFAULT_TASK, help="lm-eval task name.")
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory root for benchmark output. This path is gitignored.",
    )
    parser.add_argument("--run-name", help="Run directory name. Defaults to UTC timestamp.")
    parser.add_argument("--port", type=int, help="Gateway port. Defaults to a free local port.")
    parser.add_argument(
        "--pull-missing",
        action="store_true",
        help="Pull missing Ollama models before running. Without this, missing models fail fast.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print lm-eval commands without starting Gateway or running benchmarks.",
    )
    parser.add_argument(
        "--log-samples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ask lm-eval to write per-sample logs.",
    )
    parser.add_argument("--num-fewshot", type=int, default=8)
    parser.add_argument("--max-gen-toks", type=int, default=DEFAULT_MAX_GEN_TOKS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--num-concurrent", type=int, default=1)
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help=(
            "Per-backend Gateway request timeout while the benchmark server is running. "
            "Fusion can exceed the normal Gateway timeout on long GSM8K prompts."
        ),
    )
    parser.add_argument("--batch-size", default="1")
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=["qwen", "smollm", "gemma", "fusion"],
        default=["qwen", "smollm", "gemma", "fusion"],
        help="Targets to benchmark.",
    )
    return parser.parse_args()


def _selected_targets(config: GatewayConfig, target_names: list[str]) -> list[BenchmarkTarget]:
    model_names = {
        "qwen": "ollama-qwen-tiny",
        "smollm": "ollama-smollm-micro",
        "gemma": "ollama-gemma-micro",
        "fusion": "openrouter/fusion",
    }
    targets: list[BenchmarkTarget] = []
    for name in target_names:
        model = model_names[name]
        if model != "openrouter/fusion":
            config.resolve_model(model)
        targets.append(BenchmarkTarget(name=name, model=model))
    return targets


def _make_output_dir(output_root: Path, run_name: str | None) -> Path:
    name = run_name or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / name
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _pull_missing_models(config: GatewayConfig) -> None:
    present = _ollama_model_names()
    for model in _backend_ollama_models(config):
        if model in present:
            continue
        subprocess.run(["ollama", "pull", model], check=True)


def _assert_ollama_models_present(config: GatewayConfig) -> None:
    present = _ollama_model_names()
    missing = sorted(set(_backend_ollama_models(config)) - present)
    if missing:
        missing_text = ", ".join(missing)
        raise SystemExit(f"Missing Ollama model(s): {missing_text}. Re-run with --pull-missing.")


def _backend_ollama_models(config: GatewayConfig) -> list[str]:
    return [
        model.model
        for model in config.models.values()
        if model.base_url.startswith("http://127.0.0.1:11434")
        or model.base_url.startswith("http://localhost:11434")
    ]


def _ollama_model_names() -> set[str]:
    try:
        response = httpx.get("http://127.0.0.1:11434/api/tags", timeout=10)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise SystemExit(f"Ollama is not reachable at 127.0.0.1:11434: {exc}") from exc
    return {model["name"] for model in response.json().get("models", [])}


@contextmanager
def _gateway_process(config_path: Path, port: int, request_timeout_seconds: float):
    env = os.environ.copy()
    env["LOCAL_FUSION_CONFIG"] = str(config_path)
    env["LOCAL_FUSION_HOST"] = "127.0.0.1"
    env["LOCAL_FUSION_PORT"] = str(port)
    env["LOCAL_FUSION_REQUEST_TIMEOUT_SECONDS"] = str(request_timeout_seconds)
    env.setdefault("LOCAL_FUSION_LOG_LEVEL", "warning")
    process = subprocess.Popen(
        [sys.executable, "-m", "local_fusion_gateway.main"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        yield process
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def _wait_for_gateway(port: int) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + 30
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(url, timeout=2)
            if response.status_code == 200:
                return
        except httpx.HTTPError as exc:
            last_error = exc
        time.sleep(0.25)
    raise RuntimeError(f"Gateway did not become ready on {url}: {last_error}")


def _run_lm_eval(
    target: BenchmarkTarget,
    gateway_base_url: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> RunSummary:
    run_dir = output_dir / _safe_path_name(target.name)
    run_dir.mkdir(parents=True, exist_ok=False)
    command = _lm_eval_command(target, gateway_base_url, output_dir, args)
    started = time.perf_counter()
    env = os.environ.copy()
    env.setdefault("OPENAI_API_KEY", "local-fusion")
    completed = subprocess.run(
        command,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    (run_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (run_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    (run_dir / "command.json").write_text(
        json.dumps(command, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        return RunSummary(
            target=target.name,
            model=target.model,
            status="failed",
            elapsed_seconds=elapsed,
            output_path=str(run_dir),
            error=f"lm_eval exited with {completed.returncode}",
        )
    try:
        metrics = extract_gsm8k_metrics(run_dir)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        return RunSummary(
            target=target.name,
            model=target.model,
            status="failed",
            elapsed_seconds=elapsed,
            output_path=str(run_dir),
            error=str(exc),
        )
    return RunSummary(
        target=target.name,
        model=target.model,
        status="ok",
        elapsed_seconds=elapsed,
        output_path=str(run_dir),
        strict_match=metrics.get(STRICT_METRIC_KEY),
        strict_match_stderr=metrics.get(STRICT_STDERR_KEY),
        flexible_extract=metrics.get(FLEXIBLE_METRIC_KEY),
        flexible_extract_stderr=metrics.get(FLEXIBLE_STDERR_KEY),
    )


def _lm_eval_command(
    target: BenchmarkTarget,
    gateway_base_url: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    run_dir = output_dir / _safe_path_name(target.name)
    command = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "local-chat-completions",
        "--model_args",
        (
            f"model={target.model},"
            f"base_url={gateway_base_url}/chat/completions,"
            f"num_concurrent={args.num_concurrent}"
        ),
        "--tasks",
        args.task,
        "--num_fewshot",
        str(args.num_fewshot),
        "--limit",
        str(args.limit),
        "--batch_size",
        str(args.batch_size),
        "--apply_chat_template",
        "--gen_kwargs",
        f"max_gen_toks={args.max_gen_toks},temperature={args.temperature}",
        "--output_path",
        str(run_dir),
    ]
    if args.log_samples:
        command.append("--log_samples")
    return command


def extract_gsm8k_metrics(run_dir: Path) -> dict[str, Any]:
    for result_file in sorted(run_dir.rglob("*.json")):
        try:
            payload = json.loads(result_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        results = payload.get("results")
        if not isinstance(results, dict):
            continue
        task_result = results.get(DEFAULT_TASK)
        if not isinstance(task_result, dict):
            task_result = _first_gsm8k_result(results)
        if task_result is None:
            continue
        if STRICT_METRIC_KEY not in task_result:
            raise KeyError(f"{STRICT_METRIC_KEY} not found in {result_file}")
        return task_result
    raise FileNotFoundError(f"No lm-eval GSM8K result JSON found under {run_dir}")


def _first_gsm8k_result(results: dict[str, Any]) -> dict[str, Any] | None:
    for task_name, task_result in results.items():
        if "gsm8k" in task_name and isinstance(task_result, dict):
            return task_result
    return None


def _write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    summaries: list[RunSummary],
) -> None:
    summary_payload = {
        "task": args.task,
        "limit": args.limit,
        "num_fewshot": args.num_fewshot,
        "metric": "strict-match",
        "request_timeout_seconds": args.request_timeout_seconds,
        "generated_at": datetime.now(UTC).isoformat(),
        "runs": [asdict(summary) for summary in summaries],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# GSM8K local tiny benchmark summary",
        "",
        f"- task: `{args.task}`",
        f"- limit: `{args.limit}`",
        f"- fewshot: `{args.num_fewshot}`",
        f"- request timeout seconds: `{args.request_timeout_seconds:g}`",
        "- primary metric: `strict-match`",
        "",
        "| target | model | status | strict-match | flexible-extract | elapsed sec |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary.target} | `{summary.model}` | {summary.status} | "
            f"{_format_metric(summary.strict_match)} | "
            f"{_format_metric(summary.flexible_extract)} | {summary.elapsed_seconds:.2f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_metric(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def _safe_path_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


if __name__ == "__main__":
    raise SystemExit(main())
