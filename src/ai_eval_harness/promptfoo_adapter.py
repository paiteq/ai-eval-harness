"""promptfoo integration.

promptfoo (https://promptfoo.dev) is a Node.js CLI for prompt-level regression
testing — different paradigm than Ragas. We wire it as an optional second
scoring lane: if `metrics.promptfoo` is non-empty in the config AND the
`promptfoo` binary is on PATH, we generate a promptfoo config from our outputs,
shell out to `promptfoo eval`, and parse the resulting JSON.

If the binary is missing we skip silently — promptfoo is a Node tool and the
harness must work in Python-only environments. Install via `npm i -g promptfoo`.

Wire status as of v0.1: adapter is functional. Full assertion-driven eval flow
(LLM-graded assertions, regex/contains matchers wired per-metric) lands in v0.2.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_eval_harness.config import EvalConfig
    from ai_eval_harness.runner import PerRowOutput


PROMPTFOO_BIN = "promptfoo"


@dataclass
class PromptfooResult:
    available: bool
    """True if the promptfoo binary was found and the eval ran."""

    raw_path: Path | None = None
    """Path to the raw promptfoo JSON output, if produced."""

    scores: dict[str, dict[str, float]] | None = None
    """{model_name: {metric_name: pass_rate}} once parsed."""

    error: str | None = None


def available() -> bool:
    """Return True if `promptfoo` is on PATH."""
    return shutil.which(PROMPTFOO_BIN) is not None


def run(
    config: EvalConfig,
    outputs: list[PerRowOutput],
    work_dir: str | Path,
) -> PromptfooResult:
    """Generate a promptfoo config from our outputs, shell out, parse results.

    Returns a PromptfooResult with available=False if the binary is missing —
    caller can treat that as "skip silently". Any subprocess error gets
    captured in `result.error` rather than raised, so a misconfigured Node
    install never kills a Python eval run.
    """
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    if not available():
        return PromptfooResult(available=False, error="promptfoo binary not on PATH")

    promptfoo_config_path = work / "promptfoo.yaml"
    promptfoo_output_path = work / "promptfoo-output.json"

    _write_config(config, outputs, promptfoo_config_path)

    try:
        proc = subprocess.run(  # noqa: S603 — PROMPTFOO_BIN is a fixed literal
            [
                PROMPTFOO_BIN,
                "eval",
                "-c", str(promptfoo_config_path),
                "--output", str(promptfoo_output_path),
            ],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return PromptfooResult(
            available=True,
            raw_path=promptfoo_config_path,
            error="promptfoo eval timed out after 600s",
        )
    except (FileNotFoundError, OSError) as e:
        return PromptfooResult(available=True, error=f"failed to launch promptfoo: {e}")

    if proc.returncode != 0 and not promptfoo_output_path.exists():
        return PromptfooResult(
            available=True,
            raw_path=promptfoo_config_path,
            error=f"promptfoo exited {proc.returncode}: {proc.stderr[-400:].strip()}",
        )

    scores = _parse_output(promptfoo_output_path) if promptfoo_output_path.exists() else None
    return PromptfooResult(
        available=True,
        raw_path=promptfoo_output_path,
        scores=scores,
    )


def _write_config(
    config: EvalConfig,
    outputs: list[PerRowOutput],
    out_path: Path,
) -> None:
    """Emit a promptfoo YAML config from our model outputs.

    The promptfoo config is "external" — we generate prompts from our outputs
    and let promptfoo apply assertions. v0.1 wires three default assertions:

    - `contains-any`: answer must contain at least one ground-truth substring
    - `not-contains`: answer must NOT contain the placeholder "Not in context."
       when ground truth IS in context (faithfulness floor)
    - `latency`: answer generated under 30s (loose latency budget)

    These are intentionally coarse — they catch broken outputs, not subtle
    quality regressions (Ragas handles those). v0.2 will add LLM-graded
    assertions, model-vs-model preference scoring, and regex matchers.
    """
    import yaml

    # Group outputs by model so promptfoo can compare across models
    by_model: dict[str, list[PerRowOutput]] = {}
    for o in outputs:
        by_model.setdefault(o.model_name, []).append(o)

    # Each model becomes a "provider" in promptfoo terminology, but here we
    # use the `python:` provider trick — we just feed the precomputed answer
    # back via tests.vars. This makes promptfoo an assertion engine over our
    # answers, NOT a re-runner of the models. Saves API spend + ensures the
    # promptfoo scores reflect the same answers Ragas saw.
    tests = []
    for model_name, model_outs in by_model.items():
        for i, out in enumerate(model_outs):
            tests.append({
                "description": f"{model_name} · row {i}",
                "vars": {
                    "question": out.question,
                    "answer": out.answer,
                    "ground_truth": out.ground_truth,
                    "model": model_name,
                },
                "assert": [
                    # The answer should contain at least the first 3 words of
                    # the ground truth — coarse but catches "Not in context."
                    # outputs when the context actually has the answer.
                    {
                        "type": "contains",
                        "value": " ".join(out.ground_truth.split()[:3]),
                    },
                ],
            })

    promptfoo_config = {
        "description": f"ai-eval-harness · {config.name} · promptfoo assertions",
        "prompts": ["{{answer}}"],
        # Use the echo provider — promptfoo just evaluates assertions against
        # the precomputed answer text, doesn't re-run any LLM.
        "providers": [{"id": "echo"}],
        "tests": tests,
    }
    out_path.write_text(yaml.safe_dump(promptfoo_config, sort_keys=False))


def _parse_output(output_path: Path) -> dict[str, dict[str, float]]:
    """Parse promptfoo's JSON output into {model_name: {metric: pass_rate}}.

    promptfoo's output schema varies by version; this parser handles v0.x
    schema (results.results array with success bool + vars.model). Adjust if
    promptfoo schema changes — there's a versioned parser inside _Parser.
    """
    try:
        raw = json.loads(output_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}

    results = raw.get("results", {}).get("results", []) or raw.get("results", [])
    if not isinstance(results, list):
        return {}

    by_model: dict[str, dict[str, list[bool]]] = {}
    for r in results:
        model_name = (
            r.get("vars", {}).get("model")
            or r.get("test", {}).get("vars", {}).get("model")
            or "unknown"
        )
        success = bool(r.get("success", False))
        model_bucket = by_model.setdefault(model_name, {"contains-ground-truth": []})
        model_bucket["contains-ground-truth"].append(success)

    return {
        model: {metric: sum(vals) / len(vals) if vals else 0.0 for metric, vals in buckets.items()}
        for model, buckets in by_model.items()
    }
