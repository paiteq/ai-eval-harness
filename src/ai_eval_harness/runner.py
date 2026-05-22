"""Eval runner. Orchestrates: load dataset → generate per model → score via
Ragas (and optionally promptfoo) → record cost → emit report."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_eval_harness import promptfoo_adapter
from ai_eval_harness.config import EvalConfig, ModelSpec
from ai_eval_harness.cost import CostTracker, ModelCostSummary
from ai_eval_harness.loaders import load_rows
from ai_eval_harness.providers import get_generator


# Default RAG-style prompt used when the dataset gives us (question, contexts)
# but no model-specific prompt template. Kept boring on purpose — eval
# harness is for comparing models, not for prompt engineering.
_DEFAULT_PROMPT_TEMPLATE = """You are answering a question using only the provided context.

Context:
{contexts}

Question: {question}

Answer concisely. If the context does not contain the answer, say "Not in context."
"""


@dataclass
class PerRowOutput:
    question: str
    ground_truth: str
    contexts: list[str]
    answer: str
    model_name: str
    error: str | None = None


@dataclass
class EvalResult:
    config: EvalConfig
    rows: int
    outputs: list[PerRowOutput]
    cost: list[ModelCostSummary]
    ragas_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    """ragas_scores[model_name][metric_name] = score"""

    promptfoo_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    """promptfoo_scores[model_name][metric_name] = pass_rate (0..1).
    Empty when the promptfoo binary isn't installed or metrics.promptfoo is empty."""

    promptfoo_note: str | None = None
    """Human-readable status (e.g. "promptfoo binary not on PATH; skipped").
    Set when promptfoo was requested but didn't run."""

    def to_json(self) -> dict[str, Any]:
        return {
            "config": self.config.model_dump(),
            "rows": self.rows,
            "cost": [c.to_row() for c in self.cost],
            "ragas_scores": self.ragas_scores,
            "promptfoo_scores": self.promptfoo_scores,
            "promptfoo_note": self.promptfoo_note,
            "outputs_preview": [
                {
                    "model": o.model_name,
                    "question": o.question,
                    "answer": o.answer[:240],
                    "error": o.error,
                }
                for o in self.outputs[:20]
            ],
        }

    def write(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{self.config.name}.json"
        path.write_text(json.dumps(self.to_json(), indent=2))
        return path


def run_eval(
    config: EvalConfig,
    *,
    score: bool = True,
    work_dir: str | Path | None = None,
) -> EvalResult:
    """Run an eval end-to-end. When `score=False`, the Ragas + promptfoo
    scoring steps are skipped — useful for smoke tests that don't have
    judge-model credentials.

    `work_dir` is where promptfoo's generated config + output JSON land.
    Defaults to a per-run temp directory; pass an explicit path when you
    want to inspect the promptfoo artifacts after the run."""
    import tempfile

    rows = load_rows(config.dataset)
    tracker = CostTracker(config.models)
    outputs: list[PerRowOutput] = []

    for model in config.models:
        outputs.extend(_run_one_model(model, rows, tracker))

    ragas_scores: dict[str, dict[str, float]] = {}
    if score and config.metrics.ragas:
        ragas_scores = _score_with_ragas(config, outputs)

    promptfoo_scores: dict[str, dict[str, float]] = {}
    promptfoo_note: str | None = None
    if score and config.metrics.promptfoo:
        wd = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="ai-eval-pf-"))
        pf = promptfoo_adapter.run(config, outputs, work_dir=wd / "promptfoo")
        if not pf.available:
            promptfoo_note = "promptfoo binary not on PATH; install via `npm i -g promptfoo`"
        elif pf.error:
            promptfoo_note = f"promptfoo error: {pf.error}"
        elif pf.scores:
            promptfoo_scores = pf.scores

    return EvalResult(
        config=config,
        rows=len(rows),
        outputs=outputs,
        cost=tracker.summary(),
        ragas_scores=ragas_scores,
        promptfoo_scores=promptfoo_scores,
        promptfoo_note=promptfoo_note,
    )


def _run_one_model(
    model: ModelSpec,
    rows: list[dict[str, object]],
    tracker: CostTracker,
) -> list[PerRowOutput]:
    gen = get_generator(model.provider)
    outs: list[PerRowOutput] = []
    for row in rows:
        question = str(row["question"])
        contexts = [str(c) for c in row.get("contexts", [])]  # type: ignore[arg-type]
        ground_truth = str(row["ground_truth"])
        prompt = _DEFAULT_PROMPT_TEMPLATE.format(
            contexts="\n\n".join(contexts) or "(no context)",
            question=question,
        )
        with tracker.time(model.name) as t:
            try:
                answer, in_tok, out_tok = gen.generate(model, prompt)
                t.commit(in_tok, out_tok)
                outs.append(
                    PerRowOutput(
                        question=question,
                        ground_truth=ground_truth,
                        contexts=contexts,
                        answer=answer,
                        model_name=model.name,
                    )
                )
            except Exception as e:  # noqa: BLE001 — log + continue
                t.commit(0, 0, error=repr(e))
                outs.append(
                    PerRowOutput(
                        question=question,
                        ground_truth=ground_truth,
                        contexts=contexts,
                        answer="",
                        model_name=model.name,
                        error=repr(e),
                    )
                )
    return outs


def _score_with_ragas(
    config: EvalConfig,
    outputs: list[PerRowOutput],
) -> dict[str, dict[str, float]]:
    """Group outputs by model and score each group via Ragas. Returns
    {model_name: {metric: score}}.

    Ragas needs a judge model (typically OpenAI). If credentials are missing
    the score step returns an empty dict and the runner reports rows + cost
    only — that is intentional, so the harness still produces a useful artifact
    on machines without judge-model access.
    """
    try:
        from datasets import Dataset  # type: ignore[import-not-found]
        from ragas import evaluate  # type: ignore[import-not-found]
        from ragas import metrics as ragas_metrics  # type: ignore[import-not-found]
    except ImportError:
        return {}

    metric_map = {name: getattr(ragas_metrics, name) for name in config.metrics.ragas
                  if hasattr(ragas_metrics, name)}
    if not metric_map:
        return {}

    grouped: dict[str, list[PerRowOutput]] = {}
    for o in outputs:
        grouped.setdefault(o.model_name, []).append(o)

    scores: dict[str, dict[str, float]] = {}
    for model_name, group in grouped.items():
        successful = [o for o in group if not o.error]
        if not successful:
            continue
        ds = Dataset.from_dict(
            {
                "question": [o.question for o in successful],
                "answer": [o.answer for o in successful],
                "contexts": [o.contexts for o in successful],
                "ground_truth": [o.ground_truth for o in successful],
            }
        )
        try:
            result = evaluate(ds, metrics=list(metric_map.values()))
        except Exception as e:  # noqa: BLE001 — judge-model error shouldn't kill the run
            scores[model_name] = {"_error": str(e)}  # type: ignore[dict-item]
            continue
        scores[model_name] = {k: float(v) for k, v in dict(result).items() if isinstance(v, (int, float))}
    return scores
