"""Per-model cost + latency tracking.

Every eval run logs token usage and wall-clock per request. The summary is
keyed by ModelSpec.name so runs are comparable across configs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Self

from ai_eval_harness.config import ModelSpec


@dataclass
class RequestRecord:
    model_name: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    error: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class ModelCostSummary:
    model_name: str
    requests: int = 0
    failures: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms_total: float = 0.0
    cost_usd: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.latency_ms_total / self.requests if self.requests else 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_row(self) -> dict[str, float | int | str]:
        return {
            "model": self.model_name,
            "requests": self.requests,
            "failures": self.failures,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "cost_usd": round(self.cost_usd, 4),
        }


class CostTracker:
    """Accumulates per-model usage. Compute cost via ModelSpec rates at the
    end of a run — input/output tokens are tracked separately because most
    providers price them differently."""

    def __init__(self, models: list[ModelSpec]) -> None:
        self._rates: dict[str, ModelSpec] = {m.name: m for m in models}
        self._records: list[RequestRecord] = []

    def record(self, rec: RequestRecord) -> None:
        self._records.append(rec)

    def time(self, model_name: str) -> _Timer:
        """Context manager: `with tracker.time("claude-sonnet-4-6") as t: ...`
        Then t.commit(input_tokens=..., output_tokens=...)."""
        return _Timer(self, model_name)

    def summary(self) -> list[ModelCostSummary]:
        summaries: dict[str, ModelCostSummary] = {}
        for rec in self._records:
            s = summaries.setdefault(rec.model_name, ModelCostSummary(rec.model_name))
            s.requests += 1
            if rec.error:
                s.failures += 1
                continue
            s.input_tokens += rec.input_tokens
            s.output_tokens += rec.output_tokens
            s.latency_ms_total += rec.latency_ms

        for name, summ in summaries.items():
            rate = self._rates.get(name)
            if rate is None:
                continue  # unknown model — leave cost at 0
            summ.cost_usd = (
                summ.input_tokens / 1_000_000 * rate.input_cost_per_mtok
                + summ.output_tokens / 1_000_000 * rate.output_cost_per_mtok
            )
        return list(summaries.values())


@dataclass
class _Timer:
    tracker: CostTracker
    model_name: str
    _start_ns: int = 0

    def __enter__(self) -> Self:
        self._start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: object) -> None:
        # If commit() was never called and there's no error, we still record
        # the timing as a no-token request so latency is captured.
        if exc is not None and not getattr(self, "_committed", False):
            self.commit(0, 0, error=repr(exc))

    def commit(self, input_tokens: int, output_tokens: int, error: str | None = None) -> None:
        latency_ms = (time.perf_counter_ns() - self._start_ns) / 1_000_000
        self.tracker.record(
            RequestRecord(
                model_name=self.model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                error=error,
            )
        )
        self._committed = True
