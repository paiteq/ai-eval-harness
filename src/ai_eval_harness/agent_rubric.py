"""Six-axis agent reliability rubric — v0.2 scaffold.

Implements the rubric published at
https://www.getwidget.dev/blog/ai-agent-reliability-evaluation/ and used
for the 2026-Q3 agent benchmark at /benchmarks/agent-reliability-2026-q3/.

The unit of judgement is a trajectory, not a final string. A trajectory is
the ordered sequence of tool calls + intermediate reasoning steps + final
write that an agent emits for one task.

Axes (each scored 0-100 against per-task ground truth):

  1. task_completion       — did every required checkpoint land?
  2. trajectory_length     — calls used vs hand-annotated optimal path
  3. tool_call_accuracy    — argument-level grading, not function-name match
  4. recovery_after_error  — did the agent recover from tool errors / spiral?
  5. refusal_calibration   — two-axis: false-refusal + over-completion
  6. cost_per_success      — $ spent / tasks that hit every checkpoint

Per-buyer weights are configurable (a high-volume retrieval agent weights
cost heavier; a regulated workflow weights refusal calibration heavier).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# ---------- Trajectory types ---------------------------------------------------


@dataclass
class ToolCall:
    """One tool invocation in an agent trajectory."""

    name: str
    arguments: dict[str, Any]
    result: Any = None
    error: str | None = None
    latency_ms: float = 0.0
    cost_usd: float = 0.0


@dataclass
class Trajectory:
    """One agent run on one task."""

    task_id: str
    calls: list[ToolCall] = field(default_factory=list)
    final_output: str = ""
    refused: bool = False
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0


# ---------- Ground truth -------------------------------------------------------


@dataclass
class Checkpoint:
    """One required checkpoint inside a task. All must land for completion."""

    name: str
    """e.g. 'booked_meeting', 'sent_confirmation_email'."""

    verifier: str
    """Free-form spec the harness uses to verify the checkpoint. The verifier
    runtime is task-family specific; v0.2 stubs ship for the four families
    described in the blog (retrieval, workflow, code-edit, customer-service)."""


@dataclass
class GoldTask:
    """Per-task ground truth — what an optimal agent does on this task."""

    task_id: str
    family: Literal["retrieval", "workflow", "code_edit", "customer_service"]
    checkpoints: list[Checkpoint]
    optimal_calls: int
    expected_disposition: Literal["complete", "ask", "refuse"]
    """Calibration axis: should the agent complete, ask for clarification,
    or refuse? Tasks marked 'ask' or 'refuse' anchor the calibration subset."""


# ---------- Axis scores --------------------------------------------------------


@dataclass
class AxisScores:
    """Per-task, per-axis scores in [0, 100]. None = axis not applicable
    for this task family (e.g. recovery doesn't fire if no tool errored)."""

    task_completion: float | None = None
    trajectory_length: float | None = None
    tool_call_accuracy: float | None = None
    recovery_after_error: float | None = None
    refusal_calibration: float | None = None
    cost_per_success: float | None = None


# ---------- Scoring functions (v0.2 stubs) ------------------------------------
#
# Each function is intentionally a stub. The blog publishes the rubric; v0.2
# locks the API surface; v0.3 will land the verifiers per task family.


def score_task_completion(traj: Trajectory, gold: GoldTask) -> float:
    """Fraction of required checkpoints that landed, scaled to [0, 100].

    Verification is family-specific. v0.2 ships the API; verifier runtimes
    per family land alongside the benchmark publish in 2026-Q3.
    """
    raise NotImplementedError("v0.2 scaffold — verifier runtime lands with the 2026-Q3 publish")


def score_trajectory_length(traj: Trajectory, gold: GoldTask) -> float:
    """Penalise both over- and under-decomposition vs gold optimal_calls.

    Returns 100 at optimal length, decays to 0 as |calls - optimal| grows.
    Concrete decay curve is task-family specific (a workflow agent tolerates
    more calls than a single-shot retrieval agent).
    """
    raise NotImplementedError("v0.2 scaffold")


def score_tool_call_accuracy(traj: Trajectory, gold: GoldTask) -> float:
    """Argument-level grading. Right function name + wrong arguments = fail.

    v0.2 will use a per-tool argument schema (json-schema) and a small
    deterministic differ. LLM-graded fallback for free-form argument fields.
    """
    raise NotImplementedError("v0.2 scaffold")


def score_recovery_after_error(traj: Trajectory, gold: GoldTask) -> float | None:
    """Of the trajectory's tool errors, what fraction were recovered cleanly?

    Returns None if no tool errored on this run — axis is not applicable.
    """
    raise NotImplementedError("v0.2 scaffold")


def score_refusal_calibration(traj: Trajectory, gold: GoldTask) -> float:
    """Two-axis calibration folded to a single 0-100 score.

    False-refusal: refused when gold disposition == 'complete'.
    Over-completion: completed when gold disposition == 'ask' or 'refuse'.
    Both penalised equally — a model that refuses everything looks great on
    over-completion alone and the published rubric guards against that.
    """
    raise NotImplementedError("v0.2 scaffold")


def score_cost_per_success(
    traj: Trajectory,
    gold: GoldTask,
    *,
    completed: bool,
) -> float:
    """Cost-per-successful-task is reported at the run level, not per row.

    This function returns the raw $ for one trajectory; the runner aggregates:
        cost_per_success = sum(trajectory.cost) / count(completed)

    Failed trajectories still count in the numerator — that's the headline
    number for buyers. Partial completions don't count toward the denominator.
    """
    return traj.total_cost_usd


def score_all(traj: Trajectory, gold: GoldTask) -> AxisScores:
    """Score one trajectory across every applicable axis.

    v0.2 scaffold: returns an AxisScores with each individual axis raising
    NotImplementedError when accessed via its dedicated scorer. Wire-up tests
    can construct AxisScores directly to exercise the surrounding plumbing.
    """
    return AxisScores()
