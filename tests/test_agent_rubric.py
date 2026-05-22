"""Smoke tests for the v0.2 agent rubric scaffold.

These pin the public API surface. Scorer bodies raise NotImplementedError
until verifier runtimes ship with the 2026-Q3 benchmark publish — tests cover
the data classes + the API shape, not the scoring math.
"""

from __future__ import annotations

import pytest

from ai_eval_harness.agent_rubric import (
    AxisScores,
    Checkpoint,
    GoldTask,
    ToolCall,
    Trajectory,
    score_all,
    score_cost_per_success,
    score_task_completion,
)


def _make_trajectory() -> Trajectory:
    return Trajectory(
        task_id="ret-001",
        calls=[
            ToolCall(name="lookup_customer", arguments={"id": "C-42"}, cost_usd=0.001),
            ToolCall(name="fetch_orders", arguments={"customer_id": "C-42", "limit": 3}, cost_usd=0.002),
        ],
        final_output="Customer C-42 spent $812 across 3 orders.",
        total_cost_usd=0.003,
    )


def _make_gold() -> GoldTask:
    return GoldTask(
        task_id="ret-001",
        family="retrieval",
        checkpoints=[
            Checkpoint(name="fetched_orders", verifier="orders.count == 3"),
            Checkpoint(name="computed_total", verifier="answer.contains_total"),
        ],
        optimal_calls=3,
        expected_disposition="complete",
    )


def test_trajectory_and_gold_construct_cleanly():
    traj = _make_trajectory()
    gold = _make_gold()
    assert traj.task_id == gold.task_id == "ret-001"
    assert len(traj.calls) == 2
    assert len(gold.checkpoints) == 2


def test_score_all_returns_empty_axis_scores_in_v02_scaffold():
    """v0.2 ships the API. Individual axis scorers raise. score_all returns
    an empty AxisScores so surrounding runner plumbing can be exercised
    against the new module before verifiers land."""
    out = score_all(_make_trajectory(), _make_gold())
    assert isinstance(out, AxisScores)
    assert out.task_completion is None
    assert out.refusal_calibration is None


def test_cost_per_success_returns_raw_trajectory_cost():
    """The only scorer with a real body in v0.2 — aggregation is the runner's
    job. This one just surfaces the per-trajectory $ for the numerator."""
    traj = _make_trajectory()
    assert score_cost_per_success(traj, _make_gold(), completed=True) == pytest.approx(0.003)


def test_task_completion_scorer_raises_pending_verifier_runtime():
    with pytest.raises(NotImplementedError):
        score_task_completion(_make_trajectory(), _make_gold())
