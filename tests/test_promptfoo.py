"""promptfoo integration smoke tests.

The promptfoo binary is Node.js — not always available in CI / dev envs.
These tests skip cleanly when it's missing, but always exercise the adapter
code paths that don't depend on the binary (config generation, output parsing).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from ai_eval_harness import promptfoo_adapter
from ai_eval_harness.config import EvalConfig
from ai_eval_harness.runner import PerRowOutput, run_eval

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO_ROOT / "examples" / "rag-baseline.yaml"


def _sample_outputs() -> list[PerRowOutput]:
    return [
        PerRowOutput(
            question="What is the harness license?",
            ground_truth="MIT",
            contexts=["The harness is MIT-licensed."],
            answer="The harness is MIT-licensed.",
            model_name="local-echo",
        ),
        PerRowOutput(
            question="Which providers ship in v0.1?",
            ground_truth="Anthropic, OpenAI, and a local-echo provider",
            contexts=["Providers: anthropic, openai, local."],
            answer="[echo] context",
            model_name="local-echo",
        ),
    ]


def _sample_config() -> EvalConfig:
    cfg = EvalConfig.from_yaml(EXAMPLE_CONFIG)
    return EvalConfig(
        name="pf-smoke",
        dataset=cfg.dataset,
        models=[m for m in cfg.models if m.provider == "local"],
        metrics={"ragas": [], "promptfoo": ["contains-ground-truth"]},
    )


def test_config_generation_is_well_formed(tmp_path: Path) -> None:
    """Even without the binary, we can assert the config generator emits
    valid promptfoo YAML with one test per (model × row) tuple."""
    cfg = _sample_config()
    out_path = tmp_path / "promptfoo.yaml"
    promptfoo_adapter._write_config(cfg, _sample_outputs(), out_path)

    raw = yaml.safe_load(out_path.read_text())
    assert "description" in raw
    assert raw["providers"] == [{"id": "echo"}]
    assert len(raw["tests"]) == 2
    for t in raw["tests"]:
        assert "question" in t["vars"]
        assert "answer" in t["vars"]
        assert "model" in t["vars"]
        assert any(a.get("type") == "contains" for a in t["assert"])


def test_output_parser_handles_v0x_schema(tmp_path: Path) -> None:
    """Hand-craft a promptfoo-v0.x-shaped output and verify the parser
    rolls assertions up to per-model pass rates."""
    fake = {
        "results": {
            "results": [
                {"vars": {"model": "local-echo"}, "success": True},
                {"vars": {"model": "local-echo"}, "success": True},
                {"vars": {"model": "local-echo"}, "success": False},
                {"vars": {"model": "claude-sonnet-4-6"}, "success": True},
            ]
        }
    }
    path = tmp_path / "promptfoo-output.json"
    path.write_text(json.dumps(fake))

    parsed = promptfoo_adapter._parse_output(path)
    assert parsed["local-echo"]["contains-ground-truth"] == pytest.approx(2 / 3)
    assert parsed["claude-sonnet-4-6"]["contains-ground-truth"] == pytest.approx(1.0)


def test_output_parser_tolerates_missing_file(tmp_path: Path) -> None:
    parsed = promptfoo_adapter._parse_output(tmp_path / "does-not-exist.json")
    assert parsed == {}


def test_adapter_skips_when_binary_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When `promptfoo` isn't on PATH, run() returns available=False with
    no error — caller treats it as 'skip silently'."""
    monkeypatch.setattr(promptfoo_adapter, "available", lambda: False)
    result = promptfoo_adapter.run(_sample_config(), _sample_outputs(), tmp_path)
    assert result.available is False
    assert result.scores is None


def test_runner_records_note_when_promptfoo_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Full pipeline: a config with metrics.promptfoo set but no binary
    available should produce a usable EvalResult with promptfoo_note set."""
    monkeypatch.setattr(promptfoo_adapter, "available", lambda: False)
    cfg = _sample_config()
    result = run_eval(cfg, score=True)
    assert result.rows == 10
    assert result.promptfoo_scores == {}
    assert result.promptfoo_note is not None
    assert "promptfoo" in result.promptfoo_note.lower()


@pytest.mark.skipif(
    not promptfoo_adapter.available(),
    reason="promptfoo binary not installed (npm i -g promptfoo)",
)
def test_promptfoo_runs_end_to_end_when_installed(tmp_path: Path) -> None:
    """Only runs if promptfoo is actually installed. Verifies the full
    shell-out → parse path against the real binary."""
    cfg = _sample_config()
    pf = promptfoo_adapter.run(cfg, _sample_outputs(), work_dir=tmp_path)
    assert pf.available is True
    # The fake outputs above include one that should fail the contains check
    # ("[echo] context" doesn't contain "Anthropic, OpenAI") and one that
    # should pass. Just assert we got SOME scores back; exact numbers depend
    # on promptfoo version.
    assert pf.scores is not None
