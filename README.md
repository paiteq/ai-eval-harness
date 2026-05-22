# ai-eval-harness

[![License: MIT](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)
[![Version: v0.1](https://img.shields.io/badge/version-v0.1.0-blue.svg)](#roadmap)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

**Open-source eval harness for RAG and agent systems.** Built on Ragas + a small cost-tracking layer, with promptfoo wiring for prompt-level regression suites. Model-agnostic on principle: Claude, GPT, and open-source models score on the same prompts and corpora.

Maintained by [Paiteq](https://www.paiteq.com) and used internally on client engagements by [Paiteq](https://www.paiteq.com) and [GetWidget](https://www.getwidget.dev). MIT-licensed so you can run the same harness on your own infra and verify our published benchmark numbers.

> **Status: v0.1 — public scaffold.** Ragas wiring, cost/latency tracking, local-echo smoke provider, and one example RAG config ship in v0.1. Benchmarks produced with this harness publish at [getwidget.dev/benchmarks](https://www.getwidget.dev/benchmarks/), starting with `rag-2026-q2` in June.

## Why this exists

Most "best LLM for RAG" content online is vibes-based. Authors run a handful of prompts on a curated corpus, declare a winner, and never publish the prompts or corpus. The result is unverifiable, undated, and rots within a quarter.

We wanted a different shape:

- **Reproducible.** Anyone can clone this repo, point it at a corpus, and get scores inside our published confidence intervals.
- **Dated.** Every benchmark we publish carries the quarter in the URL and H1. Undated benchmarks are uncitable.
- **Model-agnostic.** Same prompts, same corpus, same rubric across Claude / GPT / Gemini / open-source.
- **Cost on the same axis as quality.** Recall@5 and pass@1 are meaningless without $/1k queries on the same dated run.

## What the harness covers

| Capability | Backed by | Status |
|---|---|---|
| Context precision, context recall | Ragas | ✅ v0.1 |
| Faithfulness, answer relevancy | Ragas | ✅ v0.1 |
| Cost + latency tracking per-run | custom | ✅ v0.1 |
| Local-echo provider (no-credentials smoke path) | custom | ✅ v0.1 |
| Prompt-level eval, regression suites | promptfoo | 🟡 wiring in v0.1, full eval flow v0.2 |
| Agent reliability (tool-use, multi-step, error recovery) | custom rubric | ⚪ v0.2 |

## Roadmap

| Milestone | Date | What ships |
|---|---|---|
| **v0.1** | ✅ 2026-05 (shipped) | Ragas wiring. CLI runner. Cost + latency tracking. Local-echo smoke provider. One example RAG config. |
| **v0.2** | 2026-07 | Agent reliability bench (100-task harness, tool-use + multi-step). promptfoo full eval flow. |
| **v0.3** | 2026-08 | Public dataset cards on [huggingface.co/paiteq-ai](https://huggingface.co/paiteq-ai). PyPI release. |
| **v1.0** | 2026-Q4 | Stable CLI + Python API. Versioned benchmark snapshots. |

Track milestones in [GitHub issues](https://github.com/paiteq/ai-eval-harness/issues) and [releases](https://github.com/paiteq/ai-eval-harness/releases) once they ship.

## Published benchmarks

Each benchmark uses this harness end-to-end. Full methodology, prompts, scores, and per-model cost live on the benchmark page; the dataset is mirrored to HuggingFace.

| Benchmark | Publish target | Page |
|---|---|---|
| RAG retrieval, 2026-Q2 | 2026-06 | [getwidget.dev/benchmarks/rag-2026-q2/](https://www.getwidget.dev/benchmarks/) |
| Agent reliability, 2026-Q3 | 2026-09 | [getwidget.dev/benchmarks/agent-reliability-2026-q3/](https://www.getwidget.dev/benchmarks/) |

## Quick start

```bash
# From source (PyPI publish lands in v0.3).
git clone https://github.com/paiteq/ai-eval-harness.git
cd ai-eval-harness
pip install -e ".[ragas,anthropic,openai]"

# Smoke run against the bundled local-echo provider — no API keys needed.
ai-eval run examples/rag-baseline.yaml --no-score --out runs/

# With Ragas scoring + real models (Claude, GPT). Export keys first.
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
ai-eval run examples/rag-baseline.yaml --out runs/
```

The bundled `examples/rag-baseline.yaml` runs on a 10-row local corpus
(`examples/data/rag-baseline.jsonl`) so the full pipeline is exercisable
without external data.

### Defining your own eval

Three sections, all YAML:

```yaml
name: my-rag-eval
dataset:
  source: path/to/my-corpus.jsonl  # or org/name on HuggingFace
  max_rows: 100
  columns:
    question: query        # remap if your corpus uses different names
    ground_truth: answer
    contexts: docs

models:
  - name: claude-sonnet-4-6
    provider: anthropic
    model_id: claude-sonnet-4-6
    input_cost_per_mtok: 3.00
    output_cost_per_mtok: 15.00

  - name: gpt-5
    provider: openai
    model_id: gpt-5
    input_cost_per_mtok: 2.50
    output_cost_per_mtok: 10.00

metrics:
  ragas:
    - context_precision
    - context_recall
    - faithfulness
    - answer_relevancy
```

Each row in the dataset needs three fields: `question`, `ground_truth`, and a
`contexts` list. Retrieval is precomputed in v0.1 — bring your own retriever
and pass the top-K chunks per question. End-to-end retriever wiring lands later.

## License

MIT. See [LICENSE](LICENSE). Free for personal and commercial use, no attribution required (though a link back is appreciated).

## Maintained by

This harness is built and maintained by the engineering team behind:

- **[Paiteq](https://www.paiteq.com)** — AI engineering studio (Claude, OpenAI, open-source). Eval-first delivery.
- **[GetWidget](https://www.getwidget.dev)** — AI-native development studio, founded 2017 (Dallas + Bengaluru). Open-source [Flutter UI Kit](https://github.com/ionicfirebaseapp/getwidget) (4,800+ stars).
- **[Hire Flutter Dev](https://hireflutterdev.com)** — vetted senior Flutter developers, AI-augmented delivery.

Contact: [paiteq.com/contact](https://www.paiteq.com/contact/) · [getwidget.dev/contact-us](https://www.getwidget.dev/contact-us/)
