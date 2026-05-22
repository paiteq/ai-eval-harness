# ai-eval-harness

[![License: MIT](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)
[![Status: pre-v0.1](https://img.shields.io/badge/status-pre--v0.1-yellow.svg)](#roadmap)

**Open-source eval harness for RAG and agent systems.** Built on Ragas + promptfoo + a small custom-rubric layer for agent reliability. Model-agnostic on principle: Claude, GPT, Gemini, and open-source models score on the same prompts and corpora.

Maintained by [Paiteq](https://www.paiteq.com) and used internally on client engagements by [Paiteq](https://www.paiteq.com) and [GetWidget](https://www.getwidget.dev). MIT-licensed so you can run the same harness on your own infra and verify our published benchmark numbers.

> **Status: pre-v0.1, public scaffold.** Code lands in v0.1 (June 2026). Benchmarks produced with this harness will publish at [getwidget.dev/benchmarks](https://www.getwidget.dev/benchmarks/), starting with `rag-2026-q2`.

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
| Retrieval quality (recall@k, MRR) | Ragas | v0.1 |
| Generation faithfulness, answer relevance | Ragas | v0.1 |
| Prompt-level eval, regression suites | promptfoo | v0.1 |
| Agent reliability (tool-use, multi-step, error recovery) | custom rubric | v0.2 |
| Cost + latency tracking per-run | custom | v0.1 |

## Roadmap

| Milestone | Date | What ships |
|---|---|---|
| **v0.1** | 2026-06 | Ragas + promptfoo wired. One example RAG eval config. CLI runner. Cost + latency tracking. |
| **v0.2** | 2026-07 | Agent reliability bench (100-task harness, tool-use + multi-step). |
| **v0.3** | 2026-08 | Public dataset cards on [huggingface.co/paiteq-ai](https://huggingface.co/paiteq-ai). |
| **v1.0** | 2026-Q4 | Stable CLI + Python API. Versioned benchmark snapshots. |

Track milestones in [GitHub issues](https://github.com/paiteq/ai-eval-harness/issues) and [releases](https://github.com/paiteq/ai-eval-harness/releases) once they ship.

## Published benchmarks

Each benchmark uses this harness end-to-end. Full methodology, prompts, scores, and per-model cost live on the benchmark page; the dataset is mirrored to HuggingFace.

| Benchmark | Publish target | Page |
|---|---|---|
| RAG retrieval, 2026-Q2 | 2026-06 | [getwidget.dev/benchmarks/rag-2026-q2/](https://www.getwidget.dev/benchmarks/) |
| Agent reliability, 2026-Q3 | 2026-09 | [getwidget.dev/benchmarks/agent-reliability-2026-q3/](https://www.getwidget.dev/benchmarks/) |

## Quick start (lands in v0.1)

```bash
# Pre-v0.1 placeholder. v0.1 ships June 2026.
pip install ai-eval-harness  # not published yet
ai-eval run examples/rag-baseline.yaml
```

A working scaffold with one example RAG config will land with v0.1. If you'd like early access, open an [issue](https://github.com/paiteq/ai-eval-harness/issues) and we'll loop you in.

## License

MIT. See [LICENSE](LICENSE). Free for personal and commercial use, no attribution required (though a link back is appreciated).

## Maintained by

This harness is built and maintained by the engineering team behind:

- **[Paiteq](https://www.paiteq.com)** — AI engineering studio (Claude, OpenAI, open-source). Eval-first delivery.
- **[GetWidget](https://www.getwidget.dev)** — AI-native development studio, founded 2017 (Dallas + Bengaluru). Open-source [Flutter UI Kit](https://github.com/ionicfirebaseapp/getwidget) (4,800+ stars).
- **[Hire Flutter Dev](https://hireflutterdev.com)** — vetted senior Flutter developers, AI-augmented delivery.

Contact: [paiteq.com/contact](https://www.paiteq.com/contact/) · [getwidget.dev/contact-us](https://www.getwidget.dev/contact-us/)
