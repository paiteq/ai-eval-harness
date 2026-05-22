"""ai-eval-harness — open-source eval harness for RAG and agent systems.

Maintained by Paiteq (https://www.paiteq.com).
Benchmark results published at https://www.getwidget.dev/benchmarks/.
"""

__version__ = "0.1.0"

from ai_eval_harness.config import EvalConfig, ModelSpec, RetrievalSpec
from ai_eval_harness.runner import EvalResult, run_eval

__all__ = [
    "__version__",
    "EvalConfig",
    "ModelSpec",
    "RetrievalSpec",
    "EvalResult",
    "run_eval",
]
