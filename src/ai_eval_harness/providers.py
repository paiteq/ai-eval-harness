"""Model provider adapters. Each adapter takes a prompt + ModelSpec and
returns (text, input_tokens, output_tokens). Adapters are imported lazily so
users only need to install the providers they actually use."""

from __future__ import annotations

from typing import Protocol

from ai_eval_harness.config import ModelSpec


class Generator(Protocol):
    def generate(self, model: ModelSpec, prompt: str) -> tuple[str, int, int]:
        """Return (response_text, input_tokens, output_tokens)."""
        ...


def get_generator(provider: str) -> Generator:
    if provider == "anthropic":
        return _AnthropicGenerator()
    if provider == "openai":
        return _OpenAIGenerator()
    if provider == "local":
        return _LocalEchoGenerator()
    raise ValueError(f"unknown provider: {provider!r}")


class _AnthropicGenerator:
    """Anthropic Claude. Requires ANTHROPIC_API_KEY in env."""

    def generate(self, model: ModelSpec, prompt: str) -> tuple[str, int, int]:
        try:
            import anthropic  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "Anthropic provider requires the 'anthropic' extra: "
                "pip install 'ai-eval-harness[anthropic]'"
            ) from e

        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=model.model_id,
            max_tokens=model.max_tokens,
            temperature=model.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in msg.content if hasattr(block, "text"))
        return text, msg.usage.input_tokens, msg.usage.output_tokens


class _OpenAIGenerator:
    """OpenAI GPT. Requires OPENAI_API_KEY in env."""

    def generate(self, model: ModelSpec, prompt: str) -> tuple[str, int, int]:
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "OpenAI provider requires the 'openai' extra: "
                "pip install 'ai-eval-harness[openai]'"
            ) from e

        client = OpenAI()
        resp = client.chat.completions.create(
            model=model.model_id,
            max_tokens=model.max_tokens,
            temperature=model.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        choice = resp.choices[0]
        usage = resp.usage
        return (
            choice.message.content or "",
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )


class _LocalEchoGenerator:
    """Local echo — for smoke tests. Returns the prompt back as the answer
    and reports token counts based on whitespace splitting. Cost = 0."""

    def generate(self, model: ModelSpec, prompt: str) -> tuple[str, int, int]:
        text = f"[echo] {prompt[:200]}"
        in_toks = max(1, len(prompt.split()))
        out_toks = max(1, len(text.split()))
        return text, in_toks, out_toks
