"""DeepSeek v4 Flash LLM client with exponential-backoff retry.

Used by repair.py to produce fix patches for TVM frontend converters.
Only dependency at runtime is the `openai` package.
"""
import logging
import os
import time

logger = logging.getLogger("autorepair.llm")

# Fallback key if DEEPSEEK_API_KEY is not set (user-provided).
DEFAULT_API_KEY = "[Your own API key]"
MAX_OUTPUT_TOKENS = 8192

try:
    import openai
except ImportError:  # pragma: no cover - surfaced at construction time
    openai = None


def _retry_prediction(call_fn, model_name, max_trials=3, base_sleep=2.0, backoff=2.0):
    """Call ``call_fn()`` with exponential-backoff retry; return "" on exhaustion."""
    last_err = None
    for trial in range(1, max_trials + 1):
        try:
            return call_fn()
        except Exception as e:  # network / API / rate-limit errors
            last_err = e
            if trial < max_trials:
                sleep_s = base_sleep * (backoff ** (trial - 1))
                logger.warning(
                    "DeepSeek API call failed (%s), retrying %d/%d in %.1fs",
                    e, trial, max_trials, sleep_s,
                )
                time.sleep(sleep_s)
    logger.error("DeepSeek API call (%s) failed after %d trials: %s",
                 model_name, max_trials, last_err)
    return ""


class DeepseekV4FlashClient:
    """DeepSeek v4 Flash via OpenAI-compatible API."""

    model_name = "deepseek-v4-flash"

    def __init__(self, api_key=None, base_url="https://api.deepseek.com/beta",
                 max_tokens=MAX_OUTPUT_TOKENS, max_trials=3):
        if openai is None:
            raise RuntimeError("`openai` is not installed. Run: pip install openai")
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY") or DEFAULT_API_KEY
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.max_trials = max_trials
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def model_prediction(self, prompt, system=None):
        """Return the model's text reply for ``prompt`` ("" on persistent failure)."""
        system = system or "You are a compiler expert. Compare code logic carefully."
        return _retry_prediction(
            lambda: self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=self.max_tokens,
                extra_body={"thinking": {"type": "disabled"}},
            ).choices[0].message.content,
            self.model_name,
            max_trials=self.max_trials,
        )
