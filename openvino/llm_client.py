"""
Unified LLM client module for OPENVINO audit system.

Provides:
- Common interface (model_prediction) for all LLM backends
- Global proxy that transparently delegates to the selected model
- setup_model() for interactive model selection at startup

Usage:
    from llm_client import setup_model, global_model

    setup_model()                     # show menu, pick model
    result = global_model.model_prediction(prompt)   # call current model
    name = global_model.model_name    # get current model name
"""

import openai


# ============================================================
# Shared helpers
# ============================================================

# DeepSeek/thinking models count reasoning tokens against max_tokens; a small
# budget (8192) can be consumed entirely by "thinking", leaving an empty final
# answer (finish_reason=length, content=""). 32768 leaves headroom for both.
MAX_OUTPUT_TOKENS = 32768


def _retry_prediction(make_call, model_label, max_attempts=1):
    """Call make_call() once and validate the result.

    Returns the content if non-empty, otherwise a visible "error: ..." marker
    (never a silent empty string).
    """
    last_error = "empty LLM response"
    for attempt in range(1, max_attempts + 1):
        try:
            content = make_call()
            if content and content.strip():
                return content
            print(f"  [WARN] {model_label} returned empty content")
        except Exception as e:
            print(f"  [WARN] {model_label} API call failed: {e}")
            last_error = f"API call failed: {e}"
    return f"error: {last_error}"


# ============================================================
# Model implementations
# ============================================================

class DeepseekClient:
    """DeepSeek Chat via OpenAI-compatible API (recommended)."""
    model_name = "deepseek-chat"

    def __init__(self):
        self.api_key = "[Your own API key]"
        self.base_url = "https://api.deepseek.com/beta"
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def model_prediction(self, prompt):
        return _retry_prediction(
            lambda: self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a compiler expert. Compare code logic carefully."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=MAX_OUTPUT_TOKENS,
                extra_body={"thinking": {"type": "disabled"}}
            ).choices[0].message.content,
            self.model_name,
        )


class QwenMaxClient:
    """Qwen3.7-Max via DashScope OpenAI-compatible API."""
    model_name = "qwen3.7-max"

    def __init__(self):
        self.api_key = "[Your own API key]"
        self.base_url = "https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1"
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def model_prediction(self, prompt):
        def call():
            response = self.client.responses.create(
                model=self.model_name,
                input=prompt,
                extra_body={
                    "enable_thinking": False,
                    "temperature": 0,
                    "max_tokens": MAX_OUTPUT_TOKENS
                }
            )
            for item in response.output:
                if item.type == "message":
                    return item.content[0].text
            return None
        return _retry_prediction(call, self.model_name)


class QwenFlashClient:
    """Qwen3.6-Flash via DashScope OpenAI-compatible API."""
    model_name = "qwen3.5-flash"

    def __init__(self):
        self.api_key = "[Your own API key]"
        self.base_url = "https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1"
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def model_prediction(self, prompt):
        def call():
            response = self.client.responses.create(
                model=self.model_name,
                input=prompt,
                extra_body={
                    "enable_thinking": False,
                    "temperature": 0,
                    "max_tokens": MAX_OUTPUT_TOKENS
                }
            )
            for item in response.output:
                if item.type == "message":
                    return item.content[0].text
            return None
        return _retry_prediction(call, self.model_name)


class DeepseekV4FlashClient:
    """DeepSeek v4 Flash via original DeepSeek API."""
    model_name = "deepseek-v4-flash"

    def __init__(self):
        self.api_key = "[Your own API key]"
        self.base_url = "https://api.deepseek.com/beta"
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def model_prediction(self, prompt):
        return _retry_prediction(
            lambda: self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a compiler expert. Compare code logic carefully."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=MAX_OUTPUT_TOKENS,
                extra_body={"thinking": {"type": "disabled"}}
            ).choices[0].message.content,
            self.model_name,
        )


class KamiapiGPTClient:
    """GPT-5.4-mini via kamiapi.top (OpenAI-compatible API)."""
    model_name = "gpt-5.4-mini"

    def __init__(self):
        self.api_key = "[Your own API key]"
        self.base_url = "https://www.kamiapi.top/v1"
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers={"User-Agent": "Mozilla/5.0"},
        )

    def model_prediction(self, prompt):
        return _retry_prediction(
            lambda: self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a compiler expert. Compare code logic carefully."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=MAX_OUTPUT_TOKENS,
                extra_body={"reasoning_effort": "low"}
            ).choices[0].message.content,
            self.model_name,
        )


# ============================================================
# Available model registry
# ============================================================
# Add new models here to make them available in the selection menu.
AVAILABLE_MODELS = {
    "1": ("DeepSeek Chat (Remote API - OpenAI SDK)", DeepseekClient),
    "2": ("Qwen3.7-Max (Qwen Cloud - DashScope OpenAI SDK)", QwenMaxClient),
    "3": ("Qwen3.5-Flash (Qwen Cloud - DashScope OpenAI SDK)", QwenFlashClient),
    "4": ("DeepSeek v4 Flash (Remote API - OpenAI SDK)", DeepseekV4FlashClient),
    "5": ("GPT-5.4-mini (kamiapi.top)", KamiapiGPTClient),
}


# ============================================================
# Global model proxy
# ============================================================

class _ModelProxy:
    """
    Proxy that delegates all attribute access to the selected model instance.
    This ensures modules that do `from llm_client import global_model` at import
    time still see the latest model after setup_model() is called in main().
    """
    def __init__(self):
        self._model = None

    @property
    def model_name(self):
        """Return current model name, or a placeholder if uninitialized."""
        if self._model is None:
            return "not_selected"
        return self._model.model_name

    def __getattr__(self, name):
        if name == '_model':
            raise AttributeError(name)
        if self._model is None:
            raise RuntimeError(
                "Model not initialized. Call setup_model() before using global_model."
            )
        return getattr(self._model, name)

    def model_prediction(self, prompt):
        """Delegate prediction to the selected model."""
        if self._model is None:
            raise RuntimeError(
                "Model not initialized. Call setup_model() before using global_model."
            )
        return self._model.model_prediction(prompt)


global_model = _ModelProxy()


def setup_model():
    """
    Display available models, let the user select one, and set it as the global model.

    Returns:
        The selected model instance.
    """
    print("\n" + "=" * 50)
    print("            LLM Model Selection")
    print("=" * 50)
    model_keys = sorted(AVAILABLE_MODELS.keys())
    for key in model_keys:
        name, _ = AVAILABLE_MODELS[key]
        print(f"  [{key}] {name}")
    print("-" * 50)

    choice = input("Select model number [default 1]: ").strip()
    if not choice:
        choice = "1"

    if choice in AVAILABLE_MODELS:
        _, cls = AVAILABLE_MODELS[choice]
        instance = cls()
        global_model._model = instance
        print(f"  [OK] Selected: {instance.model_name}")
    else:
        print(f"  [ERROR] Invalid choice: '{choice}', using default model DeepSeek Chat")
        instance = DeepseekClient()
        global_model._model = instance
        print(f"  [OK] Using default: {instance.model_name}")

    print("=" * 50 + "\n")
    return instance


def get_model():
    """Get the current global model instance. Raises if not initialized."""
    if global_model._model is None:
        raise RuntimeError("Model not initialized. Call setup_model() first.")
    return global_model._model
