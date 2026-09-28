"""Run the audit with a pre-selected model, non-interactively.

Usage: python run_with_model.py [model_number] [--backend <name>]

  model_number: 1=DeepSeek Chat, 2=Qwen3.7-Max, 3=Qwen3.5-Flash,
                4=DeepSeek v4 Flash, 5=GPT-5.4-mini

Any further arguments are forwarded to main.py.
"""
import sys

from llm_client import (DeepseekClient, DeepseekV4FlashClient, KamiapiGPTClient,
                        QwenFlashClient, QwenMaxClient, global_model)

MODELS = {
    "1": DeepseekClient,
    "2": QwenMaxClient,
    "3": QwenFlashClient,
    "4": DeepseekV4FlashClient,
    "5": KamiapiGPTClient,
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and not argv[0].startswith("-"):
        model_num, rest = argv[0], argv[1:]
    else:
        model_num, rest = "3", argv

    if model_num not in MODELS:
        raise SystemExit(f"Unknown model '{model_num}'. Available: {', '.join(sorted(MODELS))}")

    global_model._model = MODELS[model_num]()
    print(f"Model: {global_model.model_name}")

    import main as audit_main
    audit_main.setup_model = lambda: None
    audit_main.main(rest)


if __name__ == "__main__":
    main()
