"""
Run TVM main.py with a pre-selected model (non-interactive).
Usage: python3 run_with_model.py <model_number>
  model_number: 1=DeepSeek Chat, 2=Qwen3.7-Max, 3=Qwen3.5-Flash, 4=DeepSeek v4 Flash, 5=GPT-5.4-mini
"""
import sys
import os

model_num = sys.argv[1] if len(sys.argv) > 1 else "3"

from llm_client import (QwenFlashClient, DeepseekV4FlashClient, QwenMaxClient,
                         DeepseekClient, KamiapiGPTClient, global_model)

if model_num == "3":
    global_model._model = QwenFlashClient()
elif model_num == "4":
    global_model._model = DeepseekV4FlashClient()
elif model_num == "2":
    global_model._model = QwenMaxClient()
elif model_num == "1":
    global_model._model = DeepseekClient()
elif model_num == "5":
    global_model._model = KamiapiGPTClient()
else:
    print(f"Unknown model: {model_num}")
    sys.exit(1)

print(f"Model: {global_model.model_name}")

import main
main.setup_model = lambda: None
main.main()
