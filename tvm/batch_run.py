"""
Batch runner: run TVM audit with DeepSeek v4 Flash, Qwen3.5-Flash, and
GPT-5.4-mini (5 runs each), running the 3 models IN PARALLEL (each model's
5 runs are sequential) so the whole batch takes ~1/3 of the wall time.
Output streams to batch_run.log.
"""
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

BASE_DIR = "[this directory]"
RUNNER = os.path.join(BASE_DIR, "run_with_model.py")
LOG_FILE = os.path.join(BASE_DIR, "batch_run.log")

_log_lock = threading.Lock()

# (model_number, model_name) used by run_with_model.py
MODELS = [
    ("4", "DeepSeek v4 Flash"),
    ("3", "Qwen3.5-Flash"),
    ("5", "GPT-5.4-mini"),
]
RUNS_PER_MODEL = 5
TOTAL_RUNS = len(MODELS) * RUNS_PER_MODEL


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line)
    with _log_lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _append(text):
    with _log_lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(text)


def run_model(model_num, model_name, run_index, total):
    log(f"{'=' * 60}")
    log(f"  Start: {model_name} run {run_index}/{total}")
    log(f"{'=' * 60}")
    _append(f"\n--- {model_name} run {run_index}/{total} ---\n")
    with subprocess.Popen(
        ["python3", "-u", RUNNER, model_num],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, text=True, cwd=BASE_DIR,
    ) as proc:
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                _append(line + "\n")
        proc.wait()
    if proc.returncode == 0:
        log(f"  [OK] {model_name} run {run_index}/{total} finished")
    else:
        log(f"  [ERROR] {model_name} run {run_index}/{total} return code: {proc.returncode}")
    log("")


def run_model_runs(model_num, model_name, start_index):
    for i in range(1, RUNS_PER_MODEL + 1):
        run_model(model_num, model_name, start_index + i - 1, TOTAL_RUNS)


# Clear log
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)

log("Batch run started (3 models in parallel)")
log(f"{MODELS[0][1]} x {RUNS_PER_MODEL} runs, {MODELS[1][1]} x {RUNS_PER_MODEL} runs, "
    f"{MODELS[2][1]} x {RUNS_PER_MODEL} runs\n")

with ThreadPoolExecutor(max_workers=len(MODELS)) as ex:
    for idx, (model_num, model_name) in enumerate(MODELS):
        ex.submit(run_model_runs, model_num, model_name, idx * RUNS_PER_MODEL + 1)

log(f"{'=' * 60}")
log(f"  All {TOTAL_RUNS} runs completed!")
log(f"  Log: {LOG_FILE}")
log(f"{'=' * 60}")
