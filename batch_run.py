"""Batch runner: run the audit with several models in parallel.

Each model's runs are sequential, but the models run concurrently, so the batch
takes roughly the wall time of a single model's runs. Output streams to
batch_run.log.

Usage: python batch_run.py [--backend <name>]
"""
import argparse
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(BASE_DIR, "run_with_model.py")
LOG_FILE = os.path.join(BASE_DIR, "batch_run.log")

_log_lock = threading.Lock()

# (model_number, model_name) understood by run_with_model.py
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


def run_model(model_num, model_name, run_index, total, backend=None):
    log("=" * 60)
    log(f"  Start: {model_name} run {run_index}/{total}")
    log("=" * 60)
    _append(f"\n--- {model_name} run {run_index}/{total} ---\n")

    cmd = ["python3", "-u", RUNNER, model_num]
    if backend:
        cmd += ["--backend", backend]

    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
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


def run_model_runs(model_num, model_name, start_index, backend=None):
    for i in range(1, RUNS_PER_MODEL + 1):
        run_model(model_num, model_name, start_index + i - 1, TOTAL_RUNS, backend)


def main(argv=None):
    from frontends import available

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default=os.environ.get("AUDIT_BACKEND"),
                        help=f"tool stack under audit ({', '.join(available())})")
    args = parser.parse_args(argv)

    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    log(f"Batch run started (backend: {args.backend or 'default'})")
    log(", ".join(f"{name} x {RUNS_PER_MODEL} runs" for _, name in MODELS) + "\n")

    with ThreadPoolExecutor(max_workers=len(MODELS)) as ex:
        for idx, (model_num, model_name) in enumerate(MODELS):
            ex.submit(run_model_runs, model_num, model_name,
                      idx * RUNS_PER_MODEL + 1, args.backend)

    log("=" * 60)
    log(f"  All {TOTAL_RUNS} runs completed!")
    log(f"  Log: {LOG_FILE}")
    log("=" * 60)


if __name__ == "__main__":
    main()
