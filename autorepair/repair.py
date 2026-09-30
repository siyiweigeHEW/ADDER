#!/usr/bin/env python3
"""Manual input -> fix patch generation (TVM / OpenVINO frontend converter defect repair).

Usage:
    python repair.py                          # interactively enter all information
    python repair.py --dry-run                # only build/print the prompt; no LLM call
    python repair.py --backend TVM --frontend onnx --op Flatten \
                     --code-file code.py --doc-file doc.txt --cause-file cause.txt \
                     --src-file python/tvm/relax/frontend/onnx/onnx_frontend.py

Manual inputs:
    backend   backend (TVM / OPENVINO)
    frontend  frontend (onnx / torch / paddle ...)
    op        operator name (e.g., Flatten)
    code      original code to fix (only the parts related to the buggy operator)
    doc       key documentation information (only what is needed for the fix)
    cause     error cause attributed by the LLM (optional; if omitted, the repair LLM
              analyzes it on its own)
    src-file  path to the real source file (optional): anchors the patch to the real
              file and generates an applicable unified diff (line numbers/file headers
              are computed by the program, not dependent on LLM output format)

Design: the prompt only asks the LLM to output the 'fixed code'; it contains no diff
format requirements; the patch format (file headers, hunk line numbers) is generated
entirely by this script with difflib, deterministically.

Output: prompt.txt / response.md / fixed.py / fix.patch under repairs/{backend}_{frontend}_{op}/
"""
import argparse
import difflib
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_client import DeepseekV4FlashClient  # noqa: E402
import api_check  # noqa: E402   API existence check + misuse review + correction loop

OUT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repairs")
EOF = "EOF"  # end marker for multiline paste
CODE_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\s*\n(.*?)```", re.DOTALL)


def clean_block(text):
    """Strip leading/trailing blank lines and trailing whitespace, but **preserve the
    first line's indentation** (the first line of a code block may be `    @classmethod`)."""
    lines = text.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def read_file(path):
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        return clean_block(f.read())


def ask_line(title, default=None):
    hint = f"default {default}" if default else "required"
    val = input(f"{title} [{hint}]: ").strip()
    return val or default


def ask_text(title, optional=False):
    """Multiline input; paste and terminate with a line of EOF; optional input may be
    skipped by pressing Enter."""
    print(f"\n{title}")
    print(f"  paste content; end with a line {EOF}" + ("" if optional else " (required)"))
    lines = []
    while True:
        try:
            line = input("  > ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.strip() == EOF:
            break
        lines.append(line)
    text = clean_block("\n".join(lines))
    if not text and not optional:
        print("  !! required field cannot be empty; please re-enter")
        return ask_text(title, optional)
    return text or None


def build_prompt(backend, frontend, op, code, doc, cause):
    """Repair prompt: only asks for the 'fixed code'; the diff format is generated
    externally by the program."""
    cause_block = cause or ("(not provided) analyze the root cause from the 'Original "
                            "code' and 'Key documentation information' below.")
    return f"""# Fix the defect in the {backend}-{frontend} frontend converter "{op}"

## 1. Root cause of the defect (attribution)
{cause_block}

## 2. Original code (to be fixed; the fix must preserve the full extent of this code)
```python
{code}
```

## 3. Key documentation information (only what is needed for the fix)
{doc or "(not provided)"}

## 4. Output: the complete fixed code
The patch is generated automatically by the program; you only need to output the code
itself. Please follow this contract:
1. Scope alignment: output the complete code in the same scope as the "Original code";
   modify only the lines directly related to the root cause; preserve every other line
   (comments, blank lines, indentation, function signatures) byte-for-byte; do not
   reorder, rename, or refactor along the way.
2. Only use operators/interfaces that actually exist in this {backend} version. Prefer
   the concrete implementation approach given in "Key documentation information"; if
   unsure whether an API exists, use an equivalent low-level implementation (e.g., a
   topi operator + `bb.emit_te`), and note it outside the code block.
3. Outside the code block, first explain the root cause and the change in one sentence.
4. If the information is insufficient to determine the fix, state outside the code block
   what is missing; do not fabricate.

Output format: an explanation outside the code block + a ```python code block
(the complete fixed code)
"""


def extract_code(response):
    """Extract the ```python code block from the response (preserving the first line's
    indentation); return the whole response if there is no fence."""
    m = CODE_FENCE_RE.search(response)
    return (clean_block(m.group(1)) + "\n") if m else clean_block(response) + "\n"


ATTR_CALL_RE = re.compile(r"\battr\.([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def check_attr_accessors(original, fixed):
    """Heuristic check: whether the fixed code introduces `attr.<method>(` calls not
    used by the original code.

    A frontend attr is usually a dict (`attr.get(...)`); the LLM often fabricates
    Attrs methods such as `attr.get_int_tuple()`. This check only warns, it does not
    block — when hit, confirm the method actually exists.
    Returns the set of newly introduced attr method names.
    """
    orig = set(ATTR_CALL_RE.findall(original))
    new = set(ATTR_CALL_RE.findall(fixed)) - orig
    if new:
        print(f"  [warn] the fixed code introduces attr method calls not used by the "
              f"original code: {sorted(new)}")
        print("         frontend attr is usually a dict; confirm these methods actually "
              "exist; if unsure, use attr.get(...) instead")
    return new


def find_block(src, block):
    """Locate block by line in src (tolerating trailing-whitespace differences); return
    the (start, end) line range or None."""
    block_lines = [ln.rstrip() for ln in block.splitlines()]
    src_lines = [ln.rstrip() for ln in src.splitlines()]
    n = len(block_lines)
    for i in range(len(src_lines) - n + 1):
        if src_lines[i:i + n] == block_lines:
            return i, i + n
    return None


def patch_path(src_file):
    """Compute the patch-header path: when the tvm source contains `/python/`, truncate
    to start at `python/` (source-root-relative path, directly applicable with
    `patch -p1` from the source root); otherwise strip the leading / and use as-is."""
    if "/python/" in src_file:
        return src_file[src_file.index("/python/") + 1:]
    return src_file.lstrip("/")


def build_fix_patch(original, fixed, src_file=None):
    """Generate a unified diff with difflib.

    Prefers to replace the LLM-fixed code back into the real source file (line numbers
    and context are aligned automatically, producing an applicable patch); when no
    src-file is given or the snippet does not match, falls back to a snippet-level diff
    (for reference only).
    Returns (diff text, whether it is anchored to the real file).
    """
    original = original.strip("\n") + "\n"
    fixed = fixed.strip("\n") + "\n"
    if src_file and os.path.isfile(src_file):
        with open(src_file, encoding="utf-8") as f:
            src = f.read()
        span = find_block(src, original)
        if span is not None:
            src_lines = src.splitlines(True)
            start, end = span
            new_src = "".join(src_lines[:start] + fixed.splitlines(True) + src_lines[end:])
            path = patch_path(src_file)
            return ("".join(difflib.unified_diff(
                src.splitlines(True), new_src.splitlines(True),
                fromfile="a/" + path, tofile="b/" + path, n=3)), True)
        print(f"[warn] original code block matching the input not found in source file "
              f"{src_file}; falling back to a snippet-level diff")
    return ("".join(difflib.unified_diff(
        original.splitlines(True), fixed.splitlines(True),
        fromfile="a/<snippet>", tofile="b/<snippet>", n=3)), False)


def main():
    ap = argparse.ArgumentParser(description="Manual input -> generate frontend converter fix patch")
    ap.add_argument("--backend", help="backend: TVM / OPENVINO")
    ap.add_argument("--frontend", help="frontend: onnx / torch / paddle ...")
    ap.add_argument("--op", help="operator name: e.g., Flatten")
    ap.add_argument("--code-file", help="original code file (buggy operator related)")
    ap.add_argument("--doc-file", help="key documentation information file (optional)")
    ap.add_argument("--cause-file", help="LLM attribution file (optional)")
    ap.add_argument("--src-file", help="path to the real source file (optional; used to "
                                       "generate an applicable patch)")
    ap.add_argument("--dry-run", action="store_true", help="only build/print the prompt; "
                                                           "no LLM call")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip API verification (existence check + misuse review + "
                         "correction loop)")
    ap.add_argument("--no-review", action="store_true",
                    help="only run the static API existence check; skip the LLM misuse "
                         "review/correction")
    ap.add_argument("--verify-rounds", type=int, default=2,
                    help="max API correction rounds (default 2)")
    ap.add_argument("--tvm-root", help="tvm source python/tvm directory (defaults to the "
                                       "TVM_PYTHON_ROOT environment variable)")
    args = ap.parse_args()

    backend = args.backend or ask_line("Backend", "TVM")
    frontend = args.frontend or ask_line("Frontend", "onnx")
    op = args.op or ask_line("Operator")
    code = read_file(args.code_file) or ask_text(
        "Original code (only the parts related to the buggy operator, to be fixed)")
    doc = read_file(args.doc_file) or ask_text(
        "Key documentation information (only what is needed for the fix)", optional=True)
    cause = read_file(args.cause_file) or ask_text(
        "Error cause attributed by the LLM", optional=True)
    src_file = args.src_file
    if not src_file and sys.stdin.isatty():
        val = input("\nSource file path (optional; used to generate an applicable patch) "
                    "[Enter to skip]: ").strip()
        src_file = val or None

    prompt = build_prompt(backend, frontend, op, code, doc, cause)

    outdir = os.path.join(OUT_BASE, f"{backend}_{frontend}_{op}")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt)
    print(f"\n[prompt.txt] saved to {outdir}/")

    if args.dry_run:
        print("\n" + "=" * 60 + "\nPROMPT\n" + "=" * 60)
        print(prompt)
        print("=" * 60)
        print("(--dry-run: no LLM call)")
        return

    system = (
        f"You are a meticulous {backend} frontend converter developer. "
        "Write minimal surgical fixes: preserve every untouched line byte-for-byte, "
        "never rename or restructure code, and never use APIs that may not exist "
        f"in this {backend} version."
    )
    print(f"\n[llm] calling deepseek-v4-flash ...")
    response = DeepseekV4FlashClient().model_prediction(prompt, system=system)
    if not response.strip():
        raise SystemExit("LLM returned an empty response; check DEEPSEEK_API_KEY / network.")

    fixed_code = extract_code(response)
    check_attr_accessors(code, fixed_code)

    if not args.no_verify:
        print("\n[api_check] starting API verification (existence check + LLM misuse "
              "review + correction loop)...")
        fixed_code, api_report = api_check.run_verification(
            code, fixed_code, system,
            {"rounds": args.verify_rounds, "no_review": args.no_review,
             "tvm_root": args.tvm_root},
        )
        with open(os.path.join(outdir, "api_check.md"), "w", encoding="utf-8") as f:
            f.write(api_report)
        print(f"[api_check] done; report -> {outdir}/api_check.md")

    patch, anchored = build_fix_patch(code, fixed_code, src_file)

    with open(os.path.join(outdir, "response.md"), "w", encoding="utf-8") as f:
        f.write(response)
    with open(os.path.join(outdir, "fixed.py"), "w", encoding="utf-8") as f:
        f.write(fixed_code)
    with open(os.path.join(outdir, "fix.patch"), "w", encoding="utf-8") as f:
        f.write(patch)

    print(f"[output]  {outdir}/")
    print(f"  response.md  bytes={len(response.encode('utf-8'))}")
    print(f"  fixed.py     bytes={len(fixed_code.encode('utf-8'))}   (code fixed by the LLM)")
    print(f"  fix.patch    bytes={len(patch.encode('utf-8'))}  applyable={anchored}")
    if not args.no_verify:
        print("  api_check.md  API verification report (existence/misuse/correction rounds)")
    if not anchored:
        print("  [warn] fix.patch is not anchored to the real source file (--src-file "
              "missing or the snippet does not match); line numbers are snippet-local "
              "and cannot be applied directly to the real file")
    else:
        print("  anchored to the real source file; the patch can be applied with "
              "`patch -p1` / `git apply` (the script does not actually apply it)")


if __name__ == "__main__":
    main()
