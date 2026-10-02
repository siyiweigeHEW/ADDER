#!/usr/bin/env python3
"""Syntax check of a candidate translation unit: `py_compile` for Python, `clang++
-fsyntax-only` for C++.

Three states, so a check that could not actually run is never reported as a failure:

    ok           the checker parsed the unit cleanly
    failed       the checker reported a syntax error, returned as diagnostics
    unavailable  the checker is absent, or the unit cannot be parsed on its own (a C++
                 file whose headers are not all on the include path, an indented fragment
                 with no full file to live in, an empty candidate)

Which checker a backend uses, and the include roots its units are parsed against, come
from the backend's `SYNTAX` mapping; this module holds no backend-specific path.
"""
import os
import py_compile
import re
import shutil
import subprocess
import tempfile

OK = "ok"
FAILED = "failed"
UNAVAILABLE = "unavailable"

_CXX_TIMEOUT = 180
_ERROR_RE = re.compile(r":\s*(?:fatal\s+)?error:")
_MISSING_RE = re.compile(r"file not found|No such file or directory")


def _diagnostics(text, limit=6):
    """The error lines, or the tail of the output when none is marked as an error."""
    lines = [ln.strip() for ln in (text or "").splitlines() if _ERROR_RE.search(ln)]
    if not lines:
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return " | ".join(lines[:limit])


def _python_diagnostics(text):
    """Compact `line N: <error>` out of a py_compile traceback, dropping the temp path."""
    lines = [ln.strip() for ln in (text or "").splitlines()]
    error = next((ln for ln in reversed(lines)
                  if ln.startswith(("SyntaxError", "IndentationError", "TabError"))), "")
    m = re.search(r", line (\d+)", text or "")
    lineno = m.group(1) if m else None
    if error and lineno:
        return f"line {lineno}: {error}"
    return error or _diagnostics(text)


def check_python(text):
    """Compile `text` the way the interpreter would. Returns (status, detail)."""
    if not (text or "").strip():
        return UNAVAILABLE, "the candidate is empty"
    first = next((ln for ln in text.splitlines() if ln.strip()), "")
    if first[:1].isspace():
        return UNAVAILABLE, ("the candidate is an indented fragment with no full file to "
                             "live in")
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "candidate.py")
        with open(src, "w", encoding="utf-8") as f:
            f.write(text)
        try:
            py_compile.compile(src, cfile=os.path.join(d, "candidate.pyc"), doraise=True)
        except py_compile.PyCompileError as exc:
            return FAILED, _python_diagnostics(str(exc))
        except (OSError, ValueError) as exc:
            return UNAVAILABLE, f"py_compile could not run: {exc}"
    return OK, ""


def check_cpp(text, include_dirs=(), std="c++17"):
    """Parse `text` as a C++ translation unit. Returns (status, detail)."""
    if not (text or "").strip():
        return UNAVAILABLE, "the candidate is empty"
    exe = shutil.which("clang++") or shutil.which("g++") or shutil.which("c++")
    if not exe:
        return UNAVAILABLE, "no clang++ / g++ / c++ on PATH"
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "candidate.cpp")
        with open(src, "w", encoding="utf-8") as f:
            f.write(text)
        cmd = [exe, "-fsyntax-only", "-w", f"-std={std}"]
        cmd += [f"-I{p}" for p in include_dirs if p]
        cmd.append(src)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=_CXX_TIMEOUT)
        except subprocess.TimeoutExpired:
            return UNAVAILABLE, f"{os.path.basename(exe)} timed out after {_CXX_TIMEOUT}s"
        except OSError as exc:
            return UNAVAILABLE, f"{os.path.basename(exe)} could not run: {exc}"
    output = ((proc.stderr or "") + (proc.stdout or "")).replace(d + os.sep, "")
    if proc.returncode == 0:
        return OK, os.path.basename(exe)
    if _MISSING_RE.search(output):
        return UNAVAILABLE, ("the unit's headers are not all on the include path, so it "
                             "cannot be parsed on its own")
    return FAILED, _diagnostics(output)


def run(text, spec, extra_include_dirs=()):
    """Dispatch on the backend's `SYNTAX` mapping. Returns (status, detail)."""
    checker = (spec or {}).get("checker")
    if checker == "python":
        return check_python(text)
    if checker == "cpp":
        dirs = list(spec.get("include_dirs") or []) + list(extra_include_dirs)
        return check_cpp(text, dirs, spec.get("std", "c++17"))
    return UNAVAILABLE, "this backend declares no syntax checker"


def include_dirs_for(root, spec, frontend=None, src_file=None):
    """The `-I` roots for a candidate: the backend's declared paths (relative to the
    checkout root, `{frontend}` substituted, absent ones skipped) plus the directory the
    patched file lives in, so its own quoted includes still resolve.
    """
    dirs = []
    for rel in (spec or {}).get("include_dirs") or ():
        if root is None:
            break
        path = os.path.join(root, rel.format(frontend=frontend or ""))
        if os.path.isdir(path):
            dirs.append(path)
    if src_file:
        own = os.path.dirname(os.path.abspath(src_file))
        if os.path.isdir(own):
            dirs.append(own)
    return dirs
