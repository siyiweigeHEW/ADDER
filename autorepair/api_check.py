#!/usr/bin/env python3
"""Verification of LLM fix patches: code validity check + semantic correctness check.

A candidate patch must pass two complementary checks before it is returned.

**Code validity** — the candidate is parsed by the backend's syntax checker (`py_compile`
for Python, `clang++ -fsyntax-only` for C++), and the calls made by its added lines are
resolved against a static index of the backend's source tree (read, not imported) plus the
standard library; the APIs that exist go to the LLM for a misuse review. A syntax error,
or a 'nonexistent' / 'misused' finding, is a failure.

**Semantic correctness** — the patched code is reviewed against the source requirement
the diagnosis said the original implementation violated. The candidate passes only when
the review supports that the requirement is now satisfied.

Failures from either check are fed back for a correction round until both pass or the
round limit is hit.

This module is the driver. Everything backend-specific lives in one module per tool stack
under `tvm/` and `openvino/`; adding a backend means adding a directory there and listing
it in `_BACKEND_DIRS`. Such a module provides:

    NAME, ROOT_ENV, ROOT_HINT, REVIEW_SYSTEM, SEMANTIC_SYSTEM, SELF_TEST_CASES, SYNTAX
    detect_root()               -> path or None
    build_index(root)           -> opaque index
    describe_index(index)       -> one-line summary
    extract_calls(code, index)  -> [call names]
    resolve(call, index)        -> (status, ns, leaf, info, close)
    describe(call, index)       -> (declaration, doc, source location)

`status` is `ok` / `missing` / `unverifiable`. A backend reports `missing` only where its
index is complete enough for that verdict to be trustworthy, so a real API that merely was
not enumerated is never reported as fabricated.
"""
import difflib
import importlib.util
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from llm_client import DeepseekV4FlashClient  # noqa: E402
import syntax_check  # noqa: E402

_BACKEND_DIRS = ("tvm", "openvino")


def _load_backend(subdir):
    """Load a backend by path: both are called `api_index`, and making them importable
    packages would put `autorepair/tvm` ahead of the real `tvm` on `sys.path`.
    """
    path = os.path.join(_HERE, subdir, "api_index.py")
    spec = importlib.util.spec_from_file_location(f"_api_index_{subdir}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BACKENDS = [_load_backend(d) for d in _BACKEND_DIRS]
BACKENDS = {mod.NAME.upper(): mod for mod in _BACKENDS}
DEFAULT_BACKEND = _BACKENDS[0].NAME.upper()

CODE_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\s*\n(.*?)```", re.DOTALL)

# Correction rounds allowed per diagnosed bug. The initial candidate is generated before
# the loop, so the patch-generation budget is DEFAULT_ROUNDS + 1 = 5 attempts.
DEFAULT_ROUNDS = 4


def get_backend(name=None):
    key = (name or DEFAULT_BACKEND).upper()
    if key not in BACKENDS:
        raise SystemExit(
            f"Unknown backend '{name}'. Known: {', '.join(sorted(BACKENDS))}"
        )
    return BACKENDS[key]


def _clean_block(text):
    """Trim blank edges, keeping the first line's indentation."""
    lines = text.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _extract_code(response):
    """First fenced block, whatever its language tag; the whole response if unfenced."""
    m = CODE_FENCE_RE.search(response)
    return (_clean_block(m.group(1)) + "\n") if m else _clean_block(response) + "\n"


def added_lines(original, fixed):
    """The `+` lines of the diff."""
    orig_lines = original.splitlines()
    fix_lines = fixed.splitlines()
    sm = difflib.SequenceMatcher(a=orig_lines, b=fix_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("insert", "replace"):
            for line in fix_lines[j1:j2]:
                yield line


# ---------------------------------------------------------------- index


_INDEX_CACHE = {}


def get_index(backend=None, root=None):
    """Index for `backend`, built on first use. Empty when the tree is not configured."""
    mod = get_backend(backend)
    root = root or mod.detect_root()
    if root is None:
        return None, {}, None
    key = (mod.NAME, root)
    if key not in _INDEX_CACHE:
        _INDEX_CACHE[key] = mod.build_index(root)
    return root, _INDEX_CACHE[key], mod


# ---------------------------------------------------------------- misuse review (LLM)


def _load_json(text):
    """Parse JSON out of a response that may wrap it in a code block or prose."""
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    payload = m.group(1) if m else text
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        pass
    m2 = re.search(r"\{.*\}", payload, re.DOTALL)
    if not m2:
        return None
    try:
        return json.loads(m2.group(0))
    except json.JSONDecodeError:
        return None


def _parse_verdicts(text):
    """Tolerantly parse the JSON returned by the misuse review LLM."""
    data = _load_json(text)
    verdicts = data.get("verdicts", []) if isinstance(data, dict) else []
    return [v for v in verdicts if isinstance(v, dict) and v.get("api")]


def review_usage(original, fixed, used_apis, index, mod, client, code_lang="python"):
    """Returns (verdicts, raw_response)."""
    added = "\n".join(added_lines(original, fixed)) or fixed
    doc_blocks = []
    for name in used_apis:
        decl, doc, src = mod.describe(name, index)
        if decl is None:
            doc_blocks.append(f"### {name}\n(document unavailable: {src})")
        else:
            excerpt = (doc or "")[:800].replace("`", "'")
            doc_blocks.append(f"### {name}\ndeclaration: `{decl}`\nsource: {src}\n"
                              f"summary: {excerpt}")

    prompt = f"""# API usage review (autorepair automatic verification)

The added/changed lines of the fixed code call the following APIs. Based on each API's
signature and summary, judge whether its usage is correct.

## Original code (the fix scope; use it to infer codebase conventions)
```{code_lang}
{original}
```

## Added/changed lines of the fixed code
```{code_lang}
{added}
```

## Documentation of the used APIs (signature + summary)
{chr(10).join(doc_blocks)}

## Judging criteria (any hit -> correct=false)
1. Parameter name, type, or count does not match the signature;
2. Return type/semantics do not match what the call site expects;
3. Clearly deviates from codebase conventions (e.g., passing a DataType object where a
   string dtype is expected, calling a language built-in where a {mod.NAME} API is
   expected, hard-coding behavior that should be parameterized);
4. The API does not exist in this version.

## Output (strict JSON, only JSON)
```json
{{"verdicts": [
  {{"api": "<full API name>", "correct": true, "issue": "", "suggestion": ""}}
]}}
```
"""
    resp = client.model_prediction(prompt, system=mod.REVIEW_SYSTEM)
    return _parse_verdicts(resp), resp


# ---------------------------------------------------------------- semantic review (LLM)


def build_semantic_prompt(requirement, cause, original, fixed, code_lang="python"):
    """Prompt asking whether the patched code satisfies the requirement it violated."""
    cause_block = cause or ("(not provided) infer the violated requirement from the "
                            "'Original code' and 'Source requirement' below.")
    return f"""# Semantic correctness review (autorepair automatic verification)

A patch was generated to repair the violation below. Judge whether the patched code now
satisfies the source operator definition it was meant to conform to.

## Diagnosed violation
{cause_block}

## Source requirement (the behaviour the patched code must satisfy)
{requirement}

## Original code (before the patch)
```{code_lang}
{original}
```

## Patched code (the candidate under review)
```{code_lang}
{fixed}
```

## Judging criteria (any hit -> satisfied=false)
1. The patched behaviour still violates the requirement above;
2. The patch touches only the area the diagnosis named but does not restore the required
   behaviour;
3. The patch introduces a new conflict between the patched behaviour and the requirement;
4. The patch suppresses the symptom without satisfying the requirement.

Do not report a violation for behaviour the requirement does not cover, and do not
restate code-validity problems (nonexistent or misused APIs) here.

## Output (strict JSON, only JSON)
```json
{{"satisfied": true, "feedback": "", "conflicts": []}}
```
"""


def parse_semantic_verdict(text):
    """Returns (satisfied, feedback, conflicts). `satisfied` is None when the review
    returned nothing parsable, which is reported rather than counted as a failure."""
    data = _load_json(text)
    if not isinstance(data, dict) or "satisfied" not in data:
        return None, "the review did not return a parsable verdict", []
    feedback = str(data.get("feedback") or "").strip()
    conflicts = [str(c).strip() for c in (data.get("conflicts") or []) if str(c).strip()]
    return bool(data["satisfied"]), feedback, conflicts


def review_semantics(requirement, cause, original, fixed, mod, client, code_lang="python"):
    """Returns (satisfied, feedback, conflicts, raw_response)."""
    prompt = build_semantic_prompt(requirement, cause, original, fixed, code_lang)
    resp = client.model_prediction(prompt, system=mod.SEMANTIC_SYSTEM)
    satisfied, feedback, conflicts = parse_semantic_verdict(resp)
    return satisfied, feedback, conflicts, resp


# ---------------------------------------------------------------- correction loop


def _format_issue(issue):
    """issue = (api, kind, detail). kind: missing | misuse | syntax."""
    api, kind, detail = issue
    if kind == "syntax":
        return (
            f"❌ **syntax error** in the candidate translation unit: {detail}. Fix the "
            f"syntax before re-submitting."
        )
    if kind == "missing":
        _st, ns, leaf, info, close = detail
        extra = f"; similar APIs: {', '.join(close)}" if close else ""
        if isinstance(info, dict):
            where = f"not found in namespace `{ns}` ({len(info['names'])} APIs)"
        else:
            where = "not declared anywhere in the scanned sources"
        return (
            f"❌ **nonexistent API**: `{api}` -- {where}{extra}. Use a real API or an "
            f"equivalent low-level implementation."
        )
    return (
        f"⚠ **suspected misuse**: `{api}` -- {detail.get('issue', '')}"
        + (f"  suggestion: {detail.get('suggestion', '')}"
           if detail.get("suggestion") else "")
    )


def build_correction_prompt(original, previous_fixed, issues, code_lang="python",
                            semantic_feedback=""):
    """Prompt asking the LLM to re-output the code with the reported problems fixed."""
    issues_text = "\n".join(f"{i + 1}. {_format_issue(it)}" for i, it in enumerate(issues))
    if not issues_text:
        issues_text = "(none)"
    semantic_block = ("\n## Unresolved requirement (semantic review)\n"
                      + semantic_feedback + "\n") if semantic_feedback else ""
    return f"""# Fix the problems in the previous repair round (autorepair automatic
verification feedback)

Your previous repair round has the problems below. Fix each one, then re-output the
revised **complete code**.

## Detected problems
{issues_text}
{semantic_block}
## Original code (the fix scope; the output must stay aligned to this scope, keep every
other line byte-for-byte)
```{code_lang}
{original}
```

## Your previously output fixed code (contains the above problems)
```{code_lang}
{previous_fixed}
```

## Correction constraints
1. Modify only the lines directly related to the above problems; preserve every other
   line (comments, blank lines, indentation, function signatures) byte-for-byte.
2. Only use APIs that actually exist in this version; do not introduce a nonexistent
   API or misuse an API again.
3. For a nonexistent API: use an equivalent existing API or an implementation approach
   already used in the codebase.
4. For a misused API: fix the parameter types/count/semantics or the return handling
   per the suggestion.
5. If an unresolved requirement is listed above, change the behaviour so that the patched
   code satisfies it.

## Output format
A one-sentence explanation outside the code block + a ```{code_lang} code block
(the corrected complete code)
"""


def run_verification(original, fixed, system, opts=None):
    """Main verification flow: code validity + semantic correctness, with a shared
    correction loop. Returns (final fixed_code, report text).
    """
    opts = opts or {}
    mod = get_backend(opts.get("backend"))
    root, index, _mod = get_index(mod.NAME, opts.get("api_root"))
    max_rounds = int(opts.get("rounds", DEFAULT_ROUNDS))
    no_review = bool(opts.get("no_review"))
    requirement = (opts.get("requirement") or "").strip()
    cause = (opts.get("cause") or "").strip()
    semantic_off = bool(opts.get("no_semantic"))
    run_semantic = bool(requirement) and not semantic_off
    code_lang = opts.get("code_lang") or "python"
    render_patched = opts.get("render_patched")
    syntax = getattr(mod, "SYNTAX", {}) or {}
    syntax_dirs = syntax_check.include_dirs_for(root, syntax, opts.get("frontend"),
                                                opts.get("src_file"))
    client = DeepseekV4FlashClient()

    report = []
    report.append("# Verification Report (autorepair)\n")
    report.append(f"- backend: {mod.NAME}")
    if index:
        report.append(f"- API source root: `{root}`")
        report.append(f"- static index: {mod.describe_index(index)}")
    else:
        report.append(f"- static index: none -- {mod.ROOT_ENV} is not set to a "
                      f"{mod.ROOT_HINT}, so only the runtime check and the LLM review run")
    report.append(f"- code validity check: API existence (static scan of the source tree "
                  f"+ runtime inspect)"
                  f"{' + LLM misuse review' if not no_review else ' (LLM review skipped)'}")
    checker = syntax.get("checker")
    report.append(f"- syntax check: {checker or 'no checker declared for this backend'}"
                  + (f" ({len(syntax_dirs)} include dir(s))" if checker == "cpp" else ""))
    if run_semantic:
        report.append("- semantic correctness check: patched behaviour vs the source "
                      "requirement (LLM review)")
    elif semantic_off:
        report.append("- semantic correctness check: skipped (--no-semantic)")
    else:
        report.append("- semantic correctness check: skipped (no source requirement was "
                      "supplied)")
    report.append(f"- patch-generation budget: {max_rounds + 1} attempts (the initial "
                  f"candidate plus up to {max_rounds} revisions)\n")

    cur = fixed
    done_ok = False
    for round_idx in range(max_rounds + 1):
        added = list(added_lines(original, cur))
        calls = mod.extract_calls("\n".join(added), index) or mod.extract_calls(cur, index)
        calls = sorted(set(calls))

        report.append(f"## Round {round_idx} check\n")
        if not calls:
            report.append("The added/changed lines introduce no new API calls; the code "
                          "validity check has nothing to verify.\n")
        results = [mod.resolve(c, index) for c in calls]
        missing = []
        used = []
        verdicts = []
        if calls:
            report.append("| API | Result | Note |")
            report.append("|---|---|---|")
        for c, r in zip(calls, results):
            if r[0] == "ok":
                used.append(c)
                report.append(f"| `{c}` | ✓ exists | namespace `{r[1]}` |")
            elif r[0] == "missing":
                _st, ns, leaf, info, close = r
                close_txt = f"; similar: {', '.join(close)}" if close else ""
                if isinstance(info, dict):
                    note = f"namespace `{ns}` has {len(info['names'])} APIs; " \
                           f"`{leaf}` not found"
                else:
                    note = "not declared anywhere in the scanned sources"
                missing.append((c, "missing", r))
                report.append(f"| `{c}` | ❌ **nonexistent** | {note}{close_txt} |")
            else:
                report.append(f"| `{c}` | ? unverifiable | namespace not indexed / "
                              f"relative qualifier unknown |")

        if not no_review and used:
            verdicts, _raw = review_usage(original, cur, used, index, mod, client, code_lang)
            report.append(f"\n### LLM misuse review ({len(used)} APIs)\n")
            for v in verdicts:
                ok = v.get("correct", True)
                mark = "✓ correct usage" if ok else "⚠ **suspected misuse**"
                report.append(f"- `{v['api']}` {mark}"
                              + (f": {v.get('issue', '')}" if v.get("issue") else "")
                              + (f"  suggestion: {v.get('suggestion', '')}"
                                 if v.get("suggestion") else ""))
        missing_apis = {c for c, _, _ in missing}
        misused = [
            (v["api"], "misuse", v)
            for v in verdicts
            if not v.get("correct", True) and v["api"] not in missing_apis
        ]

        # --- syntax check (the rest of the code validity check) --------------
        syntax_issue = None
        patched_file = render_patched(cur) if render_patched else None
        target = patched_file or cur
        label = "the patched file" if patched_file else "the candidate snippet"
        if patched_file is None and syntax.get("checker") == "cpp":
            # A C++ snippet cannot be parsed on its own -- its types and headers live
            # outside it -- so say so instead of reporting the inevitable parse errors.
            status, detail = syntax_check.UNAVAILABLE, (
                "no patched source file was supplied and a C++ snippet cannot be parsed "
                "on its own; point the input at the real source file to enable this check")
        else:
            status, detail = syntax_check.run(target, syntax, syntax_dirs)
        report.append("\n### Syntax check\n")
        if status == syntax_check.OK:
            report.append(f"✓ {label} parses cleanly"
                          + (f" ({detail})" if detail else "") + ".\n")
        elif status == syntax_check.FAILED:
            report.append(f"❌ **syntax error**: {detail}\n")
            syntax_issue = ("candidate", "syntax", detail)
        else:
            report.append(f"? not checked: {detail}\n")

        # --- semantic correctness check -------------------------------------
        semantic_feedback = ""
        semantic_unresolved = False
        semantic_inconclusive = False
        if run_semantic:
            satisfied, feedback, conflicts, _raw = review_semantics(
                requirement, cause, original, cur, mod, client, code_lang)
            report.append("\n### Semantic correctness review\n")
            if satisfied is True:
                report.append("**Satisfied**: the patched behaviour meets the source "
                              "requirement.\n")
            elif satisfied is False:
                report.append("**Not satisfied**: "
                              + (feedback or "the review reported a conflict") + "\n")
                for c in conflicts:
                    report.append(f"- {c}")
                report.append("")
                detail = feedback or "the patched behaviour still violates the requirement"
                if conflicts:
                    detail += " (" + "; ".join(conflicts) + ")"
                semantic_feedback = detail
                semantic_unresolved = True
            else:
                report.append(f"? inconclusive: {feedback}\n")
                semantic_inconclusive = True

        issues = missing + misused + ([syntax_issue] if syntax_issue else [])
        if not issues and not semantic_unresolved and not semantic_inconclusive:
            done_ok = True
            report.append("\n**Passed**: code validity and semantic correctness both "
                          "hold.\n")
            break

        if semantic_inconclusive:
            report.append("The semantic review could not be parsed; stopping the iteration "
                          "without a correction round.\n")
            break

        found = []
        if issues:
            found.append(f"{len(issues)} code validity problem(s)")
        if semantic_unresolved:
            found.append("the requirement is still unresolved")
        report.append(f"\n**Found {', '.join(found)}; starting correction**\n")
        if round_idx >= max_rounds:
            report.append("(reached the correction round limit; keeping this round's "
                          "result)\n")
            break

        corr_prompt = build_correction_prompt(original, cur, issues, code_lang,
                                              semantic_feedback)
        report.append("### Correction prompt\n```text\n" + corr_prompt[:2000] + "\n```\n")
        resp = client.model_prediction(corr_prompt, system=system)
        new = _extract_code(resp)
        if not new or new.strip() == cur.strip():
            report.append("The LLM correction did not change the code; stopping the "
                          "iteration.\n")
            break
        cur = new

    if not done_ok:
        report.append("Not fully passed in the end (a syntax error, a missing/misused "
                      "API, or an unresolved requirement remains, or the round limit was "
                      "reached).\n")

    return cur, "\n".join(report)


# ---------------------------------------------------------------- self-test


def self_test(backend=None):
    mod = get_backend(backend)
    root = mod.detect_root()
    print(f"backend: {mod.NAME}")
    print(f"{mod.ROOT_ENV}: {root}")
    if not root:
        print(f"!! source tree not found; set {mod.ROOT_ENV} to {mod.ROOT_HINT}")
        return 1

    _root, index, _mod = get_index(mod.NAME, root)
    print(f"index: {mod.describe_index(index)}")

    print("\n== resolution self-test ==")
    fail = 0
    for name, expect in mod.SELF_TEST_CASES:
        st = mod.resolve(name, index)[0]
        if st != expect:
            fail += 1
        print(f"  [{'PASS' if st == expect else 'FAIL'}] {name:46s} -> "
              f"{st:12s} (expected {expect})")
    print(f"\n{len(mod.SELF_TEST_CASES) - fail}/{len(mod.SELF_TEST_CASES)} passed")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    arg = next((a for a in sys.argv[1:] if not a.startswith("-")), None)
    if "--self-test" in sys.argv:
        sys.exit(self_test(arg))
