#!/usr/bin/env python3
"""API verification of LLM fix patches: existence check + misuse review + correction loop.

Targets two kinds of LLM mistakes on APIs: 'fabrication' (using an API that does not
exist) and 'misuse' (using an existing API incorrectly):

1. After the patch is generated, extract `a.b.c(...)`-style API calls from the
   **added lines** of the fixed code;
2. Resolve the call's namespace and check whether the leaf exists in it. Two sources
   are consulted:
   - a **static index** of the backend's own API tree, when the backend declares one.
     The tree is scanned rather than imported, because these frameworks are normally
     not importable in the analysis environment;
   - **runtime `inspect`** for modules that can be imported (numpy, functools, ...);
3. For APIs that exist, extract their signature/docstring and ask the LLM to review
   whether they are misused;
4. When an API is 'nonexistent' or 'misused', feed the finding back to the LLM, asking
   it to re-output the fixed code; iterate until it passes or the round limit is hit.

Everything backend-specific comes from the active frontend module's ``API_VERIFY``
profile (see ``frontends/``), so this file names no tool stack: the tool stack name,
where its API tree lives, and which namespaces are worth indexing are all configurable
there. A backend that declares no namespaces gets no static check -- only the runtime
check and the LLM review run.

Self-test: ``python api_check.py --self-test [backend]`` (no LLM; exercises the static
index and the resolution logic against the profile's ``self_test_cases``).
"""
import ast
import difflib
import importlib
import inspect
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

# The repository root is needed for `frontends` and `prompts`, but it must come
# *after* this directory: both trees contain a `llm_client` module and they are not
# interchangeable, so `autorepair/llm_client.py` has to win the lookup.
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

import prompts  # noqa: E402
from frontends import get_frontend  # noqa: E402
from llm_client import DeepseekV4FlashClient  # noqa: E402

# ---------------------------------------------------------------- constants

# Runtime inspect roots: alias as written in code -> real module name. These are
# ordinary importable modules, independent of the backend under audit.
RUNTIME_ROOTS = {
    "np": "numpy",
    "_np": "numpy",
    "numpy": "numpy",
    "functools": "functools",
    "math": "math",
    "operator": "operator",
    "re": "re",
    "os": "os",
    "warnings": "warnings",
    "json": "json",
    "collections": "collections",
    "collections.abc": "collections.abc",
}

DOTTED_CALL_RE = re.compile(r"(?<![.\w])([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)\s*\(")

CODE_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\s*\n(.*?)```", re.DOTALL)

REVIEW_SYSTEM_TEMPLATE = (
    "You are a meticulous [ToolStack] frontend developer reviewing whether the APIs a "
    "patch calls are used correctly. Return only the JSON object."
)


def _clean_block(text):
    """Strip leading/trailing blank lines and trailing whitespace, but preserve the
    first line's indentation (same as repair.py's clean_block)."""
    lines = text.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _extract_code(response):
    """Extract the first fenced code block from the response (the fence language is not
    enforced); returns the whole response when there is no fence (same as repair.py)."""
    m = CODE_FENCE_RE.search(response)
    return (_clean_block(m.group(1)) + "\n") if m else _clean_block(response) + "\n"


# ---------------------------------------------------------------- profile


def api_profile(backend=None):
    """The active backend's API_VERIFY profile."""
    return get_frontend(backend).API_VERIFY


def call_roots(profile):
    """Dotted-call roots to resolve: the runtime modules plus the profile's extras."""
    return set(RUNTIME_ROOTS) | set(profile.get("extra_call_roots") or ())


def _is_api_root(path, profile):
    """True when `path` looks like the API tree this profile describes."""
    marker = profile.get("root_marker")
    if not marker or not path:
        return False
    return os.path.isfile(os.path.join(path, *marker))


def detect_api_root(profile):
    """Locate the backend's API tree, or None.

    The location comes from the profile's `root_env` environment variable; nothing is
    guessed, because scanning the wrong tree would produce wrong existence verdicts.
    """
    if not profile.get("namespaces"):
        return None

    env = os.environ.get(profile["root_env"])
    return env if _is_api_root(env, profile) else None


# ---------------------------------------------------------------- static index

_INDEX_CACHE = {}


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def scan_defs(filepath):
    """Return a list of (name, filepath): all top-level `def name` in the file
    (respecting __all__)."""
    if not os.path.isfile(filepath):
        return []
    try:
        tree = ast.parse(_read(filepath))
    except SyntaxError:
        return []
    all_names = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(
            node.targets[0], ast.Name
        ) and node.targets[0].id == "__all__":
            try:
                all_names = [e.value for e in node.value.elts]
            except AttributeError:
                all_names = None
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if all_names is None or node.name in all_names:
                out.append((node.name, filepath))
    return out


def parse_reexports(pkg_dir):
    """Parse the re-exports of a package's __init__.py, building the set of API names
    and a name -> source-file map.

    Handles two styles:
      from .binary import (add, subtract, ...)    # explicit list
      from .conv1d import *                       # star: scan all defs in the imported file
    Also adds: all top-level defs in every *.py of the package (to cover APIs that are
    reachable as namespace attributes even if not re-exported).
    Returns (names:set, src_map:dict[name->filepath]).
    """
    names = set()
    src_map = {}
    init = os.path.join(pkg_dir, "__init__.py")
    if os.path.isfile(init):
        try:
            tree = ast.parse(_read(init))
        except SyntaxError:
            tree = None
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level == 1:
                    if node.module:  # from .sub import (...)
                        sub = node.module.replace(".", "/")
                        for alias in node.names:
                            if alias.name == "*":
                                for nm, f in scan_defs(os.path.join(pkg_dir, sub + ".py")):
                                    names.add(nm)
                                    src_map.setdefault(nm, f)
                            else:
                                names.add(alias.name)
                                src_map.setdefault(
                                    alias.name, os.path.join(pkg_dir, sub + ".py")
                                )
                    else:  # from . import a, b, c (submodule names)
                        for alias in node.names:
                            names.add(alias.name)
    # Supplement: all defs in the package's top-level files (including non-re-exported),
    # to reduce false negatives
    if os.path.isdir(pkg_dir):
        for f in sorted(os.listdir(pkg_dir)):
            if f.endswith(".py"):
                for nm, fp in scan_defs(os.path.join(pkg_dir, f)):
                    names.add(nm)
                    src_map.setdefault(nm, fp)
    return names, src_map


def build_index(api_root, profile):
    """Build a namespace -> {names, src, dir} index.

    The namespaces come from the profile. Sub-package directories (with __init__.py)
    under each of them are indexed too, so a namespace declared as `pkg.op` also
    yields `pkg.op.image` and so on.

    Returns an empty index when the profile declares no namespaces, which disables
    the static existence check for that backend.
    """
    namespaces = {}

    def add_ns(ns_name, rel):
        pkg_dir = os.path.join(api_root, rel)
        if not os.path.isdir(pkg_dir):
            return
        names, src = parse_reexports(pkg_dir)
        namespaces[ns_name] = {"names": names, "src": src, "dir": pkg_dir}

    for alias, rel in profile.get("namespaces") or ():
        add_ns(alias, rel)
        pkg_dir = os.path.join(api_root, rel)
        if os.path.isdir(pkg_dir):
            for sub in sorted(os.listdir(pkg_dir)):
                subdir = os.path.join(pkg_dir, sub)
                if os.path.isdir(subdir) and os.path.isfile(
                    os.path.join(subdir, "__init__.py")
                ):
                    add_ns(f"{alias}.{sub}", f"{rel}/{sub}")
    return namespaces


def get_index(backend=None, api_root=None, profile=None):
    """Index for the active backend. Empty when the backend declares no namespaces."""
    prof = profile or api_profile(backend)
    if not prof.get("namespaces"):
        return {}
    root = api_root or detect_api_root(prof)
    if root is None:
        raise SystemExit(
            f"No API source tree found for {prof['tool_stack']}. Point "
            f"{prof['root_env']} (or repair.py's --api-root) at it -- the existence "
            f"check scans that tree statically. Pass --no-verify to skip the check."
        )
    if root not in _INDEX_CACHE:
        _INDEX_CACHE[root] = build_index(root, prof)
    return _INDEX_CACHE[root]


# ---------------------------------------------------------------- resolution/existence


def resolve_runtime(name):
    """Resolve at runtime via inspect (numpy/functools and other importable modules).

    Returns (status, ns, leaf, info, close) -- the same 5-tuple shape as `resolve`.
    """
    parts = name.split(".")
    root = RUNTIME_ROOTS.get(parts[0], parts[0])
    try:
        obj = importlib.import_module(root)
    except ImportError:
        return ("unverifiable", root, parts[0], None, None)
    for p in parts[1:]:
        if not hasattr(obj, p):
            return ("missing", root, p, None, None)
        obj = getattr(obj, p)
    return ("ok", root, parts[-1], obj, None)


def resolve(name, index, profile):
    """Resolve a dotted API name. Returns (status, ns, leaf, info, close).

    Always a 5-tuple, whatever the status, so callers can unpack unconditionally.
    `info` is the namespace record (static index) or the runtime object; `close`
    holds near-miss names, and is only populated for a static `missing` verdict.

    status: ok / missing / unverifiable
      ok          the API exists in the namespace
      missing     the namespace resolved, but the leaf is not in its API list (fabrication)
      unverifiable the namespace is not indexed, or the intermediate level cannot be
                   confirmed (to avoid false positives)

    A `missing` verdict is only issued for a namespace listed in the profile's
    `strict_namespaces`; elsewhere the leaf may simply not have been enumerated, so
    calling it a fabrication would risk a false positive.
    """
    strict_ns = profile.get("strict_namespaces") or set()

    if name in index:
        info = index[name]
        return ("ok", name, name, info, None)
    ns_list = sorted(index.keys(), key=len, reverse=True)
    for ns in ns_list:
        if name.startswith(ns + "."):
            rest = name[len(ns):].lstrip(".")
            info = index[ns]
            parts = rest.split(".")
            if len(parts) == 1:
                if parts[0] in info["names"]:
                    return ("ok", ns, parts[0], info, None)
                # difflib needs an ordered sequence; a set would make the suggestions
                # differ between runs.
                close = difflib.get_close_matches(parts[0], sorted(info["names"]), n=3)
                if ns in strict_ns:
                    return ("missing", ns, parts[0], info, close)
                return ("unverifiable", ns, parts[0], info, None)
            # Chained: the whole chain must be within the namespace to be ok, otherwise
            # do not confirm (to avoid false positives)
            if all(p in info["names"] for p in parts):
                return ("ok", ns, parts[-1], info, None)
            return ("unverifiable", ns, parts[0], info, None)
    root = name.split(".")[0]
    if root in call_roots(profile):
        return resolve_runtime(name)
    return ("unverifiable", None, root, None, None)


def extract_api_calls(code, profile):
    """Extract dotted API call names (e.g. `pkg.op.add(`) from code.

    Only keeps calls whose root is in `call_roots(profile)`; runtime objects such as
    bb./attr./cls./x. are skipped.
    """
    roots = call_roots(profile)
    out = []
    for m in DOTTED_CALL_RE.finditer(code):
        name = m.group(1)
        if name.split(".")[0] in roots:
            out.append(name)
    return out


def added_lines(original, fixed):
    """Return the lines added/changed in fixed relative to original (the + lines of the
    diff)."""
    orig_lines = original.splitlines()
    fix_lines = fixed.splitlines()
    sm = difflib.SequenceMatcher(a=orig_lines, b=fix_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("insert", "replace"):
            for line in fix_lines[j1:j2]:
                yield line


def extract_api_calls_from_lines(lines, profile):
    return extract_api_calls("\n".join(lines), profile)


# ---------------------------------------------------------------- API docs


def extract_def_doc(filepath, name):
    """Extract the signature and docstring of `def name` from the source file.
    Returns (sig, doc, lineno) or None."""
    if not os.path.isfile(filepath):
        return None
    try:
        tree = ast.parse(_read(filepath))
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            sig = ast.unparse(node.args)
            doc = ast.get_docstring(node) or ""
            return (f"def {name}({sig})", doc, node.lineno)
    return None


def get_api_doc_text(name, index, profile):
    """Return the API's (signature, docstring summary, source). For feeding to the
    review LLM."""
    st, ns, leaf, info, _close = resolve(name, index, profile)
    if st != "ok":
        return (None, None, st)
    if isinstance(info, dict) and "src" in info:  # static namespace
        fp = info["src"].get(leaf)
        if fp:
            res = extract_def_doc(fp, leaf)
            if res:
                sig, doc, lineno = res
                return (sig, doc, f"{os.path.relpath(fp, info['dir'])}:{lineno}")
        return (None, None, "static-no-def")
    if info is not None:  # runtime object
        try:
            doc = inspect.getdoc(info) or ""
            return (f"def {leaf}(...)", doc, f"runtime {ns}")
        except Exception:
            return (None, None, "runtime-nodoc")
    return (None, None, st)


# ---------------------------------------------------------------- misuse review (LLM)


def _parse_verdicts(text):
    """Tolerantly parse the JSON returned by the review LLM (code block or raw JSON)."""
    if not text:
        return []
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    payload = m.group(1) if m else text
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        m2 = re.search(r"\{.*\}", payload, re.DOTALL)
        if not m2:
            return []
        try:
            data = json.loads(m2.group(0))
        except json.JSONDecodeError:
            return []
    verdicts = data.get("verdicts", []) if isinstance(data, dict) else []
    return [v for v in verdicts if isinstance(v, dict) and v.get("api")]


def review_usage(original, fixed, used_apis, index, profile, client, code_lang="python"):
    """Ask the LLM to review, against the API brief docs, whether there is any misuse.
    `code_lang` is the fence language for the embedded snippets. Returns
    (verdicts, raw_response)."""
    added = "\n".join(added_lines(original, fixed)) or fixed
    doc_blocks = []
    for name in used_apis:
        sig, doc, src = get_api_doc_text(name, index, profile)
        if sig is None:
            doc_blocks.append(f"### {name}\n(document unavailable: {src})")
        else:
            excerpt = doc[:800].replace("`", "'")
            doc_blocks.append(f"### {name}\nsignature: `{sig}`\nsource: {src}\n"
                              f"summary: {excerpt}")

    tool_stack = profile["tool_stack"]
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
   string dtype is expected, calling a language built-in where a {tool_stack} API is
   expected, hard-coding behavior that should be parameterized);
4. The API does not exist in this version.

## Output (strict JSON, only JSON)
```json
{{"verdicts": [
  {{"api": "<full API name>", "correct": true, "issue": "", "suggestion": ""}}
]}}
```
"""
    system = prompts.render(REVIEW_SYSTEM_TEMPLATE, {"ToolStack": tool_stack})
    resp = client.model_prediction(prompt, system=system)
    return _parse_verdicts(resp), resp


# ---------------------------------------------------------------- correction loop


def _format_issue(issue):
    """issue = (api, kind, detail). kind: missing | misuse."""
    api, kind, detail = issue
    if kind == "missing":
        st, ns, leaf, info, close = detail
        extra = f"; similar APIs: {', '.join(close)}" if close else ""
        ns_size = len(info["names"]) if isinstance(info, dict) else 0
        return (
            f"❌ **nonexistent API**: `{api}` -- not found in namespace `{ns}` "
            f"({ns_size} APIs){extra}. Use a real API or an equivalent low-level "
            f"implementation."
        )
    # misuse
    return (
        f"⚠ **suspected misuse**: `{api}` -- {detail.get('issue', '')}"
        + (f"  suggestion: {detail.get('suggestion', '')}"
           if detail.get("suggestion") else "")
    )


def build_correction_prompt(original, previous_fixed, issues, code_lang="python"):
    """Turn the detected API problems into a prompt asking the LLM to output the
    revised complete code. `code_lang` is the fence language for the embedded
    snippets."""
    issues_text = "\n".join(f"{i + 1}. {_format_issue(it)}" for i, it in enumerate(issues))
    return f"""# Fix the API problems in the previous repair round (autorepair automatic
verification feedback)

Your previous repair round has the following API problems. Fix each one, then re-output
the revised **complete code**.

## Detected problems
{issues_text}

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

## Output format
A one-sentence explanation outside the code block + a ```{code_lang} code block
(the corrected complete code)
"""


def run_verification(original, fixed, system, opts=None):
    """Main API verification flow. Returns (final fixed_code, report text)."""
    opts = opts or {}
    backend = opts.get("backend")
    profile = api_profile(backend)
    tool_stack = profile["tool_stack"]

    client = DeepseekV4FlashClient()
    api_root = opts.get("api_root")
    index = get_index(backend, api_root, profile)
    max_rounds = int(opts.get("rounds", 2))
    no_review = bool(opts.get("no_review"))
    code_lang = opts.get("code_lang") or "python"

    report = []
    report.append("# API Verification Report (autorepair)\n")
    report.append(f"- backend: {tool_stack}")
    if index:
        report.append(f"- API source root: `{api_root or detect_api_root(profile)}`")
    else:
        report.append("- static API index: none declared for this backend, so only the "
                      "runtime check and the LLM review apply")
    report.append(f"- verification method: existence check (static scan of the source "
                  f"tree + runtime inspect)"
                  f"{' + LLM misuse review' if not no_review else ' (LLM review skipped)'}")
    report.append(f"- correction round limit: {max_rounds}\n")

    cur = fixed
    done_ok = False
    for round_idx in range(max_rounds + 1):
        added = list(added_lines(original, cur))
        calls = extract_api_calls_from_lines(added, profile) or extract_api_calls(cur, profile)
        calls = sorted(set(calls))

        report.append(f"## Round {round_idx} check\n")
        if not calls:
            report.append("The added/changed lines introduce no new dotted API calls; "
                          "nothing to verify.\n")
            done_ok = True
            break

        results = [resolve(c, index, profile) for c in calls]
        missing = []
        used = []
        report.append("| API | Result | Note |")
        report.append("|---|---|---|")
        for c, r in zip(calls, results):
            if r[0] == "ok":
                used.append(c)
                report.append(f"| `{c}` | ✓ exists | namespace `{r[1]}` |")
            elif r[0] == "missing":
                st, ns, leaf, info, close = r
                ns_size = len(info["names"]) if isinstance(info, dict) else 0
                close_txt = f"; similar: {', '.join(close)}" if close else ""
                missing.append((c, "missing", r))
                report.append(f"| `{c}` | ❌ **nonexistent** | namespace `{ns}` has "
                              f"{ns_size} APIs; `{leaf}` not found{close_txt} |")
            else:
                report.append(f"| `{c}` | ? unverifiable | namespace not indexed / "
                              f"intermediate level unknown |")

        verdicts = []
        if not no_review and used:
            verdicts, raw = review_usage(original, cur, used, index, profile, client, code_lang)
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

        issues = missing + misused
        if not issues:
            done_ok = True
            report.append("\n**Passed**: no missing or misused APIs.\n")
            break

        report.append(f"\n**Found {len(issues)} API problems; starting correction**\n")
        if round_idx >= max_rounds:
            report.append("(reached the correction round limit; keeping this round's "
                          "result)\n")
            break

        corr_prompt = build_correction_prompt(original, cur, issues, code_lang)
        report.append("### Correction prompt\n```text\n" + corr_prompt[:2000] + "\n```\n")
        resp = client.model_prediction(corr_prompt, system=system)
        new = _extract_code(resp)
        if not new or new.strip() == cur.strip():
            report.append("The LLM correction did not change the code; stopping the "
                          "iteration.\n")
            break
        cur = new

    if not done_ok:
        report.append("Not fully passed in the end (missing/misused APIs remain or the "
                      "round limit was reached).\n")

    report_text = "\n".join(report)
    return cur, report_text


# ---------------------------------------------------------------- self-test


def self_test(backend=None):
    profile = api_profile(backend)
    api_root = detect_api_root(profile)
    print(f"backend: {profile['tool_stack']}")
    print(f"{profile['root_env']}: {api_root}")

    if not profile.get("namespaces"):
        print("!! this profile declares no namespaces, so the static existence check is "
              "disabled. Fill in API_VERIFY['namespaces'] in the frontend module to "
              "enable it.")
        return 1
    if not api_root:
        print(f"!! API tree not found; set {profile['root_env']} and retry")
        return 1

    idx = get_index(backend, api_root, profile)
    print(f"indexed namespaces: {len(idx)}")
    for ns in sorted(idx.keys()):
        print(f"  {ns:22s} APIs={len(idx[ns]['names'])}")

    cases = profile.get("self_test_cases") or []
    if not cases:
        print("(no self_test_cases declared in this profile; nothing to assert)")
        return 0

    print("\n== resolution self-test ==")
    fail = 0
    for name, expect in cases:
        st = resolve(name, idx, profile)[0]
        if st != expect:
            fail += 1
        print(f"  [{'PASS' if st == expect else 'FAIL'}] {name:22s} -> "
              f"{st:12s} (expected {expect})")
    print(f"\n{len(cases) - fail}/{len(cases)} passed")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    arg = next((a for a in sys.argv[1:] if not a.startswith("-")), None)
    if "--self-test" in sys.argv:
        sys.exit(self_test(arg))
