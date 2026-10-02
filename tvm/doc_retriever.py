"""Find the block in a framework's doc dump whose operator name best matches."""
import os
import re
import Levenshtein


def _norm(s):
    """Lowercase and remove underscores/spaces for lenient matching."""
    return re.sub(r'[\s_]+', '', s.lower())

# How many recent definitions to offer when there is no opset to select on.
RECENT_DEFINITIONS = 3

# A per-version section anchor, e.g. `#l-onnx-op-abs-13` (ONNX operator reference pages
# carry one section per opset).
_OPSET_ANCHOR_RE = re.compile(r'#l-onnx-op-.*-(\d+)$')


# Known cross-framework naming variants (keys and values are NORMALIZED names,
# i.e. lowercased and stripped of underscores).
_ALIAS_TABLE = {
    'cumsum': {'cumsum'},
    'cum_sum': {'cumsum'},
    'reshape2': {'reshape', 'reshape2'},
    'transpose2': {'transpose', 'transpose2'},
    'nonzero': {'nonzero'},
    'non_zero': {'nonzero'},
    'generateproposalsv2': {'generateproposals', 'generateproposalsv2'},
    'generateproposals': {'generateproposals', 'generateproposalsv2'},
    'deformableconv2d': {'deformableconv', 'deformableconv2d', 'deformconv', 'deformconv2d'},
    'deformableconv': {'deformableconv', 'deformableconv2d', 'deformconv', 'deformconv2d'},
    'deformconv': {'deformableconv', 'deformconv'},
    'gridsampler': {'gridsample', 'gridsampler'},
    'gridsample': {'gridsample', 'gridsampler'},
    'batchnorm': {'batchnorm', 'batchnormalization'},
    'batchnormalization': {'batchnorm', 'batchnormalization'},
    'groupnorm': {'groupnorm', 'groupnormalization'},
    'groupnormalization': {'groupnorm', 'groupnormalization'},
    'instancenorm': {'instancenorm', 'instancenormalization'},
    'instancenormalization': {'instancenorm', 'instancenormalization'},
    'roialign': {'roialign'},
    'logsoftmax': {'logsoftmax'},
}


def _aliases(op_name):
    """Candidate normalized names for an op (generic stripping + known table)."""
    base = _norm(op_name)
    al = {base}
    stripped = re.sub(r'\d+$', '', base)  # reshape2 -> reshape
    if stripped and stripped != base:
        al.add(stripped)
    for extra in _ALIAS_TABLE.get(base, ()):
        al.add(extra)
    return al


def _parse_url_filename(url_line):
    """Extract the op name from a doc URL, handling torch/onnx/paddle conventions."""
    temp = url_line.split('/')[-1]
    temp = temp.split('#')[0]
    if temp.lower().endswith('.html'):
        temp = temp[:-5]
    temp = re.sub(r'_cn$', '', temp, flags=re.IGNORECASE)  # paddle _cn suffix
    temp = temp.rstrip('_')                                # leftover from name__cn
    if '__' in temp:                                       # onnx __Name convention
        temp = temp.split('__')[-1]
    temp = temp.split('.')[-1]                             # torch.namespaced.fn
    return temp.strip()


def _parse_anchor_opset(url_line):
    """The opset a versioned anchor names, or None for a block that carries no version.

    A reference page for a versioned specification has one section per revision, and the
    dump keeps the anchor in the URL (`...onnx__Abs.html#l-onnx-op-abs-13`), so the version
    survives into the dump. Anchors that are not per-version (`#l-onnx-doc-abs`) give None.
    """
    m = _OPSET_ANCHOR_RE.search(url_line)
    return int(m.group(1)) if m else None


def get_doc_from_file(file_path, op_name, brand='onnx', opset=None):
    """The definition to judge `op_name` against.

    A block whose URL names an opset is a versioned definition. When `opset` is given --
    the opset the frontend under audit supports -- the newest versioned definition not
    above it is returned, which is how a versioned specification applies a definition from
    the revision that introduced it until the next one. Without a target opset the newest
    few definitions are returned together, so a versioned operator is still documented
    instead of being judged against whichever block the page happened to list first. A dump
    with no versioned block for this operator falls back to the highest-scoring block.
    """
    if not os.path.exists(file_path):
        return f"Error: {file_path} not found."

    target_op = op_name.split("::")[-1].split(".")[-1]

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = content.split("==================================================")

    raw_pattern = re.compile(r'\b' + re.escape(target_op) + r'\b', re.IGNORECASE)
    aliases = _aliases(target_op)

    # (block index, match score, the opset this block is the definition of or None)
    candidates = []

    for i in range(1, len(blocks), 2):
        url_line = blocks[i].strip()
        url_filename = _parse_url_filename(url_line)
        norm_filename = _norm(url_filename)

        # 1) exact word-boundary match on the raw filename (most precise)
        if raw_pattern.search(url_filename):
            score = 1.0 + 0.5  # boundary bonus beats any fuzzy match
        else:
            # 2) alias / normalized fuzzy match
            if not norm_filename:
                continue
            score = None
            for alias in aliases:
                denom = max(len(alias), len(norm_filename))
                if denom == 0:
                    continue
                sim = 1 - Levenshtein.distance(alias, norm_filename) / denom
                if sim >= 0.85:
                    score = sim
                    break
            if score is None:
                continue

        candidates.append((i, score, _parse_anchor_opset(url_line)))

    chosen = []
    if opset is not None:
        # The newest definition not above the opset the frontend supports applies to it.
        applicable = [c for c in candidates if c[2] is not None and c[2] <= opset]
        if applicable:
            best = max(applicable, key=lambda c: (c[2], c[1]))
            chosen = [(best[0], best[1])]
    if not chosen:
        # No opset to select on, so offer the newest definitions rather than whichever
        # block happens to come first -- an operator with versioned definitions would
        # otherwise be judged against whatever version the page listed first.
        recent = []
        seen = set()
        for idx, score, anchor in sorted(candidates, key=lambda c: c[2] or 0, reverse=True):
            if anchor is None:
                continue
            body = blocks[idx + 1].strip()
            if body in seen:
                continue  # a dump that recorded one section per operator repeats itself
            seen.add(body)
            recent.append((idx, score))
            if len(recent) == RECENT_DEFINITIONS:
                break
        chosen = list(reversed(recent))
    if not chosen and candidates:
        best = max(candidates, key=lambda c: c[1])
        chosen = [(best[0], best[1])]

    if chosen:
        return "\n\n".join(
            f"Source URL: {blocks[idx].strip()}\n"
            f"(Similarity Score: {score:.2f})\n{blocks[idx + 1].strip()}"
            for idx, score in chosen
        )

    return f"Warning: No confident documentation found for operator '{op_name}' in {file_path}"
