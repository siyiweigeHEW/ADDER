"""
Doc retrieval module.
Searches a per-framework doc dump (URL-line-delimited blocks) for the block
that best matches an op name.

Robustness fixes over the original implementation:
  * Parses paddle `name_cn` / `name__cn` URL conventions correctly.
  * Normalizes names (case / underscores) so cross-framework naming variants
    match: cum_sum <-> cumsum, reshape2 <-> reshape, batch_norm <-> BatchNormalization,
    non_zero <-> nonzero, grid_sampler <-> grid_sample, etc.
  * Keeps an exact word-boundary match as the highest-precision route so short
    names like `sqrt` never fall through to `rsqrt`.
"""
import os
import re
import Levenshtein


def _norm(s):
    """Lowercase and remove underscores/spaces for lenient matching."""
    return re.sub(r'[\s_]+', '', s.lower())


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


def get_doc_from_file(file_path, op_name, brand='onnx'):
    """
    Retrieve the documentation block matching an operator name from a txt file,
    choosing the block with the highest similarity score.
    Prefer an exact word-boundary match (to prevent sqrt from mismatching rsqrt),
    then fall back to normalized/alias fuzzy matching (to resolve cross-framework
    naming differences such as cum_sum vs cumsum).
    """
    if not os.path.exists(file_path):
        return f"Error: {file_path} not found."

    target_op = op_name.split("::")[-1].split(".")[-1]

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = content.split("==================================================")

    raw_pattern = re.compile(r'\b' + re.escape(target_op) + r'\b', re.IGNORECASE)
    aliases = _aliases(target_op)

    best_idx = -1
    best_score = -1.0

    for i in range(1, len(blocks), 2):
        url_line = blocks[i].strip()
        url_filename = _parse_url_filename(url_line)
        norm_filename = _norm(url_filename)

        # 1) exact word-boundary match on the raw filename (most precise)
        if raw_pattern.search(url_filename):
            denom = max(len(target_op), len(url_filename))
            sim = 1 - (Levenshtein.distance(target_op.lower(), url_filename.lower()) / denom) if denom else 0.0
            score = 1.0 + 0.5  # boundary bonus beats any fuzzy match
            if score > best_score:
                best_score, best_idx = score, i
            continue

        # 2) alias / normalized fuzzy match
        if not norm_filename:
            continue
        for alias in aliases:
            denom = max(len(alias), len(norm_filename))
            if denom == 0:
                continue
            sim = 1 - Levenshtein.distance(alias, norm_filename) / denom
            if sim >= 0.85:
                if sim > best_score:
                    best_score, best_idx = sim, i
                break

    if best_idx != -1:
        url_line = blocks[best_idx].strip()
        return f"Source URL: {url_line}\n(Similarity Score: {best_score:.2f})\n{blocks[best_idx + 1].strip()}"

    return f"Warning: No confident documentation found for operator '{op_name}' in {file_path}"
