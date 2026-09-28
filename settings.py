"""Runtime configuration.

The pipeline needs an official documentation dump per frontend. Those dumps are
not shipped with this repository -- they are large scraped snapshots of each
framework's reference documentation -- so they are read from a directory given
by an environment variable. See the README for how the dumps are expected to be
laid out.
"""
import os

DOC_ROOT_ENV = "FRONTEND_AUDIT_DOC_ROOT"
DOC_FILE_SUFFIX = "doc.txt"


def doc_root():
    """Directory holding the per-frontend documentation dumps."""
    root = os.environ.get(DOC_ROOT_ENV)
    if not root:
        raise SystemExit(
            f"{DOC_ROOT_ENV} is not set. Point it at a directory containing one "
            f"<frontend>{DOC_FILE_SUFFIX} file per frontend (e.g. onnxdoc.txt)."
        )
    return root


def doc_paths(front_names):
    """Map each frontend name to the path of its documentation dump."""
    root = doc_root()
    return {name: os.path.join(root, f"{name}{DOC_FILE_SUFFIX}") for name in front_names}
