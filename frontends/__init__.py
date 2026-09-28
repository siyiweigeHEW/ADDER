"""Frontend registry.

Every backend under audit is described by one module in this package. A
frontend module owns the parts of the pipeline that are genuinely specific to a
tool stack -- how converters are located in its source tree, which helper code a
converter may delegate to, and the wording of the prompts -- while the pipeline
itself lives in the top-level modules.

Adding a backend means adding one module here and registering it in ``_MODULES``.
"""
from . import openvino, tvm

_MODULES = (openvino, tvm)

_REGISTRY = {m.NAME.lower(): m for m in _MODULES}

DEFAULT = openvino.NAME.lower()


def available():
    """Names accepted by ``get_frontend``."""
    return sorted(_REGISTRY)


def get_frontend(name=None):
    """Return the frontend module for ``name`` (case-insensitive).

    Falls back to :data:`DEFAULT` when ``name`` is None or empty.
    """
    key = (name or DEFAULT).lower()
    if key not in _REGISTRY:
        raise SystemExit(
            f"Unknown backend '{name}'. Available: {', '.join(available())}"
        )
    return _REGISTRY[key]
