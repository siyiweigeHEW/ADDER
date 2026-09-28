"""Step 1: extract the converter implementations of the selected backend.

The work is entirely backend-specific -- a C++ frontend is scanned for files,
a Python frontend has an explicit operator-to-implementation mapping -- so this
module only dispatches to the active frontend module.
"""
from frontends import get_frontend


def get_all_front_converters(backend=None, target_fronts=None):
    """Return ``{front_name: (code_body_dict, api_map_dict)}``.

    ``target_fronts`` defaults to the fronts declared by the backend.
    """
    fe = get_frontend(backend)
    return fe.get_all_front_converters(target_fronts or fe.TARGET_FRONTS)
