"""TracIn Ghost — Copyright Attribution via Training Influence.

Public API:
    from src.hooks_manager import HookManager
    from src.math_utils import form_ghost_vectors, apply_adam_correction, project
    from src.indexer import build_index
    from src.inference import attribute
"""

from src.hooks_manager import HookManager
from src.math_utils import (
    apply_adam_correction,
    build_dense_projection,
    build_sjlt_matrix,
    form_ghost_vectors,
    load_adam_second_moment,
    project,
)
from src.indexer import build_index
from src.inference import attribute

__all__ = [
    "HookManager",
    "form_ghost_vectors",
    "apply_adam_correction",
    "load_adam_second_moment",
    "build_sjlt_matrix",
    "build_dense_projection",
    "project",
    "build_index",
    "attribute",
]
