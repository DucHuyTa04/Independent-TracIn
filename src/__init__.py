"""TracIn Ghost — Copyright Attribution via Training Influence.

Public API:
    from src.hooks_manager import HookManager
    from src.math_utils import form_ghost_vectors, apply_adam_correction, project
    from src.indexer import build_index
    from src.inference import attribute
"""

from src.config_utils import (
    TracInCheckpointCallback,
    find_adam_bias_param_key,
    find_adam_param_key,
    find_last_linear_layer,
    find_target_layer,
    last_checkpoint_paths,
    resolve_checkpoints,
    resolve_target_layer,
    smart_load_weights_into_model,
)
from src.error_functions import (
    classification_error,
    get_error_fn,
    regression_error,
)
from src.hooks_manager import HookManager, MultiLayerBackwardGhostManager
from src.math_utils import (
    apply_adam_correction,
    build_dense_projection,
    build_sjlt_matrix,
    concatenate_adam_second_moments,
    form_ghost_vectors,
    form_multi_layer_ghost_vectors,
    load_adam_inverse_sqrt_scale_matrix_ghost_layout,
    load_adam_second_moment,
    load_adam_second_moment_matrix_ghost_layout,
    load_adam_second_moment_with_bias,
    project,
)
from src.indexer import build_index, build_multi_checkpoint_index
from src.inference import attribute, attribute_multi_checkpoint

__all__ = [
    "TracInCheckpointCallback",
    "classification_error",
    "regression_error",
    "get_error_fn",
    "find_adam_bias_param_key",
    "find_adam_param_key",
    "find_last_linear_layer",
    "find_target_layer",
    "resolve_target_layer",
    "resolve_checkpoints",
    "last_checkpoint_paths",
    "smart_load_weights_into_model",
    "HookManager",
    "MultiLayerBackwardGhostManager",
    "form_ghost_vectors",
    "form_multi_layer_ghost_vectors",
    "apply_adam_correction",
    "load_adam_second_moment",
    "load_adam_second_moment_matrix_ghost_layout",
    "load_adam_second_moment_with_bias",
    "load_adam_inverse_sqrt_scale_matrix_ghost_layout",
    "concatenate_adam_second_moments",
    "build_sjlt_matrix",
    "build_dense_projection",
    "project",
    "build_index",
    "build_multi_checkpoint_index",
    "attribute",
    "attribute_multi_checkpoint",
]
