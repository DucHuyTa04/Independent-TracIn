---
description: "Use when editing src/ library files: hooks_manager.py, math_utils.py, indexer.py, inference.py, faiss_store.py. Covers the 4 engineering rules, memory management, hook safety, and type requirements."
applyTo: "src/**"
---
# src/ Library Rules

## The 4 Engineering Rules (mandatory)

1. **Memory (Rule 1)**: Wrap extraction in `torch.no_grad()`. After each batch: `del` activation/error/ghost tensors, call `torch.cuda.empty_cache()`.
2. **Non-destructive hooks (Rule 2)**: Always use `HookManager` as a context manager. Never register raw hooks. Hooks must be removed in `finally`.
3. **Dim standardization (Rule 3)**: All tensors entering `form_ghost_vectors` must be 2D `[Batch, Hidden]`. Use `_flatten_to_2d()` for 3D (Transformer) and 4D (CNN) inputs.
4. **Types (Rule 4)**: Every public function has type hints and a Google-style docstring with Args/Returns.

5. **Tensor ordering (ghost vs weight)**: Ghost vectors use `(H, C)` row-major from `form_ghost_vectors`. PyTorch `nn.Linear.weight` and Adam `exp_avg_sq` use `(C, H)`. Align second moments to ghost layout before `apply_adam_correction` (transpose then flatten); pass `weight_shape` into `load_adam_second_moment` when needed.

## API Design
- Functions accept explicit parameters, never config dicts
- `error_fn` is always user-provided: `(logits, targets) -> E`
- No model-specific code in src/ — that belongs in testModels/
