# TracIn Ghost — Project Guidelines

## Architecture

A functional Python library for calculating the mathematical influence of copyrighted training data on generative model outputs, using an optimized version of TracIn (ghost dot product).

### Directory Structure
- `src/` — Core library (hooks, math, indexer, inference, FAISS store)
- `testModels/<name>/` — One folder per model test (model.py, data.py, train.py, run_index.py, run_query.py)
- `tests/` — pytest suite (test_hooks_manager.py, test_math_utils.py, test_mock_pipeline.py)

### The 4 Engineering Rules
1. **Memory**: `torch.no_grad()` during extraction, `del` intermediates, `torch.cuda.empty_cache()` every batch
2. **Non-destructive hooks**: Always use `HookManager` context manager, hooks removed in `finally`
3. **Dim standardization**: All tensors flattened to `[Batch, Hidden]` via `_flatten_to_2d()` (mean pooling for 3D/4D)
4. **Types**: Strict type hints + Google-style docstrings on all public functions

### Key Math
- Ghost dot product: `<g1, g2> = <A1, A2> · <E1, E2>` where `g = vec(E * A^T)`
- Adam correction: `g_corrected = g / (sqrt(v_t) + eps)` on full ghost vector
- SJLT projection: Achlioptas 2003, `P[i,j] = sqrt(3/K) * {+1, 0, -1}` with probabilities `{1/6, 2/3, 1/6}`
- FAISS: Inner product (`IndexFlatIP`), NOT cosine — magnitude matters for TracIn scores

## Build and Test
```bash
# Activate environment
conda activate training_env

# Run tests
python -m pytest tests/ -v

# Train MNIST example
python testModels/mnist/train.py

# Build index
python testModels/mnist/run_index.py

# Query attribution
python testModels/mnist/run_query.py --input outputs/query_input.pt
```

## Conventions
- `src/` functions accept explicit parameters — no config dicts in the library API
- Model-specific wiring lives in `testModels/<name>/`, never in `src/`
- Error function `error_fn(logits, targets) -> E` is user-provided, not built into the library
- All `src/` imports use `from src.<module> import <thing>`
