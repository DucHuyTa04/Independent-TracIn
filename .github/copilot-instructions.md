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
- Adam correction: `g_corrected = g / (sqrt(v_t) + eps)` on full ghost vector; `exp_avg_sq` is stored in PyTorch weight layout `(C, H)` — transpose to ghost layout `(H, C)` before flattening (see `load_adam_second_moment`)
- SJLT projection: Achlioptas 2003, sparse CSR row-by-row — do not allocate dense `(dim_out, dim_in)` for large `dim_in`
- FAISS: Inner product (`IndexFlatIP`), NOT cosine — magnitude matters for TracIn scores
- Query ghost vector uses **final checkpoint** only while the index sums training ghosts over checkpoints (TracIn-last-on-query hybrid); document when describing attribution

## Build and Test
```bash
# Activate environment
conda activate training_env

# Run tests
python -m pytest tests/ -v

# Train MNIST example
python testModels/small/mnist/train.py

# Build index
python testModels/small/mnist/run_index.py

# Query attribution
python testModels/small/mnist/run_query.py --input outputs/query_input.pt
```

## Conventions
- `src/` functions accept explicit parameters — no config dicts in the library API
- Model-specific wiring lives in `testModels/<name>/`, never in `src/`
- Error function `error_fn(logits, targets) -> E` is user-provided, not built into the library
- All `src/` imports use `from src.<module> import <thing>`

---

## Context guide for AI assistants

When working on this project, load files in priority order by task.

### Always read first

| File | Why |
|------|-----|
| `README.md` | Problem, solution sketch, architecture, benchmark snapshot |
| `docs/theory.md` | Full math / algorithmic detail |
| `docs/implementation.md` | Setup, benchmarks, extending models |
| `.github/copilot-instructions.md` | This file — rules + navigation |

### Core library (`src/`)

| File | Why |
|------|-----|
| `src/hooks_manager.py` | Hooks, `_flatten_to_2d()`, `MultiLayerBackwardGhostManager` |
| `src/math_utils.py` | Ghost vectors, Adam, SJLT, projection |
| `src/error_functions.py` | Error signals |
| `src/indexer.py`, `inference.py`, `faiss_store.py` | Index / query / FAISS |
| `src/config_utils.py` | Checkpoints, layer helpers |

### Benchmarks

| File | Why |
|------|-----|
| `benchmarks/ghost_faiss.py` | `compute_ghost_tracin_scores`, `auto_ghost_layers`, hybrid path |
| `benchmarks/full_gradient_tracin.py` | Reference baseline |
| `benchmarks/run_all.py` | Model list, CLI |
| `benchmarks/summarize_all.py` | Aggregate metrics, cross-model figures |
| `benchmarks/comparison.py` | `metrics.json` assembly |
| `testModels/small/mnist/run_benchmark.py` | Reference benchmark pattern |
| `testModels/<model>/run_benchmark.py` | Per-model wiring |

### New test model

Copy e.g. `testModels/small/mnist/run_benchmark.py`, register in `run_all.py` and add the name to `SMALL_MODELS`, `MEDIUM_MODELS`, or `LARGE_MODELS` in `summarize_all.py`, follow **docs/implementation.md** § "Adding a New Benchmark Model".

### Tests

`tests/test_*.py` — hooks, math, FAISS, integration.

### HPC

`submit_slurm.sh`, `slurm_run_container.sh`, `run_container.sh`, `sync_from_scratch.sh`.

### Usually skip

`outputs/`, `data/`, `logs/`, generated checkpoints, and narrow helpers (`checkpoint_schedule.py`, `subset_loader.py`) unless editing those paths.
