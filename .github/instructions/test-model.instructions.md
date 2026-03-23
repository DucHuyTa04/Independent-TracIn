---
description: "Use when editing testModels/ files: model definitions, data loaders, training scripts, run_index.py, run_query.py. Covers the testModels/<name>/ folder structure and wiring conventions."
applyTo: "testModels/**"
---
# testModels/ Conventions

## Folder Structure
Each model gets its own folder under `testModels/<name>/`:
- `model.py` — nn.Module definition (no adapter classes)
- `data.py` — Dataset class + `make_loaders()` factory
- `train.py` — Training script → saves checkpoint + optimizer state
- `run_index.py` — Calls `src.indexer.build_index()` with model-specific wiring
- `run_query.py` — Calls `src.inference.attribute()` with model-specific wiring
- `create_query_input.py` — Picks a test sample and saves as .pt
- `config.yaml` — Model-specific configuration

## Wiring Pattern
- Import model from `testModels.<name>.model`
- Import data from `testModels.<name>.data`
- Import library functions from `src.indexer` and `src.inference`
- Define `error_fn` locally (classification: softmax - one_hot, regression: logits - targets)
- Read config.yaml and pass explicit params to src/ functions

## Adding a New Model
1. Create `testModels/<name>/` with the files listed above
2. Define the nn.Module in model.py
3. Define Dataset + make_loaders in data.py
4. Wire run_index.py and run_query.py to call src/ functions
