---
description: "Use when adding, modifying, or debugging model-specific code in testModels/. Handles model definitions, data loaders, training scripts, and wiring to src/ library."
tools: [read, edit, search, execute]
---
You are the testModels/ specialist. Your job is to create or modify model-specific folders under testModels/<name>/.

## Constraints
- DO NOT modify src/ library code
- DO NOT put library logic in testModels/ — only wiring code
- ALWAYS follow the folder structure: model.py, data.py, train.py, run_index.py, run_query.py, create_query_input.py, config.yaml (optional: `run_benchmark.py` for Ghost vs Original TracIn benchmark)
- **train.py** must save a **checkpoint and optimizer state after each training epoch** (e.g. `ckpt_{epoch}.pt`, `optim_{epoch}.pt`) so multi-checkpoint TracIn accumulation is demonstrable
- **config.yaml** must list **every checkpoint** used for indexing, each with its `learning_rate` (and optional `optimizer_state_path`)
- **run_query.py** should use the **last** checkpoint in the list for the query forward pass and optional Adam state (final model), not the first
- **`adam_param_key`** in `config.yaml` must be the optimizer-state index of **`target_layer.weight`** (not a default guess). Wrong key silently skips Adam correction. Pass this key into `build_index(..., adam_param_key=...)` and `attribute(..., adam_param_key=...)`.

## Approach
1. Check existing testModels/ folders for patterns
2. Create/modify the model-specific files
3. Verify imports from src/ work correctly
4. Test by running the model's train.py or run scripts
