# TracIn Ghost — Copyright Attribution via Training Influence

A plug-and-play Python library for calculating the proportional influence of specific copyrighted training data on generative model outputs. Built on TracIn with ghost dot products, Adam correction, SJLT projection, and FAISS inner-product retrieval.

See [new_approach.md](new_approach.md) for the full mathematical blueprint.

---

## Quick Start

```bash
conda activate training_env
```

### 1. Run tests

```bash
python -m pytest tests/ -v
```

### 2. Train MNIST model

```bash
python testModels/mnist/train.py
```

### 3. Index training data

```bash
python testModels/mnist/run_index.py
```

### 4. Create query input

```bash
python testModels/mnist/create_query_input.py
```

### 5. Query attribution

```bash
python testModels/mnist/run_query.py --input outputs/query_input.pt
```

Results → `outputs/attribution_results.json`.

---

## Slurm (Cloud Computing)

### Submit jobs

```bash
bash submit_slurm.sh config.yaml tracin-full full
bash submit_slurm.sh config.yaml tracin-index index
```

### Sync results from scratch

```bash
bash sync_from_scratch.sh <job_id>
bash sync_from_scratch.sh all
```

### Local run via Apptainer

```bash
bash run_container.sh config.yaml index
```

---

## Directory Structure

```
Independent-TracIn/
├── src/                        # Core library (functional API)
│   ├── hooks_manager.py        # Hook lifecycle + dim flattening
│   ├── math_utils.py           # Ghost vectors, Adam correction, SJLT
│   ├── indexer.py              # Offline: train data → FAISS index
│   ├── inference.py            # Online: generated output → attribution %
│   └── faiss_store.py          # FAISS build/load/query wrapper
├── testModels/                 # One folder per model
│   └── mnist/                  # MNIST MLP example
│       ├── model.py, data.py, train.py
│       ├── run_index.py, run_query.py
│       └── config.yaml
├── tests/                      # pytest suite
│   ├── test_hooks_manager.py
│   ├── test_math_utils.py
│   └── test_mock_pipeline.py
├── .github/                    # Agent infrastructure
│   ├── copilot-instructions.md
│   ├── instructions/           # File-scoped instructions
│   └── agents/                 # Custom agents (src-core, test-model, etc.)
├── main.py                     # Thin CLI dispatcher
├── config.yaml                 # Default config template
└── requirements.txt
```

---

## Adding a New Model

1. Create `testModels/<name>/` with: model.py, data.py, train.py, run_index.py, run_query.py, config.yaml
2. Define your `nn.Module` in model.py
3. Wire `error_fn(logits, targets) -> E` in run_index.py and run_query.py
4. Call `src.indexer.build_index()` and `src.inference.attribute()` with explicit params

---

## Plugging in Your Own Model

1. Subclass `BaseModelAdapter`: implement `build_model()`, `target_layer()`, optionally `load_weights()`
2. Subclass `BaseTaskAdapter`: implement `prepare_targets()`, `compute_loss()`, `error_signal()`
3. Subclass `BaseDataAdapter`: implement `build_loaders()` returning `DataBundle` with `metadata["sample_id_to_rights_holder"]`
4. Add your adapter class paths to `config.yaml`

---

## Requirements

```
torch
torchvision
numpy
scipy
pyyaml
faiss-cpu
```
