# Practical Guide: Setup, Run, and Extend TracIn Ghost

**Theory (short):** [README.md](../README.md) · **Theory (full math):** [theory.md](theory.md)

This guide is the hands-on companion: **environment**, **benchmarks**, **your own model**, and **new benchmark models**.

---

## Table of Contents

1. [Prerequisites & Installation](#prerequisites--installation)
2. [Running Benchmarks](#running-benchmarks)
3. [Using TracIn Ghost on Your Own Model](#using-tracin-ghost-on-your-own-model)
4. [Adding a New Benchmark Model](#adding-a-new-benchmark-model)
5. [Slurm / HPC Setup](#slurm--hpc-setup)
6. [API Reference (Key Functions)](#api-reference)
7. [Common Pitfalls](#common-pitfalls)

---

## Prerequisites & Installation

### System Requirements

- Python 3.10+
- PyTorch 2.0+ (with CUDA if running on GPU)
- ~2 GB disk for datasets (MNIST, CIFAR-10, CIFAR-100, FashionMNIST, TinyShakespeare)

### Install Dependencies

```bash
pip install torch torchvision numpy scipy pyyaml faiss-cpu matplotlib pytest
```

For GPU-accelerated FAISS (optional, for large-scale indexing):
```bash
pip install faiss-gpu
```

### Verify Installation

```bash
cd Independent-TracIn/
python -m pytest tests/ -v
```

All tests should pass. FAISS-dependent tests (`test_faiss_store.py`, `test_integration.py`) will skip gracefully if FAISS is not installed.

---

## Running Benchmarks

Benchmarks compare **Ghost+FAISS** (our fast approximation) against **Original TracIn** (full-gradient baseline). Each benchmark trains a model from scratch, computes influence scores both ways, and measures Spearman rank correlation ρ.

### Quick Start

```bash
# Single model, smoke test (~10 seconds, CPU)
python testModels/small/mnist/run_benchmark.py --per-class 3 --epochs 10

# All models registered in benchmarks/run_all.py, smoke (~1 minute on GPU)
python benchmarks/run_all.py --smoke --device auto

# All models, full scale (~1-3 hours on GPU)
python benchmarks/run_all.py --scale full --device cuda
```

### What Each Benchmark Produces

Each model writes to `outputs/benchmarks/<model_name>/`:

- `metrics.json` — Spearman ρ, top-k overlap, wall time, peak memory, ghost coverage
- `benchmark_dashboard.png` — 2×2 figure: rank scatter, top-k overlap, score spread, text summary

After running all models, aggregate with:

```bash
python benchmarks/summarize_all.py
```

This writes:
- `outputs/benchmarks/summary.json` — Cross-model metrics
- `outputs/benchmarks/comparison_cross_model.png` — Unified figure: ρ, top-k heatmap, throughput, speedup, memory, memory ratio, summary table (model order: small → medium → large)

### Understanding the Results

The primary metric is **Spearman ρ** — how well Ghost+FAISS preserves the ranking of training sample influence compared to full-gradient TracIn.

The secondary metric is **ghost coverage** — what fraction of total model parameters are in the hooked `nn.Linear` layers. Higher coverage = better approximation. See the README's [Benchmark suite](../README.md#benchmark-suite) table.

### Diagnostic Mode

For deeper analysis, run any benchmark with `--diagnostic` to compute six ablation variants (A–F):

```bash
python testModels/small/mnist/run_benchmark.py --diagnostic --device auto
```

This produces an additional `diagnostic_variants.png` showing which approximation steps (projection, Adam correction, TracIn-last) affect quality most.

---

## Using TracIn Ghost on Your Own Model

### Overview

The pipeline has three phases:

1. **Train** your model, saving checkpoints with optimizer state
2. **Index** all training data → FAISS inner-product index (offline, once)
3. **Query** a generated output → attribution percentages (online, per query)

### Step 0: Decide Your Loss Type

| Training loss | `loss_type` | Error function |
|--------------|-------------|----------------|
| Cross-entropy (classification) | `"classification"` | `softmax(logits) - one_hot(targets)` |
| MSE (regression / reconstruction) | `"regression"` | `predictions - targets` |

**This must match how you trained the model.** Wrong choice = wrong ghost vectors = meaningless attribution.

For exotic losses (e.g. combined reconstruction + KL divergence), write a custom `error_fn(logits, targets) -> E` tensor and pass it directly to the library functions.

### Step 1: Save Checkpoints During Training

Use `TracInCheckpointCallback` so each checkpoint includes model weights, optimizer state, learning rate, and epoch loss:

```python
from src.config_utils import TracInCheckpointCallback

cb = TracInCheckpointCallback(save_dir="my_model/checkpoints", save_every=5)
for epoch in range(num_epochs):
    loss = train_one_epoch(model, optimizer, loader)
    cb.on_epoch_end(epoch, model, optimizer, loss)

# Always save the last epoch (needed for query phase)
cb.finalize(model, optimizer, num_epochs - 1, loss)

# Optional: keep only the top 5 most informative checkpoints
cb.select_best(keep=5)
```

**Heuristics:**
- Aim for **5–10 checkpoints** over training. More checkpoints = more TracIn signal, but diminishing returns past ~10.
- `select_best(keep=5)` picks checkpoints with the largest loss drops between saves, matching the TracIn paper's guidance.
- Each checkpoint saves as a single `.pt` file containing `{model_state_dict, optimizer_state_dict, epoch, learning_rate, loss}`, plus a JSON sidecar with metadata.

### Step 2: Choose Target Layers

#### Option A: Single Layer (Simple)

By default, the library hooks the **last `nn.Linear`** layer:

```python
from src.config_utils import resolve_target_layer

name, target_layer = resolve_target_layer(model)
print(f"Hooked: {name}")  # e.g. "fc2"
```

To override, pass the layer name: `resolve_target_layer(model, "my_layer_name")`.

#### Option B: Multi-Layer (Better Coverage)

For models where the last linear layer covers a small fraction of parameters, hook multiple layers:

```python
from benchmarks.ghost_faiss import auto_ghost_layers

# "last" strategy: prefer layers closest to output (recommended)
ghost_layers = auto_ghost_layers(model, target_coverage=0.5, strategy="last")

# "largest" strategy: prefer layers with most parameters
ghost_layers = auto_ghost_layers(model, target_coverage=0.5, strategy="largest")
```

`target_coverage` is the fraction of total model parameters to cover (0.0–1.0). The function greedily adds `nn.Linear` layers until the target is met.

For multi-layer ghost, use `build_index(..., multi_layer_ghost=True, ghost_layers=[...])` instead of passing a single `target_layer`.

### Step 3: Write `config.yaml`

Place this next to your `checkpoints/` directory:

```yaml
loss_type: classification   # or "regression"

paths:
  outputs_dir: outputs
  data_root: data           # where your dataset lives

ghost:
  projection_dim: 1280      # SJLT output dimension (null = no projection)
  projection_type: sjlt     # "sjlt" or "dense"
  projection_seed: 42       # reproducibility

index:
  output_path: faiss_index
  metadata_path: faiss_metadata.json

query:
  index_path: faiss_index
  metadata_path: faiss_metadata.json
  top_k: 20
```

Checkpoints are auto-detected from `checkpoints/` next to the config file. `adam_param_key` is auto-inferred from `target_layer.weight`.

### Step 4: Build the Index (Offline)

Create `run_index.py` for your model (adapt from `testModels/small/mnist/run_index.py`):

```python
from src.config_utils import find_adam_param_key, resolve_target_layer
from src.error_functions import get_error_fn
from src.indexer import build_index
from testModels.pipeline_helpers import (
    abs_config_path, build_checkpoints_list, ghost_settings, outputs_dir_from_cfg,
)

# 1. Build model and data loader
model = YourModel()
train_loader = ...  # yields (inputs, targets, sample_id) tuples
sample_metadata = {0: "artist_a", 1: "artist_a", 2: "artist_b", ...}

# 2. Resolve config
cfg_path = abs_config_path("your_model/config.yaml")
checkpoints = build_checkpoints_list(cfg_path, cfg)
_, target_layer = resolve_target_layer(model)
error_fn = get_error_fn(cfg["loss_type"])
adam_key = find_adam_param_key(model, target_layer)
proj_dim, proj_type, proj_seed = ghost_settings(cfg)

# 3. Build FAISS index
build_index(
    model=model,
    target_layer=target_layer,
    error_fn=error_fn,
    data_loader=train_loader,
    checkpoints=checkpoints,
    sample_metadata=sample_metadata,
    projection_dim=proj_dim,
    projection_type=proj_type,
    projection_seed=proj_seed,
    adam_param_key=adam_key,
    output_dir=outputs_dir_from_cfg(cfg),
    index_filename="faiss_index",
    metadata_filename="faiss_metadata.json",
    device="auto",
)
```

**Important:** `sample_metadata` is a `dict[int, str]` mapping sample IDs to rights holder names. This is how attribution scores are grouped by owner.

**DataLoader contract:** Must yield `(inputs, targets, sample_id)` tuples. `sample_id` is an integer that maps into `sample_metadata`.

### Step 5: Query (Online)

Create `run_query.py` (adapt from `testModels/small/mnist/run_query.py`):

```python
from src.config_utils import find_adam_param_key, last_checkpoint_paths, resolve_target_layer
from src.error_functions import get_error_fn
from src.inference import attribute

model = YourModel()
checkpoints = build_checkpoints_list(cfg_path, cfg)
wpath, opath = last_checkpoint_paths(checkpoints)
_, target_layer = resolve_target_layer(model)
error_fn = get_error_fn(cfg["loss_type"])
adam_key = find_adam_param_key(model, target_layer)
proj_dim, proj_type, proj_seed = ghost_settings(cfg)

results = attribute(
    model=model,
    target_layer=target_layer,
    error_fn=error_fn,
    query_inputs=query_tensor,      # shape: (N, ...) — batch of query inputs
    query_targets=query_targets,    # shape: (N, ...) — corresponding targets
    index_path="outputs/faiss_index",
    metadata_path="outputs/faiss_metadata.json",
    checkpoint_path=wpath,          # last checkpoint weights
    projection_dim=proj_dim,
    projection_type=proj_type,
    projection_seed=proj_seed,
    optimizer_state_path=opath,     # last checkpoint optimizer state
    adam_param_key=adam_key,
    top_k=20,
    device="auto",
)

# results is a list of dicts, one per query:
# {
#   "rights_holder_attribution": {"artist_a": 0.65, "artist_b": 0.35},
#   "top_samples": [(sample_id, score), ...]
# }
```

### Step 6: Use the CLI Dispatcher

If you have `run_index.py` and `run_query.py` in `testModels/<name>/`, you can use the unified CLI:

```bash
python main.py --config testModels/<name>/config.yaml --model <name> --mode index
python main.py --config testModels/<name>/config.yaml --model <name> --mode query --input query.pt
python main.py --config testModels/<name>/config.yaml --model <name> --mode full --input query.pt
```

---

## Adding a New Benchmark Model

Benchmark models live in `testModels/<name>/` and validate Ghost+FAISS against full-gradient TracIn on a specific architecture. They don't need the full pipeline (train/index/query) — just `model.py`, optionally `data.py`, and `run_benchmark.py`.

### Minimal File Structure

```
testModels/<name>/
├── model.py           # nn.Module definition
├── data.py            # Dataset class (optional if using built-in datasets)
└── run_benchmark.py   # Training + Ghost vs Original TracIn comparison
```

### Step 1: Define the Model (`model.py`)

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
```

### Step 2: Define the Dataset (`data.py`)

Must return `(input_tensor, target_tensor, sample_index)`:

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_root="data"):
        # Load your data
        ...

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]       # input tensor
        y = self.labels[idx]     # target tensor
        return x, y, idx         # idx = sample ID for TracIn
```

### Step 3: Write the Benchmark (`run_benchmark.py`)

Follow the pattern from existing benchmarks. The key structure:

```python
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from benchmarks.benchmark_profiling import profile_block
from benchmarks.checkpoint_schedule import evenly_spaced_checkpoint_epochs
from benchmarks.comparison import build_comparison
from benchmarks.full_gradient_tracin import compute_full_gradient_tracin_scores
from benchmarks.ghost_faiss import auto_ghost_layers, compute_ghost_tracin_scores
from benchmarks.subset_loader import make_baseline_loader
from benchmarks.metrics import summarize_metrics
from benchmarks.plot import plot_model_benchmark
from src.error_functions import classification_error  # or regression_error


def main():
    # 1. Parse args (--output-dir, --epochs, --lr, --device, --seed, etc.)

    # 2. Create model, dataset, data loader

    # 3. Train model, saving checkpoints at evenly spaced epochs:
    ckpt_epochs = set(evenly_spaced_checkpoint_epochs(num_epochs, 5))
    checkpoints = []
    for epoch in range(num_epochs):
        # ... training loop ...
        if epoch in ckpt_epochs:
            path = f"{ckpt_dir}/ckpt_{epoch}.pt"
            torch.save({"model_state_dict": ..., "optimizer_state_dict": ...,
                        "learning_rate": lr, "epoch": epoch, "loss": avg_loss}, path)
            checkpoints.append({"weights_path": path,
                                "optimizer_state_path": path,
                                "learning_rate": lr})

    # 4. Select ghost layers
    ghost_layers = auto_ghost_layers(model, target_coverage=0.5)

    # 5. Compute Ghost+FAISS scores
    with profile_block("ghost") as ghost_prof:
        ghost_scores = compute_ghost_tracin_scores(
            model=model,
            ghost_layers=ghost_layers,
            error_fn=classification_error,
            data_loader=train_loader,
            checkpoints=checkpoints,
            query_inputs=query_inputs,
            query_targets=query_targets,
            loss_fn=nn.CrossEntropyLoss(),
            device=device,
        )

    # 6. Compute Original TracIn scores (baseline)
    baseline_loader = make_baseline_loader(train_loader, max_n=5000)
    with profile_block("original") as orig_prof:
        original_scores = compute_full_gradient_tracin_scores(
            model=model,
            loss_fn=nn.CrossEntropyLoss(),
            data_loader=baseline_loader.loader,
            query_inputs=query_inputs,
            query_targets=query_targets,
            checkpoints=checkpoints,
            device=device,
        )

    # 7. Compare and save results
    comparison = build_comparison(ghost_scores, original_scores, ghost_prof, orig_prof, ...)
    # Save metrics.json and benchmark_dashboard.png
```

See `testModels/small/mnist/run_benchmark.py` for the complete pattern.

### Step 4: Register in `run_all.py` and `summarize_all.py`

In `benchmarks/run_all.py`, add your model to:
- `ALL_MODELS` tuple
- `scripts` dict (path to your `run_benchmark.py`)
- `n_train_models` or `per_class_models` set (depends on your CLI args)
- `needs_data_root` set (if your dataset needs `--data-root`)
- Scale presets in `_argv_for_benchmark()` (smoke and full settings)

In `benchmarks/summarize_all.py`, add the model name to the correct tier tuple (`SMALL_MODELS`, `MEDIUM_MODELS`, or `LARGE_MODELS`) so it appears in `DEFAULT_MODELS`.

---

## Demos (`demos/`)

> **GPU required.** Demos default to `--device cuda`. Training is too slow on CPU
> login nodes. Run on a GPU compute node or via Slurm.

**Recommended workflow** — pre-train once, then run any demo:

```bash
cd Independent-TracIn

# Step 1: Pre-train all demo models + build indices (run once, on GPU)
python demos/pretrain_all.py

# Step 2: Run any individual demo (skips training, uses cached results)
python demos/classification_demo.py --top-k 10
python demos/text_generation_demo.py --prompt "To be" --top-k 10
python demos/image_generation_demo.py --num-images 3 --save-grid

# Or the unified interactive demo
python demos/interactive_demo.py
```

Results are saved under `demos/outputs/<name>/` (checkpoints, FAISS index, config).
Once `pretrain_all.py` has run, subsequent demo invocations skip training and reuse cached outputs.

See [demos/README.md](../demos/README.md) for all flags, output paths, and Slurm instructions.

---

## Slurm / HPC Setup

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `TRAINING_EXTRA_SBATCH` | Extra sbatch flags (e.g. `--account=def-xxx`) | (none) |
| `TRAIN_DATA_ROOT` | Path to pre-downloaded datasets | (none) |
| `BENCH_TIME_SMOKE` | Walltime for smoke benchmarks | `01:00:00` |
| `BENCH_TIME_FULL` | Walltime for full benchmarks | `06:00:00` |
| `TRACIN_EXTRA_MODULES` | CUDA module to load (e.g. `cudacore/12.6.3`) | `cudacore/12.6.3` |
| `TRAINING_ENV_ROOT` | Conda env with PyTorch | `$SCRATCH/miniconda3/envs/training_env` |

### Submit Benchmark Jobs

```bash
# Smoke test (all models, ~1h)
TRAINING_EXTRA_SBATCH="--account=def-mygroup" \
TRAIN_DATA_ROOT=/scratch/$USER/data \
bash submit_slurm.sh benchmark smoke

# Full benchmark (all models, ~3-24h depending on GPU)
bash submit_slurm.sh benchmark full

# Chain: smoke first, full only after smoke succeeds
bash submit_slurm.sh benchmark-chain
```

### Submit Training Jobs

```bash
# Local (GPU login node)
bash run_training.sh mnist --epochs 20

# Slurm GPU job
bash run_training.sh mnist --slurm --epochs 50

# Direct Slurm submission
bash submit_slurm.sh train mnist my-training-job --epochs 50
```

### Sync Results

```bash
# Pull specific job results from $SCRATCH to local
bash sync_from_scratch.sh <job_id>

# Pull all job results
bash sync_from_scratch.sh all
```

### Pre-download Datasets

Compute nodes often have no internet access. Download datasets on the login node first:

```python
import torchvision
torchvision.datasets.MNIST(root="/scratch/$USER/data", download=True)
torchvision.datasets.CIFAR10(root="/scratch/$USER/data", download=True)
torchvision.datasets.CIFAR100(root="/scratch/$USER/data", download=True)
torchvision.datasets.FashionMNIST(root="/scratch/$USER/data", download=True)
```

For TinyShakespeare, download manually:
```bash
wget -O /scratch/$USER/data/tinyshakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

Then pass `TRAIN_DATA_ROOT=/scratch/$USER/data` or `--data-root /scratch/$USER/data`.

---

## API Reference

### Core Library (`src/`)

#### `build_index(model, target_layer, error_fn, data_loader, checkpoints, sample_metadata, ...)`
Build FAISS inner-product index from training data. Returns index file path.

#### `attribute(model, target_layer, error_fn, query_inputs, query_targets, index_path, metadata_path, checkpoint_path, ...)`
Query attribution for one or more generated outputs. Returns list of dicts with `rights_holder_attribution` and `top_samples`.

#### `TracInCheckpointCallback(save_dir, save_every)`
Training callback that saves unified checkpoints (model + optimizer + lr + loss). Methods: `on_epoch_end()`, `finalize()`, `select_best(keep)`.

#### `resolve_target_layer(model, layer_name=None)`
Find the target `nn.Linear` layer by name (or auto-detect last linear). Returns `(name, module)`.

#### `find_adam_param_key(model, target_layer)`
Auto-detect the optimizer state dict key for the target layer's weight. Returns int key into `optimizer.state[key]`.

#### `get_error_fn(loss_type)`
Returns the error function: `"classification"` → `classification_error`, `"regression"` → `regression_error`.

### Benchmarks (`benchmarks/`)

#### `auto_ghost_layers(model, target_coverage=0.5, strategy="last")`
Select `nn.Linear` layers to hook. `"last"` = output-adjacent first (recommended), `"largest"` = biggest first. Returns list of modules.

#### `compute_ghost_tracin_scores(..., max_spatial_positions=None)`
Multi-layer Ghost TracIn scoring. Conv2d layers use exact sum-over-spatial ghosts (like sequence models). Optional `max_spatial_positions` triggers mean-pooled fallback per Conv2d layer when output spatial count exceeds the cap (memory guard).

#### `compute_full_gradient_tracin_scores(model, loss_fn, data_loader, query_inputs, query_targets, checkpoints, ...)`
Textbook TracInCP baseline using full-parameter per-sample gradients.

---

## Common Pitfalls

| Problem | Symptom | Fix |
|---------|---------|-----|
| Wrong `loss_type` | Ghost scores meaningless; near-zero ρ | Must match training loss: `"classification"` for CE, `"regression"` for MSE |
| Missing optimizer state | No Adam correction applied; lower ρ | Use `TracInCheckpointCallback` when training, which saves optimizer state automatically |
| `projection_dim` mismatch | FAISS query fails or returns garbage | Use the same config for index and query |
| `target_layer` wrong | Silent wrong results | Verify with `resolve_target_layer(model)` — print the name and check it's the layer you intend |
| Too few checkpoints | Weak TracIn signal | Save at least 5 checkpoints over training |
| Low ghost coverage | ρ < 0.5 despite correct setup | Hook more layers with `auto_ghost_layers(model, target_coverage=0.9)` or accept the limitation for conv-heavy models |
| 3D tensor issues | Shape errors in ghost computation | `MultiLayerBackwardGhostManager` with `keep_raw=True` handles 3D via sum-of-outer-products automatically |
| Dataset not found on compute node | `FileNotFoundError` | Pre-download datasets on login node; pass `--data-root` |
| CUDA module not found (Slurm) | Import error for torch | Set `TRACIN_EXTRA_MODULES` to correct CUDA version, or `none` if not needed |
