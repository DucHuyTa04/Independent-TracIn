# TracIn Ghost — Demos

> **GPU required.** All demos default to `--device cuda`. Training and
> indexing are too slow on CPU (login nodes will kill the process).
> Run on a GPU compute node — either interactively (`salloc --gpus-per-node=1`)
> or via Slurm (see below).

Run from the **Independent-TracIn** repository root.

```bash
cd Independent-TracIn
```

## Prerequisites

Same as the main project: `torch`, `torchvision`, `numpy`, `scipy`, `faiss-cpu`, `matplotlib`.

---

## Recommended Workflow

Each demo trains a model, builds a FAISS index, and then runs queries.
Training is the slow part — once done, results are cached and reused.

**Step 1 — Pre-train all models (once):**

```bash
python demos/pretrain_all.py            # default: --device cuda
```

This trains all three demo models (CIFAR-10 CNN, TinyGPT, Fashion-VAE),
builds FAISS indices, and saves everything under `demos/outputs/`.
Run this once; subsequent demo runs will skip training automatically.

**Step 2 — Run any demo:**

```bash
python demos/classification_demo.py --top-k 10 --num-queries 5
python demos/text_generation_demo.py --prompt "To be or not to be" --top-k 10
python demos/image_generation_demo.py --num-images 3 --top-k 10 --save-grid
```

Or use the unified interactive demo (requires `pretrain_all.py` first):

```bash
python demos/interactive_demo.py
```

---

## Output Directories

All results are saved under `demos/outputs/`. After `pretrain_all.py`
finishes, you can inspect these directories even without re-running demos.

| Demo | Output path | Contents |
|------|-------------|----------|
| CIFAR-10 classification | `demos/outputs/cifar10_classification/` | Checkpoints, FAISS index, `demo_config.yaml` |
| TinyGPT text generation | `demos/outputs/tinygpt_demo/` | Checkpoints, FAISS index, `demo_config.yaml` |
| Fashion-VAE image gen | `demos/outputs/vae_fashion_demo/` | Checkpoints, FAISS index, `demo_config.yaml`, optional `generated_grid.png` |

---

## Standalone Demos (detailed flags)

### 1. CIFAR-10 classification

```bash
python demos/classification_demo.py --top-k 10 --num-queries 5
```

Flags: `--max-train`, `--epochs`, `--force-retrain`, `--force-reindex`, `--projection-dim`.

### 2. Text generation

```bash
python demos/text_generation_demo.py --prompt "To be or not to be" --top-k 10
```

Flags: `--n-train`, `--epochs`, `--max-new-tokens`, `--temperature`, `--force-retrain`, `--force-reindex`.

### 3. Image generation

```bash
python demos/image_generation_demo.py --num-images 3 --top-k 10 --save-grid
```

Flags: `--max-train`, `--epochs`, `--force-retrain`, `--force-reindex`, `--save-grid`.

---

## Running on Slurm (HPC)

If you cannot get an interactive GPU session, wrap the demo in a Slurm job:

```bash
sbatch --gpus-per-node=1 --cpus-per-task=4 --time=01:00:00 --wrap \
  "cd $PWD && source /path/to/your/venv/bin/activate && python demos/pretrain_all.py"
```

---

## Notes

- Demos use smaller training subsets than full benchmarks for faster iteration.
- Use `--force-retrain` and `--force-reindex` to rebuild from scratch.
- For rigorous evaluation, use `benchmarks/run_all.py` instead.
