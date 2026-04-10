# TracIn Ghost — Copyright Attribution via Training Influence

Library for **training-data influence** (TracIn-style): score how much each training sample affected a model’s behavior on a query — supporting attribution and revenue sharing for generative AI.

**Paper:** Pruthi et al., *Estimating Training Data Influence by Tracing Gradient Descent* (NeurIPS 2020).

**Docs:** This file = **theory overview** · [docs/theory.md](docs/theory.md) = full math · [docs/implementation.md](docs/implementation.md) = setup, run, extend.

---

## Table of contents

1. [The problem](#the-problem)
2. [Our solution](#our-solution)
3. [Mathematical foundations (overview)](#mathematical-foundations-overview)
4. [System architecture](#system-architecture)
5. [Benchmark suite](#benchmark-suite)
6. [Quick start](#quick-start)
7. [Further reading & structure](#further-reading--structure)
8. [Requirements](#requirements)

---

## The problem

Generative models train on large corpora that include copyrighted material. There is no standard way to say **which training examples influenced a given output** and by how much — which blocks fair compensation. We need a **mathematical** notion of contribution (not a heuristic), e.g. attributing influence scores across training samples for a query.

---

## Our solution

**TracIn Ghost** approximates multi-checkpoint TracIn by building **compact ghost vectors** from hooked layers so inner products match true per-layer gradient structure where possible, then **projects** (SJLT) and **indexes** (FAISS) for scalable retrieval. **Key insight:** for `nn.Linear` with 2D activations, full weight-gradient inner products factor into cheap dot products in $A$ and $E$. **Pipeline:** (1) **Offline** — for each training sample and checkpoint, extract ghosts, optional Adam correction, project, accumulate $\sum_t \eta_t g$, build FAISS `IndexFlatIP`. (2) **Online** — ghost for the query, search index, normalize scores to attribution weights.

Details: Adam scaling, 3D/conv sums-of-outer-products, hybrid 2D-vs-raw paths, and memory caps are in [docs/theory.md](docs/theory.md).

---

## Mathematical foundations (overview)

### TracIn (scalar score)

$$
\mathrm{TracIn}(z_i, z') = \sum_{t} \eta_t \left\langle \nabla \ell(z'; \theta_t), \nabla \ell(z_i; \theta_t) \right\rangle.
$$

### Ghost dot (single `nn.Linear`, 2D activations)

With $g = \mathrm{vec}(E^\top A)$, $\langle g_1, g_2 \rangle = \langle A_1, A_2 \rangle \cdot \langle E_1, E_2 \rangle$ (exact for that layer’s ghost).

### Multi-layer

Concatenate per-layer ghosts; dot products **sum** across hooked layers. Coverage $< 100\%$ means the ghost lives in a **subspace** of full $\nabla_\theta \ell$ — see benchmarks vs full-gradient baseline.

### Adam, SJLT, FAISS (one line each)

- **Adam:** Elementwise $g / (\sqrt{v_t}+\epsilon)$ on the flattened ghost; $v_t$ does not factorize over $A,E$. See `apply_adam_correction` in `math_utils.py`.
- **SJLT:** Sparse JL projection preserves inner products approximately; CSR storage. See `build_sjlt_matrix`.
- **FAISS:** Inner product (`IndexFlatIP`), not cosine — magnitude matters for TracIn.

---

## System architecture

### Pipeline (ASCII)

```
Offline (per model):
  For each checkpoint t, each training sample:
    forward → hooks capture A per layer; analytic E
    ghost per layer → concat → Adam correction → SJLT → accumulate η_t · g_proj
  Build FAISS index + metadata

Online (per query):
  forward last checkpoint → query ghost → project → FAISS search → top-K → normalize → attributions
```

### Core modules

| Area | Role |
|------|------|
| `src/hooks_manager.py` | `HookManager`, `MultiLayerBackwardGhostManager` (raw conv/seq blocks, LayerNorm/BN) |
| `src/math_utils.py` | Ghost formation, Adam, SJLT, projection |
| `src/error_functions.py` | Classification / regression $E$ |
| `src/indexer.py`, `inference.py`, `faiss_store.py` | Index build, query, FAISS I/O |
| `benchmarks/ghost_faiss.py` | `compute_ghost_tracin_scores`, `auto_ghost_layers`, hybrid factored path, `max_spatial_positions` |
| `benchmarks/full_gradient_tracin.py` | Full-parameter reference TracIn |

Design notes that were previously in "Key design decisions" (layer selection, coverage, TracIn-last-on-query) are expanded in [docs/theory.md](docs/theory.md) §4–§7 and [docs/implementation.md](docs/implementation.md).

---

## Benchmark suite

**Compared methods:** **Ghost+FAISS** (production-style path) vs **full-gradient multi-checkpoint TracIn** (reference). **Primary metric:** Spearman $\rho$ between the two ranking vectors over training points. **Also logged:** wall time, peak memory, ghost coverage, top-$k$ overlap.

**17 models** (tiers match `benchmarks/summarize_all.py`; on disk under `testModels/small/`, `testModels/medium/`, `testModels/large/`):  
*Small:* `synth_regression`, `linear_logistic`, `mnist`, `mnist_autoencoder`, `multi_task`  
*Medium:* `cifar10_cnn`, `resnet_cifar100`, `transformer_lm`, `vae_fashion`, `vit_cifar10`, `encoder_transformer`, `mlp_mixer_cifar10`, `gru_lm`, `unet_tiny`  
*Large:* `resnet50_cifar100`, `transformer_lm_large`, `vit_large_cifar10`

### Snapshot: SciNet full-GPU run (job **415555**)

Representative numbers from that aggregate run. Empty cells were not transcribed into the job summary; **re-run** `benchmarks/run_all.py --scale full` and `benchmarks/summarize_all.py` for your tree. Times and memory below motivated recent speed/memory work (hybrid factored path, optional `max_spatial_positions` for conv memory caps, `auto_ghost_layers` coverage, batch `del` / `empty_cache`).

| Model | Spearman ρ | Ghost (s) | Full (s) | Ghost peak (MB) | Full peak (MB) |
|-------|------------|-----------|----------|-----------------|----------------|
| synth_regression | 0.998 | 101.5 | 12.0 | — | — |
| linear_logistic | 1.000 | — | — | — | — |
| mnist | 0.928 | — | — | — | — |
| mnist_autoencoder | 1.000 | — | — | — | — |
| multi_task | — | — | — | — | — |
| cifar10_cnn | 0.829 | — | — | — | — |
| resnet_cifar100 | 0.350 | — | — | 742 | 75 |
| transformer_lm | 0.914 | 689 | 186 | — | — |
| vae_fashion | 0.140* | — | — | — | — |
| vit_cifar10 | 0.874 | 709 | 108 | — | — |
| encoder_transformer | — | 682 | 187 | — | — |
| mlp_mixer_cifar10 | — | — | — | 146 | 68 |
| gru_lm | — | 218.1 | 53.0 | — | — |
| unet_tiny | — | — | — | 1341 | 169 |
| resnet50_cifar100 | — | — | — | 4664 | 1172 |
| transformer_lm_large | — | — | — | 1986 | 968 |
| vit_large_cifar10 | — | — | — | — | — |

\*VAE row historically sensitive to layer-selection strategy; confirm with a fresh full run.

**Figures:** Per-model `outputs/benchmarks/<model>/benchmark_dashboard.png`. After `summarize_all.py`: **`comparison_cross_model.png`** — one figure with Spearman ρ, top-k heatmap, throughput, speedup, peak memory, memory ratio, and a summary table (models ordered small → medium → large). See [docs/benchmark_guide.md](docs/benchmark_guide.md).

**Demos (GPU required):** Interactive attribution scripts in [demos/](demos/README.md) — CIFAR-10 classification, TinyGPT text generation, Fashion-VAE image sampling. Pre-train once with `python demos/pretrain_all.py`, then run any demo (results cached under `demos/outputs/`).

**Run:**

```bash
python testModels/small/mnist/run_benchmark.py --per-class 3 --epochs 10   # smoke
python benchmarks/run_all.py --smoke --device auto
python benchmarks/run_all.py --scale full --device cuda
python benchmarks/summarize_all.py
```

Slurm: `bash submit_slurm.sh benchmark smoke` / `full` / `benchmark-chain`.

---

## Quick start

```bash
pip install torch torchvision numpy scipy pyyaml faiss-cpu matplotlib pytest
cd Independent-TracIn/
python -m pytest tests/ -v
```

MNIST pipeline: `testModels/small/mnist/train.py` → `run_index.py` → `create_query_input.py` → `run_query.py --input outputs/query_input.pt`.

---

## Further reading & structure

| File | Purpose |
|------|---------|
| [docs/theory.md](docs/theory.md) | Full theory, component-by-component |
| [docs/implementation.md](docs/implementation.md) | Install, HPC, new models, pitfalls, API notes |
| [demos/README.md](demos/README.md) | Interactive classification / text / image attribution demos |
| [docs/benchmark_guide.md](docs/benchmark_guide.md) | How to read dashboards and cross-model figures |
| [.github/copilot-instructions.md](.github/copilot-instructions.md) | Engineering rules + **context guide** for assistants |

Top-level layout: `src/`, `benchmarks/`, `testModels/<name>/`, `tests/`, `docs/`, `demos/`, `main.py`.

---

## Requirements

```
torch, torchvision, numpy, scipy, faiss-cpu, pyyaml, pytest, matplotlib
```

Optional: `faiss-gpu`.
