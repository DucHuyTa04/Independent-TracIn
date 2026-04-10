# Benchmark plots: Ghost+FAISS vs Original TracIn

This guide explains the **validation benchmarks** that compare two ways of scoring how much each training sample influenced a query. Use it with the figures under `outputs/benchmarks/`.

## The two methods

| Method | What it is |
|--------|------------|
| **Ghost + FAISS** | The pipeline you ship: last-layer ghost vectors, optional SJLT projection, Adam correction, FAISS inner-product search -- same path as `run_index.py` / `run_query.py`. |
| **Original TracIn** | Full-gradient, multi-checkpoint TracIn (variant F). Computes per-sample gradients over all parameters, sums lr-weighted inner products across checkpoints. The textbook reference implementation. |

**Convention:** For Ghost scores, **higher = more influential**. Spearman rho compares **rankings** only (scale-free).

**Comparison:** Each `run_benchmark.py` records **Original TracIn** (full-gradient multi-checkpoint) and **Ghost+FAISS** (production path). See `metrics.json` -> `comparison` for Spearman Ghost vs Original, wall time, peak memory, and index size. `summarize_all.py` writes **`comparison_cross_model.png`**: one multi-panel figure (Spearman ρ, top-k overlap heatmap, throughput, speedup, peak memory, memory ratio, summary table). Models are ordered **small → medium → large** (same order as `benchmarks/summarize_all.py` tier tuples).

**CLI:** `--num-queries` (default 16) averages TracIn over multiple test points; `--diagnostic` runs extra variants B-E and writes `diagnostic_variants.png`.

**Advanced note:** Exact / multi-checkpoint ghost helpers live in `benchmarks/exact_tracin.py`; full-gradient scoring in `benchmarks/full_gradient_tracin.py`.

## Why we benchmark

We want to see whether **fast gradient-based influence** (Ghost+FAISS) **ranks training points similarly** to **full-gradient TracIn** (textbook reference) on the same query. This validates the production approximations (last-layer ghost, SJLT projection, Adam correction, TracIn-last-on-query) on the toy `testModels/*` setups.

## Per-model figure: `benchmark_dashboard.png`

One PNG per model: `outputs/benchmarks/<model>/benchmark_dashboard.png` (2x2 panels).

### Panel A -- Rank agreement: Ghost+FAISS vs Original TracIn

- **X-axis:** rank by Ghost+FAISS (1 = highest influence).
- **Y-axis:** rank by Original TracIn (1 = highest influence).
- **Diagonal:** same rank in both methods.
- **Blue:** top-K by Ghost (K = min(10, n)); gray: other points.
- **Text box:** Spearman rho and n.

**What you want:** points near the diagonal; rho close to 1.

### Panel B -- Top-k overlap vs k

For each k, overlap = |top-k by Ghost intersection top-k by Original TracIn| / k. The dashed line is **k/n** (random overlap).

**What you want:** curve **above** random.

### Panel C -- Score spread (z-scored)

Box plots of **z-scores** per method (Ghost and Original TracIn use different raw units).

**What you want:** reasonable spread; compare **shape**, not who scores higher.

### Panel D -- Summary

Plain-text recap: n, Spearman rho, top-k overlaps for each k. Matches `metrics.json` without opening JSON.

## Cross-model figure (`comparison_cross_model.png`)

After `benchmarks/run_all.py` or `benchmarks/summarize_all.py`:

| Region | Content |
|--------|---------|
| Row 1 left | Horizontal bars: Spearman ρ (tier colors: small / medium / large) |
| Row 1 right | Top-k overlap heatmap (same model order as row 1 left) |
| Row 2 left | Ghost vs full-gradient wall time per training sample (log scale) |
| Row 2 right | Speedup factor (full / Ghost) |
| Row 3 left | Peak memory (Ghost vs full) |
| Row 3 right | Memory ratio Ghost / Full |
| Row 4 | Full-width summary table (tier column uses labels **small**, **medium**, **large**) |

## If results look weak

1. **Too few checkpoints** -- TracIn sums over training time; weak temporal signal hurts Ghost.
2. **Wrong `loss_type` / error_fn** -- Ghost vectors follow the wrong signal.
3. **Ghost coverage too low** -- last-layer parameters capture little of the total gradient; rho drops.
4. **Projection too aggressive** -- `projection_dim` too small loses inner-product fidelity.

## Where outputs live

| Artifact | Path |
|----------|------|
| Per-model dashboard | `outputs/benchmarks/<model>/benchmark_dashboard.png` |
| Per-model metrics | `outputs/benchmarks/<model>/metrics.json` |
| Cross-model summary | `outputs/benchmarks/summary.json` |
| Cross-model unified figure | `outputs/benchmarks/comparison_cross_model.png` |

## How to regenerate

```bash
python testModels/small/synth_regression/run_benchmark.py
python benchmarks/run_all.py
python benchmarks/summarize_all.py   # replot from existing metrics only
```

**Defaults** target stable metrics: synthetic regression uses `--n-train 600 --epochs 40`; MNIST, CIFAR-10, and **linear_logistic** use `--per-class 150 --epochs 30` (1,500 training points each). Checkpoints are saved at **five** evenly spaced epochs per run. **Ghost and Original TracIn share the same multi-query batch** (`--num-queries`), so metrics compare like-for-like.

On Slurm/GPU (e.g. Trillium), use `bash submit_slurm.sh benchmark smoke`, `benchmark full`, or `benchmark-chain` (smoke then full with `afterok`). See the README Slurm section; sync artifacts with `sync_from_scratch.sh <job_id>` (each job ID has its own `outputs/benchmarks/` under the scratch result dir).
