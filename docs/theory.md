# TracIn Ghost — Theoretical Deep Dive

This document expands the [README](../README.md) overview into full mathematical detail and maps each idea to code (function names, not line numbers). For setup and commands, see [implementation.md](implementation.md).

---

## 1. TracIn: Full Formulation

**Influence functions** approximate how a small perturbation to training data changes model parameters and thus loss on a test point. **TracIn** (Pruthi et al., NeurIPS 2020) traces this effect along the *actual* optimization trajectory.

For checkpoints \(t = 1,\ldots,T\), learning rates \(\eta_t\), and per-example losses \(\ell(z;\theta)\), the TracIn score of training example \(z_i\) on query \(z'\) is

\[
\mathrm{TracIn}(z_i, z') = \sum_{t=1}^{T} \eta_t \left\langle \nabla_\theta \ell(z'; \theta_t), \nabla_\theta \ell(z_i; \theta_t) \right\rangle.
\]

**Assumptions and caveats:** (i) Gradients are taken w.r.t. the same parameterization as training. (ii) The sum is a first-order proxy; curvature and discrete updates are ignored. (iii) Multi-checkpoint TracIn is the reference; production may use **TracIn-last-on-query** (query gradient only at \(\theta_T\)) — see README *Mathematical foundations* and `inference.attribute` / benchmark notes.

**Code:** `benchmarks/full_gradient_tracin.py` (`compute_full_gradient_tracin_scores`) implements the full-parameter baseline; `benchmarks/exact_tracin.py` holds ghost-based variants.

---

## 2. Ghost Dot Product (Linear Layer)

For `nn.Linear` with weight \(W \in \mathbb{R}^{C \times H}\), input activation \(A \in \mathbb{R}^{B \times H}\), and upstream gradient w.r.t. output \(E \in \mathbb{R}^{B \times C}\) (error signal), the per-sample weight gradient is

\[
\nabla_W \ell = E^\top A \quad\text{(batch as outer-product sum)}.
\]

Vectorizing \(g = \mathrm{vec}(E^\top A)\) (ghost layout aligned with code), for two samples the Euclidean inner product **factorizes**:

\[
\langle g^{(1)}, g^{(2)} \rangle = \langle A^{(1)}, A^{(2)} \rangle \cdot \langle E^{(1)}, E^{(2)} \rangle
\]

when \(A,E\) are 2D and the ghost is formed from the rank-one structure \(E \otimes A\). This is **exact** for that layer’s contribution to the ghost — not for the full network unless only that layer is hooked.

**Code:** `src/math_utils.py` — `form_ghost_vectors`, `form_multi_layer_ghost_vectors`; `benchmarks/ghost_faiss.py` — factored path using `mm` products on 2D blocks.

---

## 3. Extension to 3D Activations (Sequence / im2col)

When inputs are \((B, T, H)\) (transformer tokens) or im2col-unfolded conv features \((B, L, C_{\mathrm{in}} k_H k_W)\), the true per-layer weight gradient is a **sum of outer products** over positions \(t\) or \(l\):

\[
G = \sum_t E_t^\top A_t, \quad g = \mathrm{vec}(G).
\]

The ghost dot \(\langle g^{(1)}, g^{(2)}\rangle\) does **not** factor into separate \(\langle A^{(1)}, A^{(2)}\rangle \langle E^{(1)}, E^{(2)}\rangle\) in one step; the implementation uses sum-of-products / SDP-style contractions (`einsum`) or materialized blocks depending on \(T,H,C\) and device.

**Conv2d:** Same structure with \(L = H'W'\). Exact ghosts require `keep_raw=True` in `MultiLayerBackwardGhostManager` so \([B,L,\cdot]\) tensors are retained.

**Complexity:** Materializing full \(G\) per sample is \(O(L \cdot H \cdot C)\); SDP paths avoid full \(H \times C\) matrices when profitable.

**Code:** `benchmarks/ghost_faiss.py` — `_layer_ghost_dots_from_raw_blocks`, hybrid factored vs per-layer split (`_factored_2d_linear_indices`); `src/hooks_manager.py` — `MultiLayerBackwardGhostManager`.

---

## 4. Multi-Layer Ghost Vectors

Hook layers \(\ell = 1,\ldots,L\). Concatenate per-layer ghosts \(g^{(\ell)}\):

\[
g = [g^{(1)}; \ldots; g^{(L)}], \quad \langle g_i, g_j \rangle = \sum_{\ell} \langle g_i^{(\ell)}, g_j^{(\ell)} \rangle.
\]

**Coverage:** Only hooked parameters contribute. Uncovered parameters are invisible to the ghost; optional **auto fallback** computes per-sample full gradients for the remainder (expensive).

**Heuristic vs exact:** The sum over layers is exact for the **direct sum** of hooked subspaces; it is **not** the full-network gradient inner product unless coverage is 100% (or fallback fills the rest).

**Code:** `auto_ghost_layers` in `ghost_faiss.py`; `model_ghost_coverage` in `influence_variants.py`.

---

## 5. Grouped Convolutions

For `groups > 1`, channels split into independent groups. Ghost dots decompose per group; each group has its own \((A_g, E_g)\) block. Implementations must not mix channels across groups when forming outer products.

**Code:** Group-aware paths in `ghost_faiss.py` (search `groups` / conv grouping).

---

## 6. Embedding Layers

Embeddings select rows of a weight matrix \(W \in \mathbb{R}^{V \times H}\) via indices. Gradients are **sparse** in vocabulary dimension: only rows for tokens present in the batch receive non-zero updates. Ghost dots can use scatter-add / active-vocabulary optimizations.

**Code:** Embedding hooks and token-matching logic in `ghost_faiss.py` / `hooks_manager.py` (special-case modules).

---

## 7. LayerNorm and BatchNorm

LayerNorm: normalize across feature dims, then affine \(\gamma, \beta\). Ghost hooks use normalized pre-affine activations where aligned with weight-gradient structure (see `_layernorm_x_normalized`).

BatchNorm2d (eval): use running mean/var for stable “frozen” normalization; `_batchnorm2d_x_normalized`.

**Why special:** These layers couple features; naive linear ghost on raw inputs does not match \(\nabla W\) structure.

**Code:** `src/hooks_manager.py` — LayerNorm/BatchNorm branches in `MultiLayerBackwardGhostManager`.

---

## 8. Adam Second-Moment Correction

Adam scales updates by \(1 / (\sqrt{v_t} + \epsilon)\) with \(v_t\) an EMA of squared gradients **per parameter**. Applied to ghost vectors:

\[
g_{\text{corr}} = g \odot \frac{1}{\sqrt{v_t} + \epsilon}
\]

(elementwise after flattening \(v_t\) to ghost layout). **This does not factorize** into separate \(A\) and \(E\) scalings — \(v_t\) is full weight-shaped.

**Layout:** PyTorch `nn.Linear` weights are \((C_{\text{out}}, C_{\text{in}})\); ghost uses \((H, C)\) layout — `load_adam_second_moment` transposes `exp_avg_sq` accordingly.

**Code:** `src/math_utils.py` — `apply_adam_correction`, `load_adam_second_moment`; Adam branches in `ghost_faiss.py`.

---

## 9. SJLT Projection

**Sparse Johnson–Lindenstrauss** (Achlioptas, 2003): random matrix \(P \in \mathbb{R}^{K \times d}\) with sparse entries preserves pairwise inner products approximately: \(\langle Pg_i, Pg_j \rangle \approx \langle g_i, g_j \rangle\).

Stored sparse (e.g. CSR) — never materialize dense \(P\) for large \(d\).

**Code:** `build_sjlt_matrix`, `project` in `math_utils.py`.

---

## 10. Auto Fallback (Uncovered Parameters)

When hooked layers cover fraction \(< 1\) of parameters, remaining params can be handled by **per-sample backward** over the full model (or subset) to fill the gradient — cost \(\times\) training set size.

**When to disable:** If uncovered params are negligible (e.g. biases only) and benchmarks show stable \(\rho\), `auto_fallback=False` is an optional speed win.

**Code:** `_per_sample_fallback_grad_matrix` and call sites in `ghost_faiss.py`.

---

## 11. Hybrid Factored Path (2D Linear + Per-Layer)

If **any** layer needs the expensive raw-block path (3D, Embedding, LN, BN), older logic routed **all** layers through it. The **hybrid** path splits:

- **Factored indices:** 2D linear activations → fast \(\mathrm{mm}(A,A^\top) \odot \mathrm{mm}(E,E^\top)\)-style products.
- **Per-layer indices:** remaining layers → `_layer_ghost_dots_from_raw_blocks`.

Scores add: \(S = S_{\text{factored}} + S_{\text{per-layer}}\).

**Code:** `compute_ghost_tracin_scores` — `use_factored_no_adam` branch, `_factored_2d_linear_indices`.

---

## 12. Memory Model

**Per batch, per layer:** Forward/backward hooks may store `raw_A`, `raw_E` with shapes \([B,L,\cdot]\). Peak memory scales with \(B \times L \times \text{fan-in/fan-out}\).

**`max_spatial_positions`:** When \(L\) is large (early ResNet feature maps), cap \(L\) via mean-pooled fallback blocks to bound memory — slight approximation vs full spatial sum.

**Lifecycle:** `del` large tensors after dot products; periodic `torch.cuda.empty_cache()` reduces peak fragmentation on CUDA.

**Code:** `compute_ghost_tracin_scores` loops; conv benchmarks may pass `max_spatial_positions` to trade accuracy for memory (optional).

---

## Further Reading

- [README.md](../README.md) — problem, solution sketch, architecture table, benchmark snapshot.
- [implementation.md](implementation.md) — install, Slurm, adding models, API index.
