# Q&A: Ghost+FAISS vs Original TracIn

## Question 1 — Why does Ghost+FAISS use more memory than original TracIn (even with compression)?

### The fundamental architectural difference

Original TracIn and Ghost+FAISS compute the **same mathematical quantity** — the TracIn score:

$$
\mathrm{TracIn}(z_i, z') = \sum_t \eta_t \langle \nabla_\theta \ell(z'; \theta_t),\ \nabla_\theta \ell(z_i; \theta_t) \rangle
$$

But they differ **dramatically** in *when* they compute the inner product and *what* they store in memory.

---

### Original TracIn: compute-and-discard (streaming dot product)

The full-gradient baseline (`full_gradient_tracin.py`) works like this:

```
For each checkpoint t:
    Compute query gradient   g_query  (one vector, size = total params)
    For each training sample i:
        Compute training gradient  g_train_i  (same size)
        score[i] += lr_t * dot(g_query, g_train_i)
        ← immediately discard g_train_i
```

At any moment, **only two vectors** live in memory: `g_query` and the current `g_train_i`. The score accumulator is just a single float per training sample. The gradient vectors are **never stored** — they are computed, dotted, and thrown away.

**Memory footprint:** $O(P)$ where $P$ = number of model parameters. That is it — just enough to hold one gradient at a time.

**The tradeoff:** This is extremely slow. For $N$ training samples, $Q$ queries, and $T$ checkpoints, you do $N \times Q \times T$ full backward passes. Each backward pass is expensive (full model, all parameters).

---

### Ghost+FAISS: store-then-search (vector database approach)

Ghost+FAISS is designed for **production retrieval** — you build an index once, then queries are instant. The pipeline:

```
Offline (once):
    For each checkpoint t:
        For each training sample i:
            Forward + backward hooks → capture A (activation), E (error signal)
            Form ghost vector:  g_i = vec(E^T A)     ← size H*C per layer
            (optional) Adam correction:  g_i ← g_i / (sqrt(v_t) + eps)
            (optional) SJLT projection:  g_i ← P @ g_i    (compress to lower dim)
            Accumulate:  stored_i += lr_t * g_i
    Store all N accumulated vectors into FAISS IndexFlatIP

Online (per query):
    Same ghost extraction for query → search FAISS → top-K
```

**Memory footprint:** You must **store all $N$ projected ghost vectors simultaneously** in the FAISS index. Even after SJLT compression to dimension $K$, that is $O(N \times K)$ floats in memory. The FAISS index alone is:

$$
\text{Index size} = N \times K \times 4\ \text{bytes (float32)}
$$

For $N = 200$ and $K = 1280$, this is just ~1 MB. But for real-scale datasets with $N = 1{,}000{,}000$ and $K = 1280$, it is ~5 GB.

Additionally, during the offline **ghost extraction phase**, for each batch you hold:
- The model weights ($P$ params)
- Per-layer **activations** $A$: shape $[B, L, H]$ for each hooked layer
- Per-layer **error signals** $E$: shape $[B, L, C]$ for each hooked layer
- The **materialized ghost vectors** before projection (if Adam correction or 3D paths are needed): $[B, \sum_\ell H_\ell \times C_\ell]$

For convolutional and transformer layers, the $L$ (spatial positions / sequence length) dimension makes this blow up. A ResNet50 early layer might have $L = 56 \times 56 = 3136$ spatial positions, so the raw activation block is $[B, 3136, C_\text{in} \times k_H \times k_W]$ — gigabytes for a single batch.

---

### Why compression (SJLT) doesn't fully close the gap

SJLT projection helps — it reduces the stored dimension from the full ghost dim $D = \sum_\ell H_\ell C_\ell$ (which can be 100K+) down to $K$ (e.g. 1280). But:

1. **You still store $N$ vectors.** Original TracIn stores zero vectors — it streams. Ghost+FAISS stores all $N$ compressed vectors permanently in the index. This is the core memory difference.

2. **Compression happens after ghost formation.** You still need to materialize the full $D$-dimensional ghost vector first, then multiply by the sparse projection matrix. So the peak memory during the offline phase includes the full ghost.

3. **Adam correction prevents factored shortcuts.** Without Adam, the ghost dot product factorizes:
   $$\langle g_i, g_j \rangle = \langle A_i, A_j \rangle \cdot \langle E_i, E_j \rangle$$
   and you never need to materialize the $H \times C$ ghost at all. But Adam correction requires element-wise division by $\sqrt{v_t} + \epsilon$, which is shaped $(H, C)$ — you **must** build the full ghost to apply it. After correction, the factorization breaks.

4. **3D layers (conv/sequence) require raw block storage.** For layers with spatial/temporal dimensions, the hook manager stores $[B, L, H]$ and $[B, L, C]$ tensors for each hooked layer simultaneously, because it needs all of them to compute the sum-of-outer-products ghost. This is where the memory really explodes for CNNs and transformers.

---

### Summary table

| Aspect | Original TracIn | Ghost+FAISS |
|--------|----------------|-------------|
| **What's stored** | Nothing — streaming dot products | All $N$ ghost vectors in FAISS index |
| **Peak memory (inference)** | $O(P)$ — two gradient vectors | $O(N \times K)$ — entire index in RAM |
| **Peak memory (offline build)** | N/A | $O(B \times L \times (H+C) \times \text{layers})$ per batch |
| **Query time** | $O(N \times P)$ per query (recompute everything) | $O(K \times N)$ FAISS search (sub-linear with IVF) |
| **Needs compression?** | No — nothing to compress | Yes — ghost dim can be >100K |
| **Scales to $10^6$ training samples?** | No (too slow) | Yes (that's the point) |

**Bottom line:** Ghost+FAISS trades memory for speed. Original TracIn is memory-cheap but compute-expensive. The memory cost of Ghost+FAISS comes from the fundamental design choice to **precompute and store** vectors rather than **stream and discard** dot products.

---

---

## Question 2 — Adam correction: the math from scratch

### Step 1: What Adam actually does

In vanilla SGD, the parameter update is:

$$
\theta_{t+1} = \theta_t - \eta_t \cdot \nabla_\theta \ell(z_i; \theta_t)
$$

Every parameter gets the same learning rate $\eta_t$.

Adam is different. It maintains two **running averages** for each individual parameter $\theta_j$:

- **First moment** (momentum): $m_t^{(j)} = \beta_1 m_{t-1}^{(j)} + (1 - \beta_1) g_t^{(j)}$
- **Second moment** (squared gradient EMA): $v_t^{(j)} = \beta_2 v_{t-1}^{(j)} + (1 - \beta_2) (g_t^{(j)})^2$

where $g_t^{(j)} = \frac{\partial \ell}{\partial \theta_j}$ is the gradient of parameter $j$ at step $t$.

The actual update Adam applies (ignoring bias correction for simplicity) is:

$$
\theta_{t+1}^{(j)} = \theta_t^{(j)} - \eta_t \cdot \frac{m_t^{(j)}}{\sqrt{v_t^{(j)}} + \epsilon}
$$

The key insight: **each parameter is scaled differently** by $1/(\sqrt{v_t^{(j)}} + \epsilon)$. Parameters with historically large gradients get smaller effective learning rates; parameters with small gradients get larger ones.

---

### Step 2: Why TracIn needs to account for this

The TracIn formula says the influence of $z_i$ on $z'$ is:

$$
\mathrm{TracIn}(z_i, z') = \sum_t \eta_t \langle \nabla \ell(z'; \theta_t),\ \nabla \ell(z_i; \theta_t) \rangle
$$

This formula was derived assuming **SGD** — where the update direction is literally $\eta_t \cdot \nabla \ell$. The intuition: if $z_i$'s gradient points in a similar direction to $z'$'s gradient, then the SGD step taken because of $z_i$ moved the model in a direction that helps (or hurts) $z'$.

But with Adam, the update direction is **not** the raw gradient — it's the Adam-scaled gradient. The actual direction the model moved because of $z_i$ is closer to:

$$
\frac{g_t^{(j)}}{\sqrt{v_t^{(j)}} + \epsilon}
$$

So if we want TracIn to reflect the **actual influence** of a training sample under Adam, we should use the Adam-scaled gradients in the inner product, not the raw ones.

---

### Step 3: The corrected TracIn formula

Define the Adam-corrected gradient:

$$
\tilde{g}^{(j)} = \frac{g^{(j)}}{\sqrt{v_t^{(j)}} + \epsilon}
$$

In vector form, let $D_t = \mathrm{diag}\!\Big(\frac{1}{\sqrt{v_t^{(1)}} + \epsilon},\ \ldots,\ \frac{1}{\sqrt{v_t^{(P)}} + \epsilon}\Big)$ be a diagonal scaling matrix. Then:

$$
\tilde{g} = D_t \cdot g
$$

The Adam-corrected TracIn score becomes:

$$
\mathrm{TracIn}_\text{Adam}(z_i, z') = \sum_t \eta_t \langle D_t\, \nabla \ell(z'; \theta_t),\ D_t\, \nabla \ell(z_i; \theta_t) \rangle
$$

Or equivalently (since $D_t$ is diagonal and symmetric):

$$
= \sum_t \eta_t\, (\nabla \ell(z'; \theta_t))^\top D_t^2\, \nabla \ell(z_i; \theta_t)
$$

This is an inner product in a **rescaled metric** defined by $D_t^2 = \mathrm{diag}(1/(v_t^{(j)} + \epsilon^2))$.

---

### Step 4: Applying this to ghost vectors

Recall the ghost vector for a single `nn.Linear(H, C)` layer:

$$
g = \mathrm{vec}(E^\top A)
$$

where $A \in \mathbb{R}^{1 \times H}$ (activation) and $E \in \mathbb{R}^{1 \times C}$ (error signal). The ghost is a flattened $H \times C$ matrix, and its entries are:

$$
g_{h,c} = A_h \cdot E_c
$$

Adam's second moment $v_t$ for this layer's weight is also an $H \times C$ matrix (after transposing from PyTorch's $(C, H)$ layout). Each entry $v_t^{(h,c)}$ corresponds to parameter $W_{h,c}$.

The corrected ghost is:

$$
\tilde{g}_{h,c} = \frac{A_h \cdot E_c}{\sqrt{v_t^{(h,c)}} + \epsilon}
$$

This is what `apply_adam_correction` in `math_utils.py` does: element-wise division of the flattened ghost by $\sqrt{v_t} + \epsilon$.

---

### Step 5: Why Adam correction breaks the factorization trick

Without Adam, the ghost dot product has a beautiful factorization:

$$
\langle g_1, g_2 \rangle = \sum_{h,c} A_h^{(1)} E_c^{(1)} A_h^{(2)} E_c^{(2)} = \Big(\sum_h A_h^{(1)} A_h^{(2)}\Big) \cdot \Big(\sum_c E_c^{(1)} E_c^{(2)}\Big) = \langle A_1, A_2 \rangle \cdot \langle E_1, E_2 \rangle
$$

This is $O(H + C)$ instead of $O(H \times C)$ — huge savings.

With Adam correction:

$$
\langle \tilde{g}_1, \tilde{g}_2 \rangle = \sum_{h,c} \frac{A_h^{(1)} E_c^{(1)}}{\sqrt{v^{(h,c)}} + \epsilon} \cdot \frac{A_h^{(2)} E_c^{(2)}}{\sqrt{v^{(h,c)}} + \epsilon}
$$

$$
= \sum_{h,c} \frac{A_h^{(1)} A_h^{(2)} \cdot E_c^{(1)} E_c^{(2)}}{v^{(h,c)} + 2\epsilon\sqrt{v^{(h,c)}} + \epsilon^2}
$$

The denominator $v^{(h,c)}$ depends on **both** $h$ and $c$ — it couples the two dimensions. You **cannot** separate this into a product of an $H$-only sum and a $C$-only sum. The factorization is gone.

This means with Adam correction, you **must** materialize the full $H \times C$ ghost vector, apply the element-wise scaling, and then either:
- Store it in the FAISS index (production path), or
- Compute the $D$-dimensional dot product directly (benchmark path)

---

### Step 6: Layout subtlety (PyTorch vs ghost)

One practical detail: PyTorch stores `nn.Linear` weights in shape $(C_\text{out}, C_\text{in})$ — output channels first. The ghost vector is formed as $\mathrm{vec}(E^\top A)$ which has shape $(H, C)$ — input features first, in row-major order.

Adam's `exp_avg_sq` in the optimizer state dict follows PyTorch's weight layout: $(C, H)$. To align it with the ghost layout, `load_adam_second_moment` **transposes** the matrix from $(C, H)$ to $(H, C)$ before flattening.

This is the `.T` in the code:

```python
return v_np.reshape(c_out, h_in).T.flatten().astype(np.float32)
#                                 ^ transpose to ghost layout
```

If you skip this transpose, every element gets divided by the wrong $v$ value, and your scores are garbage.

---

### Step 7: What the code actually does, end to end

1. **Load $v_t$:** `load_adam_second_moment(optimizer_state_path, param_key, weight_shape)` loads `exp_avg_sq` from the saved optimizer state, transposes from PyTorch $(C, H)$ to ghost $(H, C)$, and flattens to a 1D vector of length $H \times C$.

2. **Form raw ghost:** `form_ghost_vectors(A, E)` computes $g = \mathrm{vec}(A \otimes E)$, returning shape $(N, H \times C)$.

3. **Apply correction:** `apply_adam_correction(ghost_vectors, adam_v, eps=1e-8)` computes:
   ```
   scale = 1.0 / (sqrt(adam_v) + eps)        # shape (H*C,)
   corrected = ghost_vectors * scale[None, :] # broadcast over N samples
   ```

4. **Project (optional):** `project(corrected, P)` multiplies by the SJLT matrix to compress from $H \times C$ down to $K$.

5. **Accumulate:** `stored_i += lr_t * projected_i` summed across checkpoints.

6. **Index:** All $N$ accumulated vectors go into FAISS `IndexFlatIP`.

---

### Summary

| Concept | Formula | Code |
|---------|---------|------|
| Raw ghost | $g_{h,c} = A_h \cdot E_c$ | `form_ghost_vectors(A, E)` |
| Adam second moment | $v_t^{(h,c)}$ per parameter | `load_adam_second_moment(...)` |
| Corrected ghost | $\tilde{g}_{h,c} = g_{h,c} / (\sqrt{v_t^{(h,c)}} + \epsilon)$ | `apply_adam_correction(g, v)` |
| Why needed | Adam scales each param differently; raw gradients ≠ actual update direction | — |
| Why it breaks factorization | $v^{(h,c)}$ couples $h$ and $c$; can't separate the sum | — |
| Layout trap | PyTorch weight is $(C, H)$; ghost is $(H, C)$ → must transpose $v$ | `.T` in `load_adam_second_moment` |
