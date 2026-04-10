# TracIn Ghost — Presentation Plan

> **Audience**: PhD fellows (ML-literate, not familiar with TracIn)
> **Your parts**: 2 (Ghost Product), 3 (Real Model Support), 4 (Demo), 5 (Scalability)
> **Mentor covers**: Part 1 (Motivation + Original TracIn)
> **Total time**: ~45 min + Q&A

---
---

# ════════════════════════════════════════════════
# SIMPLE VERSION (Cheat Sheet)
# ════════════════════════════════════════════════

## Part 2 — Ghost Product (~10 min)

- **Problem**: TracIn needs the inner product of per-sample gradients. For N training samples and P parameters, that's N vectors of dimension P. For ResNet50 → 23.7 million dimensions × 50K samples. Way too expensive.
- **Key trick**: For an `nn.Linear` layer, the weight gradient is an outer product: ∇W = eᵀa (error × activation). The ghost vector is g = vec(e ⊗ a). The dot product of two ghost vectors **factors**: ⟨g₁, g₂⟩ = ⟨a₁, a₂⟩ · ⟨e₁, e₂⟩. This reduces O(H×C) to O(H+C).
- **Bias**: Append a 1 to activations → outer product naturally includes bias gradient. Ghost dim becomes (H+1)×C.
- **Multi-layer**: Sum the factored products from each layer.
- **Scale**: Project ghost vectors to 512 dims (SJLT), store in FAISS index. Query time = one FAISS search.

## Part 3 — Real Model Support (~15 min)

**Deep-dive these 3:**

1. **Conv2d (im2col)**: Convolution operates over spatial patches. Use `F.unfold` to reshape patches into a 2D matrix → same outer product trick applies. Can mean-pool over spatial positions (fast, approximate) or keep 3D (exact).
2. **Inplace ops crashing backward hooks**: `register_full_backward_hook` wraps output in an autograd view. Any inplace op after (like `ReLU(inplace=True)` or `out += skip`) crashes. Fix: use `output.register_hook()` instead — tensor-level hook, no wrapper. Affects ConvTranspose2d and BatchNorm2d.
3. **Adam preconditioning**: Adam scales gradients by 1/√(vₜ+ε). This is per-parameter, doesn't factorize into A and E components. Fix: apply correction _after_ forming the full ghost vector. Trap: PyTorch stores weight as (C,H), ghost uses (H,C) → must transpose before flattening. Bias has separate Adam state → concatenate.

**Mention briefly:**
- Grouped conv: split channels by group, process independently, concatenate
- Embedding: sparse gradient — only active tokens updated, use scatter-add over active vocab
- LayerNorm: capture normalized input, concat gamma/beta gradient components
- Auto-coverage: `auto_ghost_layers()` picks layers greedily by parameter count until target coverage reached

## Part 4 — Demo (~8 min)

- Models pre-trained beforehand. Run `interactive_demo.py` live on GPU.
- 3 tasks: Classification (CIFAR-10, click image), Text Gen (Shakespeare, pick prompt), Image Gen (Fashion VAE, click generated image)
- Each shows top-5 influential training samples with percentages
- Same code handles all 3 — only model and error function change

## Part 5 — Scalability (~8 min)

- 17 models tested, 3 tiers: Small (5), Medium (9), Large (3). Param range: 1K → 24M.
- **Accuracy**: 15/17 models get Spearman ρ ≥ 0.999 vs full-gradient TracIn
- **Speed**: ResNet50 (24M params) = 18.2× faster. Bigger model = bigger win.
- **Honest about limits**: ViT/Transformer overhead (many small non-Linear layers), U-Net ρ=0.73 (complex skip connections)
- **Coverage**: 95-100% for standard architectures

---
---

# ════════════════════════════════════════════════
# FULL VERSION (Read This to Understand Everything)
# ════════════════════════════════════════════════

---

## Part 2: The Ghost Product

### 2.1 — The problem (transition from mentor's Part 1)

Your mentor just explained TracIn. Recall the formula:

$$\text{TracIn}(z_i, z') = \sum_{t=1}^{T} \eta_t \, \langle \nabla_\theta \ell(z'; \theta_t),\; \nabla_\theta \ell(z_i; \theta_t) \rangle$$

This measures how much training sample zᵢ influenced the model's prediction on query z'.

**The cost**: To score every training sample, we need N inner products. Each gradient vector has P dimensions (one per model parameter). For ResNet50: P = 23.7 million. For 50K training samples, that means storing 50,000 vectors of 23.7M floats = ~4.4 TB, and computing 50K inner products in that space. Completely impractical.

**What we want**: A way to compute these inner products _without_ ever materializing the full gradient vectors.

### 2.2 — Focus on one Linear layer

Take a single `nn.Linear` layer: y = xWᵀ + b

During backpropagation, for a single sample, the weight gradient has a simple structure — it's an **outer product**:

$$\nabla_W \ell = e^\top a$$

where:
- **a ∈ ℝᴴ** = the layer's input (the activation that came in during forward pass)
- **e ∈ ℝᶜ** = the error signal (the gradient that came back during backward pass)

The "ghost vector" is just the flattened outer product:

$$g = \text{vec}(e \otimes a) \in \mathbb{R}^{H \times C}$$

**Where do a and e come from?** We hook the layer:
- **Forward hook** captures the input activation `a`
- **Backward hook** captures the gradient `e = grad_output`

No need to compute the actual gradient — just save `a` and `e`.

### 2.3 — The factorization trick (THE core insight)

Here's the key identity that makes everything work:

$$\langle g^{(1)}, g^{(2)} \rangle = \langle a^{(1)}, a^{(2)} \rangle \cdot \langle e^{(1)}, e^{(2)} \rangle$$

**Proof sketch**: g = vec(e ⊗ a), so ⟨g₁, g₂⟩ = Σᵢⱼ a₁ᵢ·e₁ⱼ·a₂ᵢ·e₂ⱼ = (Σᵢ a₁ᵢ·a₂ᵢ)(Σⱼ e₁ⱼ·e₂ⱼ) = ⟨a₁,a₂⟩·⟨e₁,e₂⟩

**Why this is huge**: Instead of one dot product in H×C dimensions, we do two small dot products (one in H dims, one in C dims) and multiply. 

**Example**: ResNet50's final layer has H=2048, C=1000.
- Naive: 2,048,000 multiply-adds
- Ghost: 2,048 + 1,000 = 3,048 multiply-adds → **672× fewer operations**

### 2.4 — Batched computation (how it runs in practice)

With a batch of N training samples and a query:

$$\text{scores} = (A_q \cdot A_{\text{train}}^\top) \odot (E_q \cdot E_{\text{train}}^\top)$$

- Aₜᵣₐᵢₙ is (N, H) — all training activations stacked
- Eₜᵣₐᵢₙ is (N, C) — all training error signals stacked
- Two matrix multiplies → two (1, N) vectors → element-wise multiply → scores

This is embarrassingly parallel on GPU. Just two `torch.mm` calls and a `*`.

### 2.5 — Including bias gradients

For a layer with bias b, the bias gradient is just: ∇b = e (the error signal itself).

**Trick**: Append a column of 1s to the activation:

$$a_{\text{aug}} = [a;\, 1] \in \mathbb{R}^{H+1}$$

Now the outer product e ⊗ a_aug naturally includes both weight and bias gradients:
- First H×C elements = weight gradient (same as before)
- Last C elements = 1·e = bias gradient

Ghost dimension becomes (H+1)×C. The factorization still works perfectly.

**Code**: `_maybe_append_bias_ones()` in `src/hooks_manager.py` — just `torch.cat([act, ones], dim=-1)`.

### 2.6 — Multiple layers: just sum

The full model gradient is a concatenation of per-layer gradients. So:

$$\langle g_{\text{full}}^{(1)}, g_{\text{full}}^{(2)} \rangle = \sum_{\ell=1}^{L} \langle g_\ell^{(1)}, g_\ell^{(2)} \rangle$$

Each layer's contribution is computed independently using its own (a, e) pair, then summed. Layers we don't hook are simply ignored (partial coverage — more on this in Part 3).

### 2.7 — Making it scalable: projection + FAISS index

Even with the factored trick, scoring 50K samples per query is slow if done from scratch every time. Solution: **precompute and index**.

**Offline phase (done once)**:

1. Forward + backward pass through all training data
2. At each hooked layer, capture (a, e) per sample
3. Form the ghost vector g = vec(e ⊗ a) ← now we DO materialize it (once)
4. Apply Adam correction (if using Adam — see Part 3)
5. **Project**: Multiply by a random sparse matrix P to reduce dimensionality
   - Ghost might be 2M dims → project to 512 dims
   - Uses SJLT (Sparse Johnson-Lindenstrauss Transform): random ±1/√K with 2/3 zeros
   - Key property: ⟨Px, Py⟩ ≈ ⟨x, y⟩ (preserves inner products)
6. **Accumulate across checkpoints**: index_vector[i] += ηₜ · P·gᵢₜ
7. Build FAISS `IndexFlatIP` (inner product index)

**Online phase (per query)**:

1. Forward + backward on query → ghost → same projection P → g_proj
2. FAISS search: returns top-K training samples with highest ⟨g_proj_query, index_vector[i]⟩
3. Scores = inner products → normalize to percentages

**Result**: Query-time attribution takes milliseconds, not hours. The expensive part (building the index) is done once.

### 2.8 — Summary: what you just explained

```
Original TracIn:  Store N gradients of P dims each → N×P inner products
Ghost TracIn:     Hook layers → capture (a,e) → factor: ⟨g₁,g₂⟩ = ⟨a₁,a₂⟩·⟨e₁,e₂⟩
                  → project to 512 dims → FAISS index → instant queries
```

---

## Part 3: Making It Work With Real Models

Ghost factorization was originally designed for `nn.Linear` layers. Real models have Conv2d, ConvTranspose2d, Embedding, BatchNorm, LayerNorm, RNNs, grouped convolutions, skip connections with inplace ops, and Adam optimizers.

Each of these needed a specific solution. Here are the 3 most important ones (deep-dive), followed by the rest (brief).

### 3.1 — [DEEP DIVE] Conv2d: The Im2col Trick

**The problem**: Conv2d weight is a 4D tensor (C_out, C_in, kH, kW). The convolution slides a kernel over spatial positions. There's no simple y = xWᵀ structure.

**The insight**: A convolution is actually a matrix multiply _if you rearrange the input_. This is called "im2col" (Image to Columns).

**How it works**:

1. `F.unfold(input)` extracts every patch that the kernel touches, and stacks them as rows:
   - Input: (B, C_in, H, W)
   - Output: (B, C_in·kH·kW, L) where L = number of spatial positions (output H × output W)

2. Each "column" (of length C_in·kH·kW) is one patch. The convolution at position l is:
   output[:, :, l] = W_reshaped @ unfold[:, :, l]

3. This means at each spatial position, we have the same outer product structure:
   ∇W at position l = e_l ⊗ a_l

**Two strategies**:

- **Fast (approximate)**: Average-pool over L spatial positions → get a single (B, C_in·kH·kW) activation matrix → apply standard 2D factored trick. Quick, slight approximation.

- **Exact**: Keep all L positions. Ghost = Σₗ eₗ ⊗ aₗ (sum of outer products), computed via `einsum('bth,btv→bhv', A_3d, E_3d)`. More memory, but exact.

**Controlled by** `keep_raw` flag in `HookManager`. When `keep_raw=False`, mean-pool. When `True`, keep 3D.

**Code**: `_unfold_conv2d_input()` in `src/hooks_manager.py`.

### 3.2 — [DEEP DIVE] Inplace Ops Crashing Backward Hooks

**The problem**: This was one of the hardest bugs. PyTorch's `register_full_backward_hook` internally wraps the module's output tensor in a special autograd node called `BackwardHookFunction`. This creates a **view** of the output.

If ANY subsequent operation modifies that tensor in-place, PyTorch crashes:
```
RuntimeError: one of the variables needed for gradient computation has been 
modified by an inplace operation
```

**Two classes of inplace ops that trigger this**:

1. **Module-level**: `nn.ReLU(inplace=True)` — easy fix: set `m.inplace = False`
2. **Python-level**: `out += identity` (residual connections) — **cannot be patched** because the `+=` is in the model's forward code, not a module attribute

**Where this hits**:
- **ConvTranspose2d**: U-Net uses `conv_transpose → concat → relu`. The concat or subsequent inplace ops crash.
- **BatchNorm2d**: ResNet uses `bn → relu_(inplace=True)`. The `relu_` is inplace on bn's output → crash.

**The fix — tensor-level hooks**:

Instead of `module.register_full_backward_hook(fn)`, use:

```python
def forward_hook(module, input, output):
    # Register a hook on the OUTPUT TENSOR, not the module
    output.register_hook(lambda grad: save_grad(grad))
```

Tensor-level hooks (`output.register_hook()`) do NOT create the `BackwardHookFunction` wrapper. They receive the gradient directly. No view, no inplace conflict.

**Why not just use tensor hooks for everything?** `register_full_backward_hook` gives you `grad_input` and `grad_output` as tuple arguments, which is convenient. Tensor hooks only give you the gradient of the specific tensor. For most layers, the full backward hook is fine — we only switch to tensor hooks where inplace ops exist.

**Code**: `_capture_conv_transpose2d()` and `_capture_batchnorm2d()` in `src/hooks_manager.py`.

**Lesson learned**: `output.clone()` in the forward hook does NOT fix this — the hook wrapping happens _after_ the forward hook returns, so the clone gets wrapped too.

### 3.3 — [DEEP DIVE] Adam Optimizer Preconditioning

**The problem**: When using Adam optimizer, TracIn should weight each gradient by the inverse square root of Adam's second moment:

$$g_{\text{corrected}} = g \odot \frac{1}{\sqrt{v_t} + \epsilon}$$

where vₜ is Adam's exponential moving average of squared gradients (per-parameter).

**Why the factored trick breaks**: The factored identity ⟨g₁,g₂⟩ = ⟨a₁,a₂⟩·⟨e₁,e₂⟩ works because g = vec(e⊗a). But after Adam correction, g_corrected = g / √v, and v is NOT separable into activation and error components. It's a per-element scaling.

**Solution**: Form the full ghost vector first (materialize it), THEN apply the Adam correction, THEN project. The factored trick saved us from materializing during dot products; for Adam, we materialize once during indexing.

**The layout trap**: This caused a subtle bug.
- PyTorch stores `nn.Linear.weight` as shape **(C_out, H_in)** — row = output neuron
- Ghost vector uses layout **(H_in, C_out)** — row = input neuron
- Adam's `exp_avg_sq` follows PyTorch's layout: shape (C_out, H_in)

If you just flatten v without transposing, the correction is applied to the wrong ghost elements!

**Fix**: Load Adam state → reshape to (C, H) → **transpose to (H, C)** → flatten → now matches ghost layout.

```python
v = optimizer_state[param_key]["exp_avg_sq"]  # shape (C, H)
v = v.reshape(C, H).T.flatten()               # → (H, C) → (H*C,)
```

**The bias extension**: Adam has separate state for weight and bias parameters. They have different parameter keys in the optimizer state dict. When we use bias augmentation (ghost dim = (H+1)×C), we need to:
1. Find the bias parameter key: `find_adam_bias_param_key()` in `src/config_utils.py`
2. Load weight Adam state (H×C) + bias Adam state (C)
3. Concatenate: v_full = [v_weight_transposed_flat; v_bias] → length (H+1)×C

**Code**: `load_adam_second_moment_with_bias()` in `src/math_utils.py`.

### 3.4 — [BRIEF] Other layer types

**Grouped Convolutions** (e.g., ResNet bottleneck with groups=32):
- Problem: Input channels split into groups, each group only connects to its group's output channels
- Solution: Split activations and errors by group, compute ghost per group, concatenate. Same outer product trick, just G times independently.

**Embedding Layers** (e.g., word embeddings in transformers):
- Problem: Embedding gradient is sparse — only tokens present in the batch have non-zero gradients. Full vocab V could be 50K+, but a batch might use ~200 unique tokens.
- Solution: Track which tokens appear, scatter-add gradient contributions per token, compute dot products only over the **active vocabulary**. Reduces O(V) to O(|active|).
- Code: `_embedding_ghost_dots()` in `benchmarks/ghost_faiss.py`

**LayerNorm** (transformers):
- Capture the normalized input x_norm = (x - μ) / √(σ² + ε)
- Ghost = concat of two components: weight_grad = x_norm · error, bias_grad = error (summed over normalized dimensions)
- Code: `_layernorm_x_normalized()` in `src/hooks_manager.py`

**Automatic Layer Selection** (`auto_ghost_layers()`):
- Problem: which layers should we hook? Manual selection is tedious and model-specific.
- Solution: Iterate all modules, filter to supported types, sort by parameter count, greedily select until we reach target coverage (e.g., 95% of total parameters).
- This means you just pass any model and the system figures out what to hook.
- Code: `auto_ghost_layers()` in `benchmarks/ghost_faiss.py`

### 3.5 — The full algorithm (summary, show this as a slide)

```
═══════════════════════════════════════════════════
OFFLINE: Build Index (run once per model)
═══════════════════════════════════════════════════

For each checkpoint t (weights θₜ, optimizer state vₜ, learning rate ηₜ):
│
├─ Load weights → set model to eval mode
│
├─ For each training batch:
│   │
│   ├─ Forward pass → hooks capture activations A per layer
│   │
│   ├─ Compute error signal: E = error_fn(output, target)
│   │   (classification → cross-entropy gradient)
│   │   (regression → MSE gradient)
│   │   (generation → autoregressive loss gradient)
│   │
│   ├─ For each hooked layer:
│   │   ├─ Linear:           A is (B, H), E is (B, C) → factored
│   │   ├─ Conv2d:           F.unfold → (B, C_in·k², L) → mean-pool or 3D
│   │   ├─ ConvTranspose2d:  asymmetric padding + tensor hook for grad
│   │   ├─ BatchNorm2d:      normalized input + tensor hook for grad
│   │   ├─ Embedding:        sparse token indices + scatter-add
│   │   └─ LayerNorm:        normalized input, concat gamma/beta grads
│   │
│   ├─ Form ghost vector: g = concat across layers
│   │
│   ├─ Adam correction: g = g / (√vₜ + ε)  [transpose trick for layout]
│   │
│   ├─ Project: g_proj = P · g  [SJLT, 512 dims]
│   │
│   └─ Accumulate: index_vector[sample_i] += ηₜ · g_proj
│
└─ Build FAISS IndexFlatIP (inner product)
   Save index + metadata (sample IDs, labels/rights holders)

═══════════════════════════════════════════════════
ONLINE: Query Attribution (per query, instant)
═══════════════════════════════════════════════════

Query input → Forward + Backward → hooks capture (A_q, E_q)
→ Form ghost → Adam correct → Project (same P)
→ FAISS search → top-K samples + inner product scores
→ Normalize: score_i / Σ scores → percentage attribution
```

---

## Part 4: Live Demo

### What to prepare before the presentation

1. Pre-train all 3 models (Slurm job or local). Command:
   ```bash
   python demos/pretrain_all.py \
       --device cuda --data-root /scratch/duchuy/tracin_benchmark_data \
       --cifar-epochs 10 --gpt-epochs 15 --vae-epochs 20 \
       --max-train 8000 --n-train 8000
   ```

2. Verify checkpoints exist:
   ```bash
   ls demos/outputs/cifar10_classification/checkpoints/
   ls demos/outputs/tinygpt_demo/checkpoints/
   ls demos/outputs/vae_fashion_demo/checkpoints/
   ```

3. Have backup figures (from `--headless` run) in case live demo fails.

### During the presentation

Run on a **GPU node** (not login node — TinyGPT training exceeds CPU time limits).

```bash
python demos/interactive_demo.py \
    --device cuda \
    --data-root /scratch/duchuy/tracin_benchmark_data \
    --save-figures --top-k 5
```

### Task 1: Image Classification (CIFAR-10) — ~3 min

**What happens**:
1. Script shows a 4×5 grid of random CIFAR-10 test images
2. You click one (e.g., a cat)
3. It runs attribution (~5 sec)
4. Shows: your query image on left, top-5 most influential training images on right, with percentages and a colored bar chart at the bottom

**What to say**:
- "I picked this cat image. The algorithm found these 5 training images that had the most influence on the model's prediction."
- "Notice they're all cats or similar-looking animals — the algorithm correctly identifies which training data was most important."
- "This took about 5 seconds. Without the ghost trick, this would take minutes."

### Task 2: Text Generation (Shakespeare) — ~2 min

**What happens**:
1. Menu shows 4 Shakespeare-style prompts + custom option
2. You pick one (e.g., "To be or not to be")
3. Model generates ~100 characters of continuation
4. Attribution shows which training passages most influenced the generation

**What to say**:
- "Now a completely different task — text generation. Same attribution algorithm."
- "The top training snippets should be thematically similar to what was generated."

### Task 3: Image Generation (Fashion-MNIST VAE) — ~3 min

**What happens**:
1. Script generates 12 random fashion item images from the VAE's latent space
2. You click one (e.g., a sneaker)
3. Attribution shows which training images most influenced that generation

**What to say**:
- "A generative model now — which training images did the VAE 'remember' when generating this?"
- "Top results should be the same category — sneakers influencing sneaker generation."
- "Key point: the same ghost factorization code handles classification, text generation, and image generation. Only the model and loss function change."

---

## Part 5: Scalability & Benchmark Results

### Data source
- Slurm job 416781 results
- Located at: `/scratch/duchuy/tracin_ghost_tool_results/416781/outputs/benchmarks/summary.json`
- Comparison graph: `comparison_cross_model.png`

### 5.1 — Model diversity (show as a table)

| Tier | # Models | Examples | Param Range |
|------|----------|---------|-------------|
| **Small** | 5 | Synth Regression, Linear Logistic, MNIST MLP, MNIST Autoencoder, Multi-Task | ~1K – 120K |
| **Medium** | 9 | CIFAR CNN, ResNet-CIFAR100, Transformer LM, Fashion VAE, ViT, Encoder Transformer, MLP-Mixer, GRU LM, U-Net | ~20K – 2.3M |
| **Large** | 3 | ResNet50-CIFAR100, Large Transformer LM, Large ViT | ~19M – 24M |

**Architecture types**: Linear, CNN, ResNet (skip connections + grouped conv), Transformer (attention + LayerNorm), ViT (patch embedding + attention), MLP-Mixer, VAE (encoder-decoder), GRU, U-Net (ConvTranspose2d + skip connections)

**What to say**: "We tested on 17 different architectures covering basically every common neural network type."

### 5.2 — Accuracy: Spearman ρ (the headline result)

Ghost TracIn vs full-gradient Original TracIn — how well does the ranking match?

| Model | Spearman ρ | Notes |
|-------|-----------|-------|
| synth_regression | 1.000 | Perfect |
| linear_logistic | 1.000 | Perfect |
| mnist | 1.000 | Perfect |
| mnist_autoencoder | 1.000 | Perfect |
| multi_task | 1.000 | Perfect |
| cifar10_cnn | 1.000 | Perfect |
| resnet_cifar100 | 0.999 | Excellent — slight spatial pooling approx |
| transformer_lm | 1.000 | Perfect |
| vae_fashion | 1.000 | Perfect |
| vit_cifar10 | 1.000 | Perfect |
| encoder_transformer | 0.997 | Very good |
| mlp_mixer_cifar10 | 1.000 | Perfect |
| gru_lm | 1.000 | Perfect (only 17% coverage!) |
| unet_tiny | 0.729 | Moderate — complex skip connections |
| resnet50_cifar100 | 1.000 | Perfect at 24M params |
| transformer_lm_large | 1.000 | Perfect |
| vit_large_cifar10 | 1.000 | Perfect |

**What to say**: "15 out of 17 models have ρ ≥ 0.997 — the ghost ranking is essentially identical to computing the full gradient. Even the GRU with only 17% parameter coverage gets perfect correlation, because the hooked layers capture the most important gradient components."

**On U-Net (be honest)**: "U-Net is the hardest case — it has transposed convolutions, dense skip connections, and complex residual paths. Even there, we get 62-72% top-k overlap, which is still a meaningful ranking."

### 5.3 — Speed comparison

**Where ghost wins big**:

| Model | Params | Original (s) | Ghost (s) | Speedup |
|-------|--------|--------------|-----------|---------|
| resnet50_cifar100 | 23.7M | 2,900 | 160 | **18.2×** |
| multi_task | 120K | 86 | 8 | **10.3×** |
| mnist | 120K | 36 | 6 | **6.0×** |
| vae_fashion | 340K | 105 | 23 | **4.6×** |
| unet_tiny | 2.3M | 213 | 52 | **4.1×** |
| vit_large_cifar10 | 21.3M | 1,086 | 339 | **3.2×** |

**Where ghost has overhead** (be transparent):

| Model | Original (s) | Ghost (s) | Speedup | Why |
|-------|-------------|-----------|---------|-----|
| vit_cifar10 | 99 | 640 | 0.16× | Many LayerNorm layers, per-layer path |
| transformer_lm | 146 | 644 | 0.23× | Attention-heavy, FAISS indexing cost |
| encoder_transformer | 142 | 611 | 0.23× | Same pattern |
| gru_lm | 50 | 202 | 0.25× | Sequence unrolling overhead |

**What to say**: "The speedup grows with model size. ResNet50 at 24 million parameters sees an 18× speedup. For attention-heavy models like small ViTs and Transformers, there's overhead from the per-layer path — but notice that when we scale up to Large ViT (21M params), we get 3.2× speedup. The factorization benefit scales better than the overhead."

### 5.4 — The scaling trend (key insight)

Show these two progressions:

**ResNet family**:
- ResNet-CIFAR100 (840K params): 2.2× speedup
- ResNet50-CIFAR100 (23.7M params): **18.2× speedup**

**ViT family**:
- ViT-CIFAR10 (268K params): 0.16× (overhead)
- ViT-Large-CIFAR10 (21.3M params): **3.2× speedup**

**What to say**: "This is the trend line. Ghost's advantage comes from avoiding the H×C materialization. As models get bigger, H and C grow, so the savings grow quadratically while overhead stays roughly constant. Even architectures where ghost is slow at small scale become fast at large scale."

### 5.5 — Ghost coverage

| Coverage | # Models | Example |
|----------|----------|---------|
| 100% | 12 | MNIST, CIFAR CNN, VAE, MLP-Mixer |
| 94-97% | 3 | ViT (94.4%), Transformers (95-97%) |
| 17% | 1 | GRU (only input-to-hidden hooked) |

**What to say**: "auto_ghost_layers automatically hooks all supported layer types. Most models get 100% coverage. Even the GRU at 17% coverage still gives perfect Spearman correlation — telling us the input-to-hidden weights capture enough of the gradient signal."

### 5.6 — Closing summary

| Metric | Result |
|--------|--------|
| Architectures tested | 17 (every common type) |
| Accuracy (ρ ≥ 0.999) | 15/17 models |
| Best speedup | 18.2× (ResNet50, 24M params) |
| Parameter range | 1K → 24M |
| Auto-coverage | 95-100% typical |
| Query time | Milliseconds (FAISS search) |

**Final message**: "Ghost TracIn makes training data attribution practical for real models. It's accurate, it scales, and it works across every standard architecture type."

---

## Presentation Flow Diagram

```
Mentor: Motivation + Original TracIn
              │
              ▼
Part 2: Ghost Product   ◄── "How do we make TracIn fast?"
  │  Outer product structure → factored dot product
  │  Bias augmentation → multi-layer → SJLT + FAISS
  ▼
Part 3: Real Models     ◄── "What broke, and how we fixed it"
  │  Conv2d (im2col) [DEEP]
  │  Inplace ops (tensor hooks) [DEEP]
  │  Adam correction (layout trap) [DEEP]
  │  Grouped conv, Embedding, LayerNorm, Auto-coverage [BRIEF]
  ▼
Part 4: Live Demo       ◄── "Watch it work"
  │  Classification → Text Gen → Image Gen
  │  Same code, different models
  ▼
Part 5: Scalability     ◄── "It's not just a demo"
  │  17 models, ρ ≥ 0.999 for 15/17
  │  Up to 18× faster, scales with model size
  └─ ◄── "Ghost TracIn makes attribution practical"
```

---

## Practical Checklist

- [ ] Slurm job completed (demo pre-training)
- [ ] Attribution PNGs exist: `demos/outputs/*/attribution_*.png`
- [ ] Interactive demo tested on GPU node
- [ ] Benchmark comparison graph ready from job 416781
- [ ] Headless figures as backup
- [ ] Know which CIFAR image to click for best visual result (test beforehand)
