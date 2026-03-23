# Architectural Blueprint: Generative AI Copyright Attribution Plugin

**Version:** 2.0 (Corrected Math / Plug-and-Play)
**Objective:** Build a real-time, mathematically grounded system to calculate the proportional influence of specific copyrighted training data on a generated output, enabling dynamic revenue sharing.

---

## 1. Core Mathematical Foundations

The system is built on TracIn (Tracing Influence) with ghost dot products, Adam correction, and SJLT projection.

### 1.1 The Ghost Dot Product (Efficiency Shortcut)

**The Problem:** Standard TracIn requires computing per-sample gradient vectors (one entry per model parameter) and their dot products. For billions of parameters this is impossible.

**The Solution:** Exploit the structure of the last-layer gradient. For weight matrix \(W \in \mathbb{R}^{C \times H}\):
\[
\nabla_W \ell = E \cdot A^\top
\]
where \(A \in \mathbb{R}^H\) = hidden activation (input to last linear layer), \(E \in \mathbb{R}^C\) = error signal (∂loss/∂logits).

**The Math:** The dot product of two such gradients factorizes:
\[
\langle \text{vec}(E_1 A_1^\top), \text{vec}(E_2 A_2^\top) \rangle = \langle A_1, A_2 \rangle \cdot \langle E_1, E_2 \rangle
\]

**Ghost vector:** \(g = \text{vec}(E \cdot A^\top) = E \otimes A\) (Kronecker product). Then \(\langle g_1, g_2 \rangle = \langle A_1, A_2 \rangle \cdot \langle E_1, E_2 \rangle\).

**Why:** Reduces computation from full-parameter gradients to simple vector dot products. Ghost vectors can be indexed directly in FAISS for ANN search.

### 1.2 Adam Correction (Linearized Ghost Approximation)

**The Problem:** Standard TracIn assumes SGD: \(\theta_{t+1} = \theta_t - \eta \nabla \ell\). With Adam, the update is \(\theta_{t+1} = \theta_t - \eta \cdot \nabla \ell / (\sqrt{v_t} + \epsilon)\) where \(v_t\) is the second-moment estimate. Ignoring this yields wrong influence scores.

**The Solution:** Apply the same preconditioning to ghost vectors:
\[
g_{\text{corrected}} = g / (\sqrt{v_t} + \epsilon)
\]
where \(v_t\) is Adam's `exp_avg_sq` for the target layer weight, flattened to \((H \cdot C,)\).

**Important:** The correction must be applied to the **full ghost vector** \(g = \text{vec}(E \cdot A^\top)\), not just to \(E\). Adam's \(v_t\) is per-weight-element and does not factorize along \(E\)/\(A\) dimensions.

**If Adam state is unavailable** (e.g., SGD-trained model, or only final weights provided), skip this step. The pipeline handles `optimizer_state_path: null` gracefully.

### 1.3 Sparse Johnson-Lindenstrauss Transform (SJLT)

**The Problem:** Ghost vectors have dimension \(H \times C\). For large models (e.g., ResNet-50: 2048 × 1000 = 2M dims) this is too large for FAISS.

**The Solution:** Project to a fixed \(K\) dimensions using Achlioptas (2003) sparse random projection:
\[
P[i,j] = \sqrt{3/K} \cdot \begin{cases} +1 & \text{w.p. } 1/6 \\ 0 & \text{w.p. } 2/3 \\ -1 & \text{w.p. } 1/6 \end{cases}
\]
Only ~1/3 non-zero entries. Use `scipy.sparse` for memory efficiency.

**Note:** The projection matrix \(P\) has shape \((K, H \cdot C)\) — it is **model-specific** because \(H \cdot C\) depends on the model. Only the output dimension \(K\) is standardized. The claim "entirely model-agnostic" is misleading; each model needs its own \(P\).

**Dense Gaussian fallback:** For small ghost dims, use dense projection. For large dims, SJLT is essential.

---

## 2. System Pipeline

### Phase 1: Offline Indexing (Building the Vault)

Runs once per model to index the copyrighted dataset.

1. **Target Hooks:** Register `forward_hook` on the last linear layer (e.g., `model.fc2`, `model.lm_head`). Capture input \(A\) and output (logits).
2. **Forward Pass:** Pass training data through the model. No backward pass needed — \(E\) is computed analytically from the task adapter (e.g., softmax − one_hot for cross-entropy).
3. **Extract & Correct:** Form ghost vectors \(g = \text{vec}(E \cdot A^\top)\). If Adam state available, apply \(g \leftarrow g / (\sqrt{v_t} + \epsilon)\).
4. **Compress:** Project with SJLT: \(g_{\text{proj}} = P \cdot g\).
5. **Accumulate:** Sum across checkpoints: \(\text{accumulated}[i] = \sum_t \eta_t \cdot g_{\text{proj},i,t}\).
6. **Store:** Build FAISS inner-product index, save to disk with metadata `{sample_id, rights_holder_id}`.

### Phase 2: Online Query (The Plugin)

Real-time attribution for generated outputs.

1. **Load Model:** Load weights from last checkpoint.
2. **Forward Query:** Pass generated output through model. Extract \(A\), \(E\) via hooks.
3. **Ghost Vector:** Form \(g\), apply Adam correction, project with same \(P\).
4. **ANN Search:** Query FAISS with **inner product** (not cosine similarity). Use `IndexFlatIP`.
5. **Clamp Negatives:** \(\text{score}_{\text{pos}} = \max(0, \langle g_{\text{query}}, g_{\text{train}} \rangle)\).
6. **Normalize:** \(\text{attribution}_i = \text{score}_{\text{pos},i} / \sum_j \text{score}_{\text{pos},j}\). Group by rights holder for revenue sharing.

### Phase 3 (Optional/Future): Distillation

**Defer until core pipeline is validated.** Train a lightweight encoder to predict ghost vectors from raw outputs, avoiding backward pass at runtime. High-risk; influence vectors encode training dynamics, not visual features.

---

## 3. Critical Corrections Summary

| Original Claim | Correction |
|----------------|------------|
| "Apply Adam variance to E only" | Apply to full ghost vector \(g = \text{vec}(e \cdot a^\top)\) element-wise |
| "Cosine Similarity for scoring" | Use **inner product**; cosine discards magnitude |
| "SJLT makes it model-agnostic" | Projection is model-specific; only output dim \(K\) is standardized |
| "Self-supervised reconstruction loss" | Must match model's training loss: MSE for diffusion, CE for LLMs |
| "Dense Gaussian for SJLT" | Use sparse Achlioptas matrix |
| "Distillation is Phase 2" | Defer to Phase 3 (optional) |

---

## 4. Implementation Stack

- **Frameworks:** PyTorch (hooks), FAISS (inner-product index), scipy (sparse SJLT)
- **Scoring:** FAISS `IndexFlatIP` (inner product). Do not use cosine.
- **Normalization:** `model.eval()` and `torch.no_grad()` during extraction.
- **Loss:** Task-specific. Classification: cross-entropy. Diffusion: MSE (denoising). LLM: next-token CE.

---

## 5. Directory Structure

```
Independent-TracIn/
├── main.py                 # --mode index | query | full  (dispatches to model scripts)
├── config.yaml             # Template config
├── src/                    # Core library (hooks, math, indexer, inference, FAISS)
│   ├── hooks_manager.py
│   ├── math_utils.py
│   ├── indexer.py
│   ├── inference.py
│   └── faiss_store.py
├── testModels/             # One folder per model test
│   └── mnist/
│       ├── model.py
│       ├── data.py
│       ├── train.py
│       ├── run_index.py
│       ├── run_query.py
│       ├── create_query_input.py
│       └── config.yaml
├── tests/                  # pytest suite
├── run_container.sh
├── slurm_run_container.sh
└── submit_slurm.sh
```

---

## 6. Usage

```bash
# 1. Train model (outside pipeline)
python testModels/mnist/train.py --output testModels/mnist/checkpoints

# 2. Index copyrighted data
python main.py --config testModels/mnist/config.yaml --mode index

# 3. Create query input
python testModels/mnist/create_query_input.py --output outputs/query_input.pt

# 4. Query attribution
python main.py --config testModels/mnist/config.yaml --mode query --input outputs/query_input.pt

# Full pipeline (index + query)
python main.py --config testModels/mnist/config.yaml --mode full --input outputs/query_input.pt

# Slurm
bash submit_slurm.sh testModels/mnist/config.yaml tracin-index index
bash submit_slurm.sh testModels/mnist/config.yaml tracin-query query
```

---

## 7. Math Reference

**TracIn (SGD):**
\[
\text{TracIn}(z_i) = \sum_t \eta_t \langle \nabla \ell(z_{\text{test}}; \theta_t), \nabla \ell(z_i; \theta_t) \rangle
\]

**Ghost Dot Product:**
\[
\langle g_1, g_2 \rangle = \langle A_1, A_2 \rangle \cdot \langle E_1, E_2 \rangle
\]

**Adam Correction:**
\[
g_{\text{corrected}} = g / (\sqrt{v_t} + \epsilon)
\]

**Attribution Normalization:**
\[
\text{attribution}_i = \frac{\max(0, \langle g_{\text{query}}, g_i \rangle)}{\sum_j \max(0, \langle g_{\text{query}}, g_j \rangle)}
\]
