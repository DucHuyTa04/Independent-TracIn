# TracIn with AdamW: From Total Change to Preconditioned Gradients

Pedagogical note: derive **TracIn** (Pruthi et al., NeurIPS 2020) from a telescoping loss difference, substitute an **AdamW** update, and relate the result to the **second-moment correction** used in this repository ([`src/math_utils.py`](../src/math_utils.py)).

**Related docs:** [theory.md](theory.md) (ghosts, SJLT, FAISS), [qa_ghost_vs_original.md](qa_ghost_vs_original.md) (Adam correction and factorization).

---

## TL;DR

TracIn scores are built from **per-step** approximations to how much each optimizer step changed loss on a **test** point $z'$. A first-order Taylor expansion turns “change in loss” into a dot product between the **test loss gradient** and the **actual parameter update** that ran because of training data. Under **SGD**, that update is proportional to the **training** gradient, so you get the familiar sum of dot products of raw gradients, scaled by the learning rate. Under **AdamW**, the update is **preconditioned** (each coordinate scaled by roughly $1/(\sqrt{v_t}+\epsilon)$ from a diagonal second-moment estimate). After justified simplifications, the Taylor-faithful score uses a **single-sided** diagonal preconditioner $D_t$ on the **training** gradient only. This codebase optionally rescales **both** sides of the inner product for retrieval with FAISS, which yields a **symmetric** metric $D_t^2$ instead of $D_t$—a deliberate engineering choice documented below.

---

## Part 0 — Setup and notation

**Goal.** Quantify how much a **training** example $z$ influenced the model’s behavior on a **query/test** example $z'$. A standard TracIn-style proxy is: sum over training steps the (approximate) one-step **change in loss** $\mathcal{L}(\cdot, z')$ induced by the step that ran when $z$ was used.

| Symbol | Meaning |
|--------|--------|
| $w_t \in \mathbb{R}^P$ | Parameters at step $t$ (also written $\theta_t$ elsewhere). |
| $\mathcal{L}(w, z')$ | Per-example loss on $z'$ (same functional form as training). |
| $z_t$ | Training sample(s) used in the batch at step $t$ (Part 2 uses one sample for clarity). |
| $\eta_t$ | Learning rate at step $t$ (may depend on schedules). |
| $B$ | Mini-batch size (Part 5). |
| $\nabla \mathcal{L}(w, z)$ | Gradient of $\mathcal{L}$ w.r.t. $w$ for fixed $z$ (column vector of length $P$). |
| $u \cdot v$ / $u^\top v$ | Euclidean inner product of flattened vectors. |
| $\odot$ | Elementwise (Hadamard) product. |
| $g^{\odot 2}$ | $g \odot g$ (elementwise square). |

**Large language models.** In practice, frontier LLMs are trained with **AdamW** (Loshchilov & Hutter, 2017), not plain SGD. The derivation below starts optimizer-agnostic and then specializes to AdamW.

---

## Part 1 — Total change, telescoping, and Taylor (Image 1 flow)

**1. Total change in test loss** over training from $w_0$ to $w_T$:

$$
\text{Total change} = \mathcal{L}(w_0, z') - \mathcal{L}(w_T, z').
$$

This is the overall improvement (positive when loss goes down) for $z'$ along the trajectory.

**2. Telescoping sum.** For each step $t = 0, \ldots, T-1$, compare loss before and after the update that takes $w_t \mapsto w_{t+1}$:

$$
\mathcal{L}(w_0, z') - \mathcal{L}(w_T, z')
= \sum_{t=0}^{T-1} \big[\mathcal{L}(w_t, z') - \mathcal{L}(w_{t+1}, z')\big].
$$

The sum **telescopes**: intermediate $\mathcal{L}(w_{t+1}, z')$ terms cancel.

**3. First-order Taylor expansion (one step).** Expand $\mathcal{L}(\cdot, z')$ around $w_t$ and evaluate at $w_{t+1}$:

$$
\mathcal{L}(w_{t+1}, z')
= \mathcal{L}(w_t, z') + \nabla\mathcal{L}(w_t, z')^\top (w_{t+1} - w_t) + O\!\left(\|w_{t+1} - w_t\|^2\right).
$$

Rearranging,

$$
\mathcal{L}(w_{t+1}, z') - \mathcal{L}(w_t, z')
= \nabla\mathcal{L}(w_t, z')^\top (w_{t+1} - w_t) + O\!\left(\|w_{t+1} - w_t\|^2\right).
$$

**Assumption (E1) — small local steps.** Drop $O(\|\Delta w\|^2)$ and write

$$
\mathcal{L}(w_{t+1}, z') - \mathcal{L}(w_t, z') \approx \nabla\mathcal{L}(w_t, z')^\top (w_{t+1} - w_t). \tag{E1}
$$

This is the **same starting equation** for SGD, Adam, AdamW, etc.: the optimizer enters only through $(w_{t+1} - w_t)$.

---

## Part 2 — Sign flip and SGD warm-up (Image 2 flow)

**4. Flip the sign** to match the telescoping term $\mathcal{L}(w_t, z') - \mathcal{L}(w_{t+1}, z')$:

$$
\mathcal{L}(w_t, z') - \mathcal{L}(w_{t+1}, z')
\approx \nabla\mathcal{L}(w_t, z')^\top (w_t - w_{t+1}). \tag{E1}
$$

**5. Stochastic gradient descent (SGD) template.** One SGD step subtracts the gradient of the objective at $w_t$ times the learning rate. With a single training point $z_t$ at step $t$:

$$
w_{t+1} = w_t - \eta_t \, \nabla\mathcal{L}(w_t, z_t)
\quad\Rightarrow\quad
w_t - w_{t+1} = \eta_t \, \nabla\mathcal{L}(w_t, z_t).
$$

Substitute into (E1):

$$
\mathcal{L}(w_t, z') - \mathcal{L}(w_{t+1}, z')
\approx \eta_t \, \nabla\mathcal{L}(w_t, z')^\top \nabla\mathcal{L}(w_t, z_t).
$$

Summing over steps where $z_t = z$ gives the **classical TracIn** (checkpoint) score:

$$
\mathrm{TracIn}_{\mathrm{SGD}}(z, z')
\approx \sum_{t :\, z_t = z} \eta_t \,
\nabla\mathcal{L}(w_t, z')^\top \nabla\mathcal{L}(w_t, z).
$$

This is the “grow the formula from the base” picture: Taylor gives the dot with **update**; SGD replaces **update** with $\eta_t \nabla\mathcal{L}(w_t, z_t)$.

---

## Part 3 — AdamW update rule (textbook form)

At step $t$, let the **batch** gradient be

$$
g_t = \nabla \mathcal{L}_{\mathrm{batch}}(w_t)
= \frac{1}{B} \sum_{i \in \mathcal{B}_t} \nabla\mathcal{L}(w_t, z_i).
$$

**Moments** (matching standard Adam/AdamW state):

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^{\odot 2}.
$$

**Bias-corrected** moments (used in analysis and many textbook presentations):

$$
\hat m_t = \frac{m_t}{1-\beta_1^t}, \qquad
\hat v_t = \frac{v_t}{1-\beta_2^t}.
$$

**AdamW step** (decoupled weight decay, Loshchilov & Hutter 2017):

$$
w_{t+1}
= w_t - \eta_t \left( \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda w_t \right).
$$

Here $\epsilon$ is small and positive (numerical stability), and $\lambda$ is the **decoupled** weight-decay coefficient.

**PyTorch detail (implementation vs textbook).** In `torch.optim.AdamW`, `exp_avg_sq` stores **uncorrected** $v_t$, and the denominator is often written equivalently as $\sqrt{v_t}/\sqrt{1-\beta_2^t} + \epsilon$ rather than $\sqrt{\hat v_t} + \epsilon$. These differ only in **where** $\epsilon$ is inserted relative to the bias-correction scaling; once $\sqrt{v_t} \gg \epsilon$, the distinction is minor. This note uses the textbook $\sqrt{\hat v_t} + \epsilon$ for clarity.

**Why AdamW (and not “Adam + L2”) for LLMs?** In the older pattern, an L2 penalty is added **inside** the gradient, so $m_t$ and $v_t$ are built from **regularized** $g_t$, which **contaminates** the second moment used for preconditioning. AdamW applies $\lambda w_t$ **outside** the adaptive scaling; the moments $m_t, v_t$ reflect the **data gradient** more cleanly. That matters conceptually when interpreting $v_t$ as a per-coordinate scaling for “how large updates tend to be in coordinate $j$.”

---

## Part 4 — Substitute AdamW into the Taylor sum

From the AdamW update,

$$
w_t - w_{t+1}
= \eta_t \left( \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda w_t \right).
$$

Insert into the flipped Taylor approximation from Part 2 (error still $O(\|\Delta w\|^2)$ only—assumption **(E1)**); the AdamW substitution for $w_t - w_{t+1}$ is **exact** given the optimizer state.

$$
\mathcal{L}(w_t, z') - \mathcal{L}(w_{t+1}, z')
\approx
\eta_t \, \nabla\mathcal{L}(w_t, z')^\top
\left( \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda w_t \right).
$$

Define the **diagonal preconditioner** (one scalar per parameter)

$$
D_t := \mathrm{diag}\!\left( \frac{1}{\sqrt{\hat v_t} + \epsilon} \right),
$$

so $\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} = D_t \hat m_t$ in vector form. Then

$$
\mathcal{L}(w_t, z') - \mathcal{L}(w_{t+1}, z')
\approx
\eta_t \, \nabla\mathcal{L}(w_t, z')^\top \big( D_t \hat m_t + \lambda w_t \big).
$$

**Reading the expression:** the test gradient $\nabla\mathcal{L}(w_t, z')$ contracts against **the actual AdamW step direction** (preconditioned momentum plus weight decay). That is the Adam analogue of $\nabla\mathcal{L}(w_t, z')^\top \nabla\mathcal{L}(w_t, z_t)$ in SGD.

---

## Part 5 — Simplifications to recover the code’s $g / (\sqrt{v_t} + \epsilon)$ scaling

Each bullet names the assumption that upgrades $\approx$ to the working formula used in implementation.

### (S1) Drop the decoupled weight-decay term $\lambda w_t$

The vector $\lambda w_t$ depends on time $t$ but **not** on which training sample $z_i$ was drawn in the batch at $t$. TracIn-style **per-sample attribution** asks for the **data-dependent** part of the update: what changed because $z$ was included. The $\lambda w_t$ term is part of the **baseline** trajectory the optimizer would follow even if a different batch had been drawn; it is not attributed to individual samples in the same way as $D_t \hat m_t$.

**Conclusion.** For per-sample scoring, omit $\lambda w_t$ from the contract with $\nabla\mathcal{L}(w_t, z')$. This is why **AdamW and Adam** coincide for the *preconditioning part* of TracIn once the decoupled decay is treated as baseline.

*(Corner case.)* In principle, $\nabla\mathcal{L}(w_t, z')^\top w_t$ could correlate with sample-specific signals early in training; we do not treat that fine structure here.

### (S2) Replace $\hat m_t$ by the per-sample training gradient

This is **two** approximations; keep them separate.

**(S2a) Instantaneous-gradient / momentum simplification.** For moderately large $t$, bias correction $1/(1-\beta_1^t) \to 1$. If gradients vary slowly, $m_t$ may be dominated by the latest $g_t$, so $\hat m_t \approx g_t$. This mirrors common **TracIn-with-adaptive-optimizer** practice in the literature (use raw or lightly processed gradients in the influence dot product after accounting for preconditioning in $D_t$).

**(S2b) Per-sample piece of the batch gradient.** By definition,

$$
g_t = \frac{1}{B} \sum_{i \in \mathcal{B}_t} \nabla\mathcal{L}(w_t, z_i).
$$

The **contribution** of a particular $z$ that appears in $\mathcal{B}_t$ is $\frac{1}{B}\nabla\mathcal{L}(w_t, z)$. Absorb the factor $1/B$ into an **effective** learning rate for that attribution term:

$$
\tilde \eta_t := \frac{\eta_t}{B}.
$$

Other batch members at step $t$ contribute terms not credited to $z$; they act as **background** in the same spirit as classical TracIn with mini-batches.

**(S2) Net effect.** When attributing influence to $z$ at a step where $z \in \mathcal{B}_t$, use $\tilde \eta_t \, \nabla\mathcal{L}(w_t, z')^\top D_t \nabla\mathcal{L}(w_t, z)$ with the understanding $D_t \hat m_t \rightsquigarrow D_t \nabla\mathcal{L}(w_t, z)$ **together with** $\eta_t \rightsquigarrow \tilde\eta_t$.

### (S3) Replace $\hat v_t$ by $v_t$

For large $t$, $1-\beta_2^t \to 1$, so $\hat v_t \approx v_t$. Typical LLM $\beta_2$ (e.g. $0.999$) makes the correction very close to 1 after a few thousand steps—negligible on pretraining scales. Dropping it rescales $D_t$ by a **per-$t$ scalar** that does not change **within-checkpoint** rankings of training points.

---

### Combined working approximation

Under (E1), (S1)–(S3),

$$
\mathcal{L}(w_t, z') - \mathcal{L}(w_{t+1}, z')
\approx
\tilde \eta_t \,
\nabla\mathcal{L}(w_t, z')^\top D_t \, \nabla\mathcal{L}(w_t, z),
$$

(Here $z$ is the training point whose share of the batch gradient is attributed at step $t$, per **(S2b)**.)

where

$$
D_t = \mathrm{diag}\!\left( \frac{1}{\sqrt{v_t} + \epsilon} \right).
$$

**Approximation budget (explicit):** Taylor remainder (E1); momentum model (S2a); batch bookkeeping (S2b); second-moment bias (S3); omission of weight decay (S1).

---

## Part 6 — Final AdamW-preconditioned TracIn (Taylor-faithful / single-sided)

Sum over steps (or checkpoints) where training point $z$ was used:

$$
\boxed{
\mathrm{TracIn}_{\mathrm{AdamW}}(z, z')
\approx
\sum_{t \,:\, z_t = z}
\tilde \eta_t \,
\nabla\mathcal{L}(w_t, z')^\top D_t \, \nabla\mathcal{L}(w_t, z)
}
$$

with $D_t = \mathrm{diag}\big(1/(\sqrt{v_t} + \epsilon)\big)$ and $\tilde \eta_t = \eta_t/B$ when using the per-sample contribution from a batch of size $B$.

**Annotation (like TracIn-CP style diagrams).**

- $z$ — training point whose influence we score.
- $z'$ — test / query point.
- $\tilde \eta_t$ — effective learning rate (includes $1/B$ when using the sample’s share of the batch gradient).
- $\nabla\mathcal{L}(w_t, z')$, $\nabla\mathcal{L}(w_t, z)$ — loss gradients at the same checkpoint $w_t$.
- $D_t$ — **diagonal preconditioner** from Adam’s second moment $v_t$: large running $|g|$ in coordinate $j$ ⇒ smaller effective step in $j$ ⇒ down-weight that coordinate in the influence inner product unless both test and training gradients align strongly there.

**Sanity check (sign).** If $\nabla\mathcal{L}(w_t, z')$ and $D_t \nabla\mathcal{L}(w_t, z)$ are aligned (positive dot product), the update component attributed to $z$ tends to **move** $w$ in a direction that **reduces** $\mathcal{L}(\cdot, z')$ to first order—positive TracIn-style contribution.

---

## Part 7 — Single-sided $D_t$ vs symmetric $D_t^2$ in this codebase

The Taylor derivation contracts $\nabla\mathcal{L}(w_t, z')$ with **the actual update** $D_t \hat m_t \approx D_t \nabla\mathcal{L}(w_t, z)$. Only the **training-side** gradient is premultiplied by $D_t$ in the strict linearization.

**Symmetric variant used in code.** The implementation applies the elementwise scaling $1/(\sqrt{v_t}+\epsilon)$ to **both** ghost vectors (query and training) and then computes a standard inner product. Algebraically, for vectors $a, b$ and diagonal $D_t$,

$$
\langle D_t a, D_t b \rangle = a^\top D_t^2 b,
\quad
D_t^2 = \mathrm{diag}\!\left( \frac{1}{(\sqrt{v_t}+\epsilon)^2} \right).
$$

So coordinates with large historical second moment are **down-weighted twice** compared to the Taylor-faithful $a^\top D_t b$. That is an **engineering / retrieval** choice (one symmetric inner product, compatible with FAISS `IndexFlatIP`), not a mistake—see [qa_ghost_vs_original.md](qa_ghost_vs_original.md), *Question 2 — Adam correction*, Step 3.

**Taylor-faithful symmetric alternative (not implemented here).** If one applied $D_t^{1/2}$ (elementwise $1/\sqrt{\sqrt{v_t}+\epsilon}$) to **both** sides,

$$
\langle D_t^{1/2} a, D_t^{1/2} b \rangle = a^\top D_t b,
$$

recovering the single-sided metric while keeping symmetry for indexing. **This repository uses $D_t$ on both sides, hence $D_t^2$ in the bilinear form.**

---

## Part 8 — Map to `Independent-TracIn` code

| Math object | Role | Code |
|-------------|------|------|
| Ghost / flattened gradient proxy for a layer | $g$ aligned with weight grads for `nn.Linear`-style hooks | `form_ghost_vectors`, `form_multi_layer_ghost_vectors` in [`src/math_utils.py`](../src/math_utils.py) |
| Second moment $v_t$ in ghost layout | Same flatten order as $g$ (note PyTorch weight layout transpose) | `load_adam_second_moment`, `load_adam_second_moment_with_bias`, `concatenate_adam_second_moments` |
| Preconditioning $g \mapsto D_t g$ with $D_t = \mathrm{diag}(1/(\sqrt{v_t}+\epsilon))$ | Elementwise on flattened ghost | `apply_adam_correction` |

PyTorch stores linear weight as $(C_{\mathrm{out}}, H_{\mathrm{in}})$ but ghost flattening follows $(H_{\mathrm{in}}, C_{\mathrm{out}})$; `load_adam_second_moment` **transposes** `exp_avg_sq` before flatten so each ghost entry divides by the **matching** $v_t$ entry. **AdamW checkpoints** store `exp_avg_sq` the same way as Adam; the same loaders apply.

**Out of scope here:** SJLT projection, FAISS indexing, multi-layer concatenation—see [theory.md](theory.md).

---

## Part 9 — Closing summary

1. **Base:** Total change in $\mathcal{L}(w,z')$ telescopes into per-step differences; each difference is, to first order, $\nabla\mathcal{L}(w_t,z')^\top(w_t - w_{t+1})$.
2. **SGD:** $w_t - w_{t+1} = \eta_t \nabla\mathcal{L}(w_t,z_t)$ gives the classical TracIn dot product of **raw** gradients.
3. **AdamW:** $w_t - w_{t+1}$ contains $D_t \hat m_t$ plus decoupled $\lambda w_t$; drop $\lambda w_t$ for per-sample attribution; approximate $\hat m_t \to \nabla\mathcal{L}(w_t,z)$ with batch bookkeeping in $\tilde\eta_t$; use $v_t$ instead of $\hat v_t$ for $D_t$ at large $t$.
4. **Strict linearization** uses **one** $D_t$ on the training gradient; **this repo** may use $D_t$ on **both** sides of the ghost inner product, i.e. a $D_t^2$ metric—documented above so the math and implementation align honestly.

When you implement TracIn on LLMs trained with AdamW, **preconditioning** is exactly this: replace “similar raw gradients” with “similar gradients **after** Adam’s per-coordinate scaling”—and be explicit about single-sided ($D_t$) vs symmetric ($D_t^2$) variants.

---

## Rigor checklist (for readers and reviewers)

1. **(E1)** tags only the **Taylor** step (dropping $O(\|\Delta w\|^2)$). Later $\approx$ signs add **(S1)–(S3)** on top of (E1).
2. **Batch gradient:** $g_t$ is an average over $\mathcal{B}_t$; per-sample attribution uses the $1/B$ share and $\tilde\eta_t = \eta_t/B$ **(S2b)**.
3. **Bias correction:** $\hat m_t$ and $\hat v_t$ are replaced by $g_t$ and $v_t$ only under **(S2a)** and **(S3)**.
4. **Weight decay** is dropped as non-attributable baseline under **(S1)**, not because it “cancels in ranking.”
5. **Taylor-faithful** influence uses $a^\top D_t b$ with one $D_t$ on the training/update side; **this repo** often uses $a^\top D_t^2 b$ when scaling both ghosts—documented in Part 7.
6. **PyTorch vs textbook** $\epsilon$ placement relative to $\sqrt{1-\beta_2^t}$ is noted in Part 3 and does not change the high-level preconditioning story when $\sqrt{v_t}\gg\epsilon$.
