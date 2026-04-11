# TracIn Ghost — Presentation Plan

> **Audience**: PhD fellows (ML-literate, not familiar with TracIn)
> **Your parts**: 2 (Ghost Product), 3 (What we compute), 4 (Implementation + heuristics), 5 (Live Demo), 6 (Scalability)
> **Mentor covers**: Part 1 (Motivation + Original TracIn)
> **Total time**: ~45 min + Q&A


## Part 2: The Simple Overview

### Slide deck (recommended)

This is the “intuition first” block. Keep math light and reuse the running example.

- **Slide 1 — The question**: A model predicts `cat` for a striped-cat query. Which training examples mattered?
- **Slide 2 — Why Original TracIn is expensive**: One picture: query gradient vs every training gradient, repeated across checkpoints.
- **Slide 3 — One linear layer is structured**: activation + error → outer-product table (the “multiplication table” picture).
- **Slide 4 — The shortcut**: compare activations, compare errors, multiply (exact for that linear layer’s ghost).
- **Slide 5 — Running example ranking**: explain why A (striped cat) > C (orange cat) > B (dog) is the expected intuition.
- **Slide 6 — Offline vs online**: prepaid training fingerprints + one query fingerprint + search.
- **Slide 7 — Recap diagram**: reuse the end-of-Part-2 ASCII flow as the “bridge” slide into Part 3.

### The story we want the audience to feel

Start with a concrete scene:

> "A model just predicted `cat`. We now ask a very human question: which training examples taught it to say `cat`?"

Your mentor has already introduced Original TracIn. So your job in Part 2 is not to give all the math again. Your job is to tell the audience why the original idea is beautiful but too expensive, and how our algorithm turns that idea into something practical.

### Step 2.1 - The problem, in one picture

Original TracIn says:

$$
\operatorname{TracIn}(z_i, z') = \sum_t \eta_t \left\langle \nabla \ell(z_i; \theta_t), \nabla \ell(z'; \theta_t) \right\rangle
$$

That means:

- For each checkpoint, we need a gradient for the query.
- For each checkpoint, we also need a gradient for every training sample.
- Then we compare them with inner products.

That is fine in theory. It is painful in practice.

For a large model, one gradient can have millions of numbers. If there are 50,000 training samples, then "compare the query with everything" quickly becomes too slow and too memory-hungry.

### Running example for Parts 2 and 3

Use one tiny story all the way through:

- The model is a simple image classifier that predicts `cat` or `dog`.
- The **query** is a new striped cat image.
- Training sample **A** is another striped cat.
- Training sample **B** is a white dog.
- Training sample **C** is an orange cat.

The question we keep asking is:

> "Why did the model say `cat` for the query, and which of A, B, and C helped most?"

This example is small enough to keep in your head, but rich enough to follow through the whole pipeline:

- at the final linear layer,
- across multiple layers,
- through offline indexing,
- and later through convolution, Adam correction, and retrieval.

### Step 2.2 - The key insight: do not compare the whole table

Now tell the story by zooming into one linear layer.

For one sample, the gradient of a linear layer is not a random blob. It has structure:

- One vector says what entered the layer. Call it the **activation**.
- One vector says how the loss pushes back from the output. Call it the **error signal**.
- The weight gradient is their outer product.

So instead of thinking:

> "This is one huge gradient."

we think:

> "This is a table built from two much smaller pieces."

Mini-example:

```
activation a = [a1, a2, a3]
error      e = [e1, e2]

gradient table:

        e1          e2
a1   a1*e1      a1*e2
a2   a2*e1      a2*e2
a3   a3*e1      a3*e2
```

The beautiful shortcut is this:

- We do not need to compare every cell in the table.
- We compare the activation vectors.
- We compare the error vectors.
- Then we multiply those two results.

For a linear layer, this gives the exact same inner product as comparing the full weight gradients directly.

This is the heart of the algorithm.

In the running example:

- the query and training sample A should have similar activations and similar error signals,
- the query and training sample B should look much less aligned,
- training sample C may still help, but probably less than A because it is "cat-like" in a more generic way rather than specifically striped.

So even before we talk about indexing, the audience can already picture what an influence ranking should look like:

$$
\text{A most influential} \; > \; \text{C somewhat influential} \; > \; \text{B weakly influential}
$$

### Step 2.3 - Turn that insight into a reusable fingerprint

Now the story moves from one layer to a practical system.

For each training sample, we build a compact representation of its influence information:

1. Capture the activation and error signal from chosen layers.
2. Turn them into a ghost representation.
3. If needed, correct it using the optimizer state.
4. Compress it with a random projection.
5. Store it in a search index.

The important idea for the audience:

> We do the expensive work once, offline.

Then, when a new query arrives:

1. Build the same kind of ghost representation for the query.
2. Search the index.
3. Return the top training samples with the highest influence scores.

So the online question becomes:

> "Which stored training fingerprints look most aligned with this query fingerprint?"

In the running example:

- offline, we build one stored fingerprint for A, one for B, and one for C,
- online, we build one fingerprint for the striped-cat query,
- then we search for the stored fingerprints that align best with it.

So if the method works, the query should retrieve A first, then likely C, and B should rank much lower.

### Step 2.4 - Why this is more than just a linear-layer trick

If the model only had simple linear layers, we would already be done.

But real models contain:

- convolutions,
- normalization layers,
- embeddings,
- recurrent layers,
- skip connections and inplace operations,
- Adam optimizer state.

So the real contribution of the project is not only the shortcut. It is the full engineering path that makes the shortcut work on real models without changing the model architecture.

Part 3 is where you explain the math and the repository’s core computation story.

Part 4 is where you explain the engineering choices that make the story survive real architectures.

And this is where the running example evolves:

- first, we only looked at the final classifier layer,
- then in Part 3 we will imagine the same cat-vs-dog example moving backward into earlier layers,
- where features like stripes, ears, whiskers, and fur texture appear in convolutional feature maps rather than in one clean final-layer vector.

### Step 2.5 - The simple message to leave with the audience

By the end of Part 2, the audience should remember just four ideas:

- Original TracIn is about gradient inner products across checkpoints.
- A layer gradient has structure, so we do not have to treat it as an unstructured giant vector.
- We precompute training-side ghost representations and index them.
- At query time, attribution becomes a search problem.

### Speaker notes for Part 2

> "Original TracIn asks the right question: which training samples pushed the model toward this prediction? The problem is that the direct computation is huge.
>
> Our key insight is that, at a layer like a linear layer, the gradient is not just a bag of numbers. It is built from two smaller pieces: what came in, and what came back during learning. Once you notice that structure, the comparison becomes much cheaper.
>
> Then we take the next practical step: we build these ghost representations for the training set ahead of time, compress them, and store them in an index. So when a new query arrives, we do not recompute influence from scratch. We build one query ghost and search.
>
> That is the simple overview. Part 3 is where I show what we compute and how the repository turns it into an indexed pipeline. Part 4 is where I show the hooking, checkpointing, and heuristic setup that makes it work on real models."

### End-of-Part-2 overview diagram

```
QUESTION
"This query was produced by the model.
Which training samples mattered most?"
            │
            ▼
ORIGINAL TRACIN IDEA
Compare query gradient with training-sample gradients
at multiple checkpoints
            │
            ▼
OUR KEY SHORTCUT
For a hooked layer, use structured ghost information
instead of treating the gradient as one giant flat vector
            │
            ▼
OFFLINE STAGE
training sample
  → capture layer signals
  → build ghost representation
  → optional Adam correction
  → optional projection
  → add checkpoint contribution
  → store in FAISS
            │
            ▼
ONLINE STAGE
query
  → capture same signals
  → build query ghost
  → same correction / projection
  → FAISS inner-product search
  → top influential training samples
  → normalized attribution scores
            │
            ▼
BIG PICTURE
We turn attribution from "recompute everything"
into "build once, then search"
```


## Part 3: The Deep Dive

Part 2 gave the story. Part 3 now answers:

> "What exactly do we compute, and how is it implemented in this repository?"

This part should feel like opening the machine and walking through it slowly. The goal is not just to state the ideas, but to make the audience feel why each step is needed and how the pieces fit together.

### Slide deck (recommended)

This is the “math + repository core” block. Keep hooking details light here and hand those off to Part 4.

- **Slide 1 — TracIn formula in plain language**: what each symbol means, and what the inner product is asking.
- **Slide 2 — Exact linear ghost identity**: outer product structure + boxed identity.
- **Slide 3 — Running example at the final layer**: the three toy features (stripes / ears / snout) story.
- **Slide 4 — Multi-layer ghosts**: concatenate per-layer ghosts; total score is a sum of layer “votes.”
- **Slide 5 — Offline vs online in the repo**: `build_index()` accumulates checkpoint contributions; `attribute()` searches FAISS.
- **Slide 6 — Faithful multi-checkpoint variant**: one sentence + one diagram idea (optional): per-checkpoint index + query summed across checkpoints.
- **Slide 7 — Projection + retrieval**: SJLT/dense projection + FAISS inner product (not cosine).
- **Slide 8 — Honest caveats**: coverage, approximation, multi-checkpoint fidelity, negative-score clamping.

### 3.1 - Start from the TracIn formula

Remind the audience of the reference formula:

$$
\operatorname{TracIn}(z_i, z') = \sum_{t=1}^{T} \eta_t \, \left\langle \nabla_\theta \ell(z_i; \theta_t), \nabla_\theta \ell(z'; \theta_t) \right\rangle
$$

Now translate that formula into plain language:

- $z_i$ is one training sample.
- $z'$ is the query we want to explain.
- $\theta_t$ is the model at checkpoint $t$.
- $\eta_t$ is the learning rate at that checkpoint.
- The inner product asks whether the training sample and the query push the model in a similar direction.

That is a beautiful idea, because it ties attribution directly to the optimization path. But the direct implementation is expensive.

The hard part is that the gradient lives in full parameter space. For a large model, that means:

- very high-dimensional vectors,
- one comparison per training sample,
- repeated across checkpoints.

So the implementation idea is:

> Replace the full gradient wherever possible with a structured object whose inner products match the true layer-gradient inner products, but which is much cheaper to store and compare.

Keep the running example alive here:

- query = striped cat,
- A = striped cat,
- B = white dog,
- C = orange cat.

At the formula level, TracIn asks:

> "Across checkpoints, whose gradient looked most like the query's gradient?"

So if the query and A keep pushing the model in similar directions during training, A should accumulate a larger score than B.

### 3.2 - The exact linear-layer math

For one `nn.Linear` layer:

$$
y = a W^\top + b
$$

For one sample, the weight gradient is an outer product:

$$
\nabla_W \ell = e^\top a
$$

where:

- $a$ is the input activation to the layer,
- $e$ is the error signal at the layer output.

This is the key teaching point: the gradient is not a random cloud of numbers. It is built from two smaller, meaningful pieces:

- what the layer saw in the forward pass,
- and how the loss blamed the layer in the backward pass.

That is why the "multiplication table" picture is so useful.

We flatten that outer product into the **ghost vector**:

$$
g = \operatorname{vec}(e \otimes a)
$$

Now comes the exact identity:

$$
\boxed{\langle g_1, g_2 \rangle = \langle a_1, a_2 \rangle \cdot \langle e_1, e_2 \rangle}
$$

This is the reason the shortcut works.

In words:

- the full gradient table looks large,
- but its comparison can be broken into one comparison on the input side and one comparison on the error side,
- and then those two numbers are multiplied together.

So for a linear layer, we are not approximating the dot product. We are reorganizing it.

In the running example, imagine the final classifier layer has just three hidden features:

- feature 1 = "striped texture"
- feature 2 = "pointed ears"
- feature 3 = "dog-like snout"

Then:

- the striped-cat query might activate features 1 and 2 strongly,
- sample A might do the same,
- sample B might activate feature 3 instead,
- sample C might activate feature 2 but not feature 1 as strongly.

That is exactly why A should align best with the query at this layer.

#### Tiny worked example

Suppose:

$$
a_1 = [1, 2], \quad e_1 = [3, 4]
$$

$$
a_2 = [5, 6], \quad e_2 = [7, 8]
$$

Then:

- $\langle a_1, a_2 \rangle = 1\cdot5 + 2\cdot6 = 17$
- $\langle e_1, e_2 \rangle = 3\cdot7 + 4\cdot8 = 53$
- Product = $17 \cdot 53 = 901$

If you materialize the two outer products and compare them cell by cell, you also get 901.

That is the exact linear-layer ghost product.

This is the sentence I would say out loud:

> "We are not throwing information away here. We are exploiting the structure of the layer so we can compute the same answer in a cheaper way."

#### Bias handling

The code appends a column of ones to the activation so the bias gradient is included automatically:

$$
a_{\text{aug}} = [a; 1]
$$

That is implemented in `src/hooks_manager.py` by `_maybe_append_bias_ones()`.

This is a neat design choice because it avoids splitting the explanation into "weight case" and "bias case." The same ghost machinery handles both.

In the running example, you can explain bias as:

> "Even if there is a baseline tendency for the classifier to prefer `cat` or `dog`, we still want that contribution included. The appended 1 makes that happen automatically."

### 3.3 - From one layer to a full ghost vector

If we hook multiple layers, we concatenate their per-layer ghosts:

$$
g = [g^{(1)}; g^{(2)}; \dots; g^{(L)}]
$$

and then:

$$
\langle g_i, g_j \rangle
= \sum_{\ell=1}^{L} \langle g_i^{(\ell)}, g_j^{(\ell)} \rangle
$$

So the full score is a sum of layer contributions.

This is a nice way to explain it to the audience:

> "Each hooked layer gets a vote, and the total score is the sum of those votes."

This is important to say clearly:

- The method is exact for the hooked subspace.
- It is not automatically the full-network gradient unless the hooked layers cover everything, or another fallback handles the uncovered parameters.

In the running example:

- the final layer might say A is close to the query because both look cat-like,
- an earlier layer might add that A is specifically close because both contain striped texture,
- another layer might say C is still somewhat similar because both contain cat-ear features.

So the final score is not one decision made in one place. It is a sum of evidence from several layers.

### 3.4 - What the repository actually stores offline

There are two related implementations in the repo, and it is worth being honest about both, because this helps the audience understand what is "production-style practical" versus what is "closest to textbook TracIn."

#### Production-style path: `build_index()` + `attribute()`

This is the main offline/online pipeline used in `src/indexer.py` and `src/inference.py`.

Offline:

$$
v_i = \sum_t \eta_t \, P \, \tilde{g}_{i,t}
$$

where:

- $g_{i,t}$ is the ghost vector for training sample $i$ at checkpoint $t$,
- $\tilde{g}_{i,t}$ means "after optional Adam correction",
- $P$ is the optional projection matrix.

Then `build_index()` stores the accumulated vectors in a FAISS inner-product index.

Online:

- `attribute()` builds one query ghost,
- applies the same correction and projection,
- searches the FAISS index,
- clamps negative scores to zero,
- normalizes the positive top-k scores into attribution percentages.

This is the simple production story:

> Precompute the training side once, then search it.

Another way to say it:

> "Offline, we summarize what each training sample contributed across training. Online, we ask which stored summary best matches the query."

In the running example:

- A gets its own stored vector,
- B gets its own stored vector,
- C gets its own stored vector.

When the striped-cat query arrives, we do not recompare it against raw training gradients from scratch. We compare it against those stored summaries.

#### Faithful multi-checkpoint query path

The repo also provides:

- `build_multi_checkpoint_index()`
- `attribute_multi_checkpoint()`

In that variant, each checkpoint gets its own index, and the query is recomputed at each checkpoint too. That matches the multi-checkpoint TracIn formula more faithfully, because both sides of the comparison move along the checkpoint path rather than collapsing everything into one training-side summary.

This is worth saying in one sentence during the talk:

> "The repo contains both a production-style indexed approximation and a more faithful multi-checkpoint query variant."

### 3.5 - How activations and error signals are captured

At this point, the audience knows what quantities we want mathematically. The next question is: how do we actually get them out of PyTorch?

There are two main modes in the repository.

#### Simple single-layer path

Used by `HookManager`.

- A forward hook captures the activation.
- The output plus target are passed to an `error_fn`.
- For classification, that error is based on the logits and labels.
- For regression, it is based on predictions and targets.

This path is simple and fast.

The intuition is:

> "Intercept the layer input during the forward pass, then compute the output-side error in a direct analytic way."

That makes it a nice path for simple, clean attribution setups.

In the running example, this means:

- the hook reads what the final layer saw for the striped-cat query,
- the error function says how the model's output differs from the target,
- and together those two pieces define the query's ghost at that layer.

#### Multi-layer backward path

Used by `MultiLayerBackwardGhostManager`.

- Forward hooks capture activations.
- Backward hooks capture layer-level gradients coming out of the loss.
- The same training loss used for optimization is replayed.

This path is more general and is what the benchmark-side deep evaluation uses for model-agnostic support.

The intuition is:

> "Instead of manually writing down the layer error, let backpropagation deliver the blame signal at every hooked layer."

That is why this path becomes important once we move beyond simple single-layer examples.

In the running example, the multi-layer path is what lets us say:

> "Do A and the query match only at the final decision layer, or do they already look similar in earlier feature-extraction layers too?"

### 3.6 - Real model support: what broke, and how we fixed it

This is the most important "implementation" section, because this is where the project stops being a toy and becomes something that can survive real architectures.

#### A. Conv2d

Problem:

- A convolution is not one single outer product.
- Each spatial location contributes its own patch-level outer product.

So the difficulty is not that convolutions destroy the idea. The difficulty is that they change the shape of the computation.

With a linear layer, we naturally get one activation vector and one error vector for the sample.

With a convolution, the kernel slides over many positions, so one sample produces many local activation/error pairs, one per spatial location.

Fix:

- Use `F.unfold()` to turn the input into patch blocks.
- Now each spatial position behaves like a local linear operation.
- If raw spatial blocks are kept, the code can compute the exact sum-of-outer-products across positions.
- If memory becomes too large, the code can fall back to a mean-pooled approximation.

Simple picture:

```
image
  → unfold into patches
  → each patch acts like a local activation vector
  → combine with per-position error signals
  → sum over spatial positions
```

Important wording:

> "For convolutions, the core idea survives, but we must respect the spatial structure."

Another good line is:

> "A linear layer gives one table. A convolution gives one table per spatial position. `F.unfold()` reorganizes those positions so we can still use the same ghost logic."

In the running example, this is where "striped cat" becomes especially intuitive:

- one convolution patch may focus on the ear,
- another on whiskers,
- another on striped fur.

So instead of one global similarity, the convolution layer collects many small local similarities and combines them.

#### B. ConvTranspose2d and BatchNorm2d

Problem:

- Some module backward hooks become fragile when later operations modify tensors inplace.
- This shows up in architectures with residual adds or inplace activations.

This is a good place to emphasize that not every challenge is mathematical. Some are about observing the right tensors safely during autograd.

Fix:

- For these layers, the code uses `output.register_hook()` on the tensor itself.
- That avoids the autograd wrapper behavior that can trigger inplace errors.

This is a great place to sound like a careful engineer:

> "A big part of making theory work in practice is not new math, but capturing the right signals safely inside PyTorch."

#### C. LayerNorm and BatchNorm

These layers are not handled as plain linear transforms on raw input.

That is because the trainable parameters in normalization layers act on normalized quantities, not on the raw tensor directly.

Instead:

- LayerNorm uses the normalized input representation associated with its affine parameters.
- BatchNorm2d uses normalized activations based on running statistics.

So when you present them, say:

> "For normalization layers, we match the gradient structure of the affine parameters, not the raw pre-normalized tensor."

That one sentence helps the audience connect the implementation to what they already know about LayerNorm and BatchNorm gradients.

In the running example, you can phrase this as:

> "Before comparing the query and sample A, we first put the features on the normalized scale that the layer actually uses."

#### D. Embedding layers

Problem:

- Embedding gradients are sparse in vocabulary space.
- Only rows for tokens that appear in the sample receive updates.

So an embedding layer is better imagined as a giant lookup table than as one ordinary dense matrix multiply.

Fix:

- The repository has embedding-specific logic that aggregates the active token rows instead of pretending the layer is dense.

This is a good one-line explanation:

> "Embedding ghosts are sparse row updates, not ordinary dense matrix products."

If anyone asks how the running example would look in NLP instead of images, you can say:

> "Replace the striped-cat image with a prompt, and replace active image patches with active tokens. The same logic becomes sparse row updates in an embedding table."

#### E. RNN / GRU / LSTM support

The repository can hook recurrent modules, but there is an important caveat:

- the ghost approximation mainly covers the input-to-hidden style parameters,
- so coverage can be partial.

That means you should not oversell RNN support as "full exact coverage of all recurrent parameters." A fair way to say it is:

> "The repo supports recurrent models, but with partial coverage caveats that do not arise in the same way for plain linear layers."

### 3.7 - Adam correction

This is one of the most important implementation details.

Many people think of Adam as just optimizer housekeeping. Here it matters directly, because TracIn is about update directions, and Adam changes those directions coordinate by coordinate.

Adam changes the effective update direction using the second-moment estimate:

$$
\tilde{g} = g \odot \frac{1}{\sqrt{v_t} + \epsilon}
$$

Two things matter here.

#### First: Adam breaks the clean factorization

Without Adam, we can often stay in the cheap factored world.

With Adam:

- each ghost coordinate gets its own scale,
- so the simple "activation dot times error dot" factorization no longer holds in general.

That is why some code paths materialize the ghost explicitly before correction.

This is the clean explanation:

> "Without Adam, the row-and-column shortcut survives. With Adam, every cell of the table gets its own weight, so we sometimes have to rebuild the table explicitly."

In the running example, Adam correction means:

- maybe "striped texture" coordinates were historically very noisy and get down-weighted,
- while "cat ear" coordinates were stable and get weighted differently.

So the score between the query and sample A is not just about raw similarity. It is about similarity after respecting how the optimizer actually treated each parameter during training.

#### Second: layout matters

PyTorch stores a linear weight as:

$$
(C_{\text{out}}, H_{\text{in}})
$$

but the ghost flattening is aligned like:

$$
(H_{\text{in}}, C_{\text{out}})
$$

So the optimizer state must be transposed before flattening into ghost layout.

Also:

- weight state and bias state are stored separately,
- the code concatenates them so the correction matches the augmented ghost vector.

This is a good place for a strong teaching sentence:

> "If you forget that transpose, every Adam coefficient lands on the wrong ghost entry."

That is one of those bugs where the code may still run, but the mathematics is wrong.

### 3.8 - Projection and retrieval

Once ghosts are built, they may still be high-dimensional, especially when several layers are hooked. So the next question becomes:

> "How do we keep retrieval fast without destroying the score structure?"

The repo can project them using SJLT:

$$
P[i,j] \in \{+1, 0, -1\} \cdot \sqrt{3/K}
$$

with many zeros, so the projection is sparse.

Why this matters:

- inner products are approximately preserved,
- storage and search become much cheaper,
- FAISS can then do fast inner-product retrieval.

Important wording:

- say "inner product," not cosine similarity,
- because TracIn is based on raw inner products,
- and the repo uses FAISS `IndexFlatIP`.

Good teaching sentence:

> "Projection makes the fingerprints shorter. FAISS makes those shorter fingerprints searchable."

In the running example:

- A, B, and C each end up with short searchable fingerprints,
- the striped-cat query becomes one short searchable fingerprint too,
- and FAISS quickly tells us that A is closest, C is next, and B is far away.

### 3.9 - The full implementation flow

```
OFFLINE INDEX BUILD

for each checkpoint t:
  load model weights
  load optimizer state if available
  for each training batch:
    capture activations and error signals
    form per-layer ghosts
    concatenate into one ghost vector
    apply Adam correction if needed
    project if needed
    accumulate lr-weighted contribution

build FAISS inner-product index
save metadata: sample ids and rights holders


ONLINE QUERY

load query checkpoint
capture query activations and error signals
form query ghost
apply same correction and projection
search FAISS
take top-k matches
clamp negative scores to zero
normalize positive scores into attribution percentages
```

Then say the meaning of this pipeline in one sentence:

> "Training-side work is prepaid. Query-time work is mostly ghost extraction plus search."

That is why the method is practical. The expensive part is moved to the offline stage, and the online stage becomes short enough to use in a real attribution workflow.

You can even end the section by replaying the running example in one sentence:

> "So for our striped-cat query, the system does not reopen all of training. It builds one query ghost, searches the stored fingerprints, and returns A, then C, then B."

### 3.10 - Honest caveats to say out loud

This section will make the presentation stronger, not weaker.

- Hooked layers may cover only part of the parameter space, so coverage matters.
- The projection step is approximate, not exact.
- The default indexed query path is a practical approximation; the repo also contains a more faithful multi-checkpoint query path.
- Some architectures need per-layer raw handling instead of the simplest factored shortcut.
- In the user-facing attribution output, negative scores are clamped before percentage normalization.

If you say these clearly, the audience will trust the rest of the talk more, because they can see exactly where the method is exact, where it is engineered, and where it is approximate by design.

Strong closing sentence for Part 3:

> "The contribution is not pretending everything is always exact. The contribution is knowing where structure gives us exactness, where engineering gives us robustness, and where approximation is worth the speedup."

### 3.11 - Suggested speaker flow for Part 3

1. Start from the TracIn formula and restate the scaling problem.
2. Show the exact linear-layer ghost identity.
3. Explain how multiple hooked layers are concatenated.
4. Explain the offline index build and online query pipeline.
5. Walk through the real-model fixes: Conv2d, ConvTranspose2d, BatchNorm2d, LayerNorm, Embedding, RNN caveat.
6. Finish with Adam correction and the projection/indexing step.
7. End with the honest caveats slide.

### 3.12 - Where each concept lives in the code

| Concept | Main file | Key functions |
|---------|-----------|---------------|
| Ghost vector formation | `src/math_utils.py` | `form_ghost_vectors()`, `form_multi_layer_ghost_vectors()` |
| Bias augmentation | `src/hooks_manager.py` | `_maybe_append_bias_ones()` |
| Conv2d unfolding | `src/hooks_manager.py` | `_unfold_conv2d_input()`, `_register_conv2d_hooks()` |
| ConvTranspose2d handling | `src/hooks_manager.py` | `_unfold_conv_transpose2d_input()`, `_register_conv_transpose2d_hooks()` |
| BatchNorm2d safe hooks | `src/hooks_manager.py` | `_register_batchnorm2d_hooks()` |
| Adam correction | `src/math_utils.py` | `load_adam_second_moment()`, `load_adam_second_moment_with_bias()`, `apply_adam_correction()` |
| Projection | `src/math_utils.py` | `build_sjlt_matrix()`, `project()` |
| Main offline index | `src/indexer.py` | `build_index()` |
| Faithful multi-ckpt index | `src/indexer.py` | `build_multi_checkpoint_index()` |
| Main online attribution | `src/inference.py` | `attribute()` |
| Faithful multi-ckpt query | `src/inference.py` | `attribute_multi_checkpoint()` |
| Automatic layer selection / benchmark path | `benchmarks/ghost_faiss.py` | `auto_ghost_layers()`, `compute_ghost_tracin_scores()` |

### Transition into Part 4

Say this out loud as a clean handoff:

> “Part 3 told you what object we compare and how the repository stores and searches it. Part 4 is the engineering panel: hooks, checkpoints, loss alignment, Adam bookkeeping, and the knobs that keep this stable on real architectures.”

## Part 4: Implementation and heuristics (engineering)

This is the “how did you actually build it?” block. It answers the questions people ask after Parts 2–3:

- Which tensors do you hook?
- How do you choose layers?
- What checkpoints do you save?
- How do you align loss / error signals?
- How do you load Adam state safely?
- What knobs exist for projection, indexing, conv memory, and coverage?

### Slide deck (recommended)

#### Slide 1 — What we hook and how

**Core objects**:

- **Activation** \(A\): what entered the hooked module on the forward pass.
- **Error signal** \(E\): how the loss “blames” the module output (either computed analytically or via autograd).

**Two hook managers**:

- **`HookManager` (`src/hooks_manager.py`)**: single target layer.
  - Forward hook captures the layer input activation.
  - Optional backward mode captures \(\partial L / \partial \text{output}\) via `register_full_backward_hook`.
- **`MultiLayerBackwardGhostManager` (`src/hooks_manager.py`)**: multiple modules.
  - Registers forward + backward hooks per module.
  - Uses specialized paths for `Conv2d`, `ConvTranspose2d`, `Embedding`, `LayerNorm`, `BatchNorm2d`, `RNNBase`, and a generic “linear-like” path.

**Why tensor hooks sometimes matter**:

- For some modules (notably `BatchNorm2d` and `ConvTranspose2d` in this repo), the implementation uses `output.register_hook(...)` to avoid fragile interactions between backward-hook wrappers and downstream inplace ops.

Speaker line:

> “Hooks are not just instrumentation. They define what mathematical object we are comparing.”

#### Slide 2 — Single-layer vs multi-layer capture (two production modes)

| Mode | Typical API | How \(E\) is obtained | When you use it |
|------|-------------|----------------------|-----------------|
| **Single-layer** | `HookManager` + `build_index(...)` | `error_fn(logits, targets)` | Fast path; good when a single layer is enough |
| **Multi-layer** | `MultiLayerBackwardGhostManager` + `build_index(..., multi_layer_ghost=True)` | `training_loss_fn` + `loss.backward()` | Better coverage; uses true per-layer backward signals |

Speaker line:

> “Single-layer mode computes an error signal directly. Multi-layer mode replays the training loss and lets backprop deliver blame at each hooked layer.”

#### Slide 3 — How we choose which layers to hook

**Default heuristic (simple path)**:

- `resolve_target_layer(model, None)` picks the **last `nn.Linear`** (`src/config_utils.py`).

**Manual override**:

- Pass a dotted module name, e.g. `"output_proj"` (used in demos for LM heads).

**Automatic multi-layer selection (benchmarks / advanced setups)**:

- `auto_ghost_layers(model, target_coverage=..., strategy=...)` (`benchmarks/ghost_faiss.py`)
  - **`strategy="last"`**: prefer modules closer to the output (recommended).
  - **`strategy="largest"`**: prefer modules with the most parameters.
  - **`target_coverage`**: greedy add hookable modules until a fraction of total parameters is covered.
  - Practical knobs: `max_layers`, `include_conv`, `include_rnn` (RNN is optional because coverage can be partial).

Speaker line:

> “Layer choice is the biggest lever between ‘fast demo’ and ‘serious attribution coverage.’”

#### Slide 4 — Checkpoint setup and heuristics

**What must be saved**:

- Unified checkpoints via `TracInCheckpointCallback` (`src/config_utils.py`) that include:
  - model weights
  - optimizer state
  - learning rate
  - epoch + loss metadata (`tracin_checkpoints_metadata.json`)

**Why this matters**:

- TracIn needs \(\eta_t\) per checkpoint.
- Adam correction needs `exp_avg_sq` aligned to ghost layout.

**Practical heuristics (from `docs/implementation.md`)**:

- Aim for **5–10 checkpoints** across training.
- `select_best(keep=5)` keeps checkpoints with the largest **loss drops** between saves (paper-flavored pruning).

Speaker line:

> “Checkpoints are not just model snapshots. They are the timeline coordinates for TracIn.”

#### Slide 5 — Loss alignment and error functions

**Rule**:

- `loss_type` / `error_fn` must match how the model was trained (`docs/implementation.md`).

**Built-ins (`src/error_functions.py`)**:

- Classification: `softmax(logits) - one_hot(target)`
- Regression: `predictions - targets`

**Custom losses**:

- Provide `error_fn(logits, targets) -> E` with the correct shape.

**Multi-layer indexer detail (`src/indexer.py`)**:

- Uses the real `training_loss_fn` and adjusts reductions so backward matches per-example semantics when needed.

Speaker line:

> “If your error signal does not match training, you are not measuring TracIn anymore — you are measuring a broken proxy.”

#### Slide 6 — Adam keys, layout, and bias augmentation

**Finding optimizer keys**:

- `find_adam_param_key` / `find_adam_bias_param_key` scan `model.named_parameters()` order (`src/config_utils.py`).

**Ghost layout vs PyTorch weight layout**:

- Linear weights are stored `(C_out, H_in)` but ghost flattening follows `(H_in, C_out)` ordering — Adam `exp_avg_sq` must be transposed accordingly (`src/math_utils.py`).

**Bias augmentation**:

- `_maybe_append_bias_ones` appends a column of ones so bias gradients ride along with the same outer-product machinery (`src/hooks_manager.py`).

**Applying Adam to ghosts**:

- `apply_adam_correction` is elementwise on the flattened ghost (`src/math_utils.py`).

Speaker line:

> “Adam is not housekeeping here. It changes the direction we compare — and the transpose bug is the silent killer.”

#### Slide 7 — Projection + FAISS configuration

**Projection (`src/indexer.py`, `src/inference.py`, `src/math_utils.py`)**:

- `projection_dim`, `projection_type` (`sjlt` vs `dense`), `projection_seed`
- If `projection_dim` is `None` or `>= ghost_dim`, projection is skipped.

**FAISS (`src/faiss_store.py`)**:

- Default **`IndexFlatIP`**: inner product, **not cosine** (magnitude matters for TracIn-style scores).
- Optional IVF mode exists, but flat IP is the common teaching default.

**Hard requirement**:

- Indexing and querying must use the **same** projection hyperparameters.

Speaker line:

> “Projection shortens fingerprints; FAISS makes fingerprint lookup fast.”

#### Slide 8 — Architecture safeguards (the “real models” knobs)

**Convolutions**:

- `F.unfold` / patch views for conv-style structure (`src/hooks_manager.py`).
- `keep_raw=True` enables exact sum-of-outer-products style handling when raw spatial blocks are kept.
- `max_spatial_positions` caps spatial extent to avoid memory blowups (warn + fallback to mean pooling).

**Grouped convolutions**:

- Group-aware handling exists in the benchmark ghost machinery (`benchmarks/ghost_faiss.py`).

**Embeddings**:

- Sparse vocabulary updates: only touched rows matter.

**RNN caveat**:

- Hooking can be partial (often input-to-hidden emphasis); be honest about coverage.

**Benchmark-only completeness lever**:

- `auto_fallback` can fill uncovered parameters with per-sample autograd gradients (expensive, but exact for missing pieces) (`benchmarks/ghost_faiss.py`).

Speaker line:

> “The math gives you a shortcut; engineering decides where you are exact, where you approximate, and where you pay extra compute to be faithful.”

#### Slide 9 — Engineering recap checklist (leave on screen for Q&A)

1. Match `loss_type` / `error_fn` to training.
2. Choose `target_layer` vs `auto_ghost_layers` based on coverage needs.
3. Save unified checkpoints + metadata; keep enough checkpoints; keep the last one.
4. Align Adam `exp_avg_sq` to ghost layout; include bias state if bias-augmented ghosts.
5. Keep projection + FAISS settings consistent between index and query.
6. Use the right hook path for convs / norm / transpose conv / fragile inplace graphs.

## Part 5: Live Demo

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

## Part 6: Scalability & Benchmark Results

### Slide deck (recommended)

This is the “evidence” block. Prefer plots over tables on slides.

- **Slide 1 — Benchmark headline**: 17 models across common architecture families.
- **Slide 2 — Accuracy story**: show `comparison_cross_model.png` (or `comparison_accuracy.png`) and summarize the Spearman pattern.
- **Slide 3 — Honest exceptions**: U-Net + any low-coverage storylines (e.g., GRU coverage vs correlation).
- **Slide 4 — Speed story**: lead with ResNet50 speedup; show one “ghost wins big” figure.
- **Slide 5 — Scaling trend**: ResNet family + ViT family story (small overhead vs large win).
- **Slide 6 — Closing summary**: 3 bullets + final message from §5.6.

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

## Presentation Flow Diagram

```
Mentor: Motivation + Original TracIn
              │
              ▼
Part 2: Simple overview ◄── "Why is naive TracIn expensive, and what is the ghost shortcut?"
  │  Running example (striped cat vs A/B/C)
  │  Outer-product intuition → offline fingerprints → online search
  ▼
Part 3: Deep dive       ◄── "What exactly do we compute in this repo?"
  │  TracIn formula → linear ghost identity
  │  Multi-layer concatenation
  │  Offline index + online query + caveats
  ▼
Part 4: Implementation  ◄── "How did we actually build it in PyTorch?"
  │  Hooks + layer selection + checkpoints
  │  Loss alignment + Adam layout + projection/FAISS knobs
  │  Conv / norm / embedding / fallback safeguards
  ▼
Part 5: Live Demo       ◄── "Watch it work"
  │  Classification → Text Gen → Image Gen
  │  Same code, different models
  ▼
Part 6: Scalability     ◄── "It's not just a demo"
  │  17 models, ρ ≥ 0.999 for 15/17
  │  Up to 18× faster, scales with model size
  └─ ◄── "Ghost TracIn makes attribution practical"
```

## Practical Checklist

- [ ] Slurm job completed (demo pre-training)
- [ ] Attribution PNGs exist: `demos/outputs/*/attribution_*.png`
- [ ] Interactive demo tested on GPU node
- [ ] Benchmark comparison graph ready from job 416781
- [ ] Headless figures as backup
- [ ] Know which CIFAR image to click for best visual result (test beforehand)
