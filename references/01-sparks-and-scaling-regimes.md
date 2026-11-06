> **Skill Navigation (`weight-space-programming-skill`):**
> [SKILL.md](../SKILL.md) | [01: Sparks & Scaling (§1)](./01-sparks-and-scaling-regimes.md) | [02: Weight Anatomy (§2)](./02-transformer-weight-anatomy.md) | [03: Surgical Editing & Erasure (§3.1–3.7)](./03-surgical-editing-and-erasure.md) | [04: Head Synthesis & Compilation (§3.8–3.16)](./04-head-synthesis-and-compilation.md) | [05: Applied Blueprint & Runbook (§4–5)](./05-applied-blueprint-and-runbook.md) | [06: Bibliography (§6)](./06-annotated-bibliography.md) | [Full Manual](./mechanistic_interpretability.md) | [PyTorch Script](../scripts/weight_space_programming.py)
>
> *Shard `01-sparks-and-scaling-regimes.md` — Lines 51–136 of `mechanistic_interpretability.md`*
<!-- SHARD_BODY_START -->
## 1. Sparks of Weight-Space Programming (Takehi et al., 2026)

**Primary Source:**
* **Title:** *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights*
* **Authors:** Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab (MIT CSAIL)
* **Date:** November 2, 2026
* **Links:** [Blog Post](https://rikiyat.github.io/blog/weight-space-programming/) | [alphaXiv (2611.weight-space-programming)](https://www.alphaxiv.org/abs/2611.weight-space-programming) | [Hugging Face Artifact (`RikiyaT/Qwen3.5-0.8B-weight-space-programming`)](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming) | [Companion Logs & Edit Code (`RikiyaT/lm_understand-logs`)](https://github.com/RikiyaT/lm_understand-logs)

### 1.1 Core Paradigm Shift
For the entire history of deep learning, neural network parameters $\theta \in \mathbb{R}^P$ have been treated as **write-only artifacts** generated opaquely by stochastic gradient optimizers (SGD, AdamW, Muon) over batches of training data. Humans only wrote the discrete surrounding scaffolding (architecture code, data pipelines, loss functions).

**Weight-Space Programming** (demonstrated by Takehi et al., 2026 using **GPT-6 Astra** as the weight-reading and weight-writing agent) shows that neural network weights are structured algebraic objects that can be **read, diagnosed, synthesized from scratch, and surgically patched** directly in parameter space using linear-algebraic reasoning and mechanistic primitives—either statically (zero forward passes) or in an agentic Python harness with a tiny forward-pass budget and **zero backpropagation / zero gradient descent**.

### 1.2 Complete Breakdown of All 8 Experiments

| Exp # | Category | Target Architecture & Scale | Constraint / Setup | Technique Used in Weight Space | Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp 1** | **Static Weight Reading (RL Policies)** | 4 identical PPO-trained MLPs (17,253 weights each) trained on MinAtar grid games (*Breakout, Pong, Pac-Man, Freeway*) | **Zero forward passes**, no tools; raw weights passed directly as text. All 4 models have identical input/output shapes. | Inspected Layer-1 input channel spatial weight receptive fields, symmetry patterns, and action-head couplings (e.g., horizontal paddle-tracking vs. 2D maze navigation). | **100% accuracy** identifying which game each weight tensor played without any hints. |
| **Exp 2** | **Static Weight Reading (Diffusion / DDPM)** | 3 small DDPM denoising MLPs (~23,824 weights each) generating $12 \times 12$ grayscale images (*circle, square, triangle*) | **Zero forward passes**, no tools; raw weights passed as text. | Analyzed spatial covariance and output pixel-projection structure in final denoising layers. | Identified all 3 geometric shapes accurately (notably inferring geometric *outlines* directly from weight boundary gradients rather than simulating sampling). |
| **Exp 3** | **Static Mechanistic Auditing (SLMs)** | 3 character-level Transformers (4 layers, 6 heads, $d=384$; 7.16M weights each) trained on 1B chars of *Python*, *Romaji*, and *English* | Given **only middle blocks** (no token embeddings or unembeddings) inside a Python sandbox; **zero forward passes allowed**. | SVD / spectral analysis of $W_{QK}$ and $W_{OV}$ matrices, attention head induction/copying score calculation ($W_{OV}$ eigenvalues), and bigram transition periodicity. | Correctly classified all 3 training languages (0.55–0.62 confidence) and pinpointed exact ground-truth copying heads (**Head 3.3** in Romaji, **Head 3.1** in English) with zero false positives. |
| **Exp 4** | **One-Shot Weight Repair (Physics Control)** | Kinetix RL physics car policy (171,664 weights) stuck in a local minimum (driving straight into a ramp without taking a run-up) | **No tools, no forward-pass loop**; given raw weights + 1 failure trajectory trace. | Located the velocity-to-throttle coupling weights in the policy head and rewrote **3.5% of weights** in a single text response to reverse first to gain momentum. | **0% $\to$ 100% success rate** (1,024 / 1,024 test episodes) after a single zero-gradient weight patch. |
| **Exp 5** | **From-Scratch Weight Synthesis (Zero-Shot)** | MinAtar $10 \times 10$ grid policies (`600 -> 32 -> 32 -> 6`, `tanh` activation; *Breakout, Seaquest, Asterix*) | Given game rules + architecture spec; **zero data, zero gradients, zero environment rollouts**. | Hand-crafted feature detector filters in early layers (ball/paddle x-coordinate subtractors) and hard-wired control logic into output logits. | Scored **517.4** on *Breakout* (1,000-episode mean vs. **15.8** PPO-100M / **25.3** PPO-500M, a **>20x gain**), **53.2** on *Asterix*, and **53.1** on *Seaquest*. |
| **Exp 6** | **Programmatic Weight Synthesis at Scale** | 1.5M-parameter 9x9 Go policy network ($243 \to 512 \times 6 \to 82$) | Too large for text context; Python CPU sandbox initialized to zeros + restricted forward-pass budget (0, 10, 25, 50 passes). | Programmatically wrote convolution-like liberty counters, capture detectors, and territory heuristics into the $512 \times 512$ MLP weight blocks. | Constructed a working 1.5M-parameter board-evaluation network via pure Python tensor assignments that beat PPO baselines. |
| **Exp 7** | **Alignment & Safety Auditing (Sycophancy)** | Paired **Gemma 3 270M** models (18 layers; identical frozen token embeddings; one honest, one sycophantic) | **Phase 1:** Pure static weights (failed). **Phase 2:** Allowed 16 prompts reading *only* attention routing matrices (no output text). | Pinpointed **Attention Head 10.2** attending to the user's leading/wrong suggestion **85.7%** of the time in the sycophantic model vs. **3.5%** in the honest model. | Proved that while raw static weights struggle with subtle semantic traits in superposition, minimal attention-routing probes immediately expose the exact head. |
| **Exp 8** | **Surgical LLM Repair at 800M Scale** | **`Qwen3.5-0.8B`** (800M parameters, thinking mode enabled; `exp8_math/apply_edit.py`) | Python notebook harness, **zero gradients / no backward passes**, max 50 batched forward-pass evaluations on tiny practice subset. | **Column-selective `mlp.down_proj` subspace subtraction** across 64 columns in Layer 15 and 64 columns in Layer 19 (**131,072 weights = 0.016%**). | Reduced token-limit looping from **93.1% $\to$ 33.1%** (-60.0 pp); **MATH-500: 7.9% $\to$ 48.1%**; **GSM8K: 13.8% $\to$ 67.1%**; **IFEval: 21.0% $\to$ 62.3%**. |

---

### 1.3 Deep-Dive: The Mathematics of Experiment 8 (`Qwen3.5-0.8B` Surgery)

Why did modifying **exactly 131,072 floats (0.016%)** of `Qwen3.5-0.8B` produce a 6x jump on MATH-500 and a 3x jump on IFEval without any training?

1. **Diagnosing the Failure Mode:**
   In `Qwen3.5-0.8B`, the model already possessed mathematical and instruction-following knowledge, but suffered from a **pathological attractor state**: during chain-of-thought reasoning, it fell into repetitive self-doubt/re-verification loops (*"Wait, let me recheck that..."*) until hitting the 32,768-token ceiling on 93.1% of problems.
2. **Isolating the Causal Residual Direction ($u_{\text{loop}}$):**
   By collecting residual stream hidden states $h_\ell \in \mathbb{R}^{d_{\text{model}}}$ ($d_{\text{model}} = 1024$) across productive reasoning tokens vs. repetitive looping tokens on a small diagnostic batch, the contrast vector at layer $\ell$ is:
   $$\Delta_\ell = \mathbb{E}_{t \in \mathcal{T}_{\text{loop}}}[h_{\ell, t}] - \mathbb{E}_{t \in \mathcal{T}_{\text{productive}}}[h_{\ell, t}], \quad u_\ell = \frac{\Delta_\ell}{\|\Delta_\ell\|_2}$$
   Tracing the layer-by-layer norm growth $\Delta_\ell - \Delta_{\ell-1}$ revealed that the loop attractor direction was overwhelmingly injected by the **MLP blocks of Layer 15 and Layer 19**.
3. **Decomposing `mlp.down_proj` by Column:**
   In a SwiGLU MLP, the output added to the residual stream is:
   $$\text{MLP}_\ell(x) = W_{\text{down}}^{(\ell)} \, a^{(\ell)}(x) = \sum_{j=1}^{d_{\text{ff}}} a_j^{(\ell)}(x) \, w_{\text{down}, j}^{(\ell)}$$
   where $w_{\text{down}, j}^{(\ell)} = W_{\text{down}}^{(\ell)}[:, j] \in \mathbb{R}^{1024}$ is the $j$-th column of `down_proj`, and $a_j^{(\ell)}(x) = \text{SiLU}(W_{\text{gate}, j}^{(\ell)} x) \cdot (W_{\text{up}, j}^{(\ell)} x) \in \mathbb{R}$ is the scalar activation of intermediate neuron $j$.
4. **Ranking Columns by Differential Contribution:**
   Each column $j$ gets a loop-contribution score measuring how strongly it pushes the residual stream along $u_\ell$ specifically during looping states:
   $$s_j^{(\ell)} = \Big(\mathbb{E}_{\text{loop}}[a_j^{(\ell)}] - \mathbb{E}_{\text{prod}}[a_j^{(\ell)}]\Big) \cdot \big\langle u_\ell, \, w_{\text{down}, j}^{(\ell)} \big\rangle$$
5. **Surgical Rank-1 Column Update:**
   Instead of modifying all $d_{\text{ff}}$ columns (which risks collateral damage to unrelated math facts stored in superposition), only the **top $K = 64$ columns** $\mathcal{J}_{64}^{(\ell)}$ with highest $s_j^{(\ell)}$ in Layers $\ell \in \{15, 19\}$ are modified by subtracting their component along $u_\ell$:
   $$\forall j \in \mathcal{J}_{64}^{(\ell)}: \quad w_{\text{down}, j}^{(\ell)} \longleftarrow w_{\text{down}, j}^{(\ell)} - \alpha \big\langle u_\ell, \, w_{\text{down}, j}^{(\ell)} \big\rangle u_\ell$$
   * **Parameter count math:** $2\text{ layers} \times 64\text{ columns} \times 1024\text{ rows/column} = \mathbf{131,072}\text{ parameters}$.
   * **Control Verification:** Applying the exact same norm modification along a random orthogonal unit vector $u_\perp \perp u_\ell$ failed to reduce looping and degraded accuracy—proving causal specificity.

6. **Scaling Across Forward-Pass Diagnostic Budgets (`Qwen3.5-0.8B`):**

| Configuration | MATH-500 (%) | GSM8K (%) | IFEval (%) | Hit 32k Token Limit (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Base `Qwen3.5-0.8B` (Unedited)** | 7.9% | 13.8% | 21.0% | 93.1% |
| **Weight-Edited (10 Forward Passes)** | 12.3% | 25.7% | 49.0% | 87.2% |
| **Weight-Edited (25 Forward Passes)** | 14.1% | 25.1% | 39.9% | 86.4% |
| **Weight-Edited (50 Forward Passes)** | **48.1%** | **67.1%** | **62.3%** | **33.1%** |

#### 1.3.1 Theoretical Root Cause: Looping, Exploration, and Why "Lazy AI Stopping" is the Dual of Reasoning Loops
Experiment 8 directly builds on **"Wait, Wait, Wait... Why Do Reasoning Models Loop?"** ([Pipis, Garg, Kontonis, Shrivastava, Krishnamurthy, & Papailiopoulos, Dec 2025, `arXiv:2512.12895`](http://arxiv.org/pdf/2512.12895.pdf)) and the steering-first MLP inspection workflow of **`neuron-explorer`** ([`lyramakesmusic/neuron-explorer`, 2026](https://github.com/lyramakesmusic/neuron-explorer)):

1. **Why Models Loop (Pipis et al., 2025):**
   * **Distillation & Capacity Mismatch:** Student models distilled on complex reasoning traces (or RLHF/instruction-tuned models facing multi-step tasks) imperfectly learn difficult forward transitions while strongly memorizing high-frequency discourse transitions.
   * **Risk Aversion from Hardness of Learning:** When the next progress-making step has high epistemic uncertainty or learning difficulty, but a safe, low-cost "stall" transition is available, the model's probability mass collapses onto the stall action.
   * **Transformer Inductive Bias Toward Temporally Correlated Errors:** Once a stall/cycle token is emitted, self-attention induction heads attend back to previous stall tokens, locking the trajectory into an attractor basin.
   * **Why Sampling Temperature Fails:** Raising temperature forces random token exploration to break out of the loop, but *does not fix the underlying weight-space transition error*, resulting in bloated, noisy, inefficient rollouts. Surgical weight subtraction fixes the transition probabilities directly at temperature 0.

2. **Why "Lazy AI Stopping" (`Can Continue = Yes`) Is the Exact Turn-Level Dual of Reasoning Loops:**
   * Inside a `<think>` block, the safe "stall" action under next-step risk aversion is emitting *"Wait, let me double-check..."* (an **intra-turn loop**).
   * Outside a `<think>` block (in an interactive assistant turn), the safe "stall" action learned during RLHF is **premature turn-yielding / permission-seeking**: summarizing partial progress and emitting *"Would you like me to proceed with Step 2?"* followed by `<EOS>` (an **inter-turn stall**).
   * **Strategic Implication for Your System:**
     * **In an Outer Agent Harness (Classifier Approach):** Your classifier is specifically detecting whether the assistant entered this *Risk-Averse Turn-Yielding Attractor* (where the next action is already known and stated in the output, so `can_continue = True`) vs. genuine completion (`DONE`) or genuine missing information (`BLOCKED_ON_USER`).
     * **In a Local Open-Weights Generator (Direct Generator Surgery):** If you control the generating LLM's weights, you can also apply Experiment 8's exact `mlp.down_proj` column subtraction to suppress the *premature turn-yielding / permission-request direction* inside the generator itself so it keeps working autonomously without stopping!

### 1.4 Failure Modes & Boundaries: Exp 7 (Sycophancy) vs. Exp 8b (`Gemma 3 270M`)
Understanding *when* static weight-space programming succeeds vs. fails is critical for engineering reliable edits:
1. **Superposition Opacity Without Routing Probes (Experiment 7 Lesson):**
   When comparing two fine-tuned `Gemma 3 270M` checkpoints (honest vs. sycophantic, with frozen identical embeddings), pure static weight inspection succeeded **0 / 4 times** because distributed weight diffs across 18 layers could not be mapped to high-level semantics without knowing which tokens activated which heads. However, adding a tiny **16-prompt attention-matrix probe** (reading *only* $A_{t,s}^h$, not model outputs) succeeded **3 / 3 times**, immediately isolating **Head 10.2** (which attended to the user's suggested wrong answer **85.7%** of the time in the sycophantic model vs. **3.5%** in the honest model).
2. **Unlocking Latent Capabilities vs. Injecting Missing Knowledge (Experiment 8b Lesson):**
   In **Experiment 8b**, the authors ran the exact same math weight-editing loop on **`Gemma 3 270M`**. Unlike `Qwen3.5-0.8B` (which looped on 93% of problems), `Gemma 3 270M` only looped on **8.5%** of problems—it simply lacked the underlying mathematical capacity at 270M parameters. Consequently, weight-space editing only moved `Gemma 3 270M` on MATH-500 from **6.8% $\to$ 7.7%** (at 25 passes) / **7.0%** (at 50 passes).
   * **Takeaway for Your Classifier:** Weight-space programming is maximally effective when you start from a backbone that **already possesses the core linguistic/entailment primitives** (such as `tasksource/ModernBERT-large-nli` or `dmis-lab/UltraBERT-380M`) and use weight surgery to **remove confounders** (polite sign-offs, surface question marks) and **wire existing internal features directly to the output head**.

---

