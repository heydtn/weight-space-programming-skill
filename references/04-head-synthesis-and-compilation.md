> **Skill Navigation (`weight-space-programming-skill`):**
> [SKILL.md](../SKILL.md) | [01: Sparks & Scaling (§1)](./01-sparks-and-scaling-regimes.md) | [02: Weight Anatomy (§2)](./02-transformer-weight-anatomy.md) | [03: Surgical Editing & Erasure (§3.1–3.7)](./03-surgical-editing-and-erasure.md) | [04: Head Synthesis & Compilation (§3.8–3.16)](./04-head-synthesis-and-compilation.md) | [05: Applied Blueprint & Runbook (§4–5)](./05-applied-blueprint-and-runbook.md) | [06: Bibliography (§6)](./06-annotated-bibliography.md) | [Full Manual](./mechanistic_interpretability.md) | [PyTorch Script](../scripts/weight_space_programming.py)
>
> *Shard `04-head-synthesis-and-compilation.md` — Section 3 (Part 2: State-of-the-Art Techniques §3.8–§3.16), Lines 326–461 of `mechanistic_interpretability.md`*
<!-- SHARD_BODY_START -->
### 3.8 Closed-Form Classifier Head Synthesis (Shrinkage LDA & NLI Grafting)
In standard fine-tuning, attaching `AutoModelForSequenceClassification` initializes `classifier.weight` randomly and runs thousands of SGD steps. In **Weight-Space Programming**, we synthesize `classifier.weight` ($W_{\text{cls}} \in \mathbb{R}^{C \times d}$) and `classifier.bias` ($b_{\text{cls}} \in \mathbb{R}^C$) **analytically in closed form**:

#### Method A: Zero-Data NLI Head Grafting (Pure Static Weight Rewiring)
If you start from an NLI-pretrained encoder like `tasksource/ModernBERT-large-nli`, its classification head $W_{\text{NLI}} \in \mathbb{R}^{3 \times d_{\text{model}}}$ is **already trained** with rows corresponding to:
* $w_{\text{entailment}}$ (Premise entails Hypothesis)
* $w_{\text{neutral}}$ (Premise leaves Hypothesis unresolved / incomplete)
* $w_{\text{contradiction}}$ (Premise contradicts / blocks Hypothesis)

By framing the input pair as:
* **Premise:** The Assistant's final message tail
* **Hypothesis:** *"The assistant has completely finished all work and requires no further execution steps."*
You can **directly wire** $W_{\text{cls}}$ in weight space using linear combinations of $w_{\text{entailment}}, w_{\text{neutral}}, w_{\text{contradiction}}$ and token prototypes mapped into post-`model.head` space (§4.2)—giving a working classifier with **0 calibration examples**!

#### Method B: Few-Shot Closed-Form Shrinkage Fisher LDA (Analytical Bayes-Optimal Head)
Given a small calibration set of pooled post-`model.head` states $H \in \mathbb{R}^{N \times d}$ (e.g., $N = 30\text{–}200$ examples across $C = 3$ classes: `LAZY_STOP`, `DONE`, `BLOCKED_ON_USER`), after scrubbing nuisance directions via mean-preserved LEACE ($H_\perp = H P_{\text{LEACE}}^\top + \mathbf{1} b_{\text{LEACE}}^\top$):
1. Compute class means $\mu_c = \frac{1}{|S_c|} \sum_{i \in S_c} h_{\perp, i} \in \mathbb{R}^d$.
2. Compute the **Ledoit-Wolf Shrinkage Within-Class Covariance** (which guarantees invertibility even when $N \le C$ or $N < d$, using $\tau_{\text{diag}} = \max\!\big(10^{-4}, \, \frac{\text{Tr}(S_W)}{d}\big)$ so 1-shot-per-class $S_W = 0$ smoothly reduces to isotropic Mass-Mean probing):
   $$\hat{\Sigma}_W = (1 - \gamma) S_W + \gamma \, \tau_{\text{diag}} \, I_d$$
3. **Zero-Sum Centered Class Means & Between-Class Mahalanobis Temperature Calibration ($N \ll d$):** Solve $\tilde{w}_{\text{lda}, c} = \hat{\Sigma}_W^{-1}(\mu_c - \bar{\mu}) \in \mathbb{R}^d$ using centered class means ($\bar{\mu} = \frac{1}{C}\sum_{c=1}^C \mu_c$) so $\sum_{c=1}^C \tilde{w}_{\text{lda}, c} = 0$ (eliminating non-discriminative common-mode drift $\hat{\Sigma}_W^{-1}\bar{\mu}$ from $W_{\text{lda}}$ while preserving 100% identical pairwise class logit differences). Normalize by the **between-class Mahalanobis dispersion** $T$ before computing biases:
   $$T = \max\!\left(1, \, \frac{1}{4C}\sum_{c=1}^C (\mu_c - \bar{\mu})^\top \tilde{w}_{\text{lda}, c}\right), \qquad w_{\text{lda}, c} = \frac{1}{T}\tilde{w}_{\text{lda}, c}$$
4. Under a shared covariance Gaussian model, folding $h_\perp = P_{\text{LEACE}} h + b_{\text{LEACE}}$ into the linear classifier gives the exact closed-form weights and biases:
   $$W_{\text{cls}}[c, :] = w_{\text{lda}, c}^\top P_{\text{LEACE}}, \qquad b_{\text{cls}}[c] = -\frac{1}{2} (\mu_c + \bar{\mu})^\top w_{\text{lda}, c} + \log \pi_c + w_{\text{lda}, c}^\top b_{\text{LEACE}}$$
   *(Important Mathematical Note: Because $b_{\text{LEACE}} = (I_d - P_{\text{LEACE}})\mathbb{E}[X]$ lies in the right null space of $P_{\text{LEACE}}^2 = P_{\text{LEACE}}$, the bias shift uses $w_{\text{lda}, c}^\top b_{\text{LEACE}}$, NOT $W_{\text{cls}}[c, :] b_{\text{LEACE}} \equiv 0$!)*
   * **Computation Time:** `< 50 milliseconds` on CPU/GPU via `torch.linalg.solve` (one matrix solve of size $1024 \times 1024$), with **zero epochs, zero learning rate schedules, and zero overfitting**.

---

### 3.9 Geometry of Truth & Mass-Mean Probing vs. Logistic Regression (Marks & Tegmark, 2024)
In *"The Geometry of Truth"* ([Marks & Tegmark, ICLR 2024](https://arxiv.org/abs/2310.06824)), the authors prove a critical geometric pitfall explaining why standard gradient-trained linear probes (Logistic Regression / SGD) fail when used for weight surgery or steering:
* Suppose the residual stream contains a true causal direction $u_{\text{causal}}$ alongside a high-variance non-causal nuisance direction $u_{\text{noise}}$ that is weakly correlated or anti-correlated on the training split.
* **Why Logistic Regression Fails:** Gradient descent maximizes margin by rotating the probe weight vector $w_{\text{LR}}$ *away* from $u_{\text{causal}}$ in order to cancel out variance along $u_{\text{noise}}$. Editing weights along $w_{\text{LR}}$ therefore corrupts unrelated features.
* **Why Mass-Mean ($\mu_+ - \mu_-$) & Mahalanobis ($\Sigma_W^{-1}(\mu_+ - \mu_-)$) Succeed:** The difference-in-means vector points strictly along the true causal displacement vector $u_{\text{causal}}$, and multiplying by the inverse within-class covariance $\Sigma_W^{-1}$ (or applying LEACE first) removes correlated nuisance variance without rotating away from the causal axis.

---

### 3.10 Dormant Neuron Hijacking: Programming Exact Polarized GeGLU/SwiGLU Logic Gates ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$)
Section 3.1 modified only $W_{\text{down}}$ (changing *what* an existing neuron writes). When you want to program a **brand-new deterministic conjunctive (AND) logic rule** into a Transformer layer (Experiments 4, 5, and 6 of *Sparks of Weight-Space Programming*):
1. **Locate Dormant / Dead Neurons via Gauge-Invariant Product Norm:**
   By §2.6, rank dormancy by the **gauge-invariant triple product norm** $\|w_{\text{down}, j}\|_2 \|w_{\text{gate}, j}\|_2 \|w_{\text{up}, j}\|_2$ (or Wanda activation-weighted norm $\|w_{\text{down}, j}\|_2 \|a_j(X)\|_2$). Overwriting the lowest-scoring indices $j \in \mathcal{D}$ causes **zero degradation** to existing model capabilities.
2. **Program an Exact 3-Neuron Polarized GeGLU Conjunctive AND Gate:**
   In `ModernBertMLP`, `input, gate = self.Wi(x).chunk(2, dim=-1)` applies `GELU` to `Wi[:d_ff]` and leaves `Wi[d_ff:]` **strictly linear**.
   * *Why a 1-Neuron Gate Fails:* (a) Wiring `input` to $k_A$ and `gate` to $k_B$ produces $\text{GELU}(\text{pos})\cdot(\text{neg}) < 0$ whenever $\langle k_A, \tilde{x}\rangle > 0$ and $\langle k_B, \tilde{x}\rangle < 0$ (inverting the output sign!); (b) wiring both branches of a single neuron to $k_A + k_B$ computes $\psi(\kappa(z_A + z_B)) \propto (z_A^2 + z_B^2 + 2z_A z_B)$, which retains univariate squares $z_A^2, z_B^2$ and fires as an **OR** gate when only one concept is present!
   * **Exact 3-Neuron Polarized Conjunctive Construction:** Define the non-negative one-sided quadratic primitive $\psi(u) := \text{GELU}(u)\cdot u = u^2\Phi(u) \ge 0$ (achieved by setting a neuron's `GELU` row equal to its linear `gate` row). First, zero-mean center $k_A, k_B$ via $P_{\mathbf{1}^\perp}$, unit-normalize them on the active coordinate support $|\gamma_{\text{mlp}, i}| > 10^{-2}$, and apply **symmetric Löwdin orthogonalization** ($\hat{k}_A, \hat{k}_B \leftarrow \frac{u_+ \pm u_-}{\sqrt{2}}$ where $u_\pm = \frac{\hat{k}_A \pm \hat{k}_B}{\|\hat{k}_A \pm \hat{k}_B\|_2}$, ensuring $\langle \hat{k}_A, \hat{k}_B \rangle = 0$ so Concept A alone never triggers $z_B > 0$), **then** multiply by $\gamma_{\text{mlp}}^{-1}$ (without re-normalizing afterward!) to obtain $\tilde{k}_A = \text{diag}(\gamma_{\text{mlp}})^{-1} \hat{k}_A$ and $\tilde{k}_B = \text{diag}(\gamma_{\text{mlp}})^{-1} \hat{k}_B$ so that runtime multiplication by $\text{diag}(\gamma_{\text{mlp}})$ recovers the exact orthonormal detectors $\hat{k}_A, \hat{k}_B$ in $1:1$ balance. Using the **3 most dormant neurons** $(j_{\text{sum}}, j_A, j_B) \in \mathcal{D}$:
     * **Neuron $j_{\text{sum}}$:** Both branches $= \kappa(\tilde{k}_A + \tilde{k}_B)$, output column $= +\frac{1}{2}\gamma_{\text{out}} v_{\text{target}}$.
     * **Neuron $j_A$:** Both branches $= \kappa \tilde{k}_A$, output column $= -\frac{1}{2}\gamma_{\text{out}} v_{\text{target}}$.
     * **Neuron $j_B$:** Both branches $= \kappa \tilde{k}_B$, output column $= -\frac{1}{2}\gamma_{\text{out}} v_{\text{target}}$.
   * **Exact Cancellation & Domain Proof:** The net output is $\Delta_{\text{AND}}(z_A, z_B) = \frac{1}{2}\big[\psi(\kappa(z_A + z_B)) - \psi(\kappa z_A) - \psi(\kappa z_B)\big] \gamma_{\text{out}} v_{\text{target}}$. On the **axes** ($z_B = 0$ or $z_A = 0$), $\Delta_{\text{AND}} \equiv \mathbf{0}$ for all $z_A, z_B \in \mathbb{R}$; on the **non-negative feature cone** ($z_A > 0, z_B > 0$), strict superadditivity of $\psi$ on $\mathbb{R}_+$ produces a strictly positive interaction ($\to +\kappa^2 z_A z_B \gamma_{\text{out}} v_{\text{target}} > 0$ for $\kappa z_A, \kappa z_B \gg 1$, and $\big[\frac{1}{2}\kappa^2 z_A z_B + \frac{3}{2\sqrt{2\pi}}\kappa^3 z_A z_B(z_A + z_B)\big]\gamma_{\text{out}}v_{\text{target}} > 0$ near $0$). *(Note on anti-correlated bipolar features: because $\text{GELU}(u) - \text{GELU}(-u) \equiv u \implies \psi(u) + \psi(-u) \equiv u^2$, the 3-neuron gate obeys the exact global identity $\Delta_{\text{AND}}(z_A, z_B) + \Delta_{\text{AND}}(-z_A, -z_B) \equiv \kappa^2 z_A z_B \gamma_{\text{out}} v_{\text{target}}$, so at $(z_A, z_B) = (+1, -1)$ where $z_A + z_B = 0$, $\Delta_{\text{AND}}(+1, -1) = -\frac{1}{2}(\psi(\kappa) + \psi(-\kappa))\gamma_{\text{out}}v_{\text{target}} = -\frac{1}{2}\kappa^2 \gamma_{\text{out}}v_{\text{target}} < 0$; if bipolar negative dip suppression is needed when $z_A > 0, z_B < 0$ and `mlp.Wi.bias` is available or emulated via `[CLS]` sink, add a negative threshold bias $b_0 \approx -\frac{3}{2}\kappa$ to the `GELU` branch.)*

---

### 3.11 Attention QK-Circuit Routing Surgery ($W_Q^h, W_K^h$ Rank-1 Rewiring Under RoPE & GQA)
Motivated by Experiment 7 (where `Head 10.2` routed 85.7% of its attention to a misleading span), you can directly edit *where* an attention head $h$ attends without training:
* **Architectural Guardrails:**
  1. **Target Global Attention Layers:** In `ModernBERT`, only layers with `layer_idx % 3 == 0` (`0, 3, 6, ..., 27`) have global attention where `[CLS]` can attend to the end of long messages.
  2. **Edit Private $W_Q^h$ (Not Shared $W_K^{g(h)}$):** In GQA models (`Qwen3.5`, `Llama-3`, `Gemma 3`), editing $W_K^{g(h)}$ corrupts all sibling query heads in group $g(h)$. Always edit the private query matrix $W_Q^h$.
  3. **Project Onto Low-Frequency RoPE Planes:** Because target discourse phrases (`"Next steps:"`, `"Shall I proceed"`) occur at variable token distances $t - s$, project the rank-1 update onto the **lowest-frequency 2D RoPE planes** ($P_{\text{low-freq}} \in \mathbb{R}^{d_{\text{head}} \times d_{\text{head}}}$, retaining both split-half row indices $\mathcal{I}_m = [m, \, m + d_{\text{head}}/2]$ for each low-frequency plane $m$, §2.7) so relative distance rotation $R_{t-s} \approx I$ does not flip the sign of the attention dot product!
* Let $k_* = P_{\text{low-freq}} W_K^{g(h)} \tilde{x}_{\text{target}} \in \mathbb{R}^{d_{\text{head}}}$ be the low-frequency key vector of the target span and $\tilde{x}_{\text{dest}} \in \mathbb{R}^{d_{\text{model}}}$ be the normed `[CLS]` representation:
  $$W_Q^h \longleftarrow W_Q^h + \eta \frac{k_* \tilde{x}_{\text{dest}}^\top}{\|k_*\|_2 \, \|\tilde{x}_{\text{dest}}\|_2}$$
* Setting $\eta > 0$ forces head $h$ to pull information from `target` tokens into the `[CLS]` summary representation; setting $\eta < 0$ suppresses attention to distractor spans.

---

### 3.12 LoReFT & Distributed Alignment Search (DAS) Folded Into Weights (Wu et al., 2024)
**ReFT / LoReFT** ([Wu et al., NeurIPS 2024](https://arxiv.org/abs/2404.03592), grounded in **Distributed Alignment Search** by [Geiger et al., 2024](https://arxiv.org/abs/2303.02536)) intervenes on an $r$-dimensional orthonormal subspace $R \in \mathbb{R}^{r \times d_{\text{model}}}$ ($R R^\top = I_r$) via:
$$\Phi_{\text{LoReFT}}(z) = z + R^\top (W z + b - R z) = \big(I_d - R^\top R + R^\top W\big) z + R^\top b$$
To fold $\Phi_{\text{LoReFT}}$ permanently into model weights with **zero runtime hooks** (avoiding the residual skip-connection bypass noted in §3.3), apply $\Phi_{\text{LoReFT}}$ either (a) directly to a submodule output $z = \text{MLP}_\ell(x)$ or $\text{Attn}_\ell(x)$ (folding $I_d - R^\top R + R^\top W$ into $W_{\text{down}}^{(\ell)}$ or $W_O^{(\ell)}$), or (b) to the pooled post-norm representation right before a linear readout (`model.head.dense` or `model.classifier`).

---

### 3.13 Weight Oracles, Natural Language Autoencoders (NLAs), & WorkspaceBench (2026)
Five citations from *Sparks of Weight-Space Programming* form the modern stack for **translating weights and activations directly into natural-language causal hypotheses**:
1. **Weight Oracles ([Kabra, Venhoff, & Schroeder de Witt, Oct 2026, `arXiv:2610.07334`](http://arxiv.org/pdf/2610.07334.pdf)):**
   * Introduces proactive, static weight-reading LLMs trained via a staged curriculum and an external "chain-of-computation" (delegating parameter-free matrix ops to deterministic Python code).
   * Achieves **99% holdout accuracy** simulating small transformer forward passes directly from raw weights, and detects unseen zero-shot backdoors statically (**0.93 AUROC** on attention-routed backdoors; **0.81 AUROC** across stealth/adversarial variants) without requiring trigger inputs.
2. **Natural Language Autoencoders — NLAs ([Fraser-Taliente et al. / Anthropic, May 2026](https://transformer-circuits.pub/2026/nla/index.html)):**
   * Replaces numerical Sparse Autoencoder (SAE) dictionaries with a **natural language bottleneck**: an **Activation Verbalizer (AV)** LLM reads residual state $h_\ell$ and outputs an English paragraph explaining the model's latent state; an **Activation Reconstructor (AR)** LLM reconstructs $\hat{h}_\ell$ solely from that English explanation, trained unsupervised via RL to minimize $\|\hat{h}_\ell - h_\ell\|_2^2$.
   * Crucially used on Claude Opus 4.6 to detect *unverbalized evaluation awareness* and hidden intent that never appeared in surface tokens—and directly applicable to verbalizing *why* an assistant is about to stop (e.g., distinguishing *"waiting for rubber-stamp approval on a known plan"* vs. *"genuinely missing a required parameter"*).
3. **`WorkspaceBench` ([Blank, Bhatia, Ong, & Nanda, 2026](https://www.lesswrong.com/posts/Zeg2JztbdhguL48uH/workspacebench-evaluating-interpretability-methods-for-the)):**
   * Evaluates interpretability methods on how faithfully they recover the model's **Global Workspace** (intermediate latent variables and decision states across reasoning and safety tasks).
4. **Automated Neuron Explanation ([Bills et al. / OpenAI, 2023](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html)) & `neuron-explorer` ([`lyramakesmusic/neuron-explorer`, 2026](https://github.com/lyramakesmusic/neuron-explorer)):**
   * While Bills et al. pioneered observational LLM-generated explanations of MLP neurons, `neuron-explorer` (built for `Qwen3.5-4B`) inverted the workflow into **steering-first exploration**: clamping/pushing individual `mlp.down_proj` columns during rollouts to observe causal behavioral shifts first, which directly inspired the 64-column surgical edit in Experiment 8.

---

### 3.14 Forward-Pass-Only Subspace Optimization: Combining Mechanistic Coordinates with MeZO, SPSA, ES, GEPA, & Reactive Optimizers
A central question in weight-space programming is: *Once mechanistic interpretability narrows a 400M-parameter model down to a small set of causal directions/columns (e.g., 64 `down_proj` columns or a low-rank subspace), how should you explore and tune the exact scaling coefficients without backpropagation?*

Takehi et al. (2026) contrast and combine two families of gradient-free optimization:

#### Family 1: Zeroth-Order Numerical Perturbation (SPSA, ES, MeZO)
* **SPSA ([Spall, 1992](https://doi.org/10.1109/9.119632))**, **Evolution Strategies ([Salimans et al., 2017, `arXiv:1703.03864`](http://arxiv.org/pdf/1703.03864.pdf))**, and **MeZO ([Malladi et al., NeurIPS 2023, `arXiv:2305.17333`](http://arxiv.org/pdf/2305.17333.pdf))** estimate gradients using **only 2 forward passes** per step with memory footprint equal to inference:
  $$\hat{g}(\theta) = \frac{\mathcal{L}(\theta + \epsilon z) - \mathcal{L}(\theta - \epsilon z)}{2\epsilon} z, \quad z \sim \mathcal{N}(0, I_P)$$
* **Why Full-Parameter MeZO/SPSA Is Slow, But Mechanistic-Subspace SPSA Is Ultra-Fast:**
  * In full parameter space ($P = 4 \times 10^8$), variance of the SPSA estimator scales as $O(P)$, requiring $10^4\text{–}10^5$ forward passes.
  * However, when you **first project into a $k$-dimensional mechanistic basis** $U \in \mathbb{R}^{P \times k}$ ($k \le 10$, such as the layer-18/22 `down_proj` amplification scale $\alpha$, LEACE shrinkage $\gamma$, and class prior thresholds $\tau$), SPSA / CMA-ES / Nelder-Mead operates in $\mathbb{R}^k$ and converges in **10 to 50 forward passes**, optimizing **non-differentiable metrics** (like exact Macro-F1 or precision at 95% recall) directly!

#### Family 2: Reflective & Agentic Outer/Inner-Loop Optimization (OPRO, DSPy, MIPRO, GEPA, Meta-Harness, `autoresearch`, Reactive Optimizers)
Instead of blind Gaussian perturbations $z$, an LLM agent inspects diagnostic traces and rewrites the weight-editing code or optimizer logic reactively:
* **OPRO ([Yang et al., ICLR 2024](http://arxiv.org/pdf/2309.03409.pdf))**, **DSPy ([Khattab et al., ICLR 2024](http://arxiv.org/pdf/2310.03714.pdf))**, & **MIPRO ([Opsahl-Ong et al., EMNLP 2024](http://arxiv.org/pdf/2406.11695.pdf)):** Declarative compilation and Bayesian/LLM-guided optimization of multi-stage pipelines.
* **GEPA — Genetic-Pareto Reflective Evolution ([Agrawal et al., ICLR 2026, `arXiv:2507.19457`](http://arxiv.org/pdf/2507.19457.pdf)):** Replaces scalar RL (like GRPO) with **natural-language reflection on failure traces + Pareto-frontier candidate selection**, outperforming RL with orders of magnitude fewer rollouts.
* **Meta-Harness ([Lee, Nair, Zhang, Lee, Khattab, & Finn, COLM 2026, `arXiv:2603.28052`](http://arxiv.org/pdf/2603.28052.pdf)) & `autoresearch` ([Karpathy, 2026](https://github.com/karpathy/autoresearch)):** Outer-loop agentic harnesses where an LLM reads past code, execution traces, and validation scores on the filesystem, proposes a surgical code/weight modification, and ratchets improvements via git/versioned checkpoints.
* **Reactive Optimizers / `OPAL` ([Bai, Liang, Teoh, & Sitzmann, 2026](https://xingjianbai.com/reactive-optimizers/)):** Places the reasoning agent **inside the inner optimization loop**, reading live activation/weight telemetry mid-run and dynamically rewriting the parameter update rule on the fly (the exact paradigm used in Experiment 8's 50-pass notebook loop).

---

### 3.15 Constructive Weight Synthesis, RASP/Tracr Compilation, Looped Computers, & AdderBoard
When constructing neural networks or algorithmic submodules **from scratch** in weight space (Experiments 5 & 6), five foundational paradigms govern how symbolic programs compile into exact floating-point weight tensors:

1. **RASP (*Restricted Access Sequence Processing* — [Weiss, Goldberg, & Yahav, ICML 2021](https://arxiv.org/abs/2106.06981)) & Tracr ([Lindner et al. / DeepMind, NeurIPS 2023](https://arxiv.org/abs/2301.05062)):**
   * Formalizes a Transformer as a deterministic register machine over the residual stream:
     * **Residual Stream Register Allocation:** Partition $\mathbb{R}^{d_{\text{model}}} = \mathcal{R}_1 \oplus \mathcal{R}_2 \oplus \dots \oplus \mathcal{R}_m$ into orthogonal subspaces ("named registers"). Each attention head or MLP block reads from specific input subspaces via $W_Q, W_K, W_V, W_{\text{in}}$ and writes into disjoint output subspaces via $W_O, W_{\text{down}}$.
     * **`select(q_reg, k_reg, predicate)` $\to$ $W_{QK}^h$:** Compiled by setting $W_Q^h$ to extract `q_reg`, $W_K^h$ to extract `k_reg`, and scaling by inverse temperature $\beta \gg 1$ so `softmax` approaches a hard boolean/argmax lookup.
     * **`aggregate(selector, v_reg)` $\to$ $W_{OV}^h$:** Compiled by setting $W_V^h$ to read `v_reg` and $W_O^h$ to write into a fresh destination register $\mathcal{R}_{\text{dest}}$.
2. **Looped Transformers as Programmable Computers ([Giannou, Rajput, Sohn, Lee, Lee, & Papailiopoulos, ICML 2023](https://arxiv.org/abs/2301.13196)):**
   * Proves that a constant-depth Transformer with **explicitly constructed weights** can execute arbitrary Turing-complete programs (including `SUBLEQ` branching, register arithmetic, and full matrix inversion) by structuring each token position as `[program_instruction, memory_register, program_counter, scratchpad]`, using Attention to fetch the instruction/data pointed to by the `program_counter`, and using the MLP to execute the ALU operation.
3. **Fourier / Trigonometric Circuits for Exact Modular & Multi-Digit Arithmetic ([Nanda et al., ICLR 2023](https://arxiv.org/abs/2301.05217)) & `AdderBoard` ([Papailiopoulos, 2026](https://github.com/anadim/AdderBoard)):**
   * Rather than memorizing $N^2$ addition tables, minimal transformers ($<500$ parameters on `AdderBoard`) encode numbers into 2D rotation/Fourier bases $(\cos(\omega_k a), \sin(\omega_k a))$ and use the bilinear MLP product to compute $\cos(\omega_k(a+b)) = \cos(\omega_k a)\cos(\omega_k b) - \sin(\omega_k a)\sin(\omega_k b)$ analytically alongside thresholded carry-lookahead attention.
4. **HyperTuning ([Phang et al., ICML 2023](https://proceedings.mlr.press/v202/phang23a.html)), Neural Network Diffusion / `p-diff` ([Wang et al., 2024, `arXiv:2402.13144`](http://arxiv.org/pdf/2402.13144.pdf)), & NeuroGen ([Wang, Zhang, & Li, 2025, `arXiv:2505.12470`](http://arxiv.org/pdf/2505.12470.pdf)):**
   * Show that trained neural network weights live on a low-dimensional structured manifold that can be synthesized directly by hypernetworks, latent diffusion models, or instruction-tuned LLMs without backpropagation.

---

### 3.16 Cross-Architecture Weight Grafting (Orthogonal Procrustes Stitching) & Circuit Breakers
1. **Orthogonal Procrustes Weight Grafting ([Moschella et al., ICLR 2023 - *Relative Representations*](https://arxiv.org/abs/2209.15430); [Bansal et al., NeurIPS 2021 - *Model Stitching*](https://arxiv.org/abs/2106.07682)):**
   * Suppose Model A (e.g., `tasksource/ModernBERT-large-nli`) has a well-trained classification head or MLP circuit $W_A$, and you want to **graft** it onto Model B (e.g., `dmis-lab/UltraBERT-380M`) without training Model B.
   * Run $N \approx 200$ anchor sentences through both models to get mean-centered hidden states $H_A, H_B \in \mathbb{R}^{N \times d}$. Solve the **Orthogonal Procrustes Problem** $\min_{Q^\top Q = I} \|H_B Q - H_A\|_F^2$ in closed form via SVD:
     $$U \Sigma V^\top = \text{SVD}(H_B^\top H_A), \qquad Q = U V^\top \in O(d)$$
   * Multiply $Q$ directly into Model A's head/circuit weights and adjust the bias for the anchor mean shift $(\mu_A, \mu_B)$:
     $$W_B^{\text{grafted}} = W_A Q^\top, \qquad b_B^{\text{grafted}} = b_A + W_A \mu_A - W_B^{\text{grafted}} \mu_B$$
2. **Representation Rerouting ("Circuit Breakers") & RMU Unlearning ([Zou et al., NeurIPS 2024](https://arxiv.org/abs/2406.04313); [Li et al., ICML 2024 - *WMDP/RMU*](https://arxiv.org/abs/2403.03218)):**
   * When erasing an undesirable capability or failure state, instead of merely zeroing out the direction, you can **reroute** columns $j \in \mathcal{J}_{\text{target}}$ of $W_{\text{down}}$ that fire on the unwanted state so they write directly into the `<EOS>` / refusal / target-class subspace $v_{\text{reroute}}$:
     $$w_{\text{down}, j} \longleftarrow w_{\text{down}, j} - \langle u_{\text{bad}}, w_{\text{down}, j}\rangle u_{\text{bad}} + \gamma \langle u_{\text{bad}}, w_{\text{down}, j}\rangle v_{\text{reroute}}$$

---

