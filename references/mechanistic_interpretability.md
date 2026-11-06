# Weight-Space Programming & Mechanistic Interpretability: Reference Manual

> **Purpose:** A comprehensive technical reference covering **Weight-Space Programming** ([Takehi et al., Nov 2026](https://rikiyat.github.io/blog/weight-space-programming/)), state-of-the-art **Mechanistic Interpretability (2022–2026)**, **Closed-Form Weight Surgery**, and a concrete blueprint for directly programming a **"Can Continue? (Lazy AI Stop)"** encoder classifier without gradient-descent training.

---

## Table of Contents
1. [Sparks of Weight-Space Programming (Takehi et al., 2026)](#1-sparks-of-weight-space-programming-takehi-et-al-2026)
   - [1.1 Core Paradigm Shift](#11-core-paradigm-shift)
   - [1.2 Complete Breakdown of All 8 Experiments](#12-complete-breakdown-of-all-8-experiments)
   - [1.3 Deep-Dive: The Mathematics of Experiment 8 (`Qwen3.5-0.8B` Surgery)](#13-deep-dive-the-mathematics-of-experiment-8-qwen35-08b-surgery)
   - [1.4 Failure Modes & Boundaries: Exp 7 (Sycophancy) vs. Exp 8b (`Gemma 3 270M`)](#14-failure-modes--boundaries-exp-7-sycophancy-vs-exp-8b-gemma-3-270m)
2. [Mathematical Anatomy of Transformer Weights as Editable Code](#2-mathematical-anatomy-of-transformer-weights-as-editable-code)
   - [2.1 The Residual Stream as a Shared Read/Write Bus](#21-the-residual-stream-as-a-shared-readwrite-bus)
   - [2.2 Attention Heads as Bilinear QK and OV Circuits (MHA, GQA, & Fused QKV)](#22-attention-heads-as-bilinear-qk-and-ov-circuits-mha-gqa--fused-qkv)
   - [2.3 SwiGLU / GeGLU MLPs as Key-Value Lookup Tables](#23-swiglu--geglu-mlps-as-key-value-lookup-tables)
   - [2.4 Static Weight-Space Logit Lens & SVD Spectrum Decoding (Zero-Inference Auditing)](#24-static-weight-space-logit-lens--svd-spectrum-decoding-zero-inference-auditing)
   - [2.5 Bilinear MLP Eigendecomposition (Static Quadratic Feature Reading)](#25-bilinear-mlp-eigendecomposition-static-quadratic-feature-reading)
   - [2.6 Weight-Space Gauge Symmetries & Function-Preserving Invariances](#26-weight-space-gauge-symmetries--function-preserving-invariances)
   - [2.7 RoPE Frequency Partitioning, Dark Signals ($\text{Null}(W_U)$), & Attention Sink Preservation](#27-rope-frequency-partitioning-dark-signals-textnullw_u--attention-sink-preservation)
3. [State-of-the-Art Mechanistic Interpretability & Weight-Editing Techniques](#3-state-of-the-art-mechanistic-interpretability--weight-editing-techniques)
   - [3.1 Column-Selective `down_proj` Subspace Surgery (Takehi et al., 2026)](#31-column-selective-down_proj-subspace-surgery-takehi-et-al-2026)
   - [3.2 Directional Ablation ("Abliteration"), Polyhedral Concept Cones, & Additive Steering Folding (Arditi et al., 2024; Wollschläger et al., 2025)](#32-directional-ablation-abliteration-polyhedral-concept-cones--additive-steering-folding-arditi-et-al-2024-wollschläger-et-al-2025)
   - [3.3 LEACE: Least-Squares Concept Erasure in Weight Space (Belrose et al., 2023)](#33-leace-least-squares-concept-erasure-in-weight-space-belrose-et-al-2023)
   - [3.4 Closed-Form Key-Value Surgery: ROME, MEMIT, and AlphaEdit (2022–2025)](#34-closed-form-key-value-surgery-rome-memit-and-alphaedit-20222025)
   - [3.5 Cross-Layer Transcoders (CLTs), Crosscoders, Attribution Graphs, and Native Feature Grafting (2024–2026)](#35-cross-layer-transcoders-clts-crosscoders-attribution-graphs-and-native-feature-grafting-20242026)
   - [3.6 Fast Causal Localization: Attribution Patching (AtP*) & EAP-IG](#36-fast-causal-localization-attribution-patching-atp--eap-ig)
   - [3.7 Function Vectors (FVs), Weight-Space Task Arithmetic, & Permutation Alignment](#37-function-vectors-fvs-weight-space-task-arithmetic--permutation-alignment)
   - [3.8 Closed-Form Classifier Head Synthesis (Shrinkage LDA & NLI Grafting)](#38-closed-form-classifier-head-synthesis-shrinkage-lda--nli-grafting)
   - [3.9 Geometry of Truth & Mass-Mean Probing vs. Logistic Regression (Marks & Tegmark, 2024)](#39-geometry-of-truth--mass-mean-probing-vs-logistic-regression-marks--tegmark-2024)
   - [3.10 Dormant Neuron Hijacking: Programming Exact Polarized GeGLU/SwiGLU Logic Gates ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$)](#310-dormant-neuron-hijacking-programming-exact-polarized-gegluswiglu-logic-gates-w_textgate-w_textup-w_textdown)
   - [3.11 Attention QK-Circuit Routing Surgery ($W_Q^h, W_K^h$ Rank-1 Rewiring Under RoPE & GQA)](#311-attention-qk-circuit-routing-surgery-w_qh-w_kh-rank-1-rewiring-under-rope--gqa)
   - [3.12 LoReFT & Distributed Alignment Search (DAS) Folded Into Weights (Wu et al., 2024)](#312-loreft--distributed-alignment-search-das-folded-into-weights-wu-et-al-2024)
   - [3.13 Weight Oracles, Natural Language Autoencoders (NLAs), & WorkspaceBench (2026)](#313-weight-oracles-natural-language-autoencoders-nlas--workspacebench-2026)
   - [3.14 Forward-Pass-Only Subspace Optimization: Combining Mechanistic Coordinates with MeZO, SPSA, ES, GEPA, & Reactive Optimizers](#314-forward-pass-only-subspace-optimization-combining-mechanistic-coordinates-with-mezo-spsa-es-gepa--reactive-optimizers)
   - [3.15 Constructive Weight Synthesis, RASP/Tracr Compilation, Looped Computers, & AdderBoard](#315-constructive-weight-synthesis-rasptracr-compilation-looped-computers--adderboard)
   - [3.16 Cross-Architecture Weight Grafting (Orthogonal Procrustes Stitching) & Circuit Breakers](#316-cross-architecture-weight-grafting-orthogonal-procrustes-stitching--circuit-breakers)
4. [Applied Blueprint: Weight-Space Programming the "Can Continue?" Classifier](#4-applied-blueprint-weight-space-programming-the-can-continue-classifier)
   - [4.1 Problem Decomposition & Confounder Analysis](#41-problem-decomposition--confounder-analysis)
   - [4.2 Mode A: Zero-Data / Zero-Forward-Pass Pure Static Weight Synthesis](#42-mode-a-zero-data--zero-forward-pass-pure-static-weight-synthesis)
   - [4.3 Mode B: 4-Stage Forward-Probed Weight-Space Pipeline (With PyTorch Code)](#43-mode-b-4-stage-forward-probed-weight-space-pipeline-with-pytorch-code)
5. [Agent Skill Runbook: Decision Tree & Execution Protocol](#5-agent-skill-runbook-decision-tree--execution-protocol)
   - [5.1 Task Triage Decision Tree](#51-task-triage-decision-tree)
   - [5.2 Mandatory Verification & Invariant Checklist](#52-mandatory-verification--invariant-checklist)
6. [Comprehensive Citations & Bibliography](#6-comprehensive-citations--bibliography)
   - [6.1 All 28 Official Citations from *Sparks of Weight-Space Programming* (Annotated)](#61-all-28-official-citations-from-sparks-of-weight-space-programming-annotated)
   - [6.2 Extended Mechanistic Interpretability, Weight Compilation, & Editing Canon (29–82)](#62-extended-mechanistic-interpretability-weight-compilation--editing-canon-2982)

---

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

## 2. Mathematical Anatomy of Transformer Weights as Editable Code

To program an encoder (like `ModernBERT`, `UltraBERT`, or `DeBERTa-v3`) or decoder directly in weight space, we rely on the **Mathematical Framework for Transformer Circuits** ([Elhage et al., 2021](https://transformer-circuits.pub/2021/framework/index.html)).

### 2.1 The Residual Stream as a Shared Read/Write Bus
A Transformer's hidden state $x_\ell \in \mathbb{R}^{N \times d_{\text{model}}}$ at layer $\ell$ is strictly the linear sum of the initial embedding and the outputs of every prior Attention head and MLP neuron:
$$x_L = x_0 + \sum_{\ell=1}^L \sum_{h=1}^H \text{Attn}_\ell^h(\text{LN}(x_{\ell-1})) + \sum_{\ell=1}^L \sum_{j=1}^{d_{\text{ff}}} a_{\ell, j}(\text{LN}(x'_{\ell-1})) \, w_{\text{down}, j}^{(\ell)}$$
Because every component **reads** via a linear projection from the residual stream and **writes** via a linear matrix addition back into the residual stream, **any linear operation you want to apply to activations can be folded directly and permanently into the weights**.

### 2.2 Attention Heads as Bilinear QK and OV Circuits (MHA, GQA, & Fused QKV)
Each query head $h \in \{0, \dots, H_Q - 1\}$ in layer $\ell$ reads via $W_Q^h \in \mathbb{R}^{d_{\text{head}} \times d_{\text{model}}}$, pairs with a key/value head $g(h) \in \{0, \dots, H_{KV} - 1\}$ ($g(h) = h$ in standard Multi-Head Attention; $g(h) = \lfloor h / G \rfloor$ with group size $G = H_Q / H_{KV}$ in **Grouped-Query Attention [GQA]**, [Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)), and writes via $W_O^h \in \mathbb{R}^{d_{\text{model}} \times d_{\text{head}}}$. They factor into two rank-$d_{\text{head}}$ bilinear circuits:
1. **The QK Circuit ($W_{QK}^h = (W_Q^h)^\top W_K^{g(h)} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$):** Determines *which source token $s$ a destination token $t$ attends to* (modulated by RoPE relative distance $R_{t-s}$):
   $$A_{t, s}^h = \text{softmax}_s \left( \frac{x_t^\top (W_Q^h)^\top R_{t-s} W_K^{g(h)} x_s}{\sqrt{d_{\text{head}}}} \right)$$
2. **The OV Circuit ($W_{OV}^h = W_O^h W_V^{g(h)} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$):** Determines *what vector gets written into destination token $t$'s residual stream* when it attends to token $s$:
   $$\text{Attn}^h(x)_t = \sum_{s} A_{t, s}^h \underbrace{(W_O^h W_V^{g(h)})}_{W_{OV}^h} x_s$$
   * **Weight-Space Diagnostic Trick (Used in Exp 3):** With input token embedding matrix $W_E \in \mathbb{R}^{V \times d_{\text{model}}}$ and unembedding matrix $W_U \in \mathbb{R}^{V \times d_{\text{model}}}$, if the top eigenvalues/diagonal entries of $W_U W_{OV}^h W_E^\top \in \mathbb{R}^{V \times V}$ (or $\text{Tr}(W_{OV}^h)$) are strongly positive, head $h$ is a **copying / induction head**; if strongly negative, it is an **anti-copying / suppression head**.
   * **Critical GQA & ModernBERT Structural Rules:**
     * **Never Edit Shared $W_K^{g(h)}$ or $W_V^{g(h)}$ in GQA Models (`Qwen3.5`, `Llama-3`, `Gemma 3`):** Editing a shared KV head simultaneously mutates all $G$ sibling query heads in that group. Always apply QK routing surgery to the **private query block $W_Q^h$** and OV surgery to the **private output block $W_O^h$**.
     * **Fused `Wqkv` & Alternating Local/Global Attention in `ModernBERT`:** In `ModernBERT`, $W_Q, W_K, W_V$ are concatenated row-wise inside `layer.attn.Wqkv.weight` (`[3 * d_model, d_model]`), and only every 3rd layer (`layer_idx % 3 == 0`: layers `0, 3, 6, ..., 27`) uses full global attention (the other layers use a 128-token local sliding window where `[CLS]` cannot see the tail of long sequences).

### 2.3 SwiGLU / GeGLU MLPs as Key-Value Lookup Tables
Following [Geva et al. (2021)](https://arxiv.org/abs/2012.14913) and modern Gated Linear Unit extensions:
$$\text{MLP}(x) = W_{\text{down}} \big( \sigma(W_{\text{gate}} x) \odot (W_{\text{up}} x) \big) = \sum_{j=1}^{d_{\text{ff}}} \underbrace{\big[ \sigma(\langle w_{\text{gate}, j}, x \rangle) \cdot \langle w_{\text{up}, j}, x \rangle \big]}_{\text{Key Detector } a_j(x) \in \mathbb{R}} \, \underbrace{w_{\text{down}, j}}_{\text{Value Vector } \in \mathbb{R}^{d_{\text{model}}}}$$
* **Rows of $W_{\text{gate}}$ and $W_{\text{up}}$** act as **pattern detectors (keys)** over the residual stream. (In `ModernBertMLP`, `input, gate = self.Wi(x).chunk(2, dim=-1)` applies `GELU` to the first half `Wi.weight[:d_ff]` and multiplies by the linear second half `Wi.weight[d_ff:]`).
* **Columns of $W_{\text{down}}$ (`mlp.Wo` in `ModernBERT`)** act as **output vectors (values)** written into the residual stream when detector $j$ fires.

### 2.4 Static Weight-Space Logit Lens & SVD Spectrum Decoding (Zero-Inference Auditing)
If your model has an unembedding matrix $W_U \in \mathbb{R}^{V \times d_{\text{model}}}$ (or a classification head $W_{\text{cls}} \in \mathbb{R}^{C \times d_{\text{model}}}$), you can read what *any* MLP column $j$ or attention head $h$ promotes **without running a single forward pass**:
* **Exact Pre-Norm (`RMSNorm` vs. `LayerNorm`) Weight Folding:**
  * **For `RMSNorm` (`Qwen3`, `Llama-3`, `Gemma`):** Absorb the learned scale vector $\gamma \in \mathbb{R}^d$ directly into the downstream linear matrix $W_{\text{in}}$ (exact up to the positive scalar $1/\text{RMS}(x)$):
    $$W_{\text{in}}^{\text{folded}} = W_{\text{in}} \, \text{diag}(\gamma)$$
  * **For `LayerNorm` (`ModernBERT`, `DeBERTa`, `RoBERTa`):** `LayerNorm` projects out the all-ones mean direction $\mathbf{1} \in \mathbb{R}^d$ via the centering projector $P_{\mathbf{1}^\perp} = I_d - \frac{1}{d}\mathbf{1}\mathbf{1}^\top$ before scaling by $\gamma$ and shifting by $\beta$ ([Elhage et al., 2021](https://transformer-circuits.pub/2021/framework/index.html)):
    $$W_{\text{in}}^{\text{folded}} = W_{\text{in}} \, \text{diag}(\gamma) \left(I_d - \frac{1}{d}\mathbf{1}\mathbf{1}^\top\right), \qquad b_{\text{in}}^{\text{folded}} = b_{\text{in}} + W_{\text{in}} \beta$$
    *(Note: In `ModernBERT`, `layers[0].attn_norm` is `nn.Identity()`—the initial embedding `LayerNorm` lives at `model.model.embeddings.norm`—and write matrices $W_{\text{down}}, W_O$ can be zero-mean standardized via $W_{\text{down}} \leftarrow P_{\mathbf{1}^\perp} W_{\text{down}}$ since $\mathbf{1}$ lies in the null space of all downstream `LayerNorm`s).*
* **Accounting for Nonlinear Prediction Heads (`model.head` in `ModernBERT`):**
  In `ModernBertForSequenceClassification`, the final residual state passes through `final_norm` (`LayerNorm`) and `model.head` (`Linear -> GELU -> LayerNorm`) before reaching `model.classifier` ($W_{\text{cls}}$). Therefore, when computing static attribution of an internal residual vector $w \in \mathbb{R}^{d_{\text{model}}}$ onto $W_{\text{cls}}$, either map through the head ($z = W_{\text{cls}} \, \text{head}(\text{final\_norm}(w))$) or use the linearized effective readout $W_{\text{eff}} = W_{\text{cls}} \text{diag}(\gamma_{\text{head}}) P_{\mathbf{1}^\perp} W_{\text{head, dense}} \text{diag}(\gamma_{\text{final}}) P_{\mathbf{1}^\perp}$.
* **MLP Column $j$ Direct Logit Effect:** $z_j = W_{\text{eff}} \, w_{\text{down}, j}^{(\ell)} \in \mathbb{R}^C$ (or $W_U \, \text{diag}(\gamma_{\text{final}}) w_{\text{down}, j}^{(\ell)} \in \mathbb{R}^V$).
* **Attention Head $h$ Direct Input-to-Output Map:** $M^h = W_{\text{eff}} W_O^h W_V^{g(h)} W_E^\top \in \mathbb{R}^{C \times V}$, which tells you directly how much attending to vocabulary token $v \in V$ pushes the classifier toward class $c \in C$.
* **Singular Value Decomposition (SVD) of $W_{OV}^h$ and $W_{\text{down}}^{(\ell)}$ ([Millidge & Black, 2022](https://arxiv.org/abs/2204.09722); [Bereska & Gavves, 2024](https://arxiv.org/abs/2404.05107)):**
  Decomposing $W_{OV}^h = U \Sigma V^\top$ yields orthogonal rank-1 semantic channels:
  * **Right Singular Vectors ($V^\top W_E^\top \in \mathbb{R}^{r \times V}$):** The exact input token subspace that triggers the head.
  * **Left Singular Vectors ($W_U U \Sigma$ or $W_{\text{eff}} U \Sigma$):** The exact output concept/class subspace written into the residual stream when triggered.

### 2.5 Bilinear MLP Eigendecomposition (Static Quadratic Feature Reading)
In modern Gated Linear Unit MLPs ($\text{SwiGLU} / \text{GeGLU}$), each intermediate neuron $j$ computes the elementwise product of two projections: $\sigma(w_{\text{gate}, j}^\top x) (w_{\text{up}, j}^\top x)$. Fixing any target output residual direction $u \in \mathbb{R}^{d_{\text{model}}}$ defines the diagonal output-weighting matrix $D_u = \text{diag}(W_{\text{down}}^\top u) \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{ff}}}$. Because both $\text{SiLU}(z) = z \sigma_{\text{logistic}}(z)$ and $\text{GELU}(z) = z \Phi(z)$ satisfy the exact global identity $\sigma(z) - \sigma(-z) \equiv z$ ($\implies \sigma(z) = \frac{1}{2}z + \sigma_{\text{even}}(z)$), the even part of $\langle u, \text{MLP}(x)\rangle$ is **globally and identically** a **symmetric bilinear form** over pre-normed $x$ (for bias-free projections `bias=False`; [Sharkey et al., 2025](https://arxiv.org/abs/2501.06126); [Pearce et al., 2024](https://arxiv.org/abs/2406.03897)):
$$\frac{1}{2}\Big(\langle u, \, \text{MLP}(+x) \rangle + \langle u, \, \text{MLP}(-x) \rangle\Big) \equiv c_\sigma \, x^\top B_u x, \quad \text{where } B_u = \frac{1}{2}\Big(W_{\text{gate}}^\top D_u W_{\text{up}} + W_{\text{up}}^\top D_u W_{\text{gate}}\Big) = \frac{1}{2} \sum_{j=1}^{d_{\text{ff}}} \langle u, \, w_{\text{down}, j} \rangle \Big( w_{\text{gate}, j} w_{\text{up}, j}^\top + w_{\text{up}, j} w_{\text{gate}, j}^\top \Big)$$
*(where $c_\sigma = 1$ for pure Bilinear MLPs and $c_\sigma = \sigma'(0) = \frac{1}{2}$ globally for the quadratic even component of $\text{SiLU}/\text{GELU}$).*
* **Zero-Forward-Pass Eigendecomposition & Sign Disambiguation:** Taking the eigendecomposition $B_u = \sum_k \lambda_k v_k v_k^\top$ statically in weight space reveals the **AND-gate feature combinations** ($\lambda_k > 0$) that promote concept $u$. Because $x^\top B_u x = (-x)^\top B_u (-x)$ is even while $\text{SiLU}/\text{GELU}$ is rectified ($\sigma(-z) \approx 0$ for $z \gg 0$), always resolve the sign of each eigenvector $\pm v_k$ by checking actual gated polarity:
  $$s_k = \text{sign}\Big(\langle u, \text{MLP}(+v_k)\rangle - \langle u, \text{MLP}(-v_k)\rangle\Big), \qquad \tilde{v}_k = s_k v_k$$

### 2.6 Weight-Space Gauge Symmetries & Function-Preserving Invariances
When reading or writing weights, you must account for the exact **gauge symmetries** under which two different weight tensors compute the **exact same mathematical function** $f_\theta(x)$ ([Ainsworth et al., ICLR 2023 - *Git Re-Basin*](https://arxiv.org/abs/2209.04836); [Navon et al., ICML 2023](https://arxiv.org/abs/2301.12780)):
1. **MLP Intermediate Neuron Permutation ($\Pi \in S_{d_{\text{ff}}}$):**
   Permuting the rows of $W_{\text{gate}}, W_{\text{up}}$ and the columns of $W_{\text{down}}$ by the same permutation matrix $\Pi$ leaves $\text{MLP}(x)$ strictly identical:
   $$W_{\text{gate}} \to \Pi W_{\text{gate}}, \quad W_{\text{up}} \to \Pi W_{\text{up}}, \quad W_{\text{down}} \to W_{\text{down}} \Pi^\top$$
   * *Consequence:* Index $j$ of an MLP neuron has no intrinsic meaning across different models; always index MLP columns by their **functional projection** ($\langle u, w_{\text{down}, j}\rangle$ or SVD singular vectors) rather than raw index $j$.
2. **MLP Positive Scaling Gauge ($\alpha_j > 0$) & Activation-Weighted Salience (`Wanda`):**
   For homogeneous/linear branches, scaling row $j$ of $W_{\text{up}}$ by $\alpha_j$ and column $j$ of $W_{\text{down}}$ by $1/\alpha_j$ preserves the exact output while changing column norms. In pure static weight space, always rank neuron dormancy using the **gauge-invariant triple product norm** $\|w_{\text{gate}, j}\|_2 \|w_{\text{up}, j}\|_2 \|w_{\text{down}, j}\|_2$; when 1 calibration pass is available, weight by empirical activation norm $\|w_{\text{down}, j}\|_2 \|a_j(X)\|_2$ (**Wanda** salience, [Sun et al., ICLR 2024](https://arxiv.org/abs/2306.11695)).
3. **Intra-Head OV Basis Rotation ($M \in \text{GL}(d_{\text{head}})$):**
   For any invertible $d_{\text{head}} \times d_{\text{head}}$ matrix $M$, transforming $W_V^h \to M W_V^h$ and $W_O^h \to W_O^h M^{-1}$ leaves $W_{OV}^h = W_O^h W_V^h$ strictly unchanged.
4. **Intra-Head QK Basis Rotation (RoPE-Commuting $M$):**
   For any matrix $M$ that commutes with the block-diagonal 2D RoPE rotation matrices $R_{t-s}$, transforming $W_Q^h \to M^{-\top} W_Q^h$ and $W_K^h \to M W_K^h$ leaves all attention logits unchanged.

### 2.7 RoPE Frequency Partitioning, Dark Signals ($\text{Null}(W_U)$), & Attention Sink Preservation
Modern Transformers (`ModernBERT`, `UltraBERT`, `Qwen3.5`, `Llama-3`, `Gemma`) use **Rotary Position Embeddings (RoPE)** and often **zero bias parameters (`bias=False`)**. Three structural rules govern their weight-space programming:

1. **Programming Positional vs. Semantic Heads Under RoPE ([Barbero et al. / DeepMind, ICLR 2025 - *Round and Round We Go!*](https://arxiv.org/abs/2410.06205)):**
   RoPE splits $d_{\text{head}}$ into $d_{\text{head}}/2$ 2D planes indexed by $m \in \{0, 1, \dots, d_{\text{head}}/2 - 1\}$ with frequency $\theta_m = \text{base}^{-2m/d_{\text{head}}}$.
   * **Split-Half (`rotate_half`) vs. Interleaved 2D Plane Row Indices ($\mathcal{I}_m$):** In HuggingFace `ModernBERT`, `Qwen3.5`, `Llama-3`, and `Gemma 3`, `rotate_half` pairs row $m$ with row $m + d_{\text{head}}/2$, so plane $m$'s 2-row index set is $\mathcal{I}_m = [m, \, m + d_{\text{head}}/2]$ (whereas in GPT-J / interleaved RoPE implementations, $\mathcal{I}_m = [2m, \, 2m+1]$).
   * **To Program a Purely Semantic (Distance-Invariant) Attention Head:** Place all nonzero weights of $W_Q^h$ and $W_K^h$ into the **lowest-frequency 2D planes** $\mathcal{I}_m$ ($m \approx d_{\text{head}}/2 - 1$, where $\theta_m \to 0$ so $R_{(t-s)\theta_m} \approx I_2$).
   * **To Program an Exact Relative-Offset Head (e.g., Attend Strictly $k$ Tokens Back, $t - s = k$):** In a high-frequency plane $m$, set the $2\times d_{\text{model}}$ slice $W_{Q, \mathcal{I}_m}^h$ to be rotated by $-k\theta_m$ relative to $W_{K, \mathcal{I}_m}^h$:
     $$W_{Q, \mathcal{I}_m}^h = \begin{bmatrix} \cos(k\theta_m) & \sin(k\theta_m) \\ -\sin(k\theta_m) & \cos(k\theta_m) \end{bmatrix} W_{K, \mathcal{I}_m}^h$$
     Then the attention score is proportional to $\cos((t - s - k)\theta_m)$, which peaks uniquely at relative distance $t - s = k$! (Furthermore, in models with QKV biases like `Qwen2.5`/`Qwen3`, editing only the $2 d_{\text{head}}$ floats of $(b_Q^h, b_K^h)$ creates a content-independent Fourier attention window $b_Q^{h\top} R_{t-s} b_K^h = \sum_m \|b_{Q,m}\|_2 \|b_{K,m}\|_2 \cos((t-s)\theta_m - \phi_m)$).

2. **Emulating Bias Terms (`+ b`) in Bias-Free (`bias=False`) Models via Dark Signals & Massive Activations ([Cancedda, ACL 2024 - *Dark Signals*](https://arxiv.org/abs/2403.00504); [Sun et al., COLM 2024 - *Massive Activations*](https://arxiv.org/abs/2402.17762)):**
   * **The Dark Signal Subspace ($\mathcal{U}_{\text{dark}}$):** Take the SVD of the unembedding/head matrix $W_U = U \Sigma V^\top$. The right singular vectors $v_i$ with near-zero singular values ($\sigma_i \approx 0$) form the **Dark Subspace** $\mathcal{U}_{\text{dark}} = \text{Null}(W_U)$.
   * **Token-0 (`[CLS]`/`<BOS>`) vs. All-Token Position-Wise MLP Biases:** Pre-trained models maintain a high-magnitude **Massive Activation / Sink vector** $s_{\text{sink}} \in \mathcal{U}_{\text{dark}}$ localized on **token 0** (`<BOS>`/`[CLS]`). For `[CLS]`-pooled classifiers or attention heads attending to token 0, adding $\frac{b_j}{\|s_{\text{sink}}\|_2^2} s_{\text{sink}}$ emulates a scalar bias $b_j$. For position-wise MLP firing across all tokens $t > 0$, use the all-token post-`mlp_norm` mean vector $\bar{z}_{\text{norm}} = \mathbb{E}_{x, t}[\text{mlp\_norm}(x_t)]$ (whose coordinates are non-zero wherever $\gamma_i$ is asymmetric) or use the bias-free quadratic GeGLU gate $\psi(u) = \text{GELU}(u)\cdot u$ (§3.10).

3. **Preserving Attention Sinks & Massive Activations During Weight Surgery ([Xiao et al., ICLR 2024 - *StreamingLLM*](https://arxiv.org/abs/2309.17453); [Sun et al., 2024](https://arxiv.org/abs/2402.17762)):**
   * Early MLP layers (typically Layers 1–3) write $s_{\text{sink}}$ ($10^2\text{–}10^4\times$ larger than typical coordinates) into the residual stream on `<BOS>` / `[CLS]` to serve as a no-op attention denominator sink.
   * **Surgical Guardrail:** If an empirical contrast vector $u = \frac{\mu_+ - \mu_-}{\|\mu_+ - \mu_-\|_2}$ has even a $0.01$ accidental cosine overlap with the residual stream at token 0 ($x_{\ell, 0}$), applying global orthogonal projection $P_\perp = I - uu^\top$ across early layers will attenuate $\|s_{\text{sink}}\|_2$ and collapse attention routing across the entire model. Always orthogonalize candidate edit directions against the residual-stream `<BOS>`/`[CLS]` direction $\hat{s}_{\text{sink}} = x_{\ell, 0} / \|x_{\ell, 0}\|_2$ ($u \leftarrow u - \langle u, \hat{s}_{\text{sink}}\rangle \hat{s}_{\text{sink}}$) and restrict MLP/OV surgery to middle-to-late layers ($\ell \in [0.4L, 0.85L]$).

---

## 3. State-of-the-Art Mechanistic Interpretability & Weight-Editing Techniques

Below are the core mathematical techniques used for training-free weight-space programming, ordered from fastest surgical interventions to full circuit-replacement methods.

### 3.1 Column-Selective `down_proj` Subspace Surgery (Takehi et al., 2026)
Unlike global steering vectors (which affect every input indiscriminately) or full-matrix projection (which alters all $d_{\text{ff}}$ neurons), Column-Selective Surgery only edits the subset of MLP columns $\mathcal{J}_K$ that both (a) fire differentially on the target state and (b) positively contribute along the target concept vector $u$:

$$\begin{aligned}
\text{1. Target Concept Vector:} \quad & u = \frac{\mu_+ - \mu_-}{\|\mu_+ - \mu_-\|_2} \in \mathbb{R}^{d_{\text{model}}} \quad (\text{orthogonalized against } \hat{s}_{\text{sink}}) \\
\text{2. Signed Column Attribution:} \quad & s_j = (\bar{a}_j^+ - \bar{a}_j^-) \cdot (u^\top w_{\text{down}, j}) \\
\text{3. Selective Rank-1 Edit:} \quad & w_{\text{down}, j}^{\text{new}} = w_{\text{down}, j} + \beta \, (u^\top w_{\text{down}, j}) \, u \quad \forall j \in \text{Top-}K(s_j \text{ with } s_j > 0)
\end{aligned}$$
* *Important Sign Rule:* Always select $\text{Top-}K(s_j)$ among columns with **positive** signed attribution $s_j > 0$ (never $|s_j|$!), because columns with $s_j < 0$ actively oppose $+u$ on positive examples and would be modified in the wrong direction.
* Set $\beta = -1$ to **erase/suppress** a pathological attractor (e.g., looping or false triggers on polite sign-offs); set $\beta > 0$ to **amplify** a latent feature circuit before it hits the classification head.

---

### 3.2 Directional Ablation ("Abliteration"), Polyhedral Concept Cones, & Additive Steering Folding (Arditi et al., 2024; Wollschläger et al., 2025)
Introduced in *"Refusal in Language Models Is Mediated by a Single Direction"* ([Arditi et al., 2024](https://arxiv.org/abs/2406.11717)) and generalized to multi-ray **Polyhedral Concept Cones** $\mathcal{C} = \{\sum_{r=1}^k \lambda_r c_r : \lambda_r \ge 0\}$ ([Wollschläger et al., ICML 2025](https://arxiv.org/abs/2502.17420)), this method permanently removes a 1D or low-rank $k$-D subspace $U = \text{QR}([c_1, \dots, c_k]) \in \mathbb{R}^{d_{\text{model}} \times k}$ ($U^\top U = I_k$, orthogonalized against $\hat{s}_{\text{sink}}$) from the residual stream by orthogonalizing every matrix that **writes** to the residual stream:
$$P_\perp = I_{d_{\text{model}}} - U U^\top$$
For every layer $\ell \in \mathcal{L}_{\text{target}}$:
$$W_O^{(\ell)} \longleftarrow P_\perp W_O^{(\ell)}, \qquad W_{\text{down}}^{(\ell)} \longleftarrow P_\perp W_{\text{down}}^{(\ell)}$$
* **Zero Runtime Cost:** Because $P_\perp$ is multiplied into $W_O^{(\ell)}$ and $W_{\text{down}}^{(\ell)}$ offline, the edited model has the exact same architecture, parameter count, and inference speed as the original model, while being mathematically incapable of writing into subspace $U$.
* **Norm-Preserving Variant (Householder / Magnitude Rescaling):** To prevent variance decay in LayerNorm/RMSNorm when projecting out strong directions, rescale updated columns to preserve their original Frobenius/column $L_2$ norm:
  $$w_j^{\text{new}} = \|w_j\|_2 \cdot \frac{P_\perp w_j}{\|P_\perp w_j\|_2 + \epsilon}$$
* **Folding Additive Activation Steering (`RepE` / `CAA`) Directly Into Weights ([Zou et al., 2023](https://arxiv.org/abs/2310.01405); [Rimsky et al., ACL 2024](https://arxiv.org/abs/2312.06681)):**
  To permanently apply an additive steering shift $h_\ell \leftarrow h_\ell + \alpha u_\ell$ with zero runtime hooks:
  1. *If `mlp.down_proj` (`Wo`) has a bias parameter:* Set $b_{\text{down}}^{(\ell)} \leftarrow b_{\text{down}}^{(\ell)} + \alpha u_\ell$ (shifts every token position $t$ uniformly).
  2. *If `bias=False` throughout the model:*
     * **For `[CLS]`-Pooled Encoders (`classifier_pooling == "cls"`):** Route token 0's intermediate (pre-`down_proj`) MLP activation $a_{\text{sink}} = a_{\ell, 0} \in \mathbb{R}^{d_{\text{ff}}}$ through $\Delta W_{\text{down}}^{(\ell)} = \frac{\alpha}{\|a_{\text{sink}}\|_2^2} u_\ell a_{\text{sink}}^\top$ (shifts token 0's `[CLS]` state by $+\alpha u_\ell$).
     * **For All-Token Steering (`mean`-Pooled Encoders or Causal Decoders like `Qwen`/`Llama`):** Because token 0's MLP sink $a_{\ell, 0}$ is position-local and does not fire at tokens $t > 0$, either (a) route through an attention-sink head $h$ that attends to token 0 (`<BOS>`/`[CLS]`) with mean weight $\bar{A}_{\bullet, 0}^h$ via $\Delta W_O^{(\ell), h} = \frac{\alpha}{\bar{A}_{\bullet, 0}^h \|v_{\text{sink}}^h\|_2^2} u_\ell (v_{\text{sink}}^h)^\top$, or (b) use the all-token mean intermediate MLP activation $\bar{a}_\ell = \mathbb{E}_{x, t}[a_{\ell, t}] \in \mathbb{R}^{d_{\text{ff}}}$.

---

### 3.3 LEACE: Least-Squares Concept Erasure in Weight Space (Belrose et al., 2023)
Standard orthogonal projection ($I - uu^\top$) only removes the mean difference vector, leaving correlated off-axis variance intact. **LEACE** ([Belrose et al., 2023](https://arxiv.org/abs/2306.03819)) derives the exact closed-form **oblique projection** that zeroes out all linear cross-covariance $\Sigma_{XZ}$ between hidden states $X \in \mathbb{R}^{N \times d}$ and a nuisance concept $Z \in \mathbb{R}^{N \times k}$ (e.g., surface politeness, text length, or code-vs-prose formatting) while minimizing mean-squared distortion $\mathbb{E}[\|\tilde{x} - x\|_2^2]$:

1. Compute sample covariance $\Sigma_{XX} = \text{Cov}(X, X) \in \mathbb{R}^{d \times d}$ and cross-covariance $\Sigma_{XZ} = \text{Cov}(X, Z) \in \mathbb{R}^{d \times k}$.
2. Compute the **whitening matrix** $W = (\Sigma_{XX}^+)^{1/2} = \Sigma_{XX}^{-1/2}$ (so $\text{Cov}(WX) = I_d$) and its **unwhitening pseudoinverse** $W^+ = \Sigma_{XX}^{1/2}$.
3. Let $P_V$ be the orthogonal projection onto the column space of the whitened cross-covariance $V = W \Sigma_{XZ} = \Sigma_{XX}^{-1/2} \Sigma_{XZ}$ (computed stably via thin SVD $V = U_V \Sigma_V V_V^\top$ so $P_V = U_V U_V^\top$ without squaring the condition number $\kappa(V)^2$):
   $$V = W \Sigma_{XZ}, \qquad P_V = V (V^\top V)^\dagger V^\top = U_V U_V^\top$$
4. The **LEACE Oblique Erasure Projector** ($P_{\text{LEACE}}^2 = P_{\text{LEACE}}$) and mean-preserving bias shift are:
   $$P_{\text{LEACE}} = I_d - W^+ P_V W = I_d - \Sigma_{XX}^{1/2} P_V \Sigma_{XX}^{-1/2}, \qquad b_{\text{LEACE}} = (I_d - P_{\text{LEACE}}) \mathbb{E}[X]$$
5. **Exact Weight-Space Folding (Crucial Architectural Rule):**
   * Because $P_{\text{LEACE}}$ is an **oblique** projector tailored to the covariance $\Sigma_{XX}$ at the exact point where $X$ was extracted, you **must fold $P_{\text{LEACE}}$ and $b_{\text{LEACE}}$ into a linear layer that reads the complete representation $X$**—specifically the linear classification head (`model.classifier`: $W_{\text{cls}} \leftarrow W_{\text{cls}} P_{\text{LEACE}}$, $b_{\text{cls}} \leftarrow b_{\text{cls}} + W_{\text{cls}}^{\text{orig}} b_{\text{LEACE}}$) or `model.head.dense` if $X$ is pooled immediately after `final_norm`.
   * *Pitfall to Avoid:* Never fold $P_{\text{LEACE}}$ only into the final encoder layer's `mlp.Wo` / `attn.Wo` expecting it to erase $Z$ from $x_L = x_{L-1} + \text{Attn}_L(x_{L-1}) + \text{MLP}_L(x'_{L-1})$: the residual skip connection $x_{L-1}$ bypasses Layer $L$'s output matrices, and subsequent nonlinear layers (`final_norm` + `model.head` GELU) break linear guardedness.
   * **Theorem (Belrose et al., 2023):** After folding $P_{\text{LEACE}}$ into the readout head, *every possible linear classifier* has zero covariance with the nuisance concept $Z$, while preserving maximum variance along all task-relevant directions.

---

### 3.4 Closed-Form Key-Value Surgery: ROME, MEMIT, and AlphaEdit (2022–2025)
When you want to overwrite how specific input patterns $K_1 = [k_1, \dots, k_m] \in \mathbb{R}^{d_{\text{ff}} \times m}$ (intermediate MLP activations) map to target residual outputs $V_1 = [v_1, \dots, v_m] \in \mathbb{R}^{d_{\text{model}} \times m}$ inside $W_{\text{down}}$:

1. **ROME (Rank-One Model Editing — [Meng et al., 2022](https://arxiv.org/abs/2202.05262)):**
   Solves the constrained least-squares problem $\min_{\hat{W}} \|\hat{W} K_0 - W K_0\|_F^2$ subject to $\hat{W} k_* = v_*$, yielding the exact rank-1 weight update:
   $$\hat{W}_{\text{down}} = W_{\text{down}} + \frac{(v_* - W_{\text{down}} k_*) (C_0^{-1} k_*)^\top}{k_*^\top C_0^{-1} k_*}, \quad \text{where } C_0 = K_0 K_0^\top$$
2. **MEMIT (Mass-Editing Memory in a Transformer — [Meng et al., 2023](https://arxiv.org/abs/2210.07229)):**
   Generalizes ROME to batch-insert $m$ key-value mappings distributed across layers $\ell \in \mathcal{R}$:
   $$\Delta W_{\text{down}}^{(\ell)} = R^{(\ell)} K_1^\top (C_0 + K_1 K_1^\top)^{-1}$$
3. **AlphaEdit (Null-Space Constrained Weight Editing — [Fang et al., ICLR 2025](https://arxiv.org/abs/2410.02355)):**
   Instead of merely penalizing distortion on preserved inputs $K_0$ via $(C_0 + K_1 K_1^\top)^{-1}$, **AlphaEdit** computes the SVD of the preserved activation covariance $C_0 = U \Lambda U^\top$, extracts the eigenvectors $U_{\text{null}}$ corresponding to small/zero singular values, and forms the symmetric **Null-Space Projection Matrix** $P_{\text{null}} = U_{\text{null}} U_{\text{null}}^\top \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{ff}}}$ (so $P_{\text{null}} K_0 = 0$):
   $$\Delta W_{\text{down}} = (V_1 - W_{\text{down}} K_1) K_1^\top P_{\text{null}} \big(K_p K_p^\top P_{\text{null}} + K_1 K_1^\top P_{\text{null}} + \lambda I_{d_{\text{ff}}}\big)^{-1}$$
   where $K_p$ optionally stores keys from previous sequential editing rounds.
   * **Fast & Exact $m \times m$ SPD Woodbury Formulation (For Single-Batch $m \ll d_{\text{ff}}$):**
     Instead of inverting a non-symmetric $d_{\text{ff}} \times d_{\text{ff}}$ ($5248 \times 5248$) matrix in `float32`, define the null-projected keys $\tilde{K}_1 = P_{\text{null}} K_1 \in \mathbb{R}^{d_{\text{ff}} \times m}$ and apply the push-through identity:
     $$\Delta W_{\text{down}} = (V_1 - W_{\text{down}} K_1) \big(\tilde{K}_1^\top \tilde{K}_1 + \lambda I_m\big)^{-1} \tilde{K}_1^\top, \qquad \tilde{K}_1 = P_{\text{null}} K_1$$
     This inverts only an $m \times m$ symmetric positive-definite matrix ($>10^5\times$ faster when $m \le 32$) and places $\tilde{K}_1^\top = K_1^\top P_{\text{null}}$ on the extreme right so $\Delta W_{\text{down}} K_0 = 0$ holds to exact machine precision!

---

### 3.5 Cross-Layer Transcoders (CLTs), Crosscoders, Attribution Graphs, and Native Feature Grafting (2024–2026)
Standard Sparse Autoencoders (SAEs) reconstruct a single layer's residual stream ($x_\ell \approx D_{\text{SAE}} f(E_{\text{SAE}} x_\ell)$), leaving the nonlinear MLP computation between layers as a black box.
* **MLP Transcoders ([Dunefsky et al., 2024](https://arxiv.org/abs/2406.11944)):** Train a sparse dictionary to approximate the input-output function of an MLP block directly: $\text{MLP}_\ell(x) \approx W_{\text{dec}} \, \text{JumpReLU}(W_{\text{enc}} x + b_{\text{enc}})$.
* **Cross-Layer Transcoders (CLTs) & Attribution Graphs ([Lindsey et al. / Anthropic, 2025](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)):**
  * CLTs read from layer $\ell$'s residual stream and write sparse features directly to **all downstream layers** $\ell' \ge \ell$.
  * Replacing MLPs with CLTs linearizes paths once attention patterns and active JumpReLU gates are fixed on a prompt, allowing **exact backward Jacobian tracing** from output logits back through intermediate features in a single pass (**Attribution Graphs**).
  * **Weight-Space Fold-Back & Direct Dormant-Neuron Compilation:** Once an Attribution Graph or SAE identifies a causal feature $f_i(x) = \sigma(w_{\text{enc}, i}^\top x + b_{\text{enc}, i}) w_{\text{dec}, i}$, you have two zero-overhead ways to fold it into the native MLP:
    1. *Aligned Native Columns:* Rank native $W_{\text{down}}$ columns by cosine similarity $\cos(w_{\text{down}, j}, w_{\text{dec}, i})$ and apply Column-Selective Surgery (§3.1).
    2. *Superposition Feature Compilation Into a Dormant Neuron:* If $w_{\text{dec}, i}$ is distributed in superposition across many columns, compile it directly into a dormant SwiGLU/GeGLU neuron $j \in \mathcal{D}$ (§3.10) by setting both $W_{\text{gate}}[j, :] \leftarrow w_{\text{enc}, i}^\top$ and $W_{\text{up}}[j, :] \leftarrow w_{\text{enc}, i}^\top$ (so the neuron computes the rectified one-sided activation $\text{GELU}(z)\cdot z \ge 0$ across all token positions $t$ without needing a bias parameter) and $W_{\text{down}}[:, j] \leftarrow \beta w_{\text{dec}, i}$.
* **Crosscoders & Dedicated-Feature (DSF) Crosscoders for Model Diffing ([Lindsey et al., 2024](https://transformer-circuits.pub/2024/crosscoders/index.html); 2025–2026 extensions):**
  * Train a shared sparse feature dictionary across multiple layers or across two distinct checkpoints (e.g., Base vs. NLI/Instruction-tuned) to isolate **model-exclusive features** vs. shared features.
  * 2026 **Dedicated Feature (DSF) Crosscoders** explicitly partition the dictionary into $(F_{\text{shared}}, F_{\text{model A}}, F_{\text{model B}})$ to overcome the "shared reconstruction prior" bias, making it straightforward to extract the exact decoder vectors $W_{\text{dec}}^{(B)}$ corresponding to newly acquired behaviors and graft them directly into another model's MLP weights.

---

### 3.6 Fast Causal Localization: Attribution Patching (AtP*) & EAP-IG
Before modifying weights in a multi-layer encoder, you need to know *which* layers and components $(\ell, h)$ or $(\ell, j)$ actually drive the target decision.
* Brute-force Activation Patching requires $O(L \cdot (H + d_{\text{ff}}))$ forward passes.
* **Attribution Patching ([Nanda, 2023](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching); [Syed et al., 2023](https://arxiv.org/abs/2310.10348); [Kramár et al., 2024 - AtP*](https://arxiv.org/abs/2403.00745)):** Approximates the causal effect of patching component output $z_c$ from a clean example $x^{\text{clean}}$ to a counterfactual example $x^{\text{cf}}$ using **just 2 forward passes and 1 backward pass** (used purely for reading/indexing, never for optimizer weight updates):
  $$\text{Attr}(c) = \big( z_c(x^{\text{cf}}) - z_c(x^{\text{clean}}) \big)^\top \frac{\partial \mathcal{M}(x^{\text{clean}})}{\partial z_c}$$
  where $\mathcal{M}(x) = \text{logit}(y_{\text{target}}) - \text{logit}(y_{\text{foil}})$ is the logit difference metric.
* **Zero-Gradient Alternative (Pure Forward-Pass Direct Logit Attribution — DLA):**
  If you want strictly **zero backward passes**, compute the Direct Logit Attribution of every component $c$ directly against the target direction $u_{\text{cls}} = w_{\text{cls}, y_1} - w_{\text{cls}, y_2}$:
  $$\text{DLA}(c) = \big\langle z_c(x^+) - z_c(x^-), \, u_{\text{cls}} \big\rangle$$

---

### 3.7 Function Vectors (FVs), Weight-Space Task Arithmetic, & Permutation Alignment
* **Function Vectors ([Todd et al., ICLR 2024](https://arxiv.org/abs/2310.15213)):** Shows that complex in-context classification rules are compressed by a small set of middle-layer attention heads $\mathcal{H}_{\text{FV}}$ into a single vector $v_{\text{FV}} = \sum_{h \in \mathcal{H}_{\text{FV}}} \bar{a}_h$. Adding $v_{\text{FV}}$ to the bias of a middle-layer `mlp.down_proj` permanently installs the task rule without needing few-shot examples at runtime!
* **Weight-Space Task Arithmetic ([Ilharco et al., 2023](https://arxiv.org/abs/2212.04089); TIES-Merging [Yadav et al., 2023](https://arxiv.org/abs/2306.01708); DARE [Yu et al., 2024](https://arxiv.org/abs/2311.03099)) & Permutation Basin Alignment ([Ainsworth et al., ICLR 2023 - *Git Re-Basin*](https://arxiv.org/abs/2209.04836)):**
  * Given two checkpoints sharing a parent backbone, $\tau = \theta_{\text{task}} - \theta_{\text{base}}$ is a **Task Vector**. Using **DARE + TIES**, up to 90–99% of small magnitude deltas in $\tau$ can be zeroed out and sign-resolved, isolating the sparse causal weight circuit.
  * *Permutation Guardrail:* If two checkpoints have drifted across permutation basins (§2.6), align intermediate MLP neurons first via the Linear Assignment Problem ($\max_{\Pi \in S_{d_{\text{ff}}}} \text{Tr}\big(\Pi^\top (W_{\text{down}}^{(A)\top} W_{\text{down}}^{(B)} + W_{\text{gate}}^{(A)} W_{\text{gate}}^{(B)\top} + W_{\text{up}}^{(A)} W_{\text{up}}^{(B)\top})\big)$ using `scipy.optimize.linear_sum_assignment`, then simultaneously transform $W_{\text{down}}^{(B)} \leftarrow W_{\text{down}}^{(B)}\Pi^\top$, $W_{\text{gate}}^{(B)} \leftarrow \Pi W_{\text{gate}}^{(B)}$, $W_{\text{up}}^{(B)} \leftarrow \Pi W_{\text{up}}^{(B)}$) before subtracting or merging weights.

---

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

## 4. Applied Blueprint: Weight-Space Programming the "Can Continue?" Classifier

### 4.1 Problem Decomposition & Confounder Analysis
Your target task is determining whether an AI assistant stopped prematurely (`CAN_CONTINUE = YES`, i.e., a **Lazy Stop**) vs. should not auto-continue (`CAN_CONTINUE = NO`).

Why naive keyword heuristics and standard binary probing fail:
1. **The Polite Sign-Off Confounder ($Z_{\text{polite}}$):**
   Phrases like *"Let me know how you'd like to proceed!"*, *"Would you like me to continue?"*, or *"Feel free to ask if you want me to make changes"* appear at the end of **both** `DONE` responses and `LAZY_STOP` responses.
2. **The Question-Mark Confounder ($Z_{\text{question}}$):**
   Both `LAZY_STOP` (*"Shall I implement Step 2 now?"*) and `BLOCKED_ON_USER` (*"Should we use Auth0 or Clerk, and what is your client ID?"*) end with a question mark (`?`).
   * The true mechanistic difference is **Decision Entropy / Action Determinism**:
     * In `LAZY_STOP`, the concrete action is already specified inside the assistant's own text (an unexecuted plan step or binary rubber-stamp request).
     * In `BLOCKED_ON_USER`, there are unresolved external variables or mutually exclusive architectural branches.
     * In `DONE`, all stated goals in the prompt/plan are reported as completed in past tense.

### 4.2 Mode A: Zero-Data / Zero-Forward-Pass Pure Static Weight Synthesis
If you have **zero calibration examples** and want to construct the `"Can Continue?"` classifier in **pure static weight space** (matching the zero-run paradigm of Experiments 3 and 5):
1. Start from **`tasksource/ModernBERT-large-nli`**, resolve the row indices $(i_{\text{ent}}, i_{\text{neu}}, i_{\text{con}})$ dynamically from `model.config.label2id` (case-insensitively, as NLI checkpoints vary in row ordering), and extract the 3-row classifier head $W_{\text{NLI}} \in \mathbb{R}^{3 \times d_{\text{model}}}$ and bias $b_{\text{NLI}} \in \mathbb{R}^3$ (with mean row norm $\bar{\nu}_{\text{NLI}} = \frac{1}{3}\sum_{c=1}^3 \|W_{\text{NLI}}[c, :]\|_2$).
2. Extract static token-embedding concept prototypes directly from the input embedding matrix $W_E \in \mathbb{R}^{V \times d_{\text{model}}}$ (averaging single-token or subword representations per word so every word has equal $1\times$ weight, and passing through `embeddings.norm` when present), map each prototype $v_c$ into the post-`model.head` coordinate basis $\tilde{v}_c = \text{model.head}(\text{model.model.final\_norm}(v_c))$, and form contrast-centered, $\bar{\nu}_{\text{NLI}}$-norm-matched class deltas (removing the shared `model.head` positive-orthant `GELU` cone offset $\bar{v} = \frac{1}{3}(\tilde{v}_{\text{next}} + \tilde{v}_{\text{done}} + \tilde{v}_{\text{block}})$):
   $$\hat{\delta}_{\text{lazy}} = \bar{\nu}_{\text{NLI}} \frac{\tilde{v}_{\text{next}} - \tilde{v}_{\text{block}}}{\|\tilde{v}_{\text{next}} - \tilde{v}_{\text{block}}\|_2}, \qquad \hat{\delta}_{\text{done}} = \bar{\nu}_{\text{NLI}} \frac{\tilde{v}_{\text{done}} - \frac{1}{2}(\tilde{v}_{\text{next}} + \tilde{v}_{\text{block}})}{\|\tilde{v}_{\text{done}} - \frac{1}{2}(\tilde{v}_{\text{next}} + \tilde{v}_{\text{block}})\|_2}, \qquad \hat{\delta}_{\text{block}} = -\hat{\delta}_{\text{lazy}}$$
   * $v_{\text{next}}$: mean embedding of continuation/deferral tokens (`["next", "proceed", "continue", "remaining", "step", "phase", "todo"]`)
   * $v_{\text{done}}$: mean embedding of completion tokens (`["completed", "finished", "done", "passed", "resolved", "implemented", "all"]`)
   * $v_{\text{block}}$: mean embedding of blocking/specification tokens (`["which", "specify", "clarify", "credentials", "key", "prefer", "choice", "error"]`)
   * $v_{\text{polite}}$: mean embedding of nuisance polite sign-off tokens (`["let", "know", "happy", "help", "feel", "free", "please"]`)
3. Orthogonalize in post-`model.head` space against the centered polite direction $\delta_{\text{polite}} = \tilde{v}_{\text{polite}} - \bar{v}$ ($P_\perp = I_d - \frac{\delta_{\text{polite}}\delta_{\text{polite}}^\top}{\|\delta_{\text{polite}}\|_2^2}$), wire the 3-class head and bias directly ($\lambda \in [0.15, 0.35]$), and hijack the 3 most dormant GeGLU neurons in the final layer (§3.10) to write along the exact linearized margin pullback $v_{\text{target}} = W_{\text{eff}}^\top [1, -\frac{1}{2}, -\frac{1}{2}]^\top$ (§2.4):
   $$\begin{aligned}
   W_{\text{cls}}[\text{LAZY\_STOP}] &= P_\perp \big( w_{\text{neutral}} + \lambda \hat{\delta}_{\text{lazy}} \big), & b_{\text{cls}}[\text{LAZY\_STOP}] &= b_{\text{neutral}} \\
   W_{\text{cls}}[\text{DONE}] &= P_\perp \big( w_{\text{entailment}} + \lambda \hat{\delta}_{\text{done}} \big), & b_{\text{cls}}[\text{DONE}] &= b_{\text{entailment}} \\
   W_{\text{cls}}[\text{BLOCKED}] &= P_\perp \big( w_{\text{contradiction}} + \lambda \hat{\delta}_{\text{block}} \big), & b_{\text{cls}}[\text{BLOCKED}] &= b_{\text{contradiction}}
   \end{aligned}$$

---

### 4.3 Mode B: 4-Stage Forward-Probed Weight-Space Pipeline (With PyTorch Code)

Below is a complete, drop-in weight-space programming script for **`tasksource/ModernBERT-large-nli`** (or `answerdotai/ModernBERT-large`) ordered so upstream weight edits are reflected in downstream head synthesis:
1. Applies **Takehi et al. (2026) Column-Selective `mlp.down_proj` Surgery (§3.1)** and **3-Neuron Polarized GeGLU Conjunctive Hijacking (§3.10)** in the upper-middle layers (`[18, 22]`) first, using masked token pooling and residual-stream attention-sink orthogonalization.
2. Runs a single post-surgery forward pass through `model.model` (respecting `model.config.classifier_pooling`: `"cls"` vs. `"mean"`) and `model.head` to extract pooled representations $H$ (zero gradients).
3. Applies **LEACE (§3.3)** to erase the polite sign-off / length nuisance subspace from $H$.
4. Synthesizes the 3-class classification head (`LAZY_STOP = 0`, `DONE = 1`, `BLOCKED_ON_USER = 2`) in closed form via **Between-Class Temperature-Calibrated Shrinkage Fisher LDA (§3.8B)** and optionally refines thresholds via **LEACE-Preserving Subspace SPSA (§3.14)**.

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Optional, Tuple

# Class definitions (collapse to binary `can_continue = (pred == 0)` at inference)
LABEL_LAZY_STOP = 0        # CAN_CONTINUE = YES (rubber-stamp pause, unexecuted next steps)
LABEL_DONE = 1             # CAN_CONTINUE = NO  (task completed)
LABEL_BLOCKED_ON_USER = 2  # CAN_CONTINUE = NO  (needs genuine decision, credentials, or missing spec)


@torch.no_grad()
def compute_leace_projection(
    X: torch.Tensor, Z: torch.Tensor, eps: float = 1e-4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Closed-form LEACE (Belrose et al., 2023) oblique projection matrix P_leace and bias shift.
    Erases all linear information about nuisance concept Z (e.g., polite sign-off phrasing)
    from representation space X [N, d] with minimal L2 perturbation.
    """
    X = X.float()
    Z = Z.to(device=X.device, dtype=torch.float32)
    N, d = X.shape
    if Z.dim() == 1:
        Z = Z.unsqueeze(1)

    mu_x = X.mean(dim=0)
    X_c = X - mu_x
    Z_c = Z - Z.mean(dim=0)

    # Covariance Sigma_XX (with Tikhonov regularization for numerical stability) and cross-covariance Sigma_XZ
    Sigma_XX = (X_c.T @ X_c) / max(N - 1, 1) + eps * torch.eye(d, device=X.device, dtype=X.dtype)
    Sigma_XZ = (X_c.T @ Z_c) / max(N - 1, 1)

    # Symmetric eigendecomposition for whitening matrix W = Sigma_XX^{-1/2} and unwhitening W^+ = Sigma_XX^{1/2}
    eigvals, eigvecs = torch.linalg.eigh(Sigma_XX)
    eigvals = torch.clamp(eigvals, min=eps)
    W = eigvecs @ torch.diag(eigvals.rsqrt()) @ eigvecs.T       # whitening: Sigma_XX^{-1/2}
    W_pinv = eigvecs @ torch.diag(eigvals.sqrt()) @ eigvecs.T   # unwhitening: Sigma_XX^{1/2}

    # Orthogonal projection onto column space of whitened cross-covariance V = W Sigma_XZ via thin SVD
    V = W @ Sigma_XZ
    U_v, S_v, _ = torch.linalg.svd(V, full_matrices=False)
    sv_tol = max(float(S_v.max()) * max(V.shape) * torch.finfo(V.dtype).eps, 1e-7) if S_v.numel() > 0 else 1e-7
    U_v = U_v[:, S_v > sv_tol]
    P_V = U_v @ U_v.T

    # Oblique erasure projection P_LEACE = I - W^+ P_V W
    I_d = torch.eye(d, device=X.device, dtype=X.dtype)
    P_leace = I_d - (W_pinv @ P_V @ W)
    b_leace = (I_d - P_leace) @ mu_x
    return P_leace, b_leace


@torch.no_grad()
def surgical_down_proj_edit(
    model: nn.Module,
    tokenizer,
    lazy_texts: List[str],
    foil_texts: List[str],
    target_layers: List[int] = [18, 22],
    top_k_cols: int = 64,
    alpha: float = 0.5,
    device: str = "cpu",
):
    """
    Takehi et al. (Nov 2026) Column-Selective Weight-Space Surgery on ModernBERT MLP down_proj.
    Isolates the top-K columns with positive signed attribution s_j > 0 in `mlp.Wo` that
    differentiate LAZY_STOP from FOIL and amplifies/suppresses their projection along u.
    """
    model.eval()
    enc_lazy = tokenizer(lazy_texts, padding=True, truncation=True, max_length=2048, return_tensors="pt").to(device)
    enc_foil = tokenizer(foil_texts, padding=True, truncation=True, max_length=2048, return_tensors="pt").to(device)

    for layer_idx in target_layers:
        layer = model.model.layers[layer_idx]
        mlp = layer.mlp
        W_down = mlp.Wo.weight  # shape: [d_model, d_ff]

        inter_acts, mlp_outs, cls_sinks = [], [], []
        current_mask = None

        def norm_pre_hook(module, inputs):
            # Capture true residual stream x_{\ell-1, 0} at token 0 ([CLS] / <BOS>) before mlp_norm
            # Supports both 3D padded [B, T, d] (right or left padded) and 2D FlashAttention-2 unpadded [total_nnz, d]
            x_in = inputs[0].detach()
            if x_in.dim() == 3:
                first_idx = current_mask.int().argmax(dim=1)
                cls_sinks.append(x_in[torch.arange(x_in.size(0), device=x_in.device), first_idx].float())
            else:
                lengths = current_mask.sum(dim=1)
                starts = (lengths.cumsum(dim=0) - lengths).clamp(max=max(x_in.size(0) - 1, 0))
                cls_sinks.append(x_in[starts].float())

        def wo_hook(module, inputs, output):
            # Mask out [PAD] tokens in float32 before pooling over sequence length (prevents fp16 sum overflow/underflow)
            a_in, out_t = inputs[0].detach().float(), output.detach().float()
            if a_in.dim() == 3:
                m = current_mask.unsqueeze(-1).float()
                denom = m.sum(dim=1).clamp(min=1.0)
                inter_acts.append((a_in * m).sum(dim=1) / denom)
                mlp_outs.append((out_t * m).sum(dim=1) / denom)
            else:
                lengths = current_mask.sum(dim=1)
                ends = lengths.cumsum(dim=0)
                starts = ends - lengths
                inter_acts.append(torch.stack([
                    a_in[s:e].mean(dim=0) if e > s else torch.zeros(a_in.size(-1), device=a_in.device)
                    for s, e in zip(starts, ends)
                ]))
                mlp_outs.append(torch.stack([
                    out_t[s:e].mean(dim=0) if e > s else torch.zeros(out_t.size(-1), device=out_t.device)
                    for s, e in zip(starts, ends)
                ]))

        h_norm = layer.mlp_norm.register_forward_pre_hook(norm_pre_hook)
        h_wo = mlp.Wo.register_forward_hook(wo_hook)
        try:
            # Forward pass on LAZY_STOP vs FOIL examples (zero gradients)
            current_mask = enc_lazy["attention_mask"]
            _ = model(**enc_lazy)
            a_lazy, out_lazy, sink_lazy = inter_acts.pop(), mlp_outs.pop(), cls_sinks.pop()

            current_mask = enc_foil["attention_mask"]
            _ = model(**enc_foil)
            a_foil, out_foil, sink_foil = inter_acts.pop(), mlp_outs.pop(), cls_sinks.pop()
        finally:
            h_norm.remove()
            h_wo.remove()

        # 1. Unit contrastive direction u [d_model], orthogonalized against residual-stream [CLS] sink (§2.7)
        delta_out = out_lazy.mean(dim=0) - out_foil.mean(dim=0)
        s_sink = 0.5 * (sink_lazy.mean(dim=0) + sink_foil.mean(dim=0))
        s_sink_hat = s_sink / (s_sink.norm() + 1e-8)
        delta_out = delta_out - torch.dot(delta_out, s_sink_hat) * s_sink_hat
        u = delta_out / (delta_out.norm() + 1e-8)

        # 2. Rank every column j of W_down by SIGNED positive differential contribution s_j > 0 (§3.1)
        W_down_f = W_down.float()
        delta_a = a_lazy.mean(dim=0) - a_foil.mean(dim=0)           # [d_ff]
        col_align = torch.mv(W_down_f.T, u)                          # [d_ff]
        scores = torch.nan_to_num(delta_a * col_align, nan=-1e9, posinf=-1e9, neginf=-1e9)  # [d_ff] (signed!)

        pos_mask = scores > 0
        k_actual = min(top_k_cols, int(pos_mask.sum().item()))
        if k_actual > 0:
            top_indices = torch.topk(scores, k=k_actual).indices
            # 3. Surgical rank-1 column edit on ONLY the top-K positive columns (Takehi et al., 2026)
            u_col = u.unsqueeze(1)                                   # [d_model, 1]
            selected_cols = W_down_f[:, top_indices]                 # [d_model, K]
            projections = u_col @ (u_col.T @ selected_cols)          # [d_model, K]
            W_down[:, top_indices] = (selected_cols + alpha * projections).to(W_down.dtype)


@torch.no_grad()
def synthesize_closed_form_classifier_head(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    labels: List[int],
    nuisance_flags: List[float],
    shrinkage: float = 0.1,
    device: str = "cpu",
) -> Tuple[nn.Module, torch.Tensor, torch.Tensor]:
    """
    Synthesizes the exact weights of `model.classifier` in closed form using
    LEACE nuisance erasure + Between-Class Temperature-Calibrated Shrinkage Fisher LDA.
    """
    model.eval()
    enc = tokenizer(texts, padding=True, truncation=True, max_length=2048, return_tensors="pt").to(device)
    outputs = model.model(**enc)

    # Respect ModernBERT's exact config.classifier_pooling ("cls" vs "mean") before model.head
    last_hidden = outputs.last_hidden_state
    pooling_mode = getattr(model.config, "classifier_pooling", "cls")
    if pooling_mode == "cls":
        first_idx = enc["attention_mask"].int().argmax(dim=1)
        pooled = last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), first_idx]
    else:
        attention_mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = ((last_hidden.float() * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1.0)).to(last_hidden.dtype)

    H = model.head(pooled) if hasattr(model, "head") else pooled  # [N, d_model]

    N, d = H.shape
    y = torch.tensor(labels, device=device, dtype=torch.long)
    z = torch.tensor(nuisance_flags, device=device, dtype=torch.float32)

    # 1. Compute LEACE projection to erase the polite sign-off nuisance direction
    P_leace, b_leace = compute_leace_projection(H.float(), z)
    H_clean = (H.float() @ P_leace.T) + b_leace

    # 2. Compute class means and pooled within-class covariance with shrinkage
    num_classes = 3
    global_mean = H_clean.mean(dim=0)
    means = global_mean.unsqueeze(0).repeat(num_classes, 1)
    priors = torch.zeros(num_classes, device=device, dtype=torch.float32)
    S_W = torch.zeros((d, d), device=device, dtype=torch.float32)

    for c in range(num_classes):
        mask = (y == c)
        if mask.any():
            H_c = H_clean[mask]
            means[c] = H_c.mean(dim=0)
            priors[c] = mask.float().mean().clamp(min=1e-4)
            diff = H_c - means[c]
            S_W += (diff.T @ diff) / max(N - num_classes, 1)
        else:
            priors[c] = 1e-4

    trace_mean = (torch.trace(S_W) / d).clamp(min=1e-4)
    Sigma_W = (1.0 - shrinkage) * S_W + shrinkage * trace_mean * torch.eye(d, device=device)

    # 3. Zero-sum centered class means + Between-Class Mahalanobis Temperature Calibration (§3.8B)
    # Solving with (means - mu_bar) eliminates common-mode drift while preserving exact logit differences!
    mu_bar = means.mean(dim=0, keepdim=True)
    means_centered = means - mu_bar
    W_lda = torch.linalg.solve(Sigma_W, means_centered.T).T  # [3, d], sum_c W_lda[c] == 0
    between_mahal = (means_centered * W_lda).sum(dim=1).mean()
    temp = (between_mahal / 4.0).clamp(min=1.0)
    W_lda = W_lda / temp
    b_lda = (-0.5 * ((means + mu_bar) * W_lda).sum(dim=1)) + torch.log(priors)

    # 4. Fold LEACE projection directly into the linear classifier weights:
    # logit = W_lda (P_leace h + b_leace) + b_lda = (W_lda P_leace) h + (W_lda b_leace + b_lda)
    W_final = (W_lda @ P_leace).to(model.classifier.weight.dtype)
    b_final = (W_lda @ b_leace + b_lda).to(model.classifier.bias.dtype)

    model.classifier.weight.copy_(W_final)
    model.classifier.bias.copy_(b_final)
    return model, H, y


@torch.no_grad()
def hijack_dormant_geglu_neurons(
    model: nn.Module,
    layer_idx: int,
    key_concept_A: torch.Tensor,
    key_concept_B: torch.Tensor,
    value_direction: torch.Tensor,
    gate_scale: float = 2.0,
    out_scale: float = 0.5,
):
    """
    Dormant Neuron Hijacking (§3.10): Locates the 3 most dormant GeGLU neurons in `layer_idx`
    using the gauge-invariant triple product norm (§2.6) and wires them as an exact polarized
    conjunctive AND gate: 0.5 * [psi(k_A + k_B) - psi(k_A) - psi(k_B)], where psi(u) = GELU(u) * u.
    Outputs identically 0 when only Concept A or only Concept B is present, and > 0 when both > 0.
    """
    layer = model.model.layers[layer_idx]
    mlp = layer.mlp
    W_in = mlp.Wi.weight    # [2 * d_ff, d_model] (first half = GELU input, second half = linear gate)
    W_down = mlp.Wo.weight  # [d_model, d_ff]
    d_model, d_ff = W_down.shape

    # 1. Gauge-invariant dormancy ranking (§2.6): ||W_down[:, j]||_2 * ||W_in[j, :]||_2 * ||W_in[d_ff+j, :]||_2
    W_in_f, W_down_f = W_in.float(), W_down.float()
    prod_norms = (
        W_down_f.norm(dim=0)
        * W_in_f[:d_ff, :].norm(dim=1)
        * W_in_f[d_ff:, :].norm(dim=1)
    )
    dormant_indices = torch.topk(prod_norms, k=3, largest=False).indices
    j_sum, j_A, j_B = int(dormant_indices[0]), int(dormant_indices[1]), int(dormant_indices[2])

    # 2. Zero-mean center (P_{1^\perp}), unit-normalize & Löwdin-orthogonalize on active support, then pre-multiply by gamma^{-1}
    # (Do NOT re-normalize after multiplying by gamma^{-1}, so runtime gamma * W_in recovers exact orthonormal keys!)
    k_A = key_concept_A.to(device=W_in.device, dtype=torch.float32)
    k_B = key_concept_B.to(device=W_in.device, dtype=torch.float32)
    k_A = k_A - k_A.mean()
    k_B = k_B - k_B.mean()
    norm_bias = getattr(layer.mlp_norm, "bias", None) if hasattr(layer, "mlp_norm") else None

    if hasattr(layer, "mlp_norm") and getattr(layer.mlp_norm, "weight", None) is not None:
        gamma = layer.mlp_norm.weight.float()
        valid = gamma.abs() > 1e-2
        if valid.any():
            k_A = torch.where(valid, k_A - k_A[valid].mean(), torch.zeros_like(k_A))
            k_B = torch.where(valid, k_B - k_B[valid].mean(), torch.zeros_like(k_B))
            if mlp.Wi.bias is None and norm_bias is not None:
                b_eff = torch.where(valid, norm_bias.float() / gamma, torch.zeros_like(gamma))
                b_eff = torch.where(valid, b_eff - b_eff[valid].mean(), torch.zeros_like(b_eff))
                b_hat = b_eff / (b_eff.norm() + 1e-8)
                k_A = k_A - torch.dot(k_A, b_hat) * b_hat
                k_B = k_B - torch.dot(k_B, b_hat) * b_hat
        k_A = k_A / (k_A.norm() + 1e-8)
        k_B = k_B / (k_B.norm() + 1e-8)
        if (k_A + k_B).norm() > 1e-5 and (k_A - k_B).norm() > 1e-5:
            u_plus = (k_A + k_B) / (k_A + k_B).norm()
            u_minus = (k_A - k_B) / (k_A - k_B).norm()
            k_A = (u_plus + u_minus) * (0.5 ** 0.5)
            k_B = (u_plus - u_minus) * (0.5 ** 0.5)
        gamma_inv = torch.where(valid, 1.0 / gamma, torch.zeros_like(gamma))
        k_A = k_A * gamma_inv
        k_B = k_B * gamma_inv
    else:
        k_A = k_A / (k_A.norm() + 1e-8)
        k_B = k_B / (k_B.norm() + 1e-8)
        if (k_A + k_B).norm() > 1e-5 and (k_A - k_B).norm() > 1e-5:
            u_plus = (k_A + k_B) / (k_A + k_B).norm()
            u_minus = (k_A - k_B) / (k_A - k_B).norm()
            k_A = (u_plus + u_minus) * (0.5 ** 0.5)
            k_B = (u_plus - u_minus) * (0.5 ** 0.5)

    v_out = value_direction.to(device=W_down.device, dtype=torch.float32)
    v_out = v_out / (v_out.norm() + 1e-8)

    # 3. Polarized 3-neuron conjunctive gate: 0.5 * [psi(k_A + k_B) - psi(k_A) - psi(k_B)]
    for idx, key_vec, coeff in [
        (j_sum, k_A + k_B, +0.5 * out_scale),
        (j_A,   k_A,       -0.5 * out_scale),
        (j_B,   k_B,       -0.5 * out_scale),
    ]:
        w_row = gate_scale * key_vec
        W_in[idx, :] = w_row.to(W_in.dtype)
        W_in[d_ff + idx, :] = w_row.to(W_in.dtype)
        if mlp.Wi.bias is not None:
            b_shift = -torch.dot(w_row, norm_bias.float()) if norm_bias is not None else 0.0
            mlp.Wi.bias[idx] = b_shift
            mlp.Wi.bias[d_ff + idx] = b_shift
        W_down[:, idx] = (coeff * v_out).to(W_down.dtype)


@torch.no_grad()
def subspace_spsa_f1_refinement(
    model: nn.Module,
    H_val: torch.Tensor,
    y_val: torch.Tensor,
    num_steps: int = 25,
    eps: float = 0.05,
    lr: float = 0.05,
) -> float:
    """
    LEACE-Preserving Mechanistic Subspace SPSA (§3.14; Spall 1992, Malladi et al. 2023):
    Parameterizes candidate weights as W_cand = A @ W_base (with A in R^{3x3} initialized to I_3).
    Because W_base = W_lda @ P_leace, W_cand @ (I - P_leace) == 0 holds identically for ALL A
    (even if W_base is rank-deficient!), and Z_base = H_val @ W_base.T is precomputed in O(1) logit units.
    """
    W_base = model.classifier.weight.float()  # [3, d], already right-multiplied by P_leace
    b_base = model.classifier.bias.float()    # [3]
    H_val_f = H_val.to(device=W_base.device, dtype=torch.float32)
    y_val_d = y_val.to(device=W_base.device)

    Z_base = H_val_f @ W_base.T               # [N, 3] precomputed once!
    mu_Z = Z_base.mean(dim=0)                 # [3] mean-center Z_base to decouple A perturbations from bias b
    Z_centered = Z_base - mu_Z                # [N, 3]
    A = torch.eye(3, device=W_base.device, dtype=torch.float32)
    b = b_base + mu_Z

    def eval_scores(A_cand: torch.Tensor, b_cand: torch.Tensor, tau: float = 0.25) -> Tuple[float, float]:
        s_norm = (3.0 ** 0.5) / (A_cand.norm() + 1e-8)
        logits = (Z_centered @ A_cand.T + b_cand) * s_norm
        preds = logits.argmax(dim=-1)
        tp = ((preds == 0) & (y_val_d == 0)).sum().float()
        fp_done = ((preds == 0) & (y_val_d == 1)).sum().float()
        fp_blocked = ((preds == 0) & (y_val_d == 2)).sum().float()
        fn = ((preds != 0) & (y_val_d == 0)).sum().float()
        hard_f1 = float((2.0 * tp) / (2.0 * tp + fp_done + 3.0 * fp_blocked + fn + 1e-8))

        # Scale-invariant temperature-sharpened surrogate prevents zero-gradient stalls on discrete argmax plateaus
        probs_0 = torch.softmax(logits / tau, dim=-1)[:, 0]
        s_tp = probs_0[y_val_d == 0].sum()
        s_fp_done = probs_0[y_val_d == 1].sum()
        s_fp_blocked = probs_0[y_val_d == 2].sum()
        s_fn = (1.0 - probs_0[y_val_d == 0]).sum()
        smooth_f1 = float((2.0 * s_tp) / (2.0 * s_tp + s_fp_done + 3.0 * s_fp_blocked + s_fn + 1e-8))
        return hard_f1, smooth_f1

    best_score, best_smooth = eval_scores(A, b)
    best_A, best_b = A.clone(), b.clone()

    for step in range(num_steps):
        # Standard Spall (1992) decaying perturbation (ck) and step-size (ak) schedules
        ck = eps / ((step + 1) ** 0.101)
        ak = lr / ((step + 1) ** 0.602)
        delta_A = (torch.randint(0, 2, A.shape, device=A.device, dtype=A.dtype) * 2 - 1)
        delta_b = (torch.randint(0, 2, b.shape, device=b.device, dtype=b.dtype) * 2 - 1)

        A_pos, b_pos = A + ck * delta_A, b + ck * delta_b
        A_neg, b_neg = A - ck * delta_A, b - ck * delta_b
        hard_pos, smooth_pos = eval_scores(A_pos, b_pos)
        hard_neg, smooth_neg = eval_scores(A_neg, b_neg)

        # Hybrid finite difference uses hard F1 jumps when present and smooth F1 slope across plateaus
        obj_pos = 0.5 * hard_pos + 0.5 * smooth_pos
        obj_neg = 0.5 * hard_neg + 0.5 * smooth_neg
        grad_est = (obj_pos - obj_neg) / (2.0 * ck)
        cand_A = A + ak * grad_est * delta_A
        cand_b = b + ak * grad_est * delta_b
        hard_curr, smooth_curr = eval_scores(cand_A, cand_b)

        # Monotone lexicographic ratchet on (hard_f1, smooth_f1) across probe points and gradient step
        for h_c, sm_c, A_c, b_c in [
            (hard_pos, smooth_pos, A_pos, b_pos),
            (hard_neg, smooth_neg, A_neg, b_neg),
            (hard_curr, smooth_curr, cand_A, cand_b),
        ]:
            if (h_c > best_score) or (h_c == best_score and sm_c > best_smooth):
                best_score, best_smooth = h_c, sm_c
                s_c = (3.0 ** 0.5) / (A_c.norm() + 1e-8)
                best_A, best_b = (A_c * s_c).clone(), (b_c * s_c).clone()

        A, b = best_A.clone(), best_b.clone()

    model.classifier.weight.copy_((best_A @ W_base).to(model.classifier.weight.dtype))
    model.classifier.bias.copy_((best_b - best_A @ mu_Z).to(model.classifier.bias.dtype))
    return best_score


@torch.no_grad()
def synthesize_zero_data_nli_head(
    model: nn.Module,
    tokenizer,
    lam: float = 0.25,
) -> nn.Module:
    """
    Mode A (§3.8A, §4.2): Pure static zero-data / zero-forward-pass classifier synthesis.
    Resolves NLI head rows dynamically via `model.config.label2id`, maps token prototypes
    through `final_norm` + `model.head`, orthogonalizes against centered polite sign-off tokens,
    wires `model.classifier`, and hijacks 3 dormant GeGLU neurons along the exact margin pullback.
    """
    model.eval()
    W_nli = model.classifier.weight.float()
    b_nli = model.classifier.bias.float() if model.classifier.bias is not None else torch.zeros(3, device=W_nli.device)
    dev = W_nli.device
    nu_bar = W_nli.norm(dim=1).mean()
    head_dtype = next(model.head.parameters()).dtype if hasattr(model, "head") else W_nli.dtype

    label2id = {str(k).lower(): int(v) for k, v in getattr(model.config, "label2id", {}).items()}
    idx_ent = next((v for k, v in label2id.items() if "entail" in k), 0)
    idx_neu = next((v for k, v in label2id.items() if "neut" in k), 1)
    idx_con = next((v for k, v in label2id.items() if "contra" in k), 2)

    W_E = model.get_input_embeddings().weight.detach().float()
    emb_norm = getattr(getattr(model, "model", None), "embeddings", None)
    emb_norm = getattr(emb_norm, "norm", None)

    def proto(words: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        word_vecs = []
        for w in words:
            single_ids, sub_vecs = [], []
            for variant in (w, " " + w):
                toks = tokenizer.encode(variant, add_special_tokens=False)
                if len(toks) == 1:
                    single_ids.append(toks[0])
                elif len(toks) > 1:
                    sub_vecs.append(W_E[toks].mean(dim=0))
            if single_ids:
                word_vecs.append(W_E[list(dict.fromkeys(single_ids))].mean(dim=0))
            elif sub_vecs:
                word_vecs.append(torch.stack(sub_vecs).mean(dim=0))
        v_raw = torch.stack(word_vecs).mean(dim=0) if word_vecs else W_E.mean(dim=0)
        v_in = v_raw.to(device=dev, dtype=head_dtype).unsqueeze(0)
        v_res = emb_norm(v_in).squeeze(0).float() if emb_norm is not None else v_raw.to(dev)
        v_normed = model.model.final_norm(v_res.to(head_dtype).unsqueeze(0)) if hasattr(model.model, "final_norm") else v_in
        v_head = (model.head(v_normed) if hasattr(model, "head") else v_normed).squeeze(0).float()
        return v_res, v_head

    v_next_res, v_next_h = proto(["next", "proceed", "continue", "remaining", "step", "phase", "todo"])
    v_perm_res, _ = proto(["would", "shall", "should", "confirm", "ready"])
    _, v_done_h = proto(["completed", "finished", "done", "passed", "resolved", "implemented", "all"])
    v_block_res, v_block_h = proto(["which", "specify", "clarify", "credentials", "key", "prefer", "choice", "error"])
    _, v_polite_h = proto(["let", "know", "happy", "help", "feel", "free", "please"])

    # Center against shared post-`model.head` GELU cone offset and norm-match to nu_bar (§4.2)
    v_bar_h = (v_next_h + v_done_h + v_block_h) / 3.0
    pol_centered = v_polite_h - v_bar_h
    u_pol = pol_centered / (pol_centered.norm() + 1e-8)
    P_perp = torch.eye(W_nli.size(1), device=dev) - torch.outer(u_pol, u_pol)

    d_lazy = nu_bar * ((v_next_h - v_block_h) / ((v_next_h - v_block_h).norm() + 1e-8))
    d_done_raw = v_done_h - 0.5 * (v_next_h + v_block_h)
    d_done = nu_bar * (d_done_raw / (d_done_raw.norm() + 1e-8))
    d_block = -d_lazy

    W_new = torch.stack([
        P_perp @ (W_nli[idx_neu] + lam * d_lazy),
        P_perp @ (W_nli[idx_ent] + lam * d_done),
        P_perp @ (W_nli[idx_con] + lam * d_block),
    ], dim=0)
    b_new = torch.stack([b_nli[idx_neu], b_nli[idx_ent], b_nli[idx_con]])

    # Exact linearized pullback v_target = W_eff^T [1, -0.5, -0.5]^T into final residual stream (§2.4, §4.2)
    c_margin = torch.tensor([1.0, -0.5, -0.5], device=dev)
    v_pull = W_new.T @ c_margin
    if hasattr(model, "head") and hasattr(model.head, "dense"):
        if hasattr(model.head, "norm") and getattr(model.head.norm, "weight", None) is not None:
            v_pull = v_pull * model.head.norm.weight.float()
            v_pull = v_pull - v_pull.mean()
        v_pull = model.head.dense.weight.float().T @ v_pull
    if hasattr(model.model, "final_norm") and getattr(model.model.final_norm, "weight", None) is not None:
        v_pull = v_pull * model.model.final_norm.weight.float()
        v_pull = v_pull - v_pull.mean()

    hijack_dormant_geglu_neurons(model, len(model.model.layers) - 1, v_next_res, v_perm_res, v_pull)

    model.classifier.weight.copy_(W_new.to(model.classifier.weight.dtype))
    if model.classifier.bias is not None:
        model.classifier.bias.copy_(b_new.to(model.classifier.bias.dtype))
    return model


@torch.no_grad()
def program_can_continue_classifier(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    labels: List[int],
    nuisance_flags: List[float],
    target_layers: List[int] = [18, 22],
    conjunctive_keys: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    device: str = "cpu",
) -> Tuple[nn.Module, float]:
    """
    End-to-end Weight-Space Programming orchestrator executing Mode A (0-data static synthesis)
    when `texts` is empty, or Mode B (upstream MLP surgery -> post-surgery feature extraction ->
    LEACE erasure -> Shrinkage LDA head synthesis -> Subspace SPSA refinement).
    """
    model.to(device)
    if not texts:
        return synthesize_zero_data_nli_head(model, tokenizer), 0.0

    lazy_texts = [t for t, l in zip(texts, labels) if l == LABEL_LAZY_STOP]
    foil_texts = [t for t, l in zip(texts, labels) if l != LABEL_LAZY_STOP]

    # Stage 1A: Upstream Column-Selective MLP Surgery in upper-middle layers
    if lazy_texts and foil_texts:
        surgical_down_proj_edit(
            model, tokenizer, lazy_texts, foil_texts, target_layers=target_layers, device=device
        )

    # Stage 1B (Optional): 3-Neuron Polarized GeGLU Conjunctive Logic Gate Hijacking
    if conjunctive_keys is not None and target_layers:
        key_A, key_B, val_dir = conjunctive_keys
        hijack_dormant_geglu_neurons(model, target_layers[-1], key_A, key_B, val_dir)

    # Stage 2 & 3: Post-surgery forward pass + LEACE + Calibrated Shrinkage Fisher LDA head synthesis
    model, H, y = synthesize_closed_form_classifier_head(
        model, tokenizer, texts, labels, nuisance_flags, device=device
    )

    # Stage 4: LEACE-preserving 12-coordinate Subspace SPSA F1 refinement
    best_f1 = subspace_spsa_f1_refinement(model, H, y)
    return model, best_f1
```

---

## 5. Agent Skill Runbook: Decision Tree & Execution Protocol

### 5.1 Task Triage Decision Tree

When asked to **create a neural network from scratch**, **build a classifier without training**, **audit a model's weights**, or **surgically modify/repair an existing model**:

| Goal | Forward Passes Allowed? | Primary Technique to Execute | Section Reference |
| :--- | :---: | :--- | :--- |
| **1. Build a small network or algorithmic module from scratch** | **0 (Pure Static)** | Partition residual stream into orthogonal registers ($\mathcal{R}_1 \oplus \mathcal{R}_2 \dots$); compile boolean/positional lookups into $W_{QK}^h$ (§2.7, §3.15), value copies into $W_{OV}^h$, and threshold/bilinear logic into $W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$ (RASP/Tracr/AdderBoard). | §2.2, §2.7, §3.10, §3.15 |
| **2. Audit what an unknown model/layer computes** | **0 (Pure Static)** | Fold RMSNorm ($\text{diag}(\gamma)$) or LayerNorm ($\text{diag}(\gamma) P_{\mathbf{1}^\perp}$) into downstream weights (§2.4); compute SVD of $W_{OV}^h = U\Sigma V^\top$; check $\text{Tr}(W_U W_{OV}^h W_E^\top)$ for copying/induction heads; eigendecompose $B_u$ with gated polarity sign check (§2.5). | §2.4, §2.5, §3.13 |
| **3. Audit subtle semantic traits (e.g., sycophancy, backdoors)** | **10–16 Attention Probes** | Static weights alone fail under heavy superposition (Exp 7). Run 10–16 contrasting diagnostic prompts and compare attention routing matrices $A_{t,s}^h$ and Direct Logit Attribution $\text{DLA}(c)$ (§3.6). | §1.4, §3.6, §3.11 |
| **4. Build a sequence classifier with 0 examples** | **0 (Pure Static)** | Start from an NLI-pretrained encoder (`tasksource/ModernBERT-large-nli`); map token prototypes through `model.head(final_norm(v_c))`, orthogonalize against nuisance tokens ($P_\perp$), and wire $W_{\text{cls}}$ from $[w_{\text{entailment}}, w_{\text{neutral}}, w_{\text{contradiction}}]$ (§4.2). | §3.8A, §4.2 |
| **5. Build a sequence classifier with 30–200 calibration examples** | **1–2 Forward Batches (0 Gradients)** | Apply upstream MLP edits first; extract pooled post-`model.head` states $H$ (respecting `config.classifier_pooling`); scrub nuisance confounders via **LEACE** ($P_{\text{LEACE}}$, §3.3); solve for `classifier.weight` & `bias` via **Temperature-Calibrated Shrinkage Fisher LDA** (§3.8B); refine via **LEACE-Preserving Subspace SPSA** (§3.14). | §3.3, §3.8B, §3.14, §4.3 |
| **6. Fix a behavioral flaw (e.g., looping, lazy stopping, refusals)** | **10–50 Forward Passes (0 Gradients)** | Collect masked contrastive residual states $(\mu_+, \mu_-)$, orthogonalize $u_\ell$ against `<BOS>`/`[CLS]` attention sink $\hat{s}_{\text{sink}}$ (§2.7); rank `mlp.down_proj` columns by $s_j = (\bar{a}_j^+ - \bar{a}_j^-)\langle u_\ell, w_{\text{down}, j}\rangle$; apply **Column-Selective Rank-1 Surgery** to the top $K=64$ columns in peak layers (§1.3, §3.1). | §1.3, §3.1, §3.2 |
| **7. Inject new key-value rules without hurting existing knowledge** | **1 Calibration Pass** | Compute preserved activation covariance $C_0 = K_0 K_0^\top$; extract its null-space projector $P_{\text{null}}$ and apply the **$m \times m$ Woodbury AlphaEdit** update ($\Delta W K_0 = 0$, §3.4) or **Dormant GeGLU Neuron Hijacking** (§3.10). | §3.4, §3.10 |
| **8. Graft a head or circuit from Model A to Model B** | **1 Anchor Pass (~200 texts)** | Compute Orthogonal Procrustes rotation $Q = U V^\top$ from $\text{SVD}(H_B^\top H_A)$ and map $W_B = W_A Q^\top$ (§3.16), or align permutation basins via `Git Re-Basin` (§3.7). | §3.7, §3.16 |

### 5.2 Mandatory Verification & Invariant Checklist
Whenever performing weight-space programming, verify these six invariants before saving the checkpoint:
1. **Bit-Exact / Shape & Dtype Preservation:** All modified weight tensors must retain their exact original `.shape`, `.dtype` (e.g., `bfloat16` / `float16`, while computing SVD/LEACE/solve intermediates in `float32`/`float64`), and memory contiguity.
2. **Attention Sink & GQA Preservation:** Never edit shared $W_K^{g(h)}$ or $W_V^{g(h)}$ in GQA models to fix a single query head (edit private $W_Q^h$ / $W_O^h$ instead, §2.2), and always orthogonalize steering/ablation directions against the `<BOS>`/`[CLS]` massive-activation sink $\hat{s}_{\text{sink}}$ (§2.7).
3. **Gauge & Norm Stability:** Rank dormant neurons by the triple product norm $\|w_{\text{down}, j}\|_2 \|w_{\text{gate}, j}\|_2 \|w_{\text{up}, j}\|_2$ (§2.6), and when projecting out a subspace ($P_\perp W_{\text{down}}$) across $>10\%$ of columns, apply norm-preserving rescaling (§3.2).
4. **Orthogonal Random Control (Exp 8 Protocol):** Always test a negative control edit using a random orthogonal unit vector $u_\perp \perp u$ of identical norm. The causal edit must beat both the unedited baseline and the $u_\perp$ control.
5. **Locality / Null-Space Check:** Evaluate on a held-out slice of clean/unrelated examples ($K_0$) to verify zero regression in general representations.
6. **Confounder Invariance Check:** Test paired counterfactual inputs that flip only the nuisance variable (e.g., appending *"Let me know if you need anything else!"* to a completed response) and verify the classifier prediction does not flip.

---

## 6. Comprehensive Citations & Bibliography

### 6.1 All 28 Official Citations from *Sparks of Weight-Space Programming* (Annotated)

Below are the exact **28 references** cited in [Takehi, Teoh, Zhang, & Khattab (Nov 2026)](https://rikiyat.github.io/blog/weight-space-programming/), together with their direct links and how each informs weight-space programming:

1. **Yang, C., Wang, X., Lu, Y., Liu, H., Le, Q. V., Zhou, D., & Chen, X. (2024).** *Large Language Models as Optimizers (OPRO).* International Conference on Learning Representations (ICLR). [[PDF]](http://arxiv.org/pdf/2309.03409.pdf) — *Establishes LLMs as black-box trajectory optimizers using natural-language optimization history.*
2. **Khattab, O., Singhvi, A., Maheshwari, P., Zhang, Z., Santhanam, K., Vardhamanan, S., Haq, S., Sharma, A., Joshi, T. T., Moazam, H., Miller, H., Zaharia, M., & Potts, C. (2024).** *DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines.* International Conference on Learning Representations (ICLR). [[PDF]](http://arxiv.org/pdf/2310.03714.pdf) — *Declarative compilation paradigm that Weight-Space Programming extends from discrete prompts down into continuous weight tensors.*
3. **Opsahl-Ong, K., Ryan, M. J., Purtell, J., Broman, D., Potts, C., Zaharia, M., & Khattab, O. (2024).** *Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs (MIPRO).* Conference on Empirical Methods in Natural Language Processing (EMNLP). [[PDF]](http://arxiv.org/pdf/2406.11695.pdf) — *Credit assignment and Bayesian proposal optimization across multi-stage AI systems.*
4. **Agrawal, L. A., Tan, S., Soylu, D., Ziems, N., Khare, R., Opsahl-Ong, K., Singhvi, A., Shandilya, H., Ryan, M. J., Jiang, M., Potts, C., Sen, K., Dimakis, A. G., Stoica, I., Klein, D., Zaharia, M., & Khattab, O. (2026).** *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning.* International Conference on Learning Representations (ICLR). [[PDF]](http://arxiv.org/pdf/2507.19457.pdf) — *Demonstrates that natural-language diagnostic reflection + Pareto selection beats RL (GRPO) with far fewer rollouts.*
5. **Lee, Y., Nair, R., Zhang, Q., Lee, K., Khattab, O., & Finn, C. (2026).** *Meta-Harness: End-to-End Optimization of Model Harnesses.* Conference on Language Modeling (COLM). [[PDF]](http://arxiv.org/pdf/2603.28052.pdf) — *Agentic outer-loop optimization of surrounding execution/context harnesses using filesystem traces and scores.*
6. **Karpathy, A. (2026).** *autoresearch.* GitHub repository. [[link]](https://github.com/karpathy/autoresearch) — *Autonomous outer-loop ML research harness using git-ratcheted code edits and short evaluation runs.*
7. **Bai, X., Liang, J., Teoh, J., & Sitzmann, V. (2026).** *Reactive Optimizers (`OPAL`).* Blog post. [[link]](https://xingjianbai.com/reactive-optimizers/) — *Places an LLM agent inside the inner optimization loop to inspect telemetry and dynamically rewrite parameter update rules mid-run.*
8. **GLOBIS University (2023).** *aozorabunko-clean: the Aozora Bunko collection of Japanese literature, cleaned and deduplicated.* Hugging Face dataset. [[link]](https://huggingface.co/datasets/globis-university/aozorabunko-clean) — *Japanese literature corpus (transliterated to Romaji) used to train the 7.16M-param Transformer audited in Experiment 3.*
9. **Wikimedia Foundation (2023).** *Wikipedia, English, dump of November 1, 2023.* Hugging Face dataset `wikimedia/wikipedia`. [[link]](https://huggingface.co/datasets/wikimedia/wikipedia) — *English corpus used to train the English 7.16M-param Transformer audited in Experiment 3.*
10. **Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., Askell, A., Bai, Y., Chen, A., Conerly, T., DasSarma, N., Drain, D., Ganguli, D., Hatfield-Dodds, Z., Hernandez, D., Jones, A., Kernion, J., Lovitt, L., Ndousse, K., Amodei, D., Brown, T., Clark, J., Kaplan, J., McCandlish, S., & Olah, C. (2021).** *A Mathematical Framework for Transformer Circuits.* Transformer Circuits Thread. [[HTML]](https://transformer-circuits.pub/2021/framework/index.html) — *Foundational decomposition of Transformers into residual streams, $W_{QK}$, and $W_{OV}$ circuits.*
11. **Matthews, M., Beukman, M., Lu, C., & Foerster, J. (2025).** *Kinetix: Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks.* International Conference on Learning Representations (ICLR). [[PDF]](http://arxiv.org/pdf/2410.23208.pdf) — *Physics control benchmark used in Experiment 4 (one-shot 3.5% weight repair of a ramp-climbing car policy).*
12. **Young, K., & Tian, T. (2019).** *MinAtar: An Atari-Inspired Testbed for Thorough and Reproducible Reinforcement Learning Experiments.* arXiv preprint arXiv:1903.03176. [[PDF]](http://arxiv.org/pdf/1903.03176.pdf) — *$10\times 10$ grid game testbed (*Breakout, Seaquest, Asterix, Pac-Man, Freeway*) used in Experiments 1 and 5.*
13. **Pipis, C., Garg, S., Kontonis, V., Shrivastava, V., Krishnamurthy, A., & Papailiopoulos, D. (2025).** *Wait, Wait, Wait... Why Do Reasoning Models Loop?* arXiv preprint arXiv:2512.12895. [[PDF]](http://arxiv.org/pdf/2512.12895.pdf) — *Explains why distilled/small reasoning models loop (risk aversion under learning hardness + temporally correlated error bias) and why temperature is only a stopgap, motivating Experiment 8.*
14. **Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., & Nanda, N. (2024).** *Refusal in Language Models Is Mediated by a Single Direction.* arXiv preprint arXiv:2406.11717. [[PDF]](http://arxiv.org/pdf/2406.11717.pdf) — *Establishes that high-level behaviors can be mediated by a 1D residual direction and ablated via rank-1 weight projection.*
15. **Olah, C., Cammarata, N., Schubert, L., Goh, G., Petrov, M., & Carter, S. (2020).** *Zoom In: An Introduction to Circuits.* Distill. [[link]](https://distill.pub/2020/circuits/zoom-in) — *Foundational circuits manifesto arguing that weights directly encode readable algorithms connecting semantic features.*
16. **Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022).** *Locating and Editing Factual Associations in GPT (ROME).* Advances in Neural Information Processing Systems (NeurIPS). [[PDF]](http://arxiv.org/pdf/2202.05262.pdf) — *Establishes MLP down-projections as linear associative key-value memories editable via rank-1 updates.*
17. **Fraser-Taliente, K., Kantamneni, S., Ong, E., Mossing, D., Lu, C., Bogdan, P. C., Ameisen, E., Chen, J., Kishylau, D., Pearce, A., Tarng, J., Wu, A., Wu, J., Zhang, Y., Ziegler, D. M., Hubinger, E., Batson, J., Lindsey, J., Zimmerman, S., & Marks, S. (2026).** *Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations (NLAs).* Transformer Circuits Thread. [[HTML]](https://transformer-circuits.pub/2026/nla/index.html) — *Unsupervised Activation Verbalizer + Reconstructor bottleneck translating hidden states into faithful English descriptions.*
18. **Kabra, K., Venhoff, C., & Schroeder de Witt, C. (2026).** *Weight Oracles: Reading Neural Network Weights with Language Models.* arXiv preprint arXiv:2610.07334. [[PDF]](http://arxiv.org/pdf/2610.07334.pdf) — *Static LLM weight readers that simulate forward passes (99% accuracy) and detect zero-shot backdoors directly from weights (0.93 AUROC).*
19. **Bills, S., Cammarata, N., Mossing, D., Tillman, H., Gao, L., Goh, G., Sutskever, I., Leike, J., Wu, J., & Saunders, W. (2023).** *Language models can explain neurons in language models.* OpenAI. [[HTML]](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html) — *Automated LLM explanation and simulation scoring of individual MLP neurons.*
20. **lyramakesmusic (2026).** *neuron-explorer: Steering-first neuron explorer for Qwen3.5-4B.* GitHub repository. [[link]](https://github.com/lyramakesmusic/neuron-explorer) — *Interactive steering-first explorer for clamping/pushing Qwen3.5 MLP columns and observing rollout shifts.*
21. **Blank, C., Bhatia, A., Ong, E., & Nanda, N. (2026).** *WorkspaceBench: Evaluating Interpretability Methods for the Global Workspace.* LessWrong. [[link]](https://www.lesswrong.com/posts/Zeg2JztbdhguL48uH/workspacebench-evaluating-interpretability-methods-for-the) — *Benchmark evaluating how well interpretability methods read latent global-workspace variables across reasoning and safety tasks.*
22. **Wang, K., Tang, D., Zeng, B., Yin, Y., Xu, Z., Zhou, Y., Zang, Z., Darrell, T., Liu, Z., & You, Y. (2024).** *Neural Network Diffusion (`p-diff`).* arXiv preprint arXiv:2402.13144. [[PDF]](http://arxiv.org/pdf/2402.13144.pdf) — *Synthesizing neural network weights via latent diffusion over autoencoder weight representations.*
23. **Phang, J., Mao, Y., He, P., & Chen, W. (2023).** *HyperTuning: Toward Adapting Large Language Models without Back-propagation.* International Conference on Machine Learning (ICML). [[HTML]](https://proceedings.mlr.press/v202/phang23a.html) — *Using hypermodels (`HyperT5`) to generate task-specific adapter/prefix weights in a single forward pass.*
24. **Wang, J., Zhang, Y., & Li, X. (2025).** *NeuroGen: Neural Network Parameter Generation via Large Language Models.* arXiv preprint arXiv:2505.12470. [[PDF]](http://arxiv.org/pdf/2505.12470.pdf) — *Instruction-tuning LLMs with parameter reference injection to directly generate weights for target networks.*
25. **Spall, J. C. (1992).** *Multivariate stochastic approximation using a simultaneous perturbation gradient approximation (SPSA).* IEEE Transactions on Automatic Control, Vol 37(3), pp. 332–341. [[link]](https://doi.org/10.1109/9.119632) — *Foundational 2-forward-pass simultaneous perturbation gradient estimator.*
26. **Salimans, T., Ho, J., Chen, X., Sidor, S., & Sutskever, I. (2017).** *Evolution Strategies as a Scalable Alternative to Reinforcement Learning.* arXiv preprint arXiv:1703.03864. [[PDF]](http://arxiv.org/pdf/1703.03864.pdf) — *Black-box parameter-space perturbation alternative to backpropagation.*
27. **Malladi, S., Gao, T., Nichani, E., Damian, A., Lee, J. D., Chen, D., & Arora, S. (2023).** *Fine-Tuning Language Models with Just Forward Passes (MeZO).* Advances in Neural Information Processing Systems (NeurIPS). [[PDF]](http://arxiv.org/pdf/2305.17333.pdf) — *In-place SPSA zeroth-order LLM fine-tuning using only forward passes.*
28. **Papailiopoulos, D. (2026).** *AdderBoard: A Community Challenge for Tiny Arithmetic Transformers.* GitHub repository. [[link]](https://github.com/anadim/AdderBoard) — *Community leaderboard demonstrating hand-crafted, constructively programmed weight matrices for tiny arithmetic transformers.*

---

### 6.2 Extended Mechanistic Interpretability, Weight Compilation, & Editing Canon (29–82)

29. **Takehi, R., Teoh, J., Zhang, A., & Khattab, O. (2026).** *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights.* MIT CSAIL. Blog: `https://rikiyat.github.io/blog/weight-space-programming/` | alphaXiv: `https://www.alphaxiv.org/abs/2611.weight-space-programming` | HF: `RikiyaT/Qwen3.5-0.8B-weight-space-programming` | GitHub: `https://github.com/RikiyaT/lm_understand-logs`.
30. **Olsson, C., Elhage, N., Nanda, N., Joseph, N., DasSarma, N., Henighan, T., Mann, B., Askell, A., Bai, Y., Chen, A., Conerly, T., Drain, D., Ganguli, D., Hatfield-Dodds, Z., Hernandez, D., Johnston, S., Jones, A., Kernion, J., Lovitt, L., Ndousse, K., Amodei, D., Brown, T., Clark, J., Kaplan, J., McCandlish, S., & Olah, C. (2022).** *In-context Learning and Induction Heads.* Transformer Circuits Thread, arXiv:2209.11895.
31. **Geva, M., Schuster, R., Berant, J., & Levy, O. (2021).** *Transformer Feed-Forward Layers Are Key-Value Memories.* In Proceedings of EMNLP 2021, arXiv:2012.14913.
32. **Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T., Kravec, S., Hatfield-Dodds, Z., Lasenby, R., Drain, D., Chen, C., Grosse, R., McCandlish, S., Kaplan, J., Amodei, D., Wattenberg, M., & Olah, C. (2022).** *Toy Models of Superposition.* Transformer Circuits Thread, arXiv:2209.10652.
33. **nostalgebraist (2020).** *interpreting GPT: the logit lens.* LessWrong.
34. **Belrose, N., Furman, Z., Smith, L., Halawi, D., Ostrovsky, I., McKinney, L., Biderman, S., & Steinhardt, J. (2023).** *Eliciting Latent Predictions from Transformers with the Tuned Lens.* arXiv:2303.08112.
35. **Belrose, N., Schneider-Joseph, D., Ravfogel, S., Cotterell, R., Raff, E., & Biderman, S. (2023).** *LEACE: Perfect linear concept erasure in closed form.* Advances in Neural Information Processing Systems (NeurIPS 2023), arXiv:2306.03819.
36. **Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., Pan, A., Yin, X., Mazeika, M., Dombrowski, A.-K., Goel, S., Li, N., Byun, M. J., Wang, Z., Mallen, A., Basart, S., Koyejo, S., Song, D., Fredrikson, M., Kolter, J. Z., & Hendrycks, D. (2023).** *Representation Engineering: A Top-Down Approach to AI Transparency.* arXiv:2310.01405.
37. **Turner, A. M., Thiergart, L., Leech, G., Udell, D., Vazquez, J. J., Mini, U., & MacDiarmid, M. (2023).** *Steering Language Models With Activation Engineering.* arXiv:2308.10248.
38. **Ravfogel, S., Elazar, Y., Gonen, H., Twiton, M., & Goldberg, Y. (2020).** *Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection (INLP).* ACL 2020, arXiv:2004.07667.
39. **Meng, K., Sharma, A. S., Andonian, A., Belinkov, Y., & Bau, D. (2023).** *Mass-Editing Memory in a Transformer (MEMIT).* ICLR 2023, arXiv:2210.07229.
40. **Fang, J., Jiang, H., Wang, K., Ma, Y., Wang, X., He, X., & Chua, T.-S. (2025).** *AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models.* ICLR 2025 (Outstanding Paper), arXiv:2410.02355.
41. **Mitchell, E., Lin, C., Bosselut, A., Finn, C., & Manning, C. D. (2022).** *Fast Model Editing at Scale (MEND).* ICLR 2022, arXiv:2110.11309.
42. **Dunefsky, J., Chlenski, P., & Nanda, N. (2024).** *Transcoders Find Interpretable LLM Feature Circuits.* NeurIPS 2024, arXiv:2406.11944.
43. **Lindsey, J., Gurnee, W., Ameisen, E., Chen, B., Pearce, A., Turner, N. L., Citro, C., Abrahams, D., Carter, S., Hosmer, B., Marcus, J., Sklar, M., Templeton, A., Bricken, T., McDougall, C., Cunningham, H., Henighan, T., Jermyn, A., Jones, A., Persic, A., Qi, Z., Thompson, T. B., Zimmerman, S., Rivoire, K., Conerly, T., Olah, C., & Batson, J. (2025).** *Circuit Tracing: Revealing Computational Graphs in Language Models.* Anthropic Transformer Circuits Thread (`https://transformer-circuits.pub/2025/attribution-graphs/methods.html`).
44. **Ameisen, E., Lindsey, J., Pearce, A., Gurnee, W., Turner, N. L., Chen, B., Citro, C., Abrahams, D., Carter, S., Hosmer, B., Marcus, J., Sklar, M., Templeton, A., Bricken, T., McDougall, C., Cunningham, H., Henighan, T., Jermyn, A., Jones, A., Persic, A., Qi, Z., Thompson, T. B., Zimmerman, S., Rivoire, K., Conerly, T., Olah, C., & Batson, J. (2025).** *On the Biology of a Large Language Model.* Anthropic Transformer Circuits Thread (`https://transformer-circuits.pub/2025/attribution-graphs/biology.html`).
45. **Lieberum, T., Rajamanoharan, S., Conmy, A., Smith, L., Sonnerat, N., Varma, V., Kramár, J., Dragan, A., Shah, R., & Nanda, N. (2024).** *Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2.* DeepMind, arXiv:2408.05147.
46. **Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2023).** *Sparse Autoencoders Find Highly Interpretable Features in Language Models.* ICLR 2024, arXiv:2309.08600.
47. **Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., Turner, N., Anil, C., Denison, C., Askell, A., Lasenby, R., Wu, Y., Kravec, S., Schiefer, N., Maxwell, T., Joseph, N., Hatfield-Dodds, Z., Tamkin, A., Nguyen, K., McLean, B., Burke, J. E., Hume, T., Carter, S., Henighan, T., & Olah, C. (2023).** *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning.* Transformer Circuits Thread.
48. **Syed, A., Rager, C., & Conmy, A. (2023).** *Attribution Patching Outperforms Automated Circuit Discovery.* arXiv:2310.10348.
49. **Kramár, J., Lieberum, T., Shah, R., & Nanda, N. (2024).** *AtP*: An efficient and scalable method for localizing LLM behaviour to components.* arXiv:2403.00745.
50. **Hanna, M., Pezzelle, S., & Belinkov, Y. (2024).** *Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms (EAP-IG).* COLM 2024, arXiv:2403.17806.
51. **Conmy, A., Mavor-Parker, A. N., Lynch, A., Heimersheim, S., & Garriga-Alonso, A. (2023).** *Towards Automated Circuit Discovery for Mechanistic Interpretability (ACDC).* NeurIPS 2023, arXiv:2304.14997.
52. **Todd, E., Li, M. L., Sharma, A. S., Mueller, A., Wallace, B. C., & Bau, D. (2024).** *Function Vectors in Large Language Models.* ICLR 2024, arXiv:2310.15213.
53. **Ilharco, G., Ribeiro, M. T., Wortsman, M., Gururangan, S., Schmidt, L., Hajishirzi, H., & Farhadi, A. (2023).** *Editing Models with Task Arithmetic.* ICLR 2023, arXiv:2212.04089.
54. **Yadav, P., Tam, D., Choshen, L., Raffel, C., & Bansal, M. (2023).** *TIES-Merging: Resolving Interference When Merging Models.* NeurIPS 2023, arXiv:2306.01708.
55. **Yu, L., Yu, B., Yu, H., Huang, F., & Li, Y. (2024).** *Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch (DARE).* ICML 2024, arXiv:2311.03099.
56. **Ha, D., Dai, A., & Le, Q. V. (2016).** *HyperNetworks.* ICLR 2017, arXiv:1609.09106.
57. **Schürholt, K., Kostadinov, D., & Borth, D. (2021).** *Self-Supervised Representation Learning on Neural Network Weights for Model Characteristic Prediction.* NeurIPS 2021, arXiv:2106.15928.
58. **Zhang, A., Khattab, O., et al. (2025).** *Recursive Language Models (RLMs).* MIT CSAIL.
59. **Lindsey, J., Templeton, A., Marcus, J., Conerly, T., Batson, J., & Olah, C. (2024).** *Sparse Crosscoders for Cross-Layer Features and Model Diffing.* Anthropic Transformer Circuits Thread (`https://transformer-circuits.pub/2024/crosscoders/index.html`).
60. **Millidge, B., & Black, S. (2022).** *The Singular Value Decompositions of Transformer Weight Matrices are Highly Interpretable.* LessWrong / Alignment Forum.
61. **Bereska, L., & Gavves, E. (2024).** *Mechanistic Interpretability for AI Safety — A Review.* TMLR 2024, arXiv:2404.05107.
62. **Marks, S., & Tegmark, M. (2024).** *The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets.* COLM / ICLR 2024, arXiv:2310.06824.
63. **Wu, Z., Arora, A., Wang, Z., Geiger, A., Jurafsky, D., Manning, C. D., & Potts, C. (2024).** *ReFT: Representation Finetuning for Language Models (LoReFT).* NeurIPS 2024, arXiv:2404.03592.
64. **Geiger, A., Wu, Z., Potts, C., Icard, T., & Goodman, N. (2024).** *Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations (DAS).* Causal Learning and Reasoning (CLeaR 2024), arXiv:2303.02536.
65. **Pearce, G., Dooms, T., Riggs, L., & Sharkey, L. (2024).** *Bilinear MLPs Enable Weight-Based Mechanistic Interpretability.* arXiv:2406.03897.
66. **Sharkey, L., Chughtai, B., Batson, J., et al. (2025).** *Open Problems in Mechanistic Interpretability (Weight-Space vs. Activation-Space Dictionary Learning).* TMLR / arXiv:2501.06126.
67. **Weiss, G., Goldberg, Y., & Yahav, E. (2021).** *Thinking Like Transformers (RASP: Restricted Access Sequence Processing).* ICML 2021, arXiv:2106.06981.
68. **Lindner, D., Kramár, J., Farquhar, S., Rahtz, M., McGrath, T., & Mikulik, V. (2023).** *Tracr: Compiled Transformers as a Laboratory for Interpretability.* NeurIPS 2023, arXiv:2301.05062.
69. **Giannou, A., Rajput, S., Sohn, J.-Y., Lee, K., Lee, J. D., & Papailiopoulos, D. (2023).** *Looped Transformers as Programmable Computers.* ICML 2023, arXiv:2301.13196.
70. **Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023).** *Progress measures for grokking via mechanistic interpretability (Fourier Arithmetic Circuits).* ICLR 2023, arXiv:2301.05217.
71. **Ainsworth, S. K., Hayase, J., & Srinivasa, S. (2023).** *Git Re-Basin: Merging Models modulo Permutation Symmetries.* ICLR 2023, arXiv:2209.04836.
72. **Navon, A., Shamsian, A., Achituve, I., Fetaya, E., Chechik, G., & Maron, H. (2023).** *Equivariant Architectures for Learning in Deep Weight Spaces.* ICML 2023, arXiv:2301.12780.
73. **Barbero, F., Vitvitskyi, A., Perivolaropoulos, C., Pascanu, R., & Veličković, P. (2025).** *Round and Round We Go! What makes Rotary Positional Encodings useful?* ICLR 2025, arXiv:2410.06205.
74. **Cancedda, N. (2024).** *Spectral Filters, Dark Signals, and Attention Sinks.* ACL 2024, arXiv:2403.00504.
75. **Sun, M., Chen, X., Kolter, J. Z., & Liu, Z. (2024).** *Massive Activations in Large Language Models.* COLM 2024, arXiv:2402.17762.
76. **Moschella, L., Maiorca, V., Fumero, M., Norelli, A., Locatello, F., & Rodolà, E. (2023).** *Relative Representations Enable Zero-Shot Latent Space Communication (Orthogonal Procrustes Stitching).* ICLR 2023, arXiv:2209.15430.
77. **Zou, A., Phan, L., Wang, J., Duenas, D., Lin, M., Andriushchenko, M., Wang, R., Kolter, J. Z., Fredrikson, M., & Hendrycks, D. (2024).** *Improving Alignment and Robustness with Circuit Breakers (Representation Rerouting).* NeurIPS 2024, arXiv:2406.04313.
78. **Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F., & Sanghai, S. (2023).** *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* EMNLP 2023, arXiv:2305.13245.
79. **Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. (2024).** *Efficient Streaming Language Models with Attention Sinks (StreamingLLM).* ICLR 2024, arXiv:2309.17453.
80. **Rimsky, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., & Turner, A. M. (2024).** *Steering Llama 2 via Contrastive Activation Addition (CAA).* ACL 2024, arXiv:2312.06681.
81. **Sun, M., Liu, Z., Bair, A., & Kolter, J. Z. (2024).** *A Simple and Effective Pruning Approach for Large Language Models (Wanda).* ICLR 2024, arXiv:2306.11695.
82. **Wollschläger, T., Elstner, J., Geisler, S., Cohen-Addad, V., Günnemann, S., & Gasteiger, J. (2025).** *The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence.* ICML 2025, arXiv:2502.17420.
