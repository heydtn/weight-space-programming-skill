---
name: weight-space-programming-skill
description: >-
  Use this skill when the user wants to read, audit, modify, repair, or synthesize
  neural network weights directly in weight space without standard gradient-descent
  training—including mechanistic interpretability, column-selective MLP surgery,
  LEACE concept erasure, closed-form classifier head synthesis (Shrinkage LDA / NLI
  grafting), ROME/MEMIT/AlphaEdit, directional ablation (abliteration), attention QK/OV
  rewiring, or compiling algorithms into Transformer weights from scratch.
---

# Weight-Space Programming & Mechanistic Interpretability Skill

This skill provides mathematical primitives, PyTorch implementations, and verification protocols for **Weight-Space Programming**—treating neural network weight tensors as readable and writable code rather than opaque gradient-descent artifacts.

## Skill Resources

Before planning or executing any weight-space inspection, model edit, or from-scratch weight synthesis, consult the bundled reference manual and standalone PyTorch module:

* **[references/mechanistic_interpretability.md](./references/mechanistic_interpretability.md)** (1,175 lines, 82 citations)
  * **Section 1:** *Sparks of Weight-Space Programming* (Takehi et al., Nov 2026, using `GPT-6 Astra`) — All 8 experiments (`Exp 1`–`Exp 8b`), forward-pass scaling regimes (`0, 10, 25, 50` passes), surgical 131K-parameter `mlp.down_proj` repair, and the `Pipis et al. (2025)` risk-aversion duality between reasoning loops and premature assistant turn-yielding.
  * **Section 2:** *Mathematical Anatomy of Transformer Weights* — Residual stream decomposition, MHA vs. GQA ($W_{QK}^h = (W_Q^h)^\top W_K^{g(h)}$, $W_{OV}^h = W_O^h W_V^{g(h)}$), SwiGLU/GeGLU key-value memories, static Logit Lens with exact `RMSNorm` ($\text{diag}(\gamma)$) & `LayerNorm` ($\text{diag}(\gamma) P_{\mathbf{1}^\perp}$) folding, SVD spectrum & bilinear MLP eigendecomposition ($B_u$ with gated polarity sign check), weight-space gauge symmetries & `Wanda` salience, 0-based RoPE high/low-frequency head programming (`Barbero et al., 2025`), and Attention Sink / Massive Activation preservation (`Xiao et al., 2024`; `Sun et al., 2024`).
  * **Section 3:** *16 State-of-the-Art Weight-Space Programming Techniques* — Column-Selective `down_proj` Surgery (signed $s_j > 0$), Directional Ablation (Abliteration / Polyhedral Concept Cones) & Additive Steering Folding (`RepE`/`CAA`), Closed-Form LEACE (`Belrose et al., 2023`), ROME/MEMIT/AlphaEdit ($m \times m$ SPD Woodbury Null-Space Surgery), Cross-Layer Transcoders / Crosscoders / Attribution Graphs + Direct Dormant-Neuron SAE Compilation, `AtP*` & `DLA`, Function Vectors & Task Arithmetic (`TIES`/`DARE` + `Git Re-Basin` permutation alignment), Closed-Form Classifier Head Synthesis (NLI Grafting + Between-Class Mahalanobis Temperature-Calibrated Shrinkage Fisher LDA), Geometry of Truth (Mass-Mean), Exact 3-Neuron Polarized GeGLU Conjunctive AND-Gate Hijacking, GQA-Safe Low-Frequency RoPE $W_Q^h$ Routing Surgery, LoReFT/DAS Weight Folding, Weight Oracles (`arXiv:2610.07334`) & Natural Language Autoencoders (`NLAs`), Rank-Safe LEACE-Preserving Subspace Optimization (`MeZO`/`SPSA`/`ES` + `GEPA`/`Meta-Harness`/`OPAL`), Constructive Weight Compilation (`RASP`/`Tracr`/Fourier Arithmetic/`AdderBoard`/`p-diff`/`NeuroGen`), and Orthogonal Procrustes Weight Grafting.
  * **Section 4:** *Complete End-to-End PyTorch Blueprint* — 3-class `"Can Continue?"` ModernBERT classifier (`Mode A` 0-forward-pass static synthesis in norm-matched post-`model.head` space & `Mode B` `program_can_continue_classifier` orchestrating upstream Column-Selective `Wo` Surgery + 3-Neuron Polarized GeGLU Hijacking $\to$ post-surgery forward pass $\to$ LEACE + Between-Class Calibrated Shrinkage LDA + Rank-Safe Subspace SPSA).
  * **Section 5 & 6:** *Agent Skill Runbook & Annotated 82-Paper Bibliography* (including all 28 official citations from `rikiyat.github.io/blog/weight-space-programming/`).
* **[scripts/weight_space_programming.py](./scripts/weight_space_programming.py)** — Standalone PyTorch module implementing `compute_leace_projection`, `surgical_down_proj_edit`, `synthesize_closed_form_classifier_head`, `hijack_dormant_geglu_neurons`, `subspace_spsa_f1_refinement`, `synthesize_zero_data_nli_head`, and `program_can_continue_classifier`.

---

## Quick Triage & Workflow

1. **Identify the Task & Forward-Pass Budget (See Manual §5.1):**
   * **0 Forward Passes (Pure Static Weight Reading / Synthesis):**
     * Fold pre-norm weights into downstream matrices before static linear algebra: $W_{\text{in}} \leftarrow W_{\text{in}}\text{diag}(\gamma)$ for `RMSNorm`, or $W_{\text{in}} \leftarrow W_{\text{in}}\text{diag}(\gamma)(I_d - \frac{1}{d}\mathbf{1}\mathbf{1}^\top)$ for `LayerNorm` (handling `layers[0].attn_norm = nn.Identity()` in `ModernBERT`, Manual §2.4).
     * Compute SVD of $W_{OV}^h = U\Sigma V^\top$ and project onto $W_E^\top$ / $W_U$ ($W_{\text{eff}}$), or eigendecompose the globally exact symmetric bilinear MLP interaction matrix $B_u = \frac{1}{2}(W_{\text{gate}}^\top D_u W_{\text{up}} + W_{\text{up}}^\top D_u W_{\text{gate}})$ and resolve eigenvector $\pm v_k$ sign via gated polarity (Manual §2.5).
     * For **from-scratch synthesis**, partition the residual stream into orthogonal registers $\mathcal{R}_1 \oplus \mathcal{R}_2 \dots$, compile boolean/positional lookups into $W_{QK}^h$ (using high-frequency RoPE phase offsets $R(-k\theta_m)$ on split-half rows $\mathcal{I}_m = [m, m + d_{\text{head}}/2]$ for relative offsets or low-frequency planes for semantic matching, Manual §2.7), value routing into $W_{OV}^h$, and exact 3-neuron polarized conjunctive logic gates $\frac{1}{2}[\psi(\kappa(z_A+z_B)) - \psi(\kappa z_A) - \psi(\kappa z_B)]$ into $(W_{\text{gate}}, W_{\text{up}}, W_{\text{down}})$ (Manual §3.10, §3.15).
     * For **0-example sequence classifiers**, call `synthesize_zero_data_nli_head` on an NLI-pretrained encoder (`tasksource/ModernBERT-large-nli`), which dynamically resolves NLI row indices via `model.config.label2id`, maps token-embedding prototypes through `embeddings.norm` $\to$ `final_norm` $\to$ `model.head`, contrast-centers against the `model.head` `GELU` cone offset and normalizes to $\bar{\nu}_{\text{NLI}}$, orthogonalizes against centered polite tokens, wires `classifier.weight` & `classifier.bias`, and hijacks 3 dormant GeGLU neurons along the exact linearized margin pullback $W_{\text{eff}}^\top [1, -\frac{1}{2}, -\frac{1}{2}]^\top$ (Manual §3.8A, §4.2, §4.3).
   * **1–2 Forward Batches (Closed-Form Classifier / Concept Erasure / Null-Space Editing):**
     * Always execute **upstream MLP weight edits first** (Layers 18 & 22), then run a fresh forward pass (respecting `model.config.classifier_pooling`: `"cls"` vs. `"mean"` and passing through `model.head`) to extract post-surgery representations $H$ (Manual §4.3).
     * Erase nuisance confounders in closed form via **LEACE** ($W = \Sigma_{XX}^{-1/2}, W^+ = \Sigma_{XX}^{1/2}, P_V = U_V U_V^\top$ from thin SVD of $V = W\Sigma_{XZ}$, $P_{\text{LEACE}} = I_d - W^+ P_V W$, Manual §3.3) and fold $P_{\text{LEACE}}$ into a linear layer reading the complete representation (`model.classifier`), never into only the last layer's `mlp.Wo` (which is bypassed by the residual skip stream).
     * Synthesize `classifier.weight` and `classifier.bias` analytically in `<50ms` via **Zero-Sum Centered, Between-Class Mahalanobis Temperature-Calibrated Shrinkage Fisher LDA** (`torch.linalg.solve` on $\mu_c - \bar{\mu}$, with bias $-\frac{1}{2}(\mu_c + \bar{\mu})^\top w_{\text{lda}, c} + \log\pi_c + w_{\text{lda}, c}^\top b_{\text{LEACE}}$, Manual §3.8B, §4.3).
     * For injecting $m$ key-value rules into MLPs without collateral damage, use the **$m \times m$ SPD Woodbury form of AlphaEdit** ($\Delta W_{\text{down}} = (V_1 - W_{\text{down}}K_1)(\tilde{K}_1^\top \tilde{K}_1 + \lambda I_m)^{-1}\tilde{K}_1^\top$ with $\tilde{K}_1 = P_{\text{null}} K_1$, Manual §3.4) or **3-Neuron Polarized GeGLU Conjunctive Hijacking** (ranked by triple product norm $\|w_{\text{down}, j}\|_2 \|w_{\text{gate}, j}\|_2 \|w_{\text{up}, j}\|_2$, with $k_A, k_B$ unit-normalized and Löwdin-orthogonalized on active support $|\gamma_i| > 10^{-2}$ *before* multiplying by $\gamma^{-1}$, Manual §3.10).
   * **10–50 Forward Passes (Surgical Behavioral Repair / Subspace Optimization):**
     * Isolate the unit contrastive direction $u_\ell$ using attention-masked pooling, orthogonalized against the true residual-stream `<BOS>`/`[CLS]` massive-activation sink $\hat{s}_{\text{sink}} = x_{\ell-1, 0}/\|x_{\ell-1, 0}\|_2$ (Manual §2.7, §3.9).
     * Rank `mlp.down_proj` (`Wo`) columns by **signed positive** differential contribution $s_j = (\bar{a}_j^+ - \bar{a}_j^-)\langle u_\ell, w_{\text{down}, j}\rangle > 0$ and apply **Column-Selective Rank-1 Surgery** to the top $K=64$ positive columns in peak layers (Takehi et al., 2026; Manual §1.3, §3.1).
     * Optionally refine decision thresholds on non-differentiable metrics (e.g., Macro-F1) using **Scale-Invariant Mean-Centered Rank-Safe LEACE-Preserving Subspace SPSA** ($W_{\text{cand}} = A W_{\text{base}}$ with $\|A\|_F = \sqrt{3}$ evaluated on centered logits $Z_{\text{centered}} = Z_{\text{base}} - \mu_Z$, using hybrid hard/RMS-normalized smooth F1 finite differences to traverse discrete argmax plateaus while ratcheting strictly on exact discrete F1, Manual §3.14, §4.3).

2. **Mandatory Verification Protocol (See Manual §5.2):**
   * Always compute matrix inversions, SVDs, and eigendecompositions in `float32`/`float64`, then cast back to the model's exact native `.dtype` and `.shape`.
   * In **GQA models** (`Qwen3.5`, `Llama-3`, `Gemma 3`), never edit shared $W_K^{g(h)}$ or $W_V^{g(h)}$ to modify a single query head—edit private $W_Q^h$ (in low-frequency RoPE planes) or private $W_O^h$ instead.
   * Run a **Negative Control** edit using a random orthogonal unit vector $u_\perp \perp u$ of identical norm to prove causal specificity.
   * Verify **Null-Space / Locality Preservation** on unrelated clean inputs ($K_0$) and **Counterfactual Confounder Invariance** on nuisance-flipped pairs.

