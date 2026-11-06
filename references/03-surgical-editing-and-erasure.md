> **Skill Navigation (`weight-space-programming-skill`):**
> [SKILL.md](../SKILL.md) | [01: Sparks & Scaling (§1)](./01-sparks-and-scaling-regimes.md) | [02: Weight Anatomy (§2)](./02-transformer-weight-anatomy.md) | [03: Surgical Editing & Erasure (§3.1–3.7)](./03-surgical-editing-and-erasure.md) | [04: Head Synthesis & Compilation (§3.8–3.16)](./04-head-synthesis-and-compilation.md) | [05: Applied Blueprint & Runbook (§4–5)](./05-applied-blueprint-and-runbook.md) | [06: Bibliography (§6)](./06-annotated-bibliography.md) | [Full Manual](./mechanistic_interpretability.md) | [PyTorch Script](../scripts/weight_space_programming.py)
>
> *Shard `03-surgical-editing-and-erasure.md` — Lines 221–325 of `mechanistic_interpretability.md`*
<!-- SHARD_BODY_START -->
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

