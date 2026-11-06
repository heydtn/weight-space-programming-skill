> **Skill Navigation (`weight-space-programming-skill`):**
> [SKILL.md](../SKILL.md) | [01: Sparks & Scaling (§1)](./01-sparks-and-scaling-regimes.md) | [02: Weight Anatomy (§2)](./02-transformer-weight-anatomy.md) | [03: Surgical Editing & Erasure (§3.1–3.7)](./03-surgical-editing-and-erasure.md) | [04: Head Synthesis & Compilation (§3.8–3.16)](./04-head-synthesis-and-compilation.md) | [05: Applied Blueprint & Runbook (§4–5)](./05-applied-blueprint-and-runbook.md) | [06: Bibliography (§6)](./06-annotated-bibliography.md) | [Full Manual](./mechanistic_interpretability.md) | [PyTorch Script](../scripts/weight_space_programming.py)
>
> *Shard `05-applied-blueprint-and-runbook.md` — Lines 462–1079 of `mechanistic_interpretability.md`*
<!-- SHARD_BODY_START -->
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

