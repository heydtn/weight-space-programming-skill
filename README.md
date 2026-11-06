# Weight-Space Programming & Mechanistic Interpretability Skill (`weight-space-programming-skill`)

An agentic coding skill for reading, auditing, surgically modifying, repairing, and synthesizing neural network weights directly in weight space without gradient-descent training.

## Repository Structure

```text
weight-space-programming-skill/
├── SKILL.md                                   # Main skill entry point & triage workflow
├── README.md                                  # Installation & overview
├── references/
│   └── mechanistic_interpretability.md        # Comprehensive 1,175-line reference manual (82 citations)
└── scripts/
    └── weight_space_programming.py            # Standalone PyTorch weight-surgery & head-synthesis primitives
```

## What's Inside

1. **[`SKILL.md`](./SKILL.md)** — Progressive-disclosure triage guide by forward-pass budget (`0` static passes, `1–2` calibration batches, or `10–50` surgical probing passes) and mandatory 6-point verification protocol.
2. **[`references/mechanistic_interpretability.md`](./references/mechanistic_interpretability.md)** — Full reference manual covering:
   - **Section 1:** *Sparks of Weight-Space Programming* ([Takehi et al., Nov 2026](https://rikiyat.github.io/blog/weight-space-programming/)) — Complete breakdown of all 8 experiments (`Exp 1`–`Exp 8b`), forward-pass scaling regimes (`0, 10, 25, 50` passes), surgical 131K-parameter `mlp.down_proj` repair on `Qwen3.5-0.8B`, and the `Pipis et al. (2025)` risk-aversion duality between reasoning loops and premature assistant turn-yielding.
   - **Section 2:** *Mathematical Anatomy of Transformer Weights as Editable Code* — Residual stream decomposition, MHA vs. GQA bilinear QK/OV circuits ($W_{QK}^h, W_{OV}^h$), SwiGLU/GeGLU key-value lookup tables, static Logit Lens with exact `RMSNorm` ($\text{diag}(\gamma)$) & `LayerNorm` ($\text{diag}(\gamma) P_{\mathbf{1}^\perp}$) folding, globally exact symmetric bilinear MLP eigendecomposition ($B_u$), weight-space gauge symmetries & `Wanda` salience, split-half (`rotate_half`) vs. interleaved RoPE plane programming, and `<BOS>`/`[CLS]` massive-activation sink preservation.
   - **Section 3:** *16 State-of-the-Art Weight-Space Programming Techniques* — Column-Selective `down_proj` Surgery, Directional Ablation (Abliteration / Polyhedral Concept Cones) & Weight-Folded Steering (`RepE`/`CAA`), Closed-Form Oblique LEACE (`Belrose et al., 2023`), ROME/MEMIT/AlphaEdit ($m \times m$ SPD Woodbury Null-Space Surgery), Cross-Layer Transcoders / Crosscoders / Attribution Graphs + Dormant-Neuron SAE Compilation, `AtP*` & `DLA`, Function Vectors & Task Arithmetic (`TIES`/`DARE` + `Git Re-Basin`), Closed-Form Classifier Head Synthesis (Zero-Data NLI Grafting + Zero-Sum Centered, Mahalanobis Temperature-Calibrated Shrinkage Fisher LDA), Geometry of Truth (Mass-Mean), Exact 3-Neuron Polarized GeGLU Conjunctive AND-Gate Hijacking (with symmetric Löwdin orthogonalization), Low-Frequency RoPE $W_Q^h$ Routing Surgery, LoReFT/DAS Weight Folding, Weight Oracles & Natural Language Autoencoders (`NLAs`), Scale-Invariant Rank-Safe LEACE-Preserving Subspace SPSA, Constructive Weight Compilation (`RASP`/`Tracr`/Fourier Arithmetic/`AdderBoard`/`p-diff`/`NeuroGen`), and Orthogonal Procrustes Weight Grafting.
   - **Section 4:** *Applied Blueprint: Programming the "Can Continue?" Classifier* — Both **Mode A** (0-data / 0-forward-pass pure static weight synthesis with linearized margin pullback) and **Mode B** (4-stage forward-probed pipeline).
   - **Sections 5 & 6:** *Agent Skill Runbook & Annotated 82-Paper Bibliography* (including all 28 official citations from *Sparks of Weight-Space Programming*).
3. **[`scripts/weight_space_programming.py`](./scripts/weight_space_programming.py)** — Importable PyTorch implementation of:
   - `compute_leace_projection`
   - `surgical_down_proj_edit`
   - `synthesize_closed_form_classifier_head`
   - `hijack_dormant_geglu_neurons`
   - `subspace_spsa_f1_refinement`
   - `synthesize_zero_data_nli_head`
   - `program_can_continue_classifier`

## Installation

### Global Skill (All Workspaces)
Clone or symlink into:
* Claude: `~/.claude/skills/weight-space-programming-skill`
* Gemini: `~/.gemini/config/skills/weight-space-programming-skill`

```bash
git clone https://github.com/heydtn/weight-space-programming-skill.git
```

### Workspace-Specific Skill
Clone into `.agents/skills/weight-space-programming-skill` inside your repository root:

```bash
mkdir -p .agents/skills
git clone https://github.com/heydtn/weight-space-programming-skill.git
```
