# Weight-Space Programming & Mechanistic Interpretability Skill (`weight-space-programming-skill`)

An agentic coding skill for reading, auditing, surgically modifying, repairing, and synthesizing neural network weights directly in weight space without gradient-descent training.

Designed for **progressive disclosure**: `SKILL.md` acts as a lightweight router with a line-indexed sitemap so agents can load only the specific topic shard (`references/01-*.md` to `06-*.md`, each `< 39 KB`) or specific PyTorch function needed for the task without loading the entire reference manual into context at once.

## Repository Structure

```text
weight-space-programming-skill/
├── SKILL.md                                           # Main skill entry point, routing table & triage workflow
├── README.md                                          # Installation & overview
├── references/
│   ├── 01-sparks-and-scaling-regimes.md               # §1: Takehi et al. (2026) 8 experiments & forward-pass budgets
│   ├── 02-transformer-weight-anatomy.md               # §2: QK/OV, GQA, GeGLU, Norm folding, SVD/B_u, RoPE, Sinks
│   ├── 03-surgical-editing-and-erasure.md             # §3.1–3.7: down_proj surgery, Abliteration, LEACE, AlphaEdit, CLTs
│   ├── 04-head-synthesis-and-compilation.md           # §3.8–3.16: NLI grafting, Shrinkage LDA, 3-Neuron GeGLU, SPSA, RASP
│   ├── 05-applied-blueprint-and-runbook.md            # §4–5: ModernBERT Mode A/B blueprint & 6-point verification runbook
│   ├── 06-annotated-bibliography.md                   # §6: 82 annotated papers (including all 28 official WSP citations)
│   └── mechanistic_interpretability.md                # Full unified 1,172-line reference manual (82 citations)
└── scripts/
    └── weight_space_programming.py                    # Standalone PyTorch weight-surgery & head-synthesis primitives
```

## What's Inside

1. **[`SKILL.md`](./SKILL.md)** — Progressive-disclosure router, section-by-section shard and line-range lookup table, standalone PyTorch function index, triage guide by forward-pass budget (`0` static passes, `1–2` calibration batches, or `10–50` surgical probing passes), and mandatory 6-point verification protocol.
2. **Modular Topic Shards (`references/01-*.md` – `06-*.md`) & Unified Manual ([`references/mechanistic_interpretability.md`](./references/mechanistic_interpretability.md)):**
   - **[`01-sparks-and-scaling-regimes.md`](./references/01-sparks-and-scaling-regimes.md) (Section 1):** *Sparks of Weight-Space Programming* ([Takehi et al., Nov 2026](https://rikiyat.github.io/blog/weight-space-programming/)) — Complete breakdown of all 8 experiments (`Exp 1`–`Exp 8b`), forward-pass scaling regimes (`0, 10, 25, 50` passes), surgical 131K-parameter `mlp.down_proj` repair on `Qwen3.5-0.8B`, and the `Pipis et al. (2025)` risk-aversion duality between reasoning loops and premature assistant turn-yielding.
   - **[`02-transformer-weight-anatomy.md`](./references/02-transformer-weight-anatomy.md) (Section 2):** *Mathematical Anatomy of Transformer Weights as Editable Code* — Residual stream decomposition, MHA vs. GQA bilinear QK/OV circuits ($W_{QK}^h, W_{OV}^h$), SwiGLU/GeGLU key-value lookup tables, static Logit Lens with exact `RMSNorm` ($\text{diag}(\gamma)$) & `LayerNorm` ($\text{diag}(\gamma) P_{\mathbf{1}^\perp}$) folding, globally exact symmetric bilinear MLP eigendecomposition ($B_u$), weight-space gauge symmetries & `Wanda` salience, split-half (`rotate_half`) vs. interleaved RoPE plane programming, and `<BOS>`/`[CLS]` massive-activation sink preservation.
   - **[`03-surgical-editing-and-erasure.md`](./references/03-surgical-editing-and-erasure.md) (Sections 3.1–3.7):** Column-Selective `down_proj` Surgery, Directional Ablation (Abliteration / Polyhedral Concept Cones) & Weight-Folded Steering (`RepE`/`CAA`), Closed-Form Oblique LEACE (`Belrose et al., 2023`), ROME/MEMIT/AlphaEdit ($m \times m$ SPD Woodbury Null-Space Surgery), Cross-Layer Transcoders / Crosscoders / Attribution Graphs + Dormant-Neuron SAE Compilation, `AtP*` & `DLA`, and Function Vectors & Task Arithmetic (`TIES`/`DARE` + `Git Re-Basin`).
   - **[`04-head-synthesis-and-compilation.md`](./references/04-head-synthesis-and-compilation.md) (Sections 3.8–3.16):** Closed-Form Classifier Head Synthesis (Zero-Data NLI Grafting + Zero-Sum Centered, Mahalanobis Temperature-Calibrated Shrinkage Fisher LDA), Geometry of Truth (Mass-Mean), Exact 3-Neuron Polarized GeGLU Conjunctive AND-Gate Hijacking (with symmetric Löwdin orthogonalization), Low-Frequency RoPE $W_Q^h$ Routing Surgery, LoReFT/DAS Weight Folding, Weight Oracles & Natural Language Autoencoders (`NLAs`), Scale-Invariant Rank-Safe LEACE-Preserving Subspace SPSA, Constructive Weight Compilation (`RASP`/`Tracr`/Fourier Arithmetic/`AdderBoard`/`p-diff`/`NeuroGen`), and Orthogonal Procrustes Weight Grafting.
   - **[`05-applied-blueprint-and-runbook.md`](./references/05-applied-blueprint-and-runbook.md) (Sections 4 & 5):** *Applied Blueprint: Programming the "Can Continue?" Classifier* — Both **Mode A** (0-data / 0-forward-pass pure static weight synthesis with linearized margin pullback) and **Mode B** (4-stage forward-probed pipeline), plus the *Agent Skill Runbook* (Task Triage Decision Tree & 6-Point Verification Checklist).
   - **[`06-annotated-bibliography.md`](./references/06-annotated-bibliography.md) (Section 6):** *Annotated 82-Paper Bibliography* (including all 28 official citations from *Sparks of Weight-Space Programming* in §6.1 and extended canon `29–82` in §6.2).
3. **[`scripts/weight_space_programming.py`](./scripts/weight_space_programming.py)** — Importable PyTorch implementation of:
   - `compute_leace_projection` (`L17–L57`)
   - `surgical_down_proj_edit` (`L60–L157`)
   - `synthesize_closed_form_classifier_head` (`L160–L236`)
   - `hijack_dormant_geglu_neurons` (`L239–L326`)
   - `subspace_spsa_f1_refinement` (`L329–L412`)
   - `synthesize_zero_data_nli_head` (`L415–L505`)
   - `program_can_continue_classifier` (`L508–L549`)

## Installation

### Global Skill (All Workspaces)
Clone or symlink into your agent's global skills directory:
* Claude Code: `~/.claude/skills/weight-space-programming-skill`
* Gemini: `~/.gemini/config/skills/weight-space-programming-skill`

```bash
git clone https://github.com/heydtn/weight-space-programming-skill.git ~/.claude/skills/weight-space-programming-skill
# or
git clone https://github.com/heydtn/weight-space-programming-skill.git ~/.gemini/config/skills/weight-space-programming-skill
```

### Workspace-Specific Skill
Clone into `.claude/skills/weight-space-programming-skill` or `.agents/skills/weight-space-programming-skill` inside your repository root:

```bash
mkdir -p .claude/skills
git clone https://github.com/heydtn/weight-space-programming-skill.git .claude/skills/weight-space-programming-skill

# or for .agents/skills harnesses:
mkdir -p .agents/skills
git clone https://github.com/heydtn/weight-space-programming-skill.git .agents/skills/weight-space-programming-skill
```
