# baselines_alphaECA — Architectural Baseline Comparison for αECA

> Companion code submission.  
> Trains and evaluates four architectural baselines against the signal-matched transformer **αM** on the α-asynchronous ECA identification task.

---

## Overview

This repository contains a **single self-contained script** — `baselines_alphaECA.py` — that reproduces the architectural comparison in **Table 3** of the paper.

The script trains four baselines under the **identical two-phase curriculum** used to train αM, then compares them against the reported αM numbers. No other files are needed beyond the pre-generated dataset.

---

## File Structure

```
.
├── baselines_alphaECA.py     ← the only source file (this repo)
├── README.md                 ← this file
│
├── ECA_Data_New/             ← pre-generated dataset (not included, see below)
│   ├── phase1/
│   │   └── train/
│   │       ├── orbits.npy
│   │       ├── rule_bits.npy
│   │       └── alphas.npy
│   └── phase2/
│       ├── train/
│       │   ├── orbits.npy
│       │   ├── rule_bits.npy
│       │   └── alphas.npy
│       └── test/
│           ├── orbits.npy
│           ├── rule_bits.npy
│           └── alphas.npy
│
└── baseline_results/         ← created automatically on first run
    ├── results_summary.csv   ← paste into paper as Table 3
    ├── results_detail.json   ← full per-model breakdown
    ├── <model>_phase1.pt     ← best Phase 1 checkpoint per model
    └── <model>_phase2.pt     ← best Phase 2 checkpoint per model
```

---

## The Four Baselines

| Key | Model | Params | What it tests |
|---|---|---|---|
| `mlp` | MLP on handcrafted statistics | ~75K | Whether 24 engineered orbit features suffice |
| `cnn` | 3-layer 2D-CNN on raw orbit image | ~210K | Local-window limitation claim (Section 4) |
| `bilstm` | 2-layer BiLSTM, row-wise | ~310K | Whether sequential state approximates global attention |
| `vanilla_transformer` | Transformer with 1D-PE + mean-pool | ~1.76M | Ablation of TripletPE2D and StatPool specifically |

The Vanilla Transformer is the **most important baseline** — it is matched to αM in parameter count, tokenisation scheme, encoder depth, and all training hyperparameters. It differs from αM **only** in replacing TripletPE2D with flat 1D sinusoidal PE and StatPool with mean pooling. Any gap between it and αM isolates exactly the contribution of those two components.

Two **trivial baselines** (random and majority-class) are also computed from the test set without any training and are always included in the output table.

---

## Requirements

```
Python  >= 3.10
PyTorch >= 2.0
NumPy   >= 1.24
```

Install with:

```bash
pip install torch numpy
```

No other dependencies. The script uses only the Python standard library plus PyTorch and NumPy.

---

## Usage

**Run all four baselines (default):**

```bash
python baselines_alphaECA.py \
  --data_dir   ECA_Data_New \
  --output_dir baseline_results
```

**Run specific baselines only:**

```bash
python baselines_alphaECA.py --models cnn vanilla_transformer
python baselines_alphaECA.py --models mlp
```

Available model keys: `mlp`, `cnn`, `bilstm`, `vanilla_transformer`

---

## Training Protocol

All baselines follow αM's training protocol exactly. Every decision below is cross-referenced to the paper appendix.

### Two-Phase Curriculum

**Phase 1** (50 epochs) — Rule identification warm-up:
- Data: synchronous orbits only (α = 1.0)
- Noise head: **OFF** (λ_α = 0.0)
- LR: 3×10⁻⁴ → 1×10⁻⁵ cosine annealing
- Val split: 15% of Phase 1 train data (5,370 samples), val-first split order

**Phase 2** (30 epochs) — Full noise identification:
- Data: full α range {0.1, 0.2, …, 1.0}
- Noise head: **ON** (λ_α = 0.3)
- Init: best Phase 1 checkpoint (by val rule exact match)
- LR: 1×10⁻⁴ → 1×10⁻⁶ cosine annealing
- Val split: 15% of Phase 2 train data (13,425 samples), val-first split order

### Optimizer and Regularisation

| Hyperparameter | Value | Source |
|---|---|---|
| Optimizer | AdamW | Table 23 |
| β₁, β₂ | 0.9, 0.999 | Table 23 |
| ε | 1×10⁻⁸ | Table 23 |
| Weight decay | 1×10⁻² | Table 23 |
| Gradient clip | 1.0 | Table 23 |
| Effective batch size | 128 | Table 23 |
| Physical batch size | 8 | Table 23 |
| Gradient accumulation steps | 16 | Table 23 |
| Dropout | 0.1 | App C |
| Weight init | N(0, 0.02) / zeros | App C |

### Loss Function

```
L = λ_rule × BCE(rule_logits, rule_bits) + λ_α × CE(alpha_logits, alpha_idx)
```

- Phase 1: λ_rule = 1.0, λ_α = 0.0
- Phase 2: λ_rule = 1.0, λ_α = 0.3

### Validation and Checkpointing

- Val metric: **rule exact match** (both phases)
- Strategy: train all epochs, save best val checkpoint (patience = ∞)
- Phase 2 loads the best Phase 1 checkpoint, not the last epoch

---

## Metrics

| Metric | Definition |
|---|---|
| **Rule exact match** | All 8 bits of the ECA rule correct (strict) |
| **α tolerance accuracy** | Predicted α within ±0.05 of true α |

---

## Output Files

After a run, `baseline_results/` contains:

- **`results_summary.csv`** — ready to paste into paper Table 3
- **`results_detail.json`** — full breakdown including parameter counts
- **`<model>_phase1.pt`** — best Phase 1 checkpoint (keyed by val rule acc)
- **`<model>_phase2.pt`** — best Phase 2 checkpoint (keyed by val rule acc)

Example console output:

```
========================================================================
  ARCHITECTURAL COMPARISON — αECA (held-out test rules, Phase 2 test)
  Rule metric : 8-bit exact match
  Noise metric: tolerance accuracy ±0.05
========================================================================
  Model                                           Params   Rule%  α-tol%
------------------------------------------------------------------------
  Random baseline (1/256 rule, 1/10 α)                 0    0.39   10.00
  Majority-class baseline                               0    x.xx   xx.xx
  MLP on handcrafted statistics                    75,000   xx.xx   xx.xx
  CNN (2D-conv, raw orbit image)                  210,000   xx.xx   xx.xx
  BiLSTM (2-layer, row-wise)                      310,000   xx.xx   xx.xx
  Vanilla Transformer (1D-PE, mean-pool)        1,760,000   xx.xx   xx.xx
  αM — signal-matched transformer (Table 2)     1,760,000   99.22   95.08  ← ours
========================================================================
```

---

## Important Notes for Paper Integration

### Hybrid estimator gap (F11)

αM's reported noise accuracy (95.08%, Table 2) is measured **with** the hybrid estimator at inference time. All baselines in this script use **only** the neural alpha head. This means:

> The reported αM advantage on noise accuracy is a **lower bound** on the true advantage.

For an apples-to-apples comparison, obtain the neural-head-only αM noise accuracy and add a table footnote clarifying this. The rule accuracy comparison is unaffected (both use the neural rule head only).

### Paper integration (Section 4)

Add the following paragraph after the αM description in Section 4:

> *"To isolate the contribution of each architectural component, we train four architectural baselines on αECA under the identical two-phase curriculum: an MLP on handcrafted orbit statistics, a 2D-CNN treating the orbit as an image, a BiLSTM processing rows sequentially, and a vanilla transformer with standard 1-D positional encoding and mean pooling replacing TripletPE2D and StatPool respectively. αECA is chosen as the canonical single-rule single-noise setting; since the architectural argument — that global attention with signal-matched inductive biases is necessary for orbit-level statistical inference — is noise-type agnostic, results on αECA are representative. Detailed results are in Table 3."*

---

## Fix Log Summary

The script incorporates 14 fixes relative to an earlier draft. The most significant:

| Fix | Description |
|---|---|
| F1 | StatPool hidden dims corrected to 512→256 (+LN+Dropout)→128 per App C eq. 990 |
| F2 | Alpha head corrected to 3-layer (128→64→32→10), Dropout on first layer only |
| F3 | Rule head Dropout confirmed present per App C eq. 974 |
| F5 | Phase 1 val split reinstated — App D.4 is authoritative; App B.2.4 was misread |
| F6 | Phase 2 LR corrected to 1×10⁻⁴ peak, 1×10⁻⁶ min |
| F7 | λ_α corrected to 0.3 in Phase 2 (not 1.0) |
| F12 | `random_split` order corrected to `[n_val, n_tr]` (val first) per App D.4 |

Full fix log is in the module docstring of `baselines_alphaECA.py`.

---

## Reproducibility

All random seeds are fixed:

```python
torch.manual_seed(42)
np.random.seed(42)
# val splits: torch.Generator().manual_seed(42)
```

Results are fully reproducible on the same hardware. Minor numerical differences are expected across GPU architectures due to floating-point non-associativity in cuDNN.
