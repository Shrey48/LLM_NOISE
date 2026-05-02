# ECA Noise Identification — LLM & Transformer Benchmark

A benchmark repository for studying **rule and noise-parameter identification** in Elementary Cellular Automata (ECA) under four distinct noise models, using both purpose-built transformer architectures and large language models.

---

## Overview

Elementary Cellular Automata (ECA) are one-dimensional, two-state, three-neighbourhood systems that, despite minimal structure, exhibit a rich spectrum of dynamics — from fixed points and periodic attractors to complex chaotic regimes. Classical ECA dynamics are strictly local: each cell depends only on its left neighbour, itself, and its right neighbour.

This repository studies what happens when that locality is **systematically broken** through four distinct noise models, and asks: *can a model — transformer or LLM — recover the hidden rule and noise parameter from the observed orbit?*

The four noise models studied are:

| Model | Parameter | Mechanism |
|---|---|---|
| **Alpha-Asynchronous** | α ∈ {0.1 … 1.0} | Each cell independently updates with probability α; retains its state otherwise |
| **s-Skewed** | s ∈ {1 … 20} | A contiguous block of s cells (random start, mod W) is updated each step |
| **Stochastic (SCA)** | λ ∈ [0.1, 0.9] | Each cell applies rule f with prob λ or rule g with prob 1−λ (cell-level mixing of two rules) |
| **Temporal Stochastic (TSCA)** | τ ∈ [TAU_MIN, TAU_MAX] | Entire row applies rule f with prob τ or rule g with prob 1−τ (one binary draw per timestep) |

The repository contains:
- **Transformer models** (ECANet family) — purpose-built architectures trained to identify rules and noise parameters from orbit data
- **LLM evaluation** — zero-shot, few-shot, and fine-tuned open-source LLMs + frontier model evaluation
- **Datasets** — identification datasets (numpy orbits) and knowledge-check datasets (200 MCQs each)

---

## Repository Structure

```
.
├── dataset/
│   ├── eca_identification_dataset/       # Numpy orbit arrays for transformer training/testing
│   │   ├── alpha_async/                  # Phase 1 + Phase 2 splits
│   │   │   ├── phase1/{train,test}/
│   │   │   └── phase2/{train,test}/
│   │   ├── s_skewed/                     # Phase 1 + Phase 2 splits
│   │   │   ├── phase1/{train,test}/
│   │   │   └── phase2/{train,test}/
│   │   ├── stochastic/                   # Test split only (training on-the-fly)
│   │   │   └── test/
│   │   └── temp_stochastic/              # Test split only (training on-the-fly)
│   │       └── test/
│   │
│   └── eca_knowledgecheck_dataset/       # 200 MCQs per variant for LLM evaluation
│       ├── alpha_async/
│       ├── s_skewed/
│       ├── stochastic/
│       └── temp_stochastic/
│
└── codes/
    ├── transformers/                     # ECANet transformer models
    │   ├── alpha_async/
    │   ├── s_skewed/
    │   ├── stochastic/
    │   └── temp_stochastic/
    │
    ├── llm/
    │   ├── open_source/                  # Open-source LLM evaluation
    │   │   ├── zero_shot/
    │   │   ├── few_shot/
    │   │   ├── fine_tune/
    │   │   └── knowledge_check/
    │   └── frontier_evaluation/          # Frontier model evaluation (GPT-4o, Claude, Gemini)
    │       ├── zero_and_few_shot/
    │       └── knowledge_check/
    │
    └── traditional_models/               # Classical baselines
```

---

## Datasets

### Identification Dataset (`dataset/eca_identification_dataset/`)

Numpy binary arrays for transformer training and testing. All splits use **SEED=42**, grid width **W=20**, periodic boundary conditions, and a **179 train / 77 test rule split** (zero overlap).

#### Alpha-Asynchronous & s-Skewed — two-phase structure

| Split | Samples | Description |
|---|---|---|
| `alpha_async/phase1/train` | 35,800 | 179 rules × 200 samples, α=1.0 (synchronous baseline), T=100 |
| `alpha_async/phase1/test` | 7,700 | 77 rules × 100 samples, α=1.0 |
| `alpha_async/phase2/train` | 89,500 | 179 rules × 10 α-levels × 50 samples (stratified) |
| `alpha_async/phase2/test` | 15,400 | 77 rules × 10 α-levels × 20 samples |
| `s_skewed/phase1/train` | 35,800 | 179 rules × 200 samples, s=W (synchronous), T=200 |
| `s_skewed/phase1/test` | 7,700 | 77 rules × 100 samples, s=W |
| `s_skewed/phase2/train` | 89,500 | 179 rules × 20 s-values × 25 samples (stratified) |
| `s_skewed/phase2/test` | 15,400 | 77 rules × 20 s-values × 10 samples |

Files per split: `orbits.npy · rule_bits.npy · alphas.npy / s_values.npy · rule_ids.npy`

#### Stochastic & Temporal Stochastic — test split only

Training data is generated on-the-fly. Test set: **500 rule pairs × 20 samples = 10,000 samples**. Pairs are sampled to guarantee coverage of all Wolfram class combinations (A×A, A×B, A×C, B×B, B×C, C×C).

| Dataset | Key files |
|---|---|
| `stochastic/test/` | `orbits.npy · frac_stats.npy · within_var_stats.npy · lambdas.npy · rule_f_ids.npy · rule_g_ids.npy` |
| `temp_stochastic/test/` | `orbits.npy · frac_stats.npy · taus.npy · step_labels.npy · rule_f_ids.npy · rule_g_ids.npy` |

Root-level files (each variant): `train_rules.npy · test_rules.npy · train_pairs.npy · test_pairs.npy · metadata_<variant>.json`

### Knowledge-Check Dataset (`dataset/eca_knowledgecheck_dataset/`)

**800 MCQs total** (200 per noise model) for evaluating LLM conceptual understanding of ECA noise mechanics. Each entry:

```json
{
  "id": 1,
  "type": "mcq",
  "topic": 2,
  "topic_name": "α-Async Noise: Definition & Parameters",
  "difficulty": 1,
  "difficulty_name": "Easy",
  "question": "...",
  "choices": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "answer": "A",
  "explanation": "..."
}
```

| Variant | Topics | Difficulties |
|---|---|---|
| `alpha_async` | 6 | Easy / Medium / Hard |
| `s_skewed` | 15 | Easy / Medium / Hard |
| `stochastic` | 12 | Easy / Medium / Hard |
| `temp_stochastic` | 14 | Easy / Medium / Hard |

---

## Models

### Transformer Models (`codes/transformers/`)

Each variant follows the same pipeline:

```
datagen_<variant>.py  →  model_<variant>.py  →  train_<variant>.py  →  final_test_<variant>.py
```

| Script | Purpose |
|---|---|
| `datagen_<variant>.py` | Simulate ECA orbits and save numpy splits |
| `model_<variant>.py` | ECANet architecture, constants, tokenisation helpers |
| `train_<variant>.py` | Two-phase training with validation and checkpointing |
| `final_test_<variant>.py` | Evaluation across all 256 rules |
| `final_test_<variant>_classwise.py` | Per Wolfram-class accuracy breakdown |

#### ECANet Architecture

All models share a **triplet tokenisation → 2D positional encoding → Transformer encoder** backbone:

| Component | Detail |
|---|---|
| **Token** | (left, centre, right, centre_next) per cell per transition — one complete rule observation |
| **2D PE** | Sinusoidal: first D/2 dims encode transition index t, last D/2 encode cell index i |
| **Encoder** | 4 × TransformerEncoderLayer (d=128, h=4, ff=512, GELU, pre-norm) |
| **Rule head** | Mean pool → Linear(128→64) → GELU → Linear(64→8), BCEWithLogitsLoss |

Noise-specific additions:

| Variant | Token dim | T | Noise head |
|---|---|---|---|
| `alpha_async` | 4 | 100 | StatisticalPool (mean+std+max+min) → 10-class CE over α∈{0.1…1.0} |
| `s_skewed` | 5 (+ did_change) | 200 | SpatialMultiScalePool (3 branches) → 20-class CE + MSE regression on s/W |
| `stochastic` | frac_stats + within_var | — | Dual-rule classification + λ regression |
| `temp_stochastic` | frac_stats + step_labels | — | Dual-rule classification + τ regression + per-step labelling |

### LLM Evaluation (`codes/llm/`)

#### Open-Source (`open_source/`)

| Folder | Scripts |
|---|---|
| `zero_shot/` | `zero_shot_<variant>.py` × 4 |
| `few_shot/` | `few_shot_<variant>.py` × 4 |
| `fine_tune/` | `finetune_<variant>.py` × 4 |
| `knowledge_check/` | `knowledge_check_open_source.py` |

#### Frontier Models (`frontier_evaluation/`)

| Folder | Scripts |
|---|---|
| `zero_and_few_shot/` | `frontier_eval_<variant>.py` × 4 |
| `knowledge_check/` | `frontier_knowledge_check.py` |

### Traditional Baselines (`codes/traditional_models/`)

Classical statistical and mean-field baselines. See `README.md` inside this folder.

---

## Background

### ECA Rule Space

Each ECA rule is a function f: {0,1}³ → {0,1}, identified by its Wolfram code (0–255). The **Rule Mean Term (RMT)** index r = 4a + 2b + c encodes the neighbourhood (a=left, b=centre, c=right). All experiments use the **88 minimal representative rules** (one per symmetry equivalence class under reflection and complement).

### Wolfram Classification

| Class | Behaviour | Label |
|---|---|---|
| I | Converges to uniform fixed point | A |
| II | Fixed points or small periodic attractors | B |
| III/IV/LC | Chaotic / complex / long-lived transients | C |

---

## Quick Start

```bash
# 1. Generate data
cd codes/transformers/alpha_async
python datagen_alpha_async.py

# 2. Train
python train_alpha_async.py

# 3. Evaluate
python final_test_alpha_async.py

# 4. LLM knowledge check
cd codes/llm/open_source/knowledge_check
python knowledge_check_open_source.py
```

**SLURM cluster** — each transformer folder has a `submit.sh`:

```bash
cd codes/transformers/alpha_async
mkdir -p logs && sbatch submit.sh
```

---

## Requirements

```bash
pip install torch numpy matplotlib plotly pandas
```

Python ≥ 3.9, PyTorch ≥ 2.0. CUDA recommended; scripts also detect Apple MPS and fall back to CPU.

---

## License

MIT License — see `LICENSE` for details.
