# Dataset

This folder contains the datasets used in the paper. Due to GitHub's 100 MB file-size limit,
large orbit `.npy` files are **not stored here** and must be downloaded separately or regenerated
from the provided data-generation scripts.

---

## Download

The seven large `orbits.npy` files that exceed GitHub's 100 MB limit are hosted on Zenodo.
Download each file and place it at the path shown:

| # | File path in repo | Zenodo record |
|---|---|---|
| 1 | `dataset/eca_identification_dataset/alpha_async/phase1/train/orbits.npy` | [zenodo.org/records/19995983](https://zenodo.org/records/19995983) |
| 2 | `dataset/eca_identification_dataset/alpha_async/phase2/train/orbits.npy` | [zenodo.org/records/19996154](https://zenodo.org/records/19996154) |
| 3 | `dataset/eca_identification_dataset/s_skewed/phase1/train/orbits.npy` | [zenodo.org/records/19996227](https://zenodo.org/records/19996227) |
| 4 | `dataset/eca_identification_dataset/s_skewed/phase2/train/orbits.npy` | [zenodo.org/records/19996242](https://zenodo.org/records/19996242) |
| 5 | `dataset/eca_identification_dataset/s_skewed/phase2/test/orbits.npy` | [zenodo.org/records/19996332](https://zenodo.org/records/19996332) |
| 6 | `dataset/eca_identification_dataset/stochastic/test/orbits.npy` | [zenodo.org/records/20019070](https://zenodo.org/records/20019070) |
| 7 | `dataset/eca_identification_dataset/temp_stochastic/test/orbits.npy` | [zenodo.org/records/20019339](https://zenodo.org/records/20019339) |

All other `.npy` files (rule_bits, alphas, s_values, rule_ids, lambdas, taus, frac_stats, etc.) are already present in the repository.

After downloading, place each file at the exact path shown above so the directory tree matches the structure below.

---

## Regenerating from Scratch

Every dataset can be fully regenerated deterministically (seed = 42) using the
data-generation scripts in `codes/`:

```bash
# Alpha-asynchronous
cd codes/transformers/alpha_async
python datagen_alpha_async.py

# s-Skewed
cd codes/transformers/s_skewed
python datagen_s_skewed.py

# Stochastic (SCA)
cd codes/transformers/stochastic
python datagen_stochastic.py

# Temporal Stochastic (TSCA)
cd codes/transformers/temp_stochastic
python datagen_temp_stochastic.py
```

Expected generation time per variant: ~30–60 min on a modern CPU.

---

## Folder Structure

```
dataset/
│
├── eca_identification_dataset/
│   │
│   ├── alpha_async/                          ← α-Asynchronous ECA
│   │   ├── train_rules.npy                   # int32  [179]
│   │   ├── test_rules.npy                    # int32  [77]
│   │   ├── phase1/
│   │   │   ├── train/
│   │   │   │   ├── orbits.npy                # int8   [35800, 100, 20]  ← NOT on GitHub
│   │   │   │   ├── rule_bits.npy             # int8   [35800, 8]
│   │   │   │   └── alphas.npy                # float32 [35800]
│   │   │   │   └── rules_ids.npy
│   │   │   └── test/
│   │   │       ├── orbits.npy                # int8   [7700, 100, 20]
│   │   │       ├── rule_bits.npy             # int8   [7700, 8]
│   │   │       └── alphas.npy                # float32 [7700]
│   │   │       └── rules_ids.npy
│   │   └── phase2/
│   │       ├── train/
│   │       │   ├── orbits.npy                # int8   [89500, 100, 20]  ← NOT on GitHub
│   │       │   ├── rule_bits.npy             # int8   [89500, 8]
│   │       │   ├── alphas.npy                # float32 [89500]
│   │       │   └── rule_ids.npy              # int32  [89500]
│   │       └── test/
│   │           ├── orbits.npy                # int8   [15400, 100, 20]
│   │           ├── rule_bits.npy             # int8   [15400, 8]
│   │           ├── alphas.npy                # float32 [15400]
│   │           └── rule_ids.npy              # int32  [15400]
│   │
│   ├── s_skewed/                             ← s-Skewed ECA  (T=200)
│   │   ├── train_rules.npy                   # int32  [179]
│   │   ├── test_rules.npy                    # int32  [77]
│   │   ├── phase1/
│   │   │   ├── train/
│   │   │   │   ├── orbits.npy                # int8   [35800, 200, 20]  ← NOT on GitHub
│   │   │   │   ├── rule_bits.npy
│   │   │   │   └── s_values.npy
│   │   |   │   └── rule_ids.npy
│   │   │   └── test/
│   │   │       ├── orbits.npy                # int8   [7700, 200, 20]   ← NOT on GitHub
│   │   │       ├── rule_bits.npy
│   │   │       └── s_values.npy
│   │   │       └── rule_ids.npy
│   │   └── phase2/
│   │       ├── train/
│   │       │   ├── orbits.npy                # int8   [89500, 200, 20]  ← NOT on GitHub
│   │       │   ├── rule_bits.npy
│   │       │   ├── s_values.npy
│   │       │   └── rule_ids.npy
│   │       └── test/
│   │           ├── orbits.npy                # int8   [15400, 200, 20]  ← NOT on GitHub
│   │           ├── rule_bits.npy
│   │           ├── s_values.npy
│   │           └── rule_ids.npy
│   │
│   ├── stochastic/                           ← Stochastic ECA (SCA), W=50, T=200, K=8
│   │   ├── train_rules.npy                   # int32  [179]
│   │   ├── test_rules.npy                    # int32  [77]
│   │   ├── train_pairs.npy                   # int32  [500, 2]
│   │   ├── test_pairs.npy                    # int32  [500, 2]
│   │   ├── metadata_stochastic.json
│   │   └── test/
│   │       ├── orbits.npy                    # float32 [10000, 8, 200, 50]  ← NOT on GitHub
│   │       ├── rule_f_ids.npy                # int32   [10000]
│   │       ├── rule_g_ids.npy                # int32   [10000]
│   │       ├── lambdas.npy                   # float32 [10000]
│   │       ├── frac_stats.npy                # float32 [10000, 24]
│   │       ├── within_var_stats.npy          # float32 [10000, 8]
│   │       ├── rule_f_bits.npy               # float32 [10000, 8]
│   │       └── rule_g_bits.npy               # float32 [10000, 8]
│   │
│   └── temp_stochastic/                      ← Temporal Stochastic ECA (TSCA)
│       ├── train_rules.npy                   # int32  [179]
│       ├── test_rules.npy                    # int32  [77]
│       ├── train_pairs.npy                   # int32  [500, 2]
│       ├── test_pairs.npy                    # int32  [500, 2]
│       ├── metadata_temp_stochastic.json
│       └── test/
│           ├── orbits.npy                    # float32 [10000, 8, 200, 50]  ← NOT on GitHub
│           ├── rule_f_ids.npy                # int32   [10000]
│           ├── rule_g_ids.npy                # int32   [10000]
│           ├── taus.npy                      # float32 [10000]
│           ├── step_labels.npy               # int8    [10000, 8, 199]
│           └── frac_stats.npy                # float32 [10000, 24]
│
└── eca_knowledgecheck_dataset/               ← 800 MCQs (200 per noise model)
    ├── alpha_async_mcq.json
    ├── s_skewed_mcq.json
    ├── stochastic_mcq.json
    └── temp_stochastic_mcq.json
```

---

## File Format Details

### Identification Dataset (`.npy` arrays)

| File | dtype | Shape | Description |
|---|---|---|---|
| `orbits.npy` | `int8` / `float32` | `[N, T, W]` or `[N, K, T, W]` | Space-time orbit(s). Binary (0/1) or float for multi-orbit variants |
| `rule_bits.npy` | `int8` | `[N, 8]` | 8-bit binary representation of the ECA rule (LSB-first) |
| `rule_ids.npy` | `int32` | `[N]` | Wolfram rule number (0–255) |
| `alphas.npy` | `float32` | `[N]` | α value ∈ {0.1, 0.2, …, 1.0} |
| `s_values.npy` | `int32` | `[N]` | s value ∈ {1, …, 20} |
| `lambdas.npy` | `float32` | `[N]` | λ ∈ [0.1, 0.9] |
| `taus.npy` | `float32` | `[N]` | τ ∈ [TAU_MIN, TAU_MAX] |
| `frac_stats.npy` | `float32` | `[N, 24]` | Engineered statistical features (fraction active per row, etc.) |
| `step_labels.npy` | `int8` | `[N, K, T-1]` | Per-step rule label (0 = rule_f, 1 = rule_g) for TSCA |

### Knowledge-Check Dataset (`.json`)

Each file is a list of MCQ objects:

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

---

## Statistics

| Variant | Train samples | Test samples | Grid size (W×T) | Orbits/sample |
|---|---|---|---|---|
| alpha_async | 89,500 | 15,400 | 20 × 100 | 1 |
| s_skewed | 89,500 | 15,400 | 20 × 200 | 1 |
| stochastic | — | 10,000 | 50 × 200 | 8 |
| temp_stochastic | — | 10,000 | 50 × 200 | 8 |

All experiments use the **full set of 256 rules**, split into 179 training rules and 77 test rules (SEED = 42). 
