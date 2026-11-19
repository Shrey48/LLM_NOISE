# Rule and Noise Recovery in Noisy Cellular Automata

Code and data for **"Bridging the Knowledge-Reasoning Gap in Natural Systems: A Benchmark for Inductive Rule and Noise Recovery in Cellular Automata"** (Salvi, Shelat, Gaur, Roy; under review at the Journal of Data-centric Machine Learning Research, DMLR).

## The question

Many natural spatio-temporal systems are modelled as simple local update rules running on a grid. Real observations of such systems are noisy. If all you have is a noisy space-time record, can you work out **the local law that generated it** and **how much noise was mixed in**?

We study this inverse problem in Elementary Cellular Automata (ECA), the smallest setting where it can be fully controlled. The clean synchronous dynamics are broken in four ways:

- cells update asynchronously
- only a moving block of cells updates
- each cell picks between two rules at random
- the whole row switches between two rules over time

For each of these noise families, a model sees a noisy orbit and must recover:

- the hidden rule (or pair of rules)
- the magnitude of the noise

## What this repository contributes

- **Four identification benchmarks**, alphaECA, skewECA, stocECA and tempECA, one per noise family.
  - Every test score is on ECA rules the model never saw during training: the 256 rules are split 179 / 77 with no overlap.
  - Orbits, labels and Croissant metadata are included.
- **Four knowledge checks**, alphaK, skewK, stocK and tempK, with 200 questions each (800 in all).
  - They test whether a language model *knows* the mechanics of each noise model. The identification task then tests whether it can *apply* them to data.
- **A family of purpose-built transformers**: alphaM, skewM, stocM and tempM, with 0.98M to 6.02M parameters. Each is designed around the statistical signal its noise family leaves in the orbit.
- **A broad comparison** against:
  - classical and neural baselines: MLP, CNN, BiLSTM, a vanilla transformer, a zero-training algorithmic method, a decoder-only control and a ViT
  - six open-weight LLMs from 7B to 72B parameters, run zero-shot, few-shot and LoRA fine-tuned
  - API-hosted frontier models (GPT-5.1, GPT-4.1 and Claude Haiku 4.5)
  - reasoning and tool-use variants, and image-input variants

The paper's own summary, in `CITATION.cff`: LLMs reach near-perfect accuracy on the knowledge checks but perform poorly on the identification tasks. The specialised transformers reach up to 99.22% exact rule match and 96.97% noise-tolerance accuracy, with at most 6.02M parameters.

In every task the noise *family* is known. The task is to recover the rule and the noise magnitude within that family, not to classify which family produced an orbit.

---

## Contents

- [The four noise families](#the-four-noise-families)
- [The architecture](#the-architecture)
- [Datasets](#datasets)
- [Model zoo](#model-zoo)
- [Reference numbers present in the repository](#reference-numbers-present-in-the-repository)
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Where outputs go](#where-outputs-go)
- [Notes and open items](#notes-and-open-items)
- [Citation](#citation)
- [License](#license)

---

## The four noise families

An ECA is a one-dimensional row of binary cells. At each step, every cell reads its three-cell neighbourhood (left, self, right) and applies a fixed 8-entry lookup table. That table is one of 256 rules, numbered by Wolfram code. Boundaries are periodic.

| Family | Code / data name | Benchmark and knowledge check | Noise parameter | Mechanism (from the simulators in the code) | Grid |
|---|---|---|---|---|---|
| **alpha-asynchronous** | `alpha_async` | alphaECA, alphaK | alpha in {0.1, ..., 1.0} | Each cell applies the rule with probability alpha and otherwise keeps its state. alpha = 1 is the ordinary ECA. | W=20, T=100 |
| **s-skewed** | `s_skewed` | skewECA, skewK | s in {1, ..., 20} | At each step, a contiguous block of s cells is updated. The block starts at a random position and wraps around the row. s = W is synchronous. | W=20, T=200 |
| **Stochastic (SCA)** | `stochastic` | stocECA, stocK | lambda in [0.1, 0.9] | Two rules, f and g. Each cell independently applies g with probability lambda and f otherwise. | W=50, T=200, 8 orbits per sample |
| **Temporal stochastic (TSCA)** | `temp_stochastic` | tempECA, tempK | tau in [0.1, 0.9] | Two rules, f and g. Once per timestep a single coin flip picks which rule updates the entire row. | W=50, T=200, 8 orbits per sample |

In the two-rule families, `(f, g, p)` and `(g, f, 1 - p)` generate the same orbits. Rules are therefore scored as an unordered pair, and the models predict the symmetric magnitude `min(p, 1 - p)`.

Each task has these targets and metrics:

| Target | Metric |
|---|---|
| Rule | 8-bit exact match. For the two-rule families, unordered-pair exact match. |
| Noise | alpha, lambda and tau: within 0.05. s: within 1. |

---

## The architecture

The four models share one idea: **match the model to the signal the noise leaves in the orbit.** In the code they are the classes `ECANetNew`, `ECANetSkew`, `ECANetSCA` and `ECANetTSCA`. The paper calls them alphaM, skewM, stocM and tempM.

### Single-rule families: alphaM and skewM

```
orbit [T, W] ─► tokens (left, centre, right, centre_next[, did_change]) ─► Linear ─► 2D sinusoidal PE (time ⊕ cell)
             ─► 4 × pre-norm Transformer encoder (d=128, 4 heads, FF 512)
                   ├─► mean pool ─► MLP ─► 8 rule bits (BCE)
                   └─► noise pool ─► noise head
```

- **Token = one observation of the rule.** Each token records a neighbourhood and the centre cell's next state. Under these noise models, an observed transition is either the rule's output or an unchanged cell.
- **2D positional encoding** (`TripletPE2D`). Half the dimensions encode the transition index and half encode the cell index, so time and space are not merged into one axis.
- **Noise-specific pooling.**
  - alphaM uses `StatisticalPool`: mean, std, max and min over the tokens, followed by a 10-class head over alpha.
  - skewM adds a `did_change` token feature, because updated cells form contiguous blocks of length s. Its `SpatialMultiScalePool` has global, temporal and spatial-convolution branches, followed by a 20-class head over s plus an MSE regression on s/W.
- **Two-phase curriculum.** Phase 1 trains the rule head alone on synchronous orbits. Phase 2 continues from the best Phase 1 checkpoint over the full noise range, with both heads active.
- **Evaluation** (`final_test_*.py`).
  - The rule is the majority vote over 20 fresh orbits per (rule, noise level).
  - The noise level comes from a hybrid estimator: given the predicted rule, the fraction of cells that changed when the rule said they should, snapped to the grid.
  - When that ratio is undefined (a rule that never changes any cell), the neural head is used instead.

### Two-rule families: stocM and tempM

```
8 orbits ─► PatternEmbedding ─► CellEncoder (4 layers, d=128, learned-query pooling over 50 cells)
         ─► TimestepEncoder (6 layers, d=256, 8 heads, over 199 transitions)
         ─► ConfidenceWeightedOrbitFusion (across the 8 orbits)
         ─► rule heads for f and g, conditioned on the predicted noise magnitude
orbit statistics ─► closed-form noise estimate + bounded learned correction ─► lambda / tau
                 ─► direct per-pattern rule decoder (+ distributional decoder for SCA)
```

- **Closed-form noise estimate first, learning second.**
  - In both families, the per-pattern conditional mean of the next state, `frac_stats`, mixes f and g in proportion to the noise.
  - In SCA, the conditional variance `within_var_stats` estimates `lambda(1 - lambda)` directly.
  - A small MLP corrects the closed-form estimate within a bound that widens when the estimate is less confident.
- **Hierarchical encoding.** The encoder runs over cells within a timestep, then over timesteps, then fuses information across the 8 orbits of a sample.
- **tempM** also predicts which rule fired at each timestep. This is trained against the stored `step_labels` and starts from the per-step residual signal.
- **Training.** Two stages of 80 epochs each, on data generated on the fly (1,000 rule pairs per epoch), with an orientation-symmetric rule loss and a curriculum over the noise range.

| Model | Class | Parameters (counted by instantiating the class) |
|---|---|---|
| alphaM | `ECANetNew` | 977,906 |
| skewM | `ECANetSkew` | 1,510,557 |
| stocM | `ECANetSCA` | 6,021,207 |
| tempM | `ECANetTSCA` | 6,018,840 |

---

## Datasets

### Identification datasets (`dataset/eca_identification_dataset/`)

All splits use seed 42 and the 179 / 77 rule split.

| Variant | Split | Samples | Orbit array | Labels |
|---|---|---|---|---|
| `alpha_async` | phase1/train | 35,800 (179 rules x 200, alpha=1.0) | int8 `[N,100,20]` | `rule_bits`, `rule_ids`, `alphas` |
| | phase1/test | 7,700 (77 x 100) | | |
| | phase2/train | 89,500 (179 x 10 alphas x 50) | | |
| | phase2/test | 15,400 (77 x 10 x 20) | | |
| `s_skewed` | phase1/train | 35,800 (s=20) | int8 `[N,200,20]` | `rule_bits`, `rule_ids`, `s_values` |
| | phase1/test | 7,700 | | |
| | phase2/train | 89,500 (179 x 20 s-values x 25) | | |
| | phase2/test | 15,400 (77 x 20 x 10) | | |
| `stochastic` | test | 10,000 (500 pairs x 20) | float32 `[N,8,200,50]` | `rule_f_*`, `rule_g_*`, `lambdas`, `frac_stats [N,24]`, `within_var_stats [N,8]` |
| `temp_stochastic` | test | 10,000 (500 pairs x 20) | float32 `[N,8,200,50]` | `rule_f_*`, `rule_g_*`, `taus`, `frac_stats [N,24]`, `step_labels [N,8,199]` |

Each variant folder also holds:

- `train_rules.npy` and `test_rules.npy`
- `train_pairs.npy` and `test_pairs.npy` (two-rule families only)
- a `metadata_<variant>.json`

In the two-rule families, the rule pairs cover every combination of Wolfram classes (AxA, AxB, AxC, BxB, BxC, CxC), and training data is generated on the fly rather than stored.

**Large orbit files.** Seven `orbits.npy` files exceed GitHub's 100 MB limit and are hosted on Zenodo. Download each one into the path shown under `dataset/eca_identification_dataset/`, or regenerate them deterministically with the `datagen_*.py` scripts.

| File | Zenodo |
|---|---|
| `alpha_async/phase1/train/orbits.npy` | https://zenodo.org/records/19995983 |
| `alpha_async/phase2/train/orbits.npy` | https://zenodo.org/records/19996154 |
| `s_skewed/phase1/train/orbits.npy` | https://zenodo.org/records/19996227 |
| `s_skewed/phase2/train/orbits.npy` | https://zenodo.org/records/19996242 |
| `s_skewed/phase2/test/orbits.npy` | https://zenodo.org/records/19996332 |
| `stochastic/test/orbits.npy` | https://zenodo.org/records/20019070 |
| `temp_stochastic/test/orbits.npy` | https://zenodo.org/records/20019339 |

[`dataset/README.md`](dataset/README.md) has the full per-file tree and dtypes.

### Knowledge-check datasets (`dataset/eca_knowledgecheck_dataset/`)

Each family has 200 questions. A question is either 4-option multiple choice, graded by exact match, or short answer, graded 0/1/2 by an LLM judge from a different model family. The scripts check at startup that the judge's family differs from every model under test.

| Dataset | File | Topics | MCQ / short answer | Easy / Medium / Hard |
|---|---|---|---|---|
| alphaK | `alpha_async/eca_alpha_async_knowledge_check.json` | 6 | 165 / 35 | 66 / 82 / 52 |
| skewK | `s_skewed/eca_s_skew_knowledge_check.json` | 15 | 140 / 60 | 53 / 86 / 61 |
| stocK | `stochastic/eca_stochastic_knowledge_check.json` | 12 | 148 / 52 | 51 / 91 / 58 |
| tempK | `temp_stochastic/eca_temp_stochastic_knowledge_check.json` | 14 | 134 / 66 | 49 / 85 / 66 |

These counts were taken from the JSON files (see [Dataset notes](#dataset-notes)).

```json
{"id": 1, "type": "mcq", "topic": 2, "topic_name": "α-Async Noise: Definition & Parameters",
 "difficulty": 1, "difficulty_name": "Easy", "question": "...",
 "choices": {"A": "...", "B": "...", "C": "...", "D": "..."}, "answer": "A", "explanation": "..."}
```

### Croissant metadata

Each dataset has a [Croissant 1.1](http://mlcommons.org/croissant/1.1) description, with Responsible-AI fields, in the repository root:

| Identification | Knowledge check |
|---|---|
| `alphaECA-croissant.jsonld` | `alphaK-croissant.jsonld` |
| `skewECA-croissant.jsonld` | `skewK-croissant.jsonld` |
| `stocECA-croissant.jsonld` | `stocK-croissant.jsonld` |
| `tempECA-croissant.jsonld` | `tempK-croissant.jsonld` |

Every `contentUrl` points either to `raw.githubusercontent.com/Shrey48/LLM_NOISE/main/...` or to the Zenodo records above.

---

## Model zoo

### Proposed transformers (`codes/transformers/<variant>/`)

| Model | Training schedule (from the scripts) |
|---|---|
| alphaM | Phase 1: 50 epochs, LR 3e-4 to 1e-5. Phase 2: 30 epochs, LR 1e-4 to 1e-6. Cosine schedule, AdamW, effective batch 128. Loss weights: rule 1.0; alpha 0 in Phase 1, 0.3 in Phase 2. |
| skewM | Same schedule. Phase 2 noise losses: s cross-entropy 0.3, s regression 0.1. |
| stocM | Stage 1: 80 epochs at LR 3e-4. Stage 2: 80 epochs at LR 3e-5, with LR 1e-5 for the lambda MLP. Batch 4 with 32 accumulation steps, warmup plus cosine schedule. |
| tempM | Same as stocM, plus a per-timestep assignment loss (weight 3.0 in Stage 1, 1.5 in Stage 2). |

### Baselines (`codes/traditional_models/`)

| Script | What it is |
|---|---|
| `baselines_alpha_async.py` | MLP on handcrafted statistics, 2D CNN, BiLSTM and a vanilla transformer (1D positional encoding, mean pool), all trained on alphaECA with alphaM's exact two-phase protocol |
| `algorithmic_baseline_alpha_skew.py` | No training and no parameters. The rule comes from the direction of observed flips for each neighbourhood. alpha is the fraction of transitions matching the rule. s is the largest circular span of changed cells. Covers alphaECA and skewECA. |
| `decoder_only_control_alpha.py` | Causal transformer trained from scratch with next-token loss on `[orbit][SEP][rule bits][SEP][alpha bits]` (about 1.05M parameters) |
| `vit_from_scratch_alpha.py` | ViT trained from scratch on the orbit rendered as an 800x160 image, with one 8x8 patch per cell (about 1.08M parameters) |
| `smoke_test_decoder_only.py` | CPU sanity check for the decoder-only control |

### Language models

**Open-weight models**, loaded from `$MODELS_DIR/<name>` with Hugging Face `transformers`:

| Model | Parameters |
|---|---|
| Qwen2.5-7B-Instruct | 7B |
| Qwen2.5-72B-Instruct | 72B |
| Llama-3.1-8B-Instruct | 8B |
| Llama-3.1-70B-Instruct | 70B |
| Mistral-7B-Instruct-v0.3 | 7B |
| Mixtral-8x7B-Instruct-v0.1 | 8x7B (mixture of experts) |

| Setting | Scripts | Protocol |
|---|---|---|
| Zero-shot | `codes/llm/open_source/zero_shot/zero_shot_<variant>.py` | Greedy decoding with a 64-token JSON answer. The orbit is given as a digit grid. Two-rule families show one of the 8 orbits. |
| Few-shot | `codes/llm/open_source/few_shot/few_shot_eval.py` (alpha), `few_shot_<variant>.py` | `--k_shots k`. Shots come from the training split or training pairs only. |
| LoRA fine-tuning | `codes/llm/open_source/fine_tune/finetune_<variant>.py` | LoRA with r=16, alpha=32 and dropout 0.05, on the attention and MLP projections. LR 2e-4, 3 epochs. 70B/72B models are loaded in 8-bit. |
| Reasoning and tool use (alphaECA) | `codes/traditional_models/zero_shot_alpha_async_reasoning_eval.py` (`--condition reasoning_only / tools_only / both`), `zero_shot_alpha_async_cot_tools.py` | Step-by-step reasoning and/or a Python tool that runs model-written code against the orbit |
| Knowledge check | `codes/llm/open_source/knowledge_check/knowledge_check_open_source.py` | Qwen2.5 7B/72B, Llama-3.1 8B/70B and Mixtral-8x7B, accessed through OpenRouter |

**Frontier models**, via the OpenAI and Anthropic SDKs:

| Script | Models |
|---|---|
| `frontier_evaluation/zero_and_few_shot/frontier_eval_alpha_async.py` | GPT-5.1 (`gpt-5.1`, optional reasoning effort) and Claude Haiku 4.5 (`claude-haiku-4-5-20251001`). Zero-shot and k = 1, 3, 5. |
| `frontier_evaluation/zero_and_few_shot/frontier_eval_{s_skewed,stochastic,temp_stochastic}.py` | The same zero- and few-shot protocol. Model identifiers are withheld while the paper is under review (see [Code notes](#code-notes)). |
| `frontier_evaluation/knowledge_check/frontier_knowledge_check.py` | Claude Haiku 4.5 and GPT-4.1 (`gpt-4.1-2025-04-14`) |
| `traditional_models/vlm_eval_alpha_{gpt51,claude}.py` | GPT-5.1 and Claude Haiku 4.5, given the orbit as an **image** |

Every identification evaluation scores the same test items: `results/fixed_test_indices.npy`, drawn once with seed 42.

---

## Reference numbers present in the repository

The repository contains no result files: no logs, JSON outputs or checkpoints. The numbers below are the reference values hard-coded in the frontier-evaluation scripts, which those scripts print next to new frontier runs. They come from the paper; the scripts' SHA-256 verification fields are still unset.

**Trained models**

| Model | Rule accuracy | Noise accuracy | Noise MAE | Source |
|---|---|---|---|---|
| alphaM | 99.22% | 95.08% (alpha within 0.05) | - | `ALPHAM_REF`, `frontier_eval_alpha_async.py` |
| skewM | 94.55% | 96.97% (s within 1) | - | `SKEWM_REF`, `frontier_eval_s_skewed.py` |
| stocM | 95.49% (pair) | 87.83% (lambda within 0.05) | 0.0289 | `STOCM_REF`, `frontier_eval_stochastic.py` |
| tempM | 89.05% (pair) | 82.30% (tau within 0.05) | 0.0311 | `TEMPM_REF`, `frontier_eval_temp_stochastic.py` |

**Open-weight LLMs**, from the `PAPER_RESULTS` lists. Values are rule accuracy / noise accuracy in %. alphaECA and skewECA use N=500; stocECA and tempECA use N=200.

| Model | alphaECA 0-shot | alphaECA 5-shot | skewECA 0-shot | skewECA 5-shot | stocECA 0-shot | stocECA 5-shot |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | 1.00 / 11.60 | 0.20 / 11.40 | 0.00 / 13.00 | 0.80 / 13.60 | 0.00 / 0.00 | 0.00 / 0.00 |
| Qwen2.5-72B | 1.00 / 12.20 | 0.60 / 16.80 | 0.80 / 15.20 | 0.60 / 13.20 | 0.00 / 27.50 | 0.00 / 28.00 |
| Llama-3.1-8B | 0.00 / 11.60 | 0.00 / 12.00 | 0.00 / 15.60 | 0.00 / 11.60 | 0.00 / 0.00 | 0.00 / 0.00 |
| Llama-3.1-70B | 0.40 / 10.00 | 0.00 / 0.00 | 0.00 / 14.80 | 0.00 / 6.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| Mistral-7B | 0.00 / 0.00 | 0.00 / 9.80 | 0.00 / 15.80 | 0.00 / 19.20 | 0.00 / 0.00 | 0.00 / 0.00 |
| Mixtral-8x7B | 0.00 / 0.00 | 0.00 / 11.20 | 0.00 / 0.20 | 0.00 / 3.80 | 0.00 / 0.00 | 0.00 / 0.00 |

- **tempECA:** all six open-weight models score 0.00 / 0.00, both zero-shot and 5-shot.
- **Fine-tuned models:**

| Model | Benchmark | Rule / pair accuracy | Noise accuracy |
|---|---|---|---|
| Qwen2.5-72B | alphaECA | 8.20 | 22.80 |
| Qwen2.5-7B | skewECA | 7.4 | 20.2 |
| Qwen2.5-72B | skewECA | 13.2 | 24.8 |
| Qwen2.5-72B | stocECA | 14.60 (pair) | 32.60 (lambda) |

---

## Repository layout

```
LLM_NOISE/
├── README.md                          this file
├── REPRODUCIBILITY.md                 seeds, hardware, compute budget, checklist
├── CITATION.cff                       citation metadata
├── LICENSE                            MIT (code) + CC-BY-4.0 (datasets)
├── requirements.txt                   pip dependencies
├── {alpha,skew,stoc,temp}ECA-croissant.jsonld   Croissant metadata, identification benchmarks
├── {alpha,skew,stoc,temp}K-croissant.jsonld     Croissant metadata, knowledge checks
├── dataset/
│   ├── README.md                      Zenodo download table, file tree, dtypes, statistics
│   ├── eca_identification_dataset/    orbits + labels: alpha_async, s_skewed, stochastic, temp_stochastic
│   └── eca_knowledgecheck_dataset/    200-question JSON + metadata per noise family
└── codes/
    ├── transformers/<variant>/        proposed models, one folder per noise family
    │   ├── datagen_<variant>.py               simulate orbits and write the data folder
    │   ├── model_<variant>.py                 architecture, constants, simulator, tokeniser
    │   ├── train_<variant>.py                 two-phase / two-stage training with resume
    │   ├── final_test_<variant>.py            held-out-rule evaluation
    │   └── final_test_<variant>_classwise.py  the same, broken down by Wolfram class
    ├── traditional_models/            baselines and extra alphaECA controls
    │   ├── baselines_alpha_async.py           MLP / CNN / BiLSTM / vanilla transformer
    │   ├── algorithmic_baseline_alpha_skew.py zero-training lookup baseline
    │   ├── decoder_only_control_alpha.py      next-token decoder-only control
    │   ├── smoke_test_decoder_only.py         CPU smoke test for the above
    │   ├── vit_from_scratch_alpha.py          ViT on rendered orbits
    │   ├── Run_vit_from_scratch.sh            SLURM template for the ViT
    │   ├── vlm_eval_alpha_{gpt51,claude}.py   image-input frontier evaluation
    │   ├── zero_shot_alpha_async_reasoning_eval.py  reasoning / tool-use factorial
    │   ├── zero_shot_alpha_async_cot_tools.py       reasoning + tools (single condition)
    │   └── run_reasoning_eval.sh              SLURM template: models x conditions
    └── llm/
        ├── open_source/{zero_shot, few_shot, fine_tune, knowledge_check}/
        └── frontier_evaluation/{zero_and_few_shot, knowledge_check}/
```

---

## Setup

```bash
git clone https://github.com/Shrey48/LLM_NOISE.git
cd LLM_NOISE
pip install -r requirements.txt
```

The code needs Python 3.9 or later and PyTorch 2.0 or later. The transformer scripts use CUDA, then Apple MPS, then CPU, whichever is available first. [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) lists the hardware used and the approximate compute budget.

### Data location

Each script reads its data from a single place. By default, that is the folder `datagen_*.py` writes next to the script. Set **`ECA_DATA_DIR`** to point any script at another copy, such as the one shipped in `dataset/` (after downloading the Zenodo files).

| Family | Default folder | Shipped copy |
|---|---|---|
| alpha_async | `ECA_Data_New/` | `dataset/eca_identification_dataset/alpha_async` |
| s_skewed | `ECA_Data_Skew/` | `dataset/eca_identification_dataset/s_skewed` |
| stochastic | `SCA_Data/` | `dataset/eca_identification_dataset/stochastic` |
| temp_stochastic | `TSCA_Data/` | `dataset/eca_identification_dataset/temp_stochastic` |

Where the default folder sits depends on the script:

- **Transformer scripts** use the folder next to the script. In the two-rule families, it sits under `$BASE_DIR`, which defaults to the script's folder.
- **Open-weight LLM scripts** use the folder under `$BASE_DIR`.
- **Baseline, VLM and frontier scripts** take a `--data_dir` argument, which defaults to `$ECA_DATA_DIR` or the folder name.

```bash
# example: train alphaM on the shipped data instead of regenerating it
export ECA_DATA_DIR=$PWD/dataset/eca_identification_dataset/alpha_async
python codes/transformers/alpha_async/train_alpha_async.py
```

Running a `datagen_*.py` script with `ECA_DATA_DIR` set writes to that folder. Generation is deterministic (seed 42), so it reproduces the shipped files.

---

## Reproducing the experiments

### 1. Proposed transformers

Outputs are written next to each script.

```bash
cd codes/transformers/alpha_async            # alphaM
python datagen_alpha_async.py                # data (skip if using ECA_DATA_DIR)
python train_alpha_async.py                  # checkpoints_new/phase{1,2}_best.pt; resumes if present
python final_test_alpha_async.py             # 77 test rules x 10 alphas x 20 orbits
python final_test_alpha_async_classwise.py   # per Wolfram class

cd ../s_skewed                               # skewM
python datagen_s_skewed.py && python train_s_skewed.py
python final_test_s_skewed.py && python final_test_s_skewed_classwise.py

cd ../stochastic                             # stocM; training data generated on the fly
python datagen_stochastic.py && python train_stochastic.py
python final_test_stochastic.py && python final_test_stochastic_classwise.py

cd ../temp_stochastic                        # tempM
python datagen_temp_stochastic.py && python train_temp_stochastic.py
python final_test_temp_stochastic.py && python final_test_temp_stochastic_classwise.py
```

### 2. Baselines

Run these from `codes/traditional_models/`, or pass `--data_dir` explicitly.

```bash
python baselines_alpha_async.py --data_dir ECA_Data_New --output_dir baseline_results
python baselines_alpha_async.py --models cnn vanilla_transformer          # subset
python algorithmic_baseline_alpha_skew.py --dataset alpha --data_dir ECA_Data_New
python algorithmic_baseline_alpha_skew.py --dataset skew  --data_dir ECA_Data_Skew
python smoke_test_decoder_only.py --data_dir ECA_Data_New                 # quick CPU check
python decoder_only_control_alpha.py --data_dir ECA_Data_New --epochs 30
python vit_from_scratch_alpha.py --data_dir ECA_Data_New --epochs 30 --batch_size 32
```

For SLURM, set `BASE`, `VENV` and the partition in `Run_vit_from_scratch.sh`, then submit it.

### 3. Open-weight LLMs

Use one `BASE_DIR` per noise family. It holds that family's data folder, and `results/` is created inside it.

```bash
export BASE_DIR=/path/to/ECA_alpha_async     # contains ECA_Data_New/ (or set ECA_DATA_DIR)
export MODELS_DIR=/path/to/models            # one subfolder per model name above

python codes/llm/open_source/zero_shot/zero_shot_alpha_async.py --model all --n_samples 500
python codes/llm/open_source/few_shot/few_shot_eval.py          --model all --n_samples 500 --k_shots 5
python codes/llm/open_source/fine_tune/finetune_alpha_async.py  --model Qwen2.5-72B-Instruct --n_train 5000 --n_test 500 --epochs 3
```

| Family | Scripts | Paper sample sizes |
|---|---|---|
| s_skewed | `zero_shot_s_skewed.py`, `few_shot_s_skewed.py`, `finetune_s_skewed.py` | `--n_samples 500` |
| stochastic | `zero_shot_stochastic.py`, `few_shot_stochastic.py`, `finetune_stochastic.py` | `--n_samples 200`; fine-tune `--n_train 2000 --n_test 200` |
| temp_stochastic | `zero_shot_temp_stochastic.py`, `few_shot_temp_stochastic.py`, `finetune_temp_stochastic.py` | same as stochastic |

The first run creates `results/fixed_test_indices.npy`, and every later script reuses it. Make that first run use the paper's N.

Reasoning and tool-use study (alphaECA):

```bash
for c in reasoning_only tools_only both; do
  python codes/traditional_models/zero_shot_alpha_async_reasoning_eval.py --model all --condition $c --n_samples 50
done
```

`run_reasoning_eval.sh` is the SLURM version. The tool conditions run code that the model writes. The scripts deny dangerous builtins and set a timeout, but this is not a security sandbox, so use an isolated machine.

### 4. Frontier models

```bash
export OPENAI_API_KEY=...  ANTHROPIC_API_KEY=...
cd codes/llm/frontier_evaluation/zero_and_few_shot
python frontier_eval_alpha_async.py     --data_dir ECA_Data_New  --n_samples 500
python frontier_eval_s_skewed.py        --data_dir ECA_Data_Skew --n_samples 500
python frontier_eval_stochastic.py      --data_dir SCA_Data      --n_samples 200
python frontier_eval_temp_stochastic.py --data_dir TSCA_Data     --n_samples 200
```

- `--models` and `--settings` select a subset of models or settings.
- `--k_shots` sets the few-shot k values.
- `--skip_verify` skips the reference-number check.

Image-input runs:

```bash
python codes/traditional_models/vlm_eval_alpha_gpt51.py  --data_dir ECA_Data_New --n_samples 30
python codes/traditional_models/vlm_eval_alpha_claude.py --data_dir ECA_Data_New --n_samples 30
```

### 5. Knowledge checks

Each run reads one question file, given by `DATASET_PATH`.

```bash
KC=$PWD/dataset/eca_knowledgecheck_dataset

# open-weight models via OpenRouter (judge default: gpt-4o; override with JUDGE_MODEL / JUDGE_API_BASE / JUDGE_FAMILY)
export OPENROUTER_API_KEY=...  JUDGE_API_KEY=...
DATASET_PATH=$KC/alpha_async/eca_alpha_async_knowledge_check.json OUTPUT_DIR=results/alphaK \
  python codes/llm/open_source/knowledge_check/knowledge_check_open_source.py

# Claude Haiku 4.5 + GPT-4.1 (judge default: mistral-large-latest)
export ANTHROPIC_API_KEY=...  OPENAI_API_KEY=...  JUDGE_API_KEY=...
DATASET_PATH=$KC/s_skewed/eca_s_skew_knowledge_check.json OUTPUT_DIR=results/skewK \
  python codes/llm/frontier_evaluation/knowledge_check/frontier_knowledge_check.py
```

---

## Where outputs go

| Experiment | Output |
|---|---|
| Data generation | `ECA_Data_New/`, `ECA_Data_Skew/`, `SCA_Data/`, `TSCA_Data/` next to the script, or `$ECA_DATA_DIR` |
| Transformer training | alphaM: `checkpoints_new/`. skewM: `checkpoints_skew/`. stocM: `checkpoints_sca_v1/stage{1,2}.pt`. tempM: `checkpoints_tsca_v9/stage{1,2}.pt`. |
| Transformer final test | alphaM and skewM: `split test result.txt` and `.json`. stocM and tempM: printed to stdout. |
| Class-wise test | `final_test_{new,skew,sca,tsca}_classwise.log` |
| Architectural baselines | `baseline_results/results_summary.csv`, `results_detail.json`, `<model>_phase{1,2}.pt` |
| Other baselines and VLM | `algorithmic_baseline_results/`, `decoder_only_results/`, `vit_results/`, `vlm_results/` |
| Open-weight LLMs | `$BASE_DIR/results/{zero_shot, few_shot, finetune, ...}/<model>_*.json`, checkpointed every 10 samples. LoRA adapters go in `$BASE_DIR/checkpoints/finetune/`. |
| Frontier LLMs | `--output_dir` (default `frontier_results*/`): `results_summary.csv`, `results_detail.json`, `checkpoints/`, `raw/` |
| Knowledge checks | `$OUTPUT_DIR/results.json` or `results_frontier.json` |

These folders are listed in `.gitignore`.

---

## Notes and open items

### Dataset notes

- **alphaK counts.** The question file contains 165 MCQ and 35 short-answer questions (66 easy, 82 medium, 52 hard). Its `metadata_alpha_async.json` and `alphaK-croissant.jsonld` record 178 / 22 and 80 / 68 / 52. The other three knowledge checks agree with their metadata.
- **Dataset licence.** `LICENSE` and the Croissant files give CC-BY-4.0; the per-variant `metadata_*.json` files record `"license": "MIT"`.
- **Short-answer judge.** The knowledge-check metadata names Grok-3 as the short-answer judge. The scripts read the judge from `JUDGE_MODEL`, with defaults `gpt-4o` (open-weight harness) and `mistral-large-latest` (frontier harness).

### Code notes

- **Anonymised frontier models.** The frontier scripts for the skew, stochastic and temp-stochastic families keep their model identifiers anonymised for blind review (`[REDACTED]` and `*-anon` placeholders in `MODEL_CONFIGS`, and in `frontier_eval_temp_stochastic.py` the OpenAI-compatible `base_url`). Fill these in, and for `frontier_eval_stochastic.py` set `LLM_A_MODEL_ID` and `LLM_A_API_KEY`, before running them. They will be disclosed on acceptance.
- **Stochastic data folder.** The open-weight LLM scripts for the stochastic family now read `SCA_Data/`, the name `datagen_stochastic.py` and the frontier script use. Earlier versions read `ECA_Data_Stoc/` (and, for few-shot, `ECA_Data_Skew/`). Set `ECA_DATA_DIR` to keep an existing layout.
- **Reference values.** The hard-coded reference values in the frontier scripts carry `sha256: None`. Running them prints a warning when the matching result files are absent.

---

## Citation

```bibtex
@article{salvi2026bridging,
  title   = {Bridging the Knowledge-Reasoning Gap in Natural Systems: A Benchmark for Inductive Rule and Noise Recovery in Cellular Automata},
  author  = {Salvi, Shrey Vimalbhai and Shelat, Shlok and Gaur, Manas and Roy, Souvik},
  year    = {2026},
  note    = {Under review at the Journal of Data-centric Machine Learning Research (DMLR)},
  url     = {https://github.com/Shrey48/LLM_NOISE}
}
```

Machine-readable metadata: [`CITATION.cff`](CITATION.cff).

## License

The code is released under the MIT License, and the datasets in `dataset/` under CC-BY-4.0. See [`LICENSE`](LICENSE).
