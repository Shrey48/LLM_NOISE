# Reproducibility Guide

This document provides all information needed to reproduce the results in the paper.

---

## Hardware & Environment

| Component | Specification |
|---|---|
| GPU | NVIDIA A100 80 GB (or equivalent) |
| CPU | Any modern multi-core CPU for data generation |
| RAM | ≥ 32 GB recommended |
| Storage | ~50 GB for all datasets |
| OS | Linux (Ubuntu 20.04+) or macOS |
| Python | ≥ 3.9 |
| PyTorch | ≥ 2.0 (CUDA 11.8+ recommended) |

> Scripts auto-detect CUDA → Apple MPS → CPU and fall back gracefully.

---

## Installation

```bash
git clone https://anonymous.4open.science/r/LLM_NOISE-9DC1/
cd LLM_NOISE
pip install -r requirements.txt
```

---

## Random Seeds

All experiments use a **global seed of 42** for NumPy, PyTorch, and Python `random`.
Shot sampling for few-shot evaluations uses seed `42 + 99 = 141`.
Results are deterministic given the same hardware and software versions.

---

## Step-by-Step Reproduction

### 1. Transformer Models (ECANet family)

Each noise variant follows the same 3-step pipeline:

```bash
# Alpha-Asynchronous (αM — Table 1)
cd codes/transformers/alpha_async
python datagen_alpha_async.py        # ~45 min CPU
python train_alpha_async.py          # ~6 h on A100
python final_test_alpha_async.py     # ~10 min

# s-Skewed (skewM — Table 2)
cd codes/transformers/s_skewed
python datagen_s_skewed.py
python train_s_skewed.py
python final_test_s_skewed.py

# Stochastic (stocM — Table 3)
cd codes/transformers/stochastic
python datagen_stochastic.py
python train_stochastic.py
python final_test_stochastic.py

# Temporal Stochastic (tempM — Table 4)
cd codes/transformers/temp_stochastic
python datagen_temp_stochastic.py
python train_temp_stochastic.py
python final_test_temp_stochastic.py
```

**SLURM users:** each folder contains a `submit.sh`:

```bash
mkdir -p logs && sbatch submit.sh
```

### 2. Architectural Baselines (Table 3 in paper)

```bash
cd codes/traditional_models
python baselines_alphaECA.py --data_dir ECA_Data_New --output_dir baseline_results
```

Output: `baseline_results/results_summary.csv` (paste directly into paper table).

### 3. Open-Source LLM Evaluation

```bash
# Zero-shot (all 4 variants)
cd codes/llm/open_source/zero_shot
python zero_shot_alpha_async.py --model all --n_samples 200
python zero_shot_s_skewed.py    --model all --n_samples 200
python zero_shot_stochastic.py  --model all --n_samples 200
python zero_shot_temp_stochastic.py --model all --n_samples 200

# Few-shot (k = 1, 3, 5)
cd codes/llm/open_source/few_shot
python few_shot_alpha_async.py --model all --n_samples 200

# Fine-tuning (LoRA)
cd codes/llm/open_source/fine_tune
python finetune_alpha_async.py --model Qwen2.5-7B-Instruct --n_train 5000 --epochs 3

# Knowledge check (800 MCQs)
cd codes/llm/open_source/knowledge_check
python knowledge_check_open_source.py
```

### 4. Frontier LLM Evaluation

```bash
cd codes/llm/frontier_evaluation/zero_and_few_shot
python frontier_eval_alpha_async.py    --n_samples 500
python frontier_eval_s_skewed.py       --n_samples 500
python frontier_eval_stochastic.py     --n_samples 500
python frontier_eval_temp_stochastic.py --n_samples 500

cd codes/llm/frontier_evaluation/knowledge_check
python frontier_knowledge_check.py
```

> **API keys required:** Set `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY` as environment variables.

---

## Approximate Compute Budget

| Experiment | Hardware | Time |
|---|---|---|
| Data generation (all 4 variants) | CPU | ~3 h total |
| Transformer training (all 4 variants) | 1× A100 80 GB | ~30 h total |
| Open-source LLM eval (7B models, N=200) | 1× A100 80 GB | ~2 h per model/variant |
| Open-source LLM eval (70B models, N=200) | 2× A100 80 GB | ~8 h per model/variant |
| LoRA fine-tuning (7B, N=5000, 3 epochs) | 1× A100 80 GB | ~4 h |
| Frontier LLM eval (N=500) | API calls | ~1–3 h (rate-limited) |

---

## Key Fixed Test Indices

All scripts share a `fixed_test_indices.npy` file (generated on first run, then reused)
to ensure every method is evaluated on **exactly the same test samples**.
This file is committed to the repository for reproducibility.

---

## Checklist (NeurIPS Reproducibility)

- [x] Code is publicly available
- [x] Dataset is publicly available (Zenodo DOI: 10.5281/zenodo.XXXXXXX)
- [x] All random seeds are fixed and documented (SEED = 42)
- [x] Hyperparameters are reported in paper Appendix (Tables 23–26)
- [x] Prompts are reported verbatim in paper Appendix F
- [x] Compute budget is reported above
- [x] Hardware specifications are reported above
- [x] `requirements.txt` is provided
- [x] Step-by-step instructions are provided in this file
