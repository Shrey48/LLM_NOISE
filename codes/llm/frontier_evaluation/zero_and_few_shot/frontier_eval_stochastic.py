"""
frontier_eval_stocECA.py  —  Anonymous submission, under review
===============================================================
Zero-shot and few-shot (k=1,3,5) evaluation of frontier LLMs on the stocECA
identification task (spatial stochastic noise, two-rule mixture).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILE STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ECA_stochastic/
│
├── SCA_Data/                           ← stocECA dataset (read-only)
│   ├── train_rules.npy                 int32 [179]
│   ├── test_rules.npy                  int32 [77]
│   ├── train_pairs.npy                 int32 [500, 2]  ← shot source
│   ├── test_pairs.npy                  int32 [500, 2]
│   └── test/
│       ├── orbits.npy                  float32 [10000, 8, 200, 50]
│       ├── rule_f_ids.npy              int32   [10000]
│       ├── rule_g_ids.npy              int32   [10000]
│       ├── lambdas.npy                 float32 [10000]
│       ├── frac_stats.npy              float32 [10000, 24]
│       ├── within_var_stats.npy        float32 [10000, 8]
│       ├── rule_f_bits.npy             float32 [10000, 8]
│       └── rule_g_bits.npy             float32 [10000, 8]
│
├── results/
│   ├── fixed_test_indices.npy          ← SHARED with open-source scripts
│   └── zero_shot/<Model>_zero_shot.json
│   └── few_shot/<Model>_<k>shot.json
│
├── frontier_results_stoc/              ← THIS script writes here
│   ├── fixed_test_indices.npy
│   ├── results_summary.csv
│   ├── results_detail.json
│   ├── checkpoints/
│   │   └── <model>_<setting>.json
│   └── raw/
│       └── <model>_<setting>_<idx:04d>.txt
│
└── frontier_eval_stocECA.py            ← THIS FILE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROMPT FIDELITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALL prompts are taken VERBATIM from Appendix F (stocECA section).
Do not modify SYSTEM_PROMPT, _orbit_to_str, make_user_prompt, build_messages,
or parse_response without updating the paper appendix to match.

  SYSTEM_PROMPT   : copied character-for-character from Appendix F
                    + LLM-A-ADD-9 hardening (no-reasoning instruction)
  orbit encoding  : first 50 rows of orbit[0] (k=0), raw "0"/"1" strings
  user prompt     : "Space-time orbit (showing 50 of 200 rows x 50 cells)..."
  few-shot format : multi-turn chat [system][user][assistant]…[user]
  response format : {"rule_f": <int>, "rule_g": <int>, "lambda": <float>}
  parse logic     : Stage-1 JSON (first-object scan) → Stage-2 regex fallback
  test indices    : shared fixed_test_indices.npy (SEED=42)
  shot sampling   : rng(SEED+99), single draw of k from train_pairs.npy,
                    placeholder zero orbits (T×W) — matches few_shot_stoc.py
  pair accuracy   : unordered-pair: {pred_f,pred_g} == {true_f,true_g}
  lambda eval     : snap to nearest 0.05 grid, then symmetric ±0.05 tolerance:
                    min(|λ̂−λ|, |λ̂−(1−λ)|) ≤ LAM_TOL

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Setup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  pip install openai anthropic numpy

  export OPENAI_API_KEY=sk-...
  export LLM_A_API_KEY=<key>            ← LLM-A-ADD-8

Usage
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # Full run — all models, zero-shot + k=1,3,5:
  python frontier_eval_stocECA.py --data_dir SCA_Data --n_samples 200

  # LLM-A only, zero-shot:
  python frontier_eval_stocECA.py --models llm_a --settings zero_shot

  # LLM-A few-shot k=5, 100 samples:
  python frontier_eval_stocECA.py --models llm_a --settings few_shot --k_shots 5 --n_samples 100

  # LLM-A + LLM-B comparison, all shot settings:
  python frontier_eval_stocECA.py --models llm_a llm_b

  # Resume interrupted run (already-done samples are auto-skipped):
  python frontier_eval_stocECA.py --models llm_a --n_samples 200

  # GPT-based model with reasoning:
  python frontier_eval_stocECA.py --models gpt_model --gpt_reasoning low
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SEED            : int         = 42
LAM_TOL         : float       = 0.05    # symmetric tolerance (FIX-4)
LAM_MIN         : float       = 0.1
LAM_MAX         : float       = 0.9
LAM_GRID_STEP   : float       = 0.05    # snap grid spacing
W               : int         = 50
T               : int         = 200
K               : int         = 8       # orbits per sample; we show orbit[0]
DISPLAY_ROWS    : int         = 50      # first 50 rows shown to LLM
MAX_K_SHOTS     : int         = 5
DEFAULT_K_LIST  : list[int]   = [1, 3, 5]

# ⚑ FIX-1: unified token budget for all models.
MAX_TOKENS_ALL  : int         = 256

# Bootstrap parameters (FIX-3)
BOOTSTRAP_REPS  : int         = 2000
BOOTSTRAP_SEED  : int         = SEED + 7

# Retry parameters (FIX-6)
MAX_RETRIES     : int         = 3
RETRY_BASE_S    : float       = 5.0
RETRY_MAX_S     : float       = 120.0

# Lambda snap grid: 0.10, 0.15, 0.20, ..., 0.90
LAM_GRID: list[float] = [
    round(i * LAM_GRID_STEP, 2)
    for i in range(
        round(LAM_MIN / LAM_GRID_STEP),
        round(LAM_MAX / LAM_GRID_STEP) + 1,
    )
]

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — verbatim from Appendix F (stocECA section)
# LLM-A-ADD-9: hardened with explicit no-reasoning / no-explanation directive.
# DO NOT MODIFY further — any change breaks comparability with paper Table 29
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT: str = (
    "You are an expert in cellular automata. "
    "You are given a space-time orbit of an Elementary Cellular Automaton (ECA) "
    "under stochastic noise.\n\n"
    "In this stochastic model, at each timestep each cell independently chooses "
    "one of two rules. With probability lambda it uses rule G, otherwise it uses "
    "rule F. The parameter lambda is a continuous value between 0.1 and 0.9. "
    "The system (F, G, lambda) is statistically identical to (G, F, 1-lambda), "
    "so you should predict the rule with the smaller number as rule_f and the "
    "larger as rule_g.\n\n"
    "The orbit is a grid of 0s and 1s. Each row is one timestep. "
    "Grid width is 50 cells.\n\n"
    "Identify:\n"
    "1. Rule F (the first ECA rule, integer 0-255)\n"
    "2. Rule G (the second ECA rule, integer 0-255, different from F)\n"
    "3. Lambda value (float between 0.1 and 0.9, rounded to 2 decimal places)\n\n"
    "Output ONLY a single JSON object. No reasoning, no explanation, no markdown, "
    "no extra text before or after. Your entire response must be exactly:\n"
    '{"rule_f": <integer>, "rule_g": <integer>, "lambda": <float>}'
)

# ─────────────────────────────────────────────────────────────────────────────
# Orbit encoding — verbatim from Appendix F _orbit_to_str (stocECA)
# Shows first DISPLAY_ROWS=50 rows of orbit[0] (k=0 of K=8).
# ─────────────────────────────────────────────────────────────────────────────
def _orbit_to_str(orbit: np.ndarray) -> str:
    """
    orbit shape: [K, T, W] or [T, W].
    Always shows first 50 rows of first orbit (index 0 if K-dim present).
    """
    if orbit.ndim == 3:
        rows = orbit[0, :DISPLAY_ROWS, :]   # first orbit, first 50 rows
    else:
        rows = orbit[:DISPLAY_ROWS, :]
    return "\n".join("".join(str(int(c)) for c in row) for row in rows)


# ─────────────────────────────────────────────────────────────────────────────
# Prompts — verbatim from Appendix F (stocECA)
# ─────────────────────────────────────────────────────────────────────────────
def make_user_prompt(orbit: np.ndarray) -> str:
    return (
        f"Space-time orbit (showing {DISPLAY_ROWS} of {T} rows x {W} cells):\n\n"
        f"{_orbit_to_str(orbit)}\n\n"
        "What are rule F, rule G, and lambda?"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Few-shot message builder — verbatim from Appendix F Section F.2 (stocECA)
# ─────────────────────────────────────────────────────────────────────────────
def build_messages(
    shots: list[dict[str, Any]],
    orbit: np.ndarray,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    for s in shots:
        messages.append({
            "role": "user",
            "content": make_user_prompt(np.array(s["orbit"])),
        })
        messages.append({
            "role": "assistant",
            "content": json.dumps({
                "rule_f":  int(s["rule_f"]),
                "rule_g":  int(s["rule_g"]),
                "lambda":  float(s["lambda"]),
            }),
        })
    messages.append({
        "role": "user",
        "content": make_user_prompt(orbit),
    })
    return messages


# ─────────────────────────────────────────────────────────────────────────────
# Lambda snap helper (FIX-4)
# ─────────────────────────────────────────────────────────────────────────────
def _snap_lambda(raw: float) -> tuple[float, bool]:
    snapped = min(LAM_GRID, key=lambda g: abs(g - raw))
    changed = abs(snapped - raw) > 1e-9
    return snapped, changed


# ─────────────────────────────────────────────────────────────────────────────
# Unordered-pair rule accuracy helper
# ─────────────────────────────────────────────────────────────────────────────
def _pair_correct(
    pred_f: int | None,
    pred_g: int | None,
    true_f: int,
    true_g: int,
) -> bool:
    if pred_f is None or pred_g is None:
        return False
    return (pred_f == true_f and pred_g == true_g) or \
           (pred_f == true_g and pred_g == true_f)


# ─────────────────────────────────────────────────────────────────────────────
# Symmetric lambda accuracy helper (FIX-4)
# ─────────────────────────────────────────────────────────────────────────────
def _lambda_ok(pred_lam: float | None, true_lam: float) -> bool:
    if pred_lam is None:
        return False
    return min(abs(pred_lam - true_lam),
               abs(pred_lam - (1.0 - true_lam))) <= LAM_TOL


def _lambda_mae(pred_lam: float | None, true_lam: float) -> float | None:
    if pred_lam is None:
        return None
    return float(min(abs(pred_lam - true_lam),
                     abs(pred_lam - (1.0 - true_lam))))


# ─────────────────────────────────────────────────────────────────────────────
# Response parser — FIX-10: first-valid-JSON-object scan.
# ─────────────────────────────────────────────────────────────────────────────
def _try_parse_json(
    text: str,
) -> tuple[int | None, int | None, float | None, bool]:
    # Strip markdown fences that some models add (LLM-A-ADD-4)
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text).strip()

    start = text.find("{")
    if start == -1:
        return None, None, None, False

    candidates = [i for i, ch in enumerate(text) if ch == "}" and i > start]
    for end in candidates:
        substr = text[start : end + 1]
        try:
            payload  = json.loads(substr)
            rule_f   = int(payload.get("rule_f", -1))
            rule_g   = int(payload.get("rule_g", -1))
            lam_raw  = float(payload.get("lambda", -1.0))
            if (0 <= rule_f <= 255 and
                    0 <= rule_g <= 255 and
                    LAM_MIN - 0.5 <= lam_raw <= LAM_MAX + 0.5):
                lam, changed = _snap_lambda(
                    max(LAM_MIN, min(LAM_MAX, lam_raw))
                )
                return rule_f, rule_g, lam, changed
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    return None, None, None, False


def parse_response(
    response: str,
) -> tuple[int | None, int | None, float | None, bool]:
    """
    Returns (pred_rule_f, pred_rule_g, pred_lambda, snap_changed).
    """
    rule_f, rule_g, lam, snap_changed = _try_parse_json(response)
    if rule_f is not None:
        return rule_f, rule_g, lam, snap_changed

    # Stage 2: regex fallback
    rf_m  = re.search(r'"rule_f"\s*:\s*(\d+)',   response)
    rg_m  = re.search(r'"rule_g"\s*:\s*(\d+)',   response)
    lam_m = re.search(r'"lambda"\s*:\s*([\d.]+)', response)
    if rf_m and rg_m and lam_m:
        rule_f  = int(rf_m.group(1))
        rule_g  = int(rg_m.group(1))
        lam_raw = float(lam_m.group(1))
        if (0 <= rule_f <= 255 and
                0 <= rule_g <= 255 and
                LAM_MIN - 0.5 <= lam_raw <= LAM_MAX + 0.5):
            lam, changed = _snap_lambda(
                max(LAM_MIN, min(LAM_MAX, lam_raw))
            )
            return rule_f, rule_g, lam, changed

    return None, None, None, False


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap confidence intervals (FIX-3)
# ─────────────────────────────────────────────────────────────────────────────
def bootstrap_ci(
    values: list[bool],
    reps:   int   = BOOTSTRAP_REPS,
    seed:   int   = BOOTSTRAP_SEED,
    alpha:  float = 0.05,
) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    n   = len(arr)
    if n == 0:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0}
    rng   = np.random.default_rng(seed)
    means = np.array([
        rng.choice(arr, size=n, replace=True).mean()
        for _ in range(reps)
    ])
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return {"mean": float(arr.mean()), "lower": lo, "upper": hi}


# ─────────────────────────────────────────────────────────────────────────────
# SHA-256 helper (FIX-8)
# ─────────────────────────────────────────────────────────────────────────────
def file_sha256(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Exponential back-off (FIX-6)
# ─────────────────────────────────────────────────────────────────────────────
def _backoff_sleep(attempt: int, retry_after: float | None = None) -> None:
    delay = min(
        float(retry_after) if retry_after is not None
        else RETRY_BASE_S * (2 ** attempt),
        RETRY_MAX_S,
    )
    print(f"      back-off: sleeping {delay:.1f}s (attempt {attempt + 1})")
    time.sleep(delay)


def _extract_retry_after(exc: Exception) -> float | None:
    try:
        response = getattr(exc, "response", None)
        if response is not None:
            ra = response.headers.get("Retry-After")
            if ra is not None:
                return float(ra)
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def get_fixed_test_indices(
    data_dir:   str,
    n_samples:  int,
    index_file: str,
) -> np.ndarray:
    canonical = Path(data_dir).parent / "results" / "fixed_test_indices.npy"
    for candidate in [canonical, Path(index_file)]:
        if candidate.exists():
            idx = np.load(str(candidate))
            if len(idx) >= n_samples:
                print(f"    [index] Loaded {len(idx)} indices from {candidate}")
                return idx[:n_samples]

    Path(index_file).parent.mkdir(parents=True, exist_ok=True)
    rng    = np.random.default_rng(SEED)
    orbits = np.load(os.path.join(data_dir, "test", "orbits.npy"))
    n      = min(n_samples, len(orbits))
    idx    = rng.choice(len(orbits), size=n, replace=False)
    np.save(index_file, idx)
    print(f"    [index] Generated {n} fresh indices → {index_file}")
    return idx


def load_shots_for_k(
    data_dir: str,
    k: int,
) -> list[dict[str, Any]]:
    """
    ⚑ FIX-9: Draw k shots from train_pairs.npy with rng(SEED+99), size=k.
    Placeholder ZERO ORBITS used — matches few_shot_stoc.py exactly.
    """
    train_pairs_path = os.path.join(data_dir, "train_pairs.npy")
    train_pairs = np.load(train_pairs_path)   # [500, 2]

    rng      = np.random.default_rng(SEED + 99)
    shot_idx = rng.choice(len(train_pairs), size=k, replace=False)

    shots: list[dict[str, Any]] = []
    for si in shot_idx:
        rule_f = int(train_pairs[si, 0])
        rule_g = int(train_pairs[si, 1])
        lam    = float(round(rng.uniform(LAM_MIN, LAM_MAX), 2))
        lam    = float(_snap_lambda(max(LAM_MIN, min(LAM_MAX, lam)))[0])
        placeholder = np.zeros((T, W), dtype=np.float32)
        shots.append({
            "orbit":  placeholder,
            "rule_f": rule_f,
            "rule_g": rule_g,
            "lambda": lam,
        })
    return shots


def load_data(
    data_dir:   str,
    n_samples:  int,
    index_file: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx          = get_fixed_test_indices(data_dir, n_samples, index_file)
    test_dir     = os.path.join(data_dir, "test")
    test_orbits  = np.load(os.path.join(test_dir, "orbits.npy"))
    test_rule_f  = np.load(os.path.join(test_dir, "rule_f_ids.npy"))
    test_rule_g  = np.load(os.path.join(test_dir, "rule_g_ids.npy"))
    test_lambdas = np.load(os.path.join(test_dir, "lambdas.npy"))

    return (
        test_orbits[idx],
        test_rule_f[idx].astype(int),
        test_rule_g[idx].astype(int),
        test_lambdas[idx].astype(float),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_checkpoint(path: str) -> tuple[list, set]:
    if not os.path.exists(path):
        return [], set()
    try:
        with open(path) as f:
            data = json.load(f)
        samples = data.get("samples", [])
        done    = {s["sample_idx"] for s in samples}
        print(f"    Resuming: {len(done)} samples already completed.")
        return samples, done
    except Exception:
        return [], set()


def save_checkpoint(
    path:         str,
    model_name:   str,
    model_ver:    str,
    setting:      str,
    k_shots:      int,
    n_total:      int,
    shots_used:   list,
    results:      list,
    snap_changes: int,
) -> None:
    n         = len(results)
    pc_vals   = [r["pair_correct"]  for r in results]
    la_vals   = [r["lambda_ok"]     for r in results]
    pair_acc  = sum(pc_vals) / n * 100 if n else 0.0
    lam_acc   = sum(la_vals) / n * 100 if n else 0.0

    mae_vals  = [r["lambda_mae"] for r in results if r["lambda_mae"] is not None]
    lam_mae   = float(np.mean(mae_vals)) if mae_vals else 0.0

    pair_ci   = bootstrap_ci(pc_vals)
    lam_ci    = bootstrap_ci(la_vals)
    api_errs  = sum(1 for r in results if r.get("api_error", False))

    payload = {
        "model":               model_name,
        "model_version":       model_ver,
        "setting":             setting,
        "k_shots":             k_shots,
        "n_evaluated":         n,
        "n_total":             n_total,
        "pair_accuracy_UO_pct":      round(pair_acc, 4),
        "lambda_accuracy_pm005_pct": round(lam_acc, 4),
        "lambda_mae_sym":      round(lam_mae, 6),
        "pair_ci95": {
            "mean":  round(pair_ci["mean"]  * 100, 4),
            "lower": round(pair_ci["lower"] * 100, 4),
            "upper": round(pair_ci["upper"] * 100, 4),
        },
        "lambda_ci95": {
            "mean":  round(lam_ci["mean"]  * 100, 4),
            "lower": round(lam_ci["lower"] * 100, 4),
            "upper": round(lam_ci["upper"] * 100, 4),
        },
        "parse_failures":  sum(1 for r in results if r["pred_f"] is None),
        "api_errors":      api_errs,
        "snap_changes":    snap_changes,
        "few_shot_examples": [
            {"rule_f": s["rule_f"], "rule_g": s["rule_g"], "lambda": s["lambda"]}
            for s in shots_used
        ],
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "samples":   results,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# API clients
# ─────────────────────────────────────────────────────────────────────────────

class GPTClient:
    """
    GPT-based model via OpenAI Chat Completions.
    FIX-1: max_completion_tokens=256.  FIX-2: returns actual model version.
    """
    def __init__(self, model_id: str, reasoning_effort: str = "none"):
        try:
            from openai import OpenAI as _OAI
        except ImportError:
            raise ImportError("pip install openai")
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError("OPENAI_API_KEY not set")
        self.client           = _OAI(api_key=key)
        self.model_id         = model_id
        self.reasoning_effort = reasoning_effort

    def __call__(self, messages: list[dict[str, str]]) -> tuple[str, str]:
        kw: dict[str, Any] = dict(
            model                 = self.model_id,
            messages              = messages,
            reasoning_effort      = self.reasoning_effort,
            max_completion_tokens = MAX_TOKENS_ALL,
        )
        if self.reasoning_effort == "none":
            kw["temperature"] = 0.0
        resp = self.client.chat.completions.create(**kw)
        text = resp.choices[0].message.content or ""
        ver  = resp.model
        return text, ver


# ─────────────────────────────────────────────────────────────────────────────
# LLMClientA — LLM-A-ADD-1 through LLM-A-ADD-5, LLM-A-ADD-9, LLM-A-ADD-10
# ─────────────────────────────────────────────────────────────────────────────
class LLMClientA:
    """
    Frontier LLM-A via the official SDK.
    Install: pip install anthropic

    LLM-A-ADD-1:  Uses the official client SDK (not OpenAI-compatible).
    LLM-A-ADD-2:  Model identifier anonymised for blind review.
    LLM-A-ADD-3:  API does NOT accept role="system" inside the messages list.
                  This client extracts the system message and passes it as the
                  top-level `system` kwarg automatically, so callers can use
                  the same standard message format as for other clients without
                  any changes.
    LLM-A-ADD-4:  Markdown fences stripped for consistency with other clients.
    LLM-A-ADD-5:  Returns (text, resp.model) — actual serving version string
                  confirmed by the API, matching FIX-2 behaviour.
    LLM-A-ADD-9:  SYSTEM_PROMPT hardened with "no reasoning / no explanation"
                  directive (see constant above).
    LLM-A-ADD-10: Assistant prefill — appends {"role":"assistant","content":"{"}
                  to the message list before the API call. The API treats this
                  as already-generated output, so the model is forced to
                  continue from "{" and cannot emit prose first. The opening
                  brace is re-prepended to the returned completion. This
                  eliminates chain-of-thought leakage that caused 100% parse
                  failures when using the original prompt alone.
    """

    # Model identifier anonymised — will be disclosed upon acceptance.
    _MODEL_ENV_VAR = "LLM_A_MODEL_ID"
    _DEFAULT_MODEL = "llm-a-model"   # placeholder; set via env var at runtime

    def __init__(self, model_id: str):
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        key = os.environ.get("LLM_A_API_KEY")
        if not key:
            raise EnvironmentError("LLM_A_API_KEY not set")
        self.client   = _anthropic.Anthropic(api_key=key)
        self.model_id = model_id
        print(f"    [LLMClientA] model_id='{model_id}'")

    def __call__(self, messages: list[dict]) -> tuple[str, str]:
        # LLM-A-ADD-3: split system message out of messages list.
        system_text  = ""
        chat_messages: list[dict] = []
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
            else:
                chat_messages.append({"role": m["role"], "content": m["content"]})

        if not chat_messages:
            raise ValueError(
                "[LLMClientA] messages must contain at least one non-system message"
            )

        # LLM-A-ADD-10: assistant prefill — forces JSON-only output.
        chat_messages.append({"role": "assistant", "content": "{"})

        create_kwargs: dict[str, Any] = {
            "model":       self.model_id,
            "max_tokens":  MAX_TOKENS_ALL,   # FIX-1: uniform token budget
            "messages":    chat_messages,
            "temperature": 0.0,
        }
        if system_text:
            create_kwargs["system"] = system_text

        try:
            resp = self.client.messages.create(**create_kwargs)

            text = ""
            for block in resp.content:
                if hasattr(block, "text"):
                    text = block.text
                    break

            # LLM-A-ADD-10 (cont.): re-attach the opening brace
            text = "{" + text

            # LLM-A-ADD-4: strip markdown fences
            text = re.sub(r"^```(?:json)?\s*", "", text.strip())
            text = re.sub(r"\s*```$", "", text).strip()

            # LLM-A-ADD-5: resp.model is the actual serving version (FIX-2)
            ver = resp.model or self.model_id
            return text, ver

        except Exception as e:
            print(f"    [LLMClientA] API call failed: {type(e).__name__}: {e}")
            raise


# ─────────────────────────────────────────────────────────────────────────────
# Model registry — LLM-A-ADD-6: llm_a entry added.
# Model identifiers anonymised for blind review.
# ─────────────────────────────────────────────────────────────────────────────
MODEL_CONFIGS: dict[str, dict] = {
    "gpt_model": {
        "display": "GPT-based Model",
        "api_id":  "gpt-model-anon",   # anonymised
        "client":  "gpt",
        "rpm":     30,
    },
    # ── LLM-A-ADD-6 ──────────────────────────────────────────────────────
    "llm_a": {
        "display": "Frontier LLM-A",
        "api_id":  "llm-a-model-anon", # anonymised — disclosed upon acceptance
        "client":  "llm_a",
        "rpm":     50,
    },
}


def build_client(cfg: dict, gpt_reasoning: str = "none"):
    """LLM-A-ADD-7: extended with 'llm_a' branch."""
    c, a = cfg["client"], cfg["api_id"]
    if c == "gpt":    return GPTClient(a, reasoning_effort=gpt_reasoning)
    if c == "llm_a":  return LLMClientA(a)   # LLM-A-ADD-7
    raise ValueError(f"Unknown client type: {c}")


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation loop  (one model × one setting)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_one(
    *,
    model_key:      str,
    setting:        str,
    k_shots:        int,
    test_orbits:    np.ndarray,
    test_rule_f:    np.ndarray,
    test_rule_g:    np.ndarray,
    test_lambdas:   np.ndarray,
    shots_used:     list[dict],
    ckpt_dir:       str,
    raw_dir:        str,
    gpt_reasoning:  str = "none",
) -> dict:

    cfg     = MODEL_CONFIGS[model_key]
    display = cfg["display"]
    if model_key == "gpt_model" and gpt_reasoning != "none":
        display = f"GPT-based Model (reasoning={gpt_reasoning})"

    delay   = 60.0 / cfg["rpm"]
    n_total = len(test_orbits)

    reason_sfx = (
        f"_re{gpt_reasoning}"
        if model_key == "gpt_model" and gpt_reasoning != "none"
        else ""
    )
    ckpt_path = os.path.join(
        ckpt_dir, f"{model_key}{reason_sfx}_{setting}.json"
    )

    print(f"\n  [{display}] [{setting}]  k={k_shots}  N={n_total}")

    try:
        client = build_client(cfg, gpt_reasoning=gpt_reasoning)
    except (ImportError, EnvironmentError) as e:
        print(f"    SKIP: {e}")
        return {
            "model":     display,
            "setting":   setting,
            "k_shots":   k_shots,
            "pair_acc":  None,
            "lam_acc":   None,
            "skipped":   True,
            "error":     str(e),
        }

    results, done_set = load_checkpoint(ckpt_path)

    observed_model_versions: set[str] = set()
    snap_change_count: int = sum(r.get("snap_changed", False) for r in results)

    t0 = time.time()

    for i, (orbit, true_f, true_g, true_lam) in enumerate(
        zip(test_orbits, test_rule_f, test_rule_g, test_lambdas)
    ):
        if i in done_set:
            continue

        # ── Build messages ────────────────────────────────────────────────
        if k_shots == 0:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": make_user_prompt(orbit)},
            ]
        else:
            messages = build_messages(shots_used, orbit)

        # ── API call with exponential back-off (FIX-6) ────────────────────
        raw_output = ""
        api_error  = False
        model_ver  = cfg["api_id"]

        for attempt in range(MAX_RETRIES + 1):
            try:
                raw_output, model_ver = client(messages)
                api_error = False
                observed_model_versions.add(model_ver)
                break
            except Exception as exc:
                api_error = True
                if attempt < MAX_RETRIES:
                    retry_after = _extract_retry_after(exc)
                    print(
                        f"    Sample {i}: attempt {attempt + 1} failed "
                        f"({type(exc).__name__}: {exc})."
                    )
                    _backoff_sleep(attempt, retry_after)
                else:
                    print(
                        f"    Sample {i}: API error after {MAX_RETRIES + 1} "
                        f"attempts. Recording failure."
                    )

        # ── Save raw response (full — FIX-5) ──────────────────────────────
        raw_fname = f"{model_key}{reason_sfx}_{setting}_{i:04d}.txt"
        with open(os.path.join(raw_dir, raw_fname), "w", encoding="utf-8") as f:
            f.write(
                f"TRUE_F: {int(true_f)}  TRUE_G: {int(true_g)}  "
                f"TRUE_LAMBDA: {float(true_lam):.2f}\n"
                f"MODEL_VERSION: {model_ver}\n"
                f"API_ERROR: {api_error}\n"
            )
            f.write("─" * 60 + "\n")
            f.write(raw_output)

        # ── Parse (FIX-10: first-object scan) ─────────────────────────────
        pred_f, pred_g, pred_lam, snap_changed = parse_response(raw_output)
        if snap_changed:
            snap_change_count += 1

        pair_ok  = _pair_correct(pred_f, pred_g, int(true_f), int(true_g))
        lam_ok   = _lambda_ok(pred_lam, float(true_lam))
        lam_mae  = _lambda_mae(pred_lam, float(true_lam))

        results.append({
            "sample_idx":    i,
            "true_f":        int(true_f),
            "true_g":        int(true_g),
            "pred_f":        pred_f,
            "pred_g":        pred_g,
            "true_lambda":   float(round(float(true_lam), 4)),
            "pred_lambda":   pred_lam,
            "pair_correct":  pair_ok,
            "lambda_ok":     lam_ok,
            "lambda_mae":    lam_mae,
            "snap_changed":  snap_changed,
            "api_error":     api_error,
            "model_version": model_ver,
            "raw_output":    raw_output,
        })
        done_set.add(i)

        # Checkpoint every 10 samples
        if (i + 1) % 10 == 0 or (i + 1) == n_total:
            ver_str = (
                sorted(observed_model_versions)[-1]
                if observed_model_versions else cfg["api_id"]
            )
            save_checkpoint(
                ckpt_path, display, ver_str, setting,
                k_shots, n_total, shots_used, results,
                snap_change_count,
            )

        # Progress every 20 samples
        if (i + 1) % 20 == 0 or (i + 1) == n_total:
            nd       = len(results)
            pair_acc = sum(r["pair_correct"] for r in results) / nd * 100
            lam_acc  = sum(r["lambda_ok"]    for r in results) / nd * 100
            pf       = sum(1 for r in results if r["pred_f"] is None)
            ae       = sum(1 for r in results if r.get("api_error", False))
            eta      = (time.time() - t0) / (i + 1) * (n_total - i - 1)
            print(
                f"    [{i+1:3d}/{n_total}]  pair(UO)={pair_acc:5.1f}%  "
                f"lam(±0.05sym)={lam_acc:5.1f}%  parse_fail={pf}  "
                f"api_err={ae}  snap={snap_change_count}  ETA={eta:.0f}s"
            )

        time.sleep(delay)

    # ── Final statistics ───────────────────────────────────────────────────
    nd        = len(results)
    pc_vals   = [r["pair_correct"] for r in results]
    la_vals   = [r["lambda_ok"]    for r in results]
    pair_acc  = sum(pc_vals) / nd * 100 if nd else 0.0
    lam_acc   = sum(la_vals) / nd * 100 if nd else 0.0
    mae_vals  = [r["lambda_mae"] for r in results if r["lambda_mae"] is not None]
    lam_mae   = float(np.mean(mae_vals)) if mae_vals else 0.0
    pf        = sum(1 for r in results if r["pred_f"] is None)
    ae        = sum(1 for r in results if r.get("api_error", False))

    pair_ci   = bootstrap_ci(pc_vals)
    lam_ci    = bootstrap_ci(la_vals)

    ver_str = (
        sorted(observed_model_versions)[-1]
        if observed_model_versions else cfg["api_id"]
    )

    print(
        f"  ✓ {display} [{setting}]  "
        f"pair={pair_acc:.2f}% [{pair_ci['lower']*100:.2f},{pair_ci['upper']*100:.2f}]  "
        f"lam={lam_acc:.2f}% [{lam_ci['lower']*100:.2f},{lam_ci['upper']*100:.2f}]  "
        f"lam_mae={lam_mae:.4f}  "
        f"parse_fail={pf}  api_err={ae}  snap={snap_change_count}  ver={ver_str}"
    )

    return {
        "model":         display,
        "model_version": ver_str,
        "setting":       setting,
        "k_shots":       k_shots,
        "n":             nd,
        "pair_acc":      round(pair_acc, 4),
        "lam_acc":       round(lam_acc,  4),
        "lam_mae":       round(lam_mae,  6),
        "pair_ci95": {
            "mean":  round(pair_ci["mean"]  * 100, 4),
            "lower": round(pair_ci["lower"] * 100, 4),
            "upper": round(pair_ci["upper"] * 100, 4),
        },
        "lambda_ci95": {
            "mean":  round(lam_ci["mean"]  * 100, 4),
            "lower": round(lam_ci["lower"] * 100, 4),
            "upper": round(lam_ci["upper"] * 100, 4),
        },
        "parse_rate":   round((nd - pf) / nd * 100, 4) if nd else 0.0,
        "parse_fail":   pf,
        "api_errors":   ae,
        "snap_changes": snap_change_count,
        "skipped":      False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Open-source reference numbers — paper Table 29 (stocECA, N=200)
# FIX-8: populate sha256 before camera-ready submission.
# ─────────────────────────────────────────────────────────────────────────────
PAPER_RESULTS: list[dict] = [
    # zero-shot
    {"model": "Qwen2.5-7B",    "setting": "zero_shot", "pair_acc": 0.00, "lam_acc":  0.00,
     "sha256": None, "source_ckpt": "results/zero_shot/Qwen2.5-7B-Instruct_zero_shot.json"},
    {"model": "Qwen2.5-72B",   "setting": "zero_shot", "pair_acc": 0.00, "lam_acc": 27.50,
     "sha256": None, "source_ckpt": "results/zero_shot/Qwen2.5-72B-Instruct_zero_shot.json"},
    {"model": "Llama-3.1-8B",  "setting": "zero_shot", "pair_acc": 0.00, "lam_acc":  0.00,
     "sha256": None, "source_ckpt": "results/zero_shot/Llama-3.1-8B-Instruct_zero_shot.json"},
    {"model": "Llama-3.1-70B", "setting": "zero_shot", "pair_acc": 0.00, "lam_acc":  0.00,
     "sha256": None, "source_ckpt": "results/zero_shot/Llama-3.1-70B-Instruct_zero_shot.json"},
    {"model": "Mistral-7B",    "setting": "zero_shot", "pair_acc": 0.00, "lam_acc":  0.00,
     "sha256": None, "source_ckpt": "results/zero_shot/Mistral-7B-Instruct-v0.3_zero_shot.json"},
    {"model": "Mixtral-8x7B",  "setting": "zero_shot", "pair_acc": 0.00, "lam_acc":  0.00,
     "sha256": None, "source_ckpt": "results/zero_shot/Mixtral-8x7B-Instruct-v0.1_zero_shot.json"},
    # 5-shot
    {"model": "Qwen2.5-7B",    "setting": "5_shot",    "pair_acc": 0.00, "lam_acc":  0.00,
     "sha256": None, "source_ckpt": "results/few_shot/Qwen2.5-7B-Instruct_5shot.json"},
    {"model": "Qwen2.5-72B",   "setting": "5_shot",    "pair_acc": 0.00, "lam_acc": 28.00,
     "sha256": None, "source_ckpt": "results/few_shot/Qwen2.5-72B-Instruct_5shot.json"},
    {"model": "Llama-3.1-8B",  "setting": "5_shot",    "pair_acc": 0.00, "lam_acc":  0.00,
     "sha256": None, "source_ckpt": "results/few_shot/Llama-3.1-8B-Instruct_5shot.json"},
    {"model": "Llama-3.1-70B", "setting": "5_shot",    "pair_acc": 0.00, "lam_acc":  0.00,
     "sha256": None, "source_ckpt": "results/few_shot/Llama-3.1-70B-Instruct_5shot.json"},
    {"model": "Mistral-7B",    "setting": "5_shot",    "pair_acc": 0.00, "lam_acc":  0.00,
     "sha256": None, "source_ckpt": "results/few_shot/Mistral-7B-Instruct-v0.3_5shot.json"},
    {"model": "Mixtral-8x7B",  "setting": "5_shot",    "pair_acc": 0.00, "lam_acc":  0.00,
     "sha256": None, "source_ckpt": "results/few_shot/Mixtral-8x7B-Instruct-v0.1_5shot.json"},
    # fine-tuned
    {"model": "Qwen2.5-72B (fine-tuned)", "setting": "fine_tuned",
     "pair_acc": 14.60, "lam_acc": 32.60,
     "sha256": None, "source_ckpt": "results/fine_tuned/Qwen2.5-72B-Instruct_ft.json"},
]

STOCM_REF: dict = {
    "model":    "stocM — signal-matched transformer",
    "setting":  "trained",
    "pair_acc": 95.49,
    "lam_acc":  87.83,
    "lam_mae":  0.0289,
}


def verify_paper_results(paper_results: list[dict]) -> None:
    print("\n[verify] Checking paper reference numbers against checkpoint files …")
    any_warned = False
    for r in paper_results:
        ckpt = r.get("source_ckpt", "")
        if not ckpt or not os.path.exists(ckpt):
            continue
        sha = file_sha256(ckpt)
        if r.get("sha256") and sha != r["sha256"]:
            print(
                f"  ⚠ SHA-256 MISMATCH for {r['model']} [{r['setting']}]\n"
                f"    expected: {r['sha256']}\n"
                f"    actual:   {sha}"
            )
            any_warned = True
        try:
            with open(ckpt) as f:
                data = json.load(f)
            file_pair = round(
                data.get("pair_accuracy_UO",
                data.get("pair_accuracy_UO_pct", None)), 2)
            file_lam  = round(
                data.get("lambda_accuracy_pm005",
                data.get("lambda_accuracy_pm005_pct", None)), 2)
            if file_pair is not None and abs(file_pair - r["pair_acc"]) > 0.01:
                print(
                    f"  ⚠ pair_acc mismatch for {r['model']} [{r['setting']}]: "
                    f"hardcoded={r['pair_acc']} file={file_pair}"
                )
                any_warned = True
            if file_lam is not None and abs(file_lam - r["lam_acc"]) > 0.01:
                print(
                    f"  ⚠ lam_acc mismatch for {r['model']} [{r['setting']}]: "
                    f"hardcoded={r['lam_acc']} file={file_lam}"
                )
                any_warned = True
        except Exception as e:
            print(f"  ⚠ Could not parse {ckpt}: {e}")
            any_warned = True
    if not any_warned:
        print("  ✓ All available checkpoint files match hardcoded values.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frontier LLM evaluation on stocECA — anonymous submission, under review"
    )
    parser.add_argument(
        "--data_dir", default="SCA_Data",
        help="stocECA data root (contains test/ and train_pairs.npy)",
    )
    parser.add_argument(
        "--output_dir", default="frontier_results_stoc",
        help="Output root (checkpoints/, raw/, summary files)",
    )
    parser.add_argument(
        "--n_samples", type=int, default=200,
        help="Test samples per condition",
    )
    parser.add_argument(
        "--models", nargs="+",
        default=list(MODEL_CONFIGS.keys()),
        choices=list(MODEL_CONFIGS.keys()),
        # LLM-A-ADD-8: llm_a included automatically via MODEL_CONFIGS.keys()
    )
    parser.add_argument(
        "--settings", nargs="+",
        default=["zero_shot", "few_shot"],
        choices=["zero_shot", "few_shot"],
    )
    parser.add_argument(
        "--k_shots", nargs="+", type=int,
        default=DEFAULT_K_LIST,
        help="k values for few-shot (default: 1 3 5; max=5)",
    )
    parser.add_argument(
        "--gpt_reasoning",
        default="none",
        choices=["none", "low", "medium", "high"],
    )
    parser.add_argument(
        "--skip_verify", action="store_true",
        help="Skip SHA-256 / value verification of paper reference numbers",
    )
    args = parser.parse_args()

    bad_k = [k for k in args.k_shots if k > MAX_K_SHOTS or k < 1]
    if bad_k:
        parser.error(
            f"Invalid --k_shots values: {bad_k}. "
            f"Must be integers in [1, {MAX_K_SHOTS}]."
        )

    # ── Directory layout ──────────────────────────────────────────────────
    ckpt_dir   = os.path.join(args.output_dir, "checkpoints")
    raw_dir    = os.path.join(args.output_dir, "raw")
    index_file = os.path.join(args.output_dir, "fixed_test_indices.npy")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(raw_dir,  exist_ok=True)

    if not args.skip_verify:
        verify_paper_results(PAPER_RESULTS)

    # ── Load test data ────────────────────────────────────────────────────
    print(f"\n[data] Loading from {args.data_dir}/test/")
    test_orbits, test_rule_f, test_rule_g, test_lambdas = load_data(
        args.data_dir, args.n_samples, index_file
    )
    print(
        f"[data] {len(test_orbits)} test samples  |  "
        f"orbits shape: {test_orbits.shape}  (using orbit[0] per sample)"
    )
    print(
        f"[cfg]  max_tokens={MAX_TOKENS_ALL} (uniform — FIX-1)  |  "
        f"bootstrap B={BOOTSTRAP_REPS} (FIX-3)  |  SEED={SEED}"
    )
    if "few_shot" in args.settings:
        k_str = ", ".join(str(k) for k in sorted(args.k_shots))
        print(
            f"[few-shot] k={k_str}  |  "
            f"shots from train_pairs.npy, rng(SEED+99), placeholder zero orbits  "
            f"(matches few_shot_stoc.py — FIX-9)"
        )
    if "gpt_model" in args.models:
        print(f"[GPT-based Model] reasoning_effort={args.gpt_reasoning}")

    # ── Build job list ────────────────────────────────────────────────────
    jobs: list[tuple[str, str, int, list]] = []
    for model_key in args.models:
        if "zero_shot" in args.settings:
            jobs.append((model_key, "zero_shot", 0, []))
        if "few_shot" in args.settings:
            for k in sorted(args.k_shots):
                shots_k = load_shots_for_k(args.data_dir, k)
                jobs.append((model_key, f"{k}_shot", k, shots_k))

    print(f"\n[run] {len(jobs)} conditions queued:")
    for model_key, setting, k, shots_k in jobs:
        if k > 0:
            shot_summary = ", ".join(
                f"f={s['rule_f']} g={s['rule_g']} λ={s['lambda']}"
                for s in shots_k
            )
            print(
                f"      {MODEL_CONFIGS[model_key]['display']:<28} {setting}  "
                f"shots=[{shot_summary}]"
            )
        else:
            print(f"      {MODEL_CONFIGS[model_key]['display']:<28} {setting}")

    # ── Run ───────────────────────────────────────────────────────────────
    all_metrics: list[dict] = []
    t_total = time.time()

    for model_key, setting, k, shots_k in jobs:
        m = evaluate_one(
            model_key      = model_key,
            setting        = setting,
            k_shots        = k,
            test_orbits    = test_orbits,
            test_rule_f    = test_rule_f,
            test_rule_g    = test_rule_g,
            test_lambdas   = test_lambdas,
            shots_used     = shots_k,
            ckpt_dir       = ckpt_dir,
            raw_dir        = raw_dir,
            gpt_reasoning  = args.gpt_reasoning,
        )
        all_metrics.append(m)

    elapsed = (time.time() - t_total) / 60
    print(f"\nTotal wall time: {elapsed:.1f} min")

    # ── Console table ─────────────────────────────────────────────────────
    COL = 100
    print("\n" + "=" * COL)
    print("  stocECA IDENTIFICATION — FRONTIER LLM EVALUATION")
    print(
        f"  Pair: unordered-pair exact match  |  "
        f"Lambda: sym ±0.05 (FIX-4)  |  N={args.n_samples}"
    )
    print(
        f"  Prompts: verbatim Appendix F  |  "
        f"Orbit: first 50 rows of orbit[0]  |  "
        f"max_tokens={MAX_TOKENS_ALL} (FIX-1)"
    )
    print(
        f"  CIs: 95% bootstrap percentile, B={BOOTSTRAP_REPS}  |  "
        f"Shots: train_pairs.npy, rng(SEED+99), placeholder zeros (FIX-9)"
    )
    print("=" * COL)
    print(
        f"  {'Model':<36} {'Setting':<12} {'Pair%(UO)':>10}  "
        f"{'[95% CI]':>16}  {'λ%':>7}  {'[95% CI]':>16}"
    )
    print("-" * COL)

    print("  — Open-source LLMs (paper Table 29, N=200) —")
    for r in PAPER_RESULTS:
        print(
            f"  {r['model']:<36} {r['setting']:<12} "
            f"{r['pair_acc']:>10.2f}  {'':>16}  "
            f"{r['lam_acc']:>7.2f}"
        )

    print("\n  — Frontier LLMs (this evaluation) —")
    for m in all_metrics:
        if m.get("skipped"):
            print(
                f"  {m['model']:<36} {m['setting']:<12}  "
                f"SKIPPED ({m.get('error', '')})"
            )
        else:
            pc  = m["pair_ci95"]
            lc  = m["lambda_ci95"]
            p_ci = f"[{pc['lower']:.2f},{pc['upper']:.2f}]"
            l_ci = f"[{lc['lower']:.2f},{lc['upper']:.2f}]"
            snap = m.get("snap_changes", 0)
            print(
                f"  {m['model']:<36} {m['setting']:<12} "
                f"{m['pair_acc']:>10.2f}  {p_ci:>16}  "
                f"{m['lam_acc']:>7.2f}  {l_ci:>16}  "
                f"snap={snap}  ver={m.get('model_version','?')}"
            )

    r = STOCM_REF
    print(
        f"\n  {r['model']:<36} {r['setting']:<12} "
        f"{r['pair_acc']:>10.2f}  {'':>16}  "
        f"{r['lam_acc']:>7.2f}  <- ours  (lam_mae={r['lam_mae']:.4f})"
    )
    print("=" * COL)

    # ── Save results_detail.json ──────────────────────────────────────────
    detail_path = os.path.join(args.output_dir, "results_detail.json")
    with open(detail_path, "w") as f:
        json.dump(
            {
                "dataset":           "stocECA",
                "frontier_metrics":  all_metrics,
                "paper_reference":   PAPER_RESULTS,
                "stocM_reference":   STOCM_REF,
                "n_samples":         args.n_samples,
                "k_shots_evaluated": sorted(args.k_shots),
                "methodology": {
                    "token_budget": (
                        f"FIX-1: max_tokens={MAX_TOKENS_ALL} applied uniformly."
                    ),
                    "llm_a_integration": (
                        "LLM-A-ADD: LLMClientA uses official SDK. "
                        "Model identifier anonymised for blind review. "
                        "System prompt extracted and passed as top-level `system` kwarg. "
                        "Markdown fences stripped. resp.model echoed as model_version (FIX-2). "
                        "LLM-A-ADD-9: SYSTEM_PROMPT hardened with no-reasoning directive. "
                        "LLM-A-ADD-10: Assistant prefill '{' forces JSON-only output."
                    ),
                    "model_versioning":     "FIX-2: actual model version per sample.",
                    "confidence_intervals": f"FIX-3: 95% bootstrap CIs, B={BOOTSTRAP_REPS}.",
                    "lambda_evaluation":    f"FIX-4: snap to {LAM_GRID_STEP} grid, sym ±{LAM_TOL}.",
                    "raw_output_storage":   "FIX-5: full output, no truncation.",
                    "api_error_tracking":   f"FIX-6: exponential back-off, max_retries={MAX_RETRIES}.",
                    "reference_verification": "FIX-8: sha256 fingerprints in PAPER_RESULTS.",
                    "shot_design":          "FIX-9: rng(SEED+99), placeholder zero orbits.",
                    "parse_logic":          "FIX-10: first-valid-JSON-object scan.",
                },
                "seed":           SEED,
                "shot_seed":      SEED + 99,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_reps": BOOTSTRAP_REPS,
                "max_tokens":     MAX_TOKENS_ALL,
                "gpt_reasoning":  args.gpt_reasoning,
                "max_retries":    MAX_RETRIES,
                "paper_table":    "Table 29 (stocECA, N=200)",
                "run_timestamp":  datetime.now(tz=timezone.utc).isoformat(),
            },
            f,
            indent=2,
        )

    # ── Save results_summary.csv ──────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, "results_summary.csv")
    with open(csv_path, "w") as f:
        f.write(
            "Model,Model Version,Setting,k_shots,N,"
            "Pair UO Acc (%),Pair CI95 Lower,Pair CI95 Upper,"
            "Lambda Acc (%),Lambda CI95 Lower,Lambda CI95 Upper,"
            "Lambda MAE (sym),Parse Failures,API Errors,Snap Changes\n"
        )
        for r in PAPER_RESULTS:
            k_val = "0" if "zero" in r["setting"] else r["setting"].split("_")[0]
            f.write(
                f"{r['model']},—,{r['setting']},{k_val},{args.n_samples},"
                f"{r['pair_acc']},—,—,{r['lam_acc']},—,—,—,—,—,—\n"
            )
        for m in all_metrics:
            if not m.get("skipped"):
                pc = m["pair_ci95"]
                lc = m["lambda_ci95"]
                f.write(
                    f"{m['model']},{m.get('model_version','?')},"
                    f"{m['setting']},{m['k_shots']},{m['n']},"
                    f"{m['pair_acc']},{pc['lower']},{pc['upper']},"
                    f"{m['lam_acc']},{lc['lower']},{lc['upper']},"
                    f"{m.get('lam_mae', '')},"
                    f"{m['parse_fail']},{m['api_errors']},"
                    f"{m.get('snap_changes', 0)}\n"
                )
        r = STOCM_REF
        f.write(
            f"{r['model']},—,{r['setting']},—,—,"
            f"{r['pair_acc']},—,—,{r['lam_acc']},—,—,"
            f"{r['lam_mae']},—,—,—\n"
        )

    print(f"\nSaved: {detail_path}")
    print(f"Saved: {csv_path}")
    print(f"Checkpoints: {ckpt_dir}/")
    print(f"Raw responses: {raw_dir}/")


if __name__ == "__main__":
    main()