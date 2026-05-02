"""
frontier_eval_tempECA.py  —  Anonymous submission
=================================================
Zero-shot and few-shot (k=1,3,5) evaluation of frontier LLMs on the tempECA
identification task (temporally stochastic ECA).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILE STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ECA_TEMP/
│
├── TSCA_Data/                          ← dataset (read-only)
│   ├── train_rules.npy                 # int32 [179]
│   ├── test_rules.npy                  # int32 [77]
│   ├── train_pairs.npy                 # int32 [500, 2]
│   ├── test_pairs.npy                  # int32 [500, 2]
│   └── test/
│       ├── orbits.npy                  # float32 [10000, 8, 200, 50]
│       ├── rule_f_ids.npy              # int32   [10000]
│       ├── rule_g_ids.npy              # int32   [10000]
│       ├── taus.npy                    # float32 [10000]
│       └── step_labels.npy             # int8    [10000, 8, 199]
│
├── results/
│   ├── fixed_test_indices.npy          ← SHARED — used by ALL scripts
│   ├── zero_shot/<Model>_zero_shot.json
│   └── few_shot/<Model>_<k>shot.json
│
├── frontier_results/                   ← THIS script writes here
│   ├── fixed_test_indices.npy
│   ├── results_summary.csv
│   ├── results_detail.json
│   ├── checkpoints/
│   │   └── <model>_<setting>.json
│   └── raw/
│       └── <model>_<setting>_<idx:04d>.txt
│
└── frontier_eval_tempECA.py            ← THIS FILE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROMPT FIDELITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALL prompts are taken VERBATIM from Appendix F.  Do not modify SYSTEM_PROMPT,
_orbit_to_str, make_user_prompt, build_messages, or parse_response without
updating the paper appendix to match.

  SYSTEM_PROMPT   : verbatim from Appendix F (tempECA variant)
  orbit encoding  : raw "0"/"1" strings, first 50 rows of orbit[0]
  user prompt     : "Space-time orbit (50 rows x 50 cells):\n\n{orbit}\n\n..."
  few-shot format : multi-turn chat [system][user][assistant]…[user]
  response format : {"rule_f": <int>, "rule_g": <int>, "tau": <float>}
  parse logic     : Stage-1 JSON → Stage-2 regex fallback
  test indices    : shared fixed_test_indices.npy (SEED=42)
  shot sampling   : np.random.default_rng(SEED+99), draw MAX_K_SHOTS=5 once;
                    nested slicing: k=1→[:1], k=3→[:3], k=5→[:5]
  tau snap        : nearest multiple of TAU_SNAP_STEP=0.05 (FIX-T3/FIX-4)
  tau tolerance   : ±0.05 symmetric: min(|p-t|, |p-(1-t)|) ≤ TAU_TOL
  UO rule match   : (pred_f==true_f and pred_g==true_g) OR
                    (pred_f==true_g and pred_g==true_f)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Setup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  pip install openai anthropic numpy

  export OPENAI_API_KEY=sk-...
  export ANTHROPIC_API_KEY=sk-ant-...

Usage
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # Full run — all models, zero-shot + k=1,3,5:
  python frontier_eval_tempECA.py --data_dir TSCA_Data --n_samples 200

  # Model-A only, zero-shot:
  python frontier_eval_tempECA.py --models model_a --settings zero_shot

  # Model-B few-shot k=5, 100 samples:
  python frontier_eval_tempECA.py --models model_b --settings few_shot --k_shots 5 --n_samples 100

  # Both models, all shot settings:
  python frontier_eval_tempECA.py --models model_a model_b

  # Resume interrupted run (already-done samples are auto-skipped):
  python frontier_eval_tempECA.py --models model_a --n_samples 200

  # Model-C with reasoning:
  python frontier_eval_tempECA.py --models model_c --model_c_reasoning low
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
TAU_TOL         : float       = 0.05          # symmetric tolerance
TAU_MIN         : float       = 0.1
TAU_MAX         : float       = 0.9
TAU_SNAP_STEP   : float       = 0.05          # FIX-T3: snap grid
W               : int         = 50
T               : int         = 200
K               : int         = 8             # orbits per sample (show 1)
MAX_K_SHOTS     : int         = 5
DEFAULT_K_LIST  : list[int]   = [1, 3, 5]

# ⚑ FIX-1: unified token budget across all frontier APIs.
MAX_TOKENS_ALL  : int         = 256

# Bootstrap parameters (FIX-3)
BOOTSTRAP_REPS  : int         = 2000
BOOTSTRAP_SEED  : int         = SEED + 7

# ECA rule table — used for generating real few-shot orbits (FIX-T1)
_RULE_TABLE: dict[int, np.ndarray] = {}

def _get_rule_table(rule: int) -> np.ndarray:
    """Return the 8-entry lookup table for an ECA rule (index = neighbourhood)."""
    if rule not in _RULE_TABLE:
        bits = np.array([(rule >> i) & 1 for i in range(8)], dtype=np.int8)
        _RULE_TABLE[rule] = bits
    return _RULE_TABLE[rule]


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — verbatim from Appendix F (tempECA variant)
# DO NOT MODIFY — any change breaks comparability with paper Table F.tempECA
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT: str = (
    "You are an expert in cellular automata. "
    "You are given a space-time orbit of an Elementary Cellular Automaton (ECA) "
    "under temporally stochastic noise.\n\n"
    "In this temporally stochastic model, at each timestep the ENTIRE row is "
    "updated by one of two rules. With probability tau the row uses rule G; "
    "otherwise the entire row uses rule F. "
    "This is a row-level coin flip — all cells in a row use the same rule at "
    "each timestep.\n\n"
    "The parameter tau is a continuous value between 0.1 and 0.9. "
    "The system (F, G, tau) is statistically identical to (G, F, 1-tau), "
    "so predict the smaller rule number as rule_f and the larger as rule_g.\n\n"
    "The orbit is a grid of 0s and 1s. Each row is one timestep. "
    "Grid width is 50 cells.\n\n"
    "Identify:\n"
    "1. Rule F (the first ECA rule, integer 0-255)\n"
    "2. Rule G (the second ECA rule, integer 0-255, different from F)\n"
    "3. Tau value (float between 0.1 and 0.9, rounded to 2 decimal places)\n\n"
    "Output ONLY a single JSON object. No reasoning, no explanation, no markdown, "
    "no extra text before or after. Your entire response must be exactly:\n"
    '{"rule_f": <integer>, "rule_g": <integer>, "tau": <float>}'
)


# ─────────────────────────────────────────────────────────────────────────────
# Orbit encoding — verbatim from Appendix F _orbit_to_str (tempECA variant)
# Shows first 50 rows of orbit[0] (the first of K=8 orbits). FIX-T4.
# ─────────────────────────────────────────────────────────────────────────────
def _orbit_to_str(orbit: np.ndarray) -> str:
    """
    orbit: shape (T, W) — a single orbit (not the full K-stack).
    Shows the first min(T, 50) rows, matching open-source zero_shot_temp.py.
    """
    rows = min(len(orbit), 50)
    return "\n".join("".join(str(int(c)) for c in row) for row in orbit[:rows])


def make_user_prompt(orbit: np.ndarray) -> str:
    """orbit: shape (T, W) — a single orbit."""
    return (
        f"Space-time orbit (50 rows x {W} cells):\n\n"
        f"{_orbit_to_str(orbit)}\n\n"
        "What are rule F, rule G, and tau?"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Few-shot message builder — verbatim from Appendix F Section F.2
# Multi-turn: [system][user][assistant] … [user]
# ─────────────────────────────────────────────────────────────────────────────
def build_messages(
    shots: list[dict[str, Any]],
    orbit: np.ndarray,       # shape (T, W) — orbit[0] of the test sample
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
                "rule_f": int(s["rule_f"]),
                "rule_g": int(s["rule_g"]),
                "tau":    float(s["tau"]),
            }),
        })
    messages.append({
        "role": "user",
        "content": make_user_prompt(orbit),
    })
    return messages


# ─────────────────────────────────────────────────────────────────────────────
# Tau snapping and metric helpers
# ─────────────────────────────────────────────────────────────────────────────
def snap_tau(val: float) -> float:
    """
    Snap val to the nearest TAU_SNAP_STEP=0.05 grid point,
    then clamp to [TAU_MIN, TAU_MAX].  FIX-T3 / FIX-4.
    """
    snapped = round(round(val / TAU_SNAP_STEP) * TAU_SNAP_STEP, 2)
    return float(max(TAU_MIN, min(TAU_MAX, snapped)))


def tau_sym_err(pred: float, true: float) -> float:
    """Symmetric tau error accounting for the (F,G,tau)≡(G,F,1-tau) symmetry."""
    return min(abs(pred - true), abs(pred - (1.0 - true)))


def rules_match_uo(pred_f: int, pred_g: int, true_f: int, true_g: int) -> bool:
    """Unordered-pair (UO) match in either orientation."""
    return (
        (pred_f == true_f and pred_g == true_g) or
        (pred_f == true_g and pred_g == true_f)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Response parser — verbatim from Appendix F parse_response (tempECA variant)
# Stage 1: JSON parse.  Stage 2: regex fallback.
# Pred tau is snapped before return (FIX-T3 / FIX-4).
# ─────────────────────────────────────────────────────────────────────────────
def parse_response(response: str) -> tuple[int | None, int | None, float | None]:
    # Strip markdown fences that some models add (e.g. ```json ... ```)
    response = re.sub(r"^```(?:json)?\s*", "", response.strip())
    response = re.sub(r"\s*```$", "", response).strip()

    try:
        start = response.find("{")
        end   = response.rfind("}") + 1
        if start != -1 and end > start:
            payload = json.loads(response[start:end])
            rf  = int(payload.get("rule_f", -1))
            rg  = int(payload.get("rule_g", -1))
            tau = float(payload.get("tau",    -1.0))
            if 0 <= rf <= 255 and 0 <= rg <= 255 and TAU_MIN <= tau <= TAU_MAX:
                return rf, rg, snap_tau(tau)
    except Exception:
        pass

    rfm  = re.search(r'"rule_f"\s*:\s*(\d+)',   response)
    rgm  = re.search(r'"rule_g"\s*:\s*(\d+)',   response)
    taum = re.search(r'"tau"\s*:\s*([\d.]+)',   response)
    if rfm and rgm and taum:
        rf  = int(rfm.group(1))
        rg  = int(rgm.group(1))
        tau = float(taum.group(1))
        if 0 <= rf <= 255 and 0 <= rg <= 255 and TAU_MIN <= tau <= TAU_MAX:
            return rf, rg, snap_tau(tau)

    return None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# Real orbit generation for few-shot examples (FIX-T1)
# ─────────────────────────────────────────────────────────────────────────────
def _simulate_temp_eca(
    rule_f: int,
    rule_g: int,
    tau: float,
    width: int,
    steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate one temporally stochastic ECA orbit.

    Returns shape (steps+1, width) of dtype int8.
    Row t+1 is produced by flipping a coin: with prob tau the ENTIRE row
    uses rule_g, otherwise it uses rule_f — matching the tempECA definition.
    Initial configuration: uniform random binary, non-trivial
    (at least one 0 and one 1).
    """
    f_table = _get_rule_table(rule_f)
    g_table = _get_rule_table(rule_g)

    # Non-trivial initial configuration
    while True:
        row = rng.integers(0, 2, size=width, dtype=np.int8)
        if 0 < row.sum() < width:
            break

    orbit = np.empty((steps + 1, width), dtype=np.int8)
    orbit[0] = row

    for t in range(steps):
        use_g = rng.random() < tau
        table = g_table if use_g else f_table
        # Periodic boundary: left = roll right, right = roll left
        left   = np.roll(orbit[t], 1)
        centre = orbit[t]
        right  = np.roll(orbit[t], -1)
        # Neighbourhood index: left*4 + centre*2 + right
        nbr_idx = (left * 4 + centre * 2 + right).astype(np.int8)
        orbit[t + 1] = table[nbr_idx]

    return orbit


def generate_shot_orbits(
    train_pairs: np.ndarray,          # shape (N_pairs, 2)
    n_shots: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """
    Draw n_shots training rule pairs and generate real tempECA orbits for each.
    Tau is sampled uniformly from the interior of [TAU_MIN, TAU_MAX].

    Returns a list of dicts with keys: orbit (T×W ndarray), rule_f, rule_g, tau.
    """
    shot_pair_idx = rng.choice(len(train_pairs), size=n_shots, replace=False)
    shots: list[dict[str, Any]] = []

    for idx in shot_pair_idx:
        rf  = int(train_pairs[idx, 0])
        rg  = int(train_pairs[idx, 1])
        # Canonical ordering: smaller rule as rule_f, larger as rule_g
        if rf > rg:
            rf, rg = rg, rf
        # Sample tau, avoiding the degenerate τ=0.5 boundary
        while True:
            tau = float(rng.uniform(TAU_MIN + 0.05, TAU_MAX - 0.05))
            tau = snap_tau(tau)
            if abs(tau - 0.5) > 1e-6:
                break
        orbit = _simulate_temp_eca(rf, rg, tau, W, T, rng)
        shots.append({
            "orbit":  orbit,          # shape (T+1, W) → sliced to (T, W) for prompt
            "rule_f": rf,
            "rule_g": rg,
            "tau":    tau,
        })

    return shots


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap confidence intervals (FIX-3)
# ─────────────────────────────────────────────────────────────────────────────
def bootstrap_ci(
    values: list[bool | float],
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
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def get_fixed_test_indices(
    data_dir:   str,
    n_samples:  int,
    index_file: str,
) -> np.ndarray:
    """
    Load or create the shared fixed test index file.

    Priority:
      1. results/fixed_test_indices.npy  (written by open-source scripts)
      2. frontier_results/fixed_test_indices.npy  (written here)
      3. Generate fresh with SEED=42 → write to index_file
    """
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


def load_data(
    data_dir:   str,
    n_samples:  int,
    index_file: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """
    Returns (test_orbits, test_rule_f, test_rule_g, test_taus, all_shots).

    test_orbits: shape (n_samples, K, T, W)
    all_shots  : MAX_K_SHOTS=5 entries with real simulated tempECA orbits (FIX-T1).
    Callers slice to desired k: k=1→[:1], k=3→[:3], k=5→[:5].
    """
    idx      = get_fixed_test_indices(data_dir, n_samples, index_file)
    test_dir = os.path.join(data_dir, "test")

    test_orbits  = np.load(os.path.join(test_dir, "orbits.npy"))     # [N, K, T, W]
    test_rule_f  = np.load(os.path.join(test_dir, "rule_f_ids.npy"))
    test_rule_g  = np.load(os.path.join(test_dir, "rule_g_ids.npy"))
    test_taus    = np.load(os.path.join(test_dir, "taus.npy"))
    train_pairs  = np.load(os.path.join(data_dir, "train_pairs.npy"))  # [500, 2]

    # Generate real few-shot orbits (FIX-T1)
    shot_rng = np.random.default_rng(SEED + 99)
    all_shots = generate_shot_orbits(train_pairs, MAX_K_SHOTS, shot_rng)

    return (
        test_orbits[idx],
        test_rule_f[idx].astype(int),
        test_rule_g[idx].astype(int),
        test_taus[idx].astype(float),
        all_shots,
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
    path:        str,
    model_name:  str,
    model_ver:   str,          # FIX-2
    setting:     str,
    k_shots:     int,
    n_total:     int,
    shots_used:  list,
    results:     list,
) -> None:
    n   = len(results)
    pc  = [r["pair_correct"] for r in results]
    tok = [r["tau_ok"]       for r in results]
    mae = [r["tau_mae"] for r in results if r["tau_mae"] is not None]

    pair_acc = sum(pc)  / n * 100 if n else 0.0
    tau_acc  = sum(tok) / n * 100 if n else 0.0
    tau_mae  = float(np.mean(mae)) if mae else 0.0

    pair_ci = bootstrap_ci(pc)
    tau_ci  = bootstrap_ci(tok)

    api_errors = sum(1 for r in results if r.get("api_error", False))

    payload = {
        "model":                model_name,
        "model_version":        model_ver,
        "setting":              setting,
        "k_shots":              k_shots,
        "n_evaluated":          n,
        "n_total":              n_total,
        "pair_accuracy_UO_pct": round(pair_acc, 4),
        "tau_accuracy_pct":     round(tau_acc,  4),
        "tau_mae_sym":          round(tau_mae,  6),
        "pair_ci95": {
            "mean":  round(pair_ci["mean"]  * 100, 4),
            "lower": round(pair_ci["lower"] * 100, 4),
            "upper": round(pair_ci["upper"] * 100, 4),
        },
        "tau_ci95": {
            "mean":  round(tau_ci["mean"]  * 100, 4),
            "lower": round(tau_ci["lower"] * 100, 4),
            "upper": round(tau_ci["upper"] * 100, 4),
        },
        "parse_failures":     sum(1 for r in results if r["pred_f"] is None),
        "api_errors":         api_errors,
        "few_shot_examples": [
            {"rule_f": s["rule_f"], "rule_g": s["rule_g"], "tau": s["tau"]}
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

class ModelCClient:
    """
    Model C via OpenAI Chat Completions interface.
    FIX-1: max_completion_tokens=MAX_TOKENS_ALL=256 uniformly.
    FIX-2: returns (text, actual_model_id_from_api).
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
            max_completion_tokens = MAX_TOKENS_ALL,    # FIX-1
        )
        if self.reasoning_effort == "none":
            kw["temperature"] = 0.0
        resp = self.client.chat.completions.create(**kw)
        text = resp.choices[0].message.content or ""
        ver  = resp.model                               # FIX-2
        return text, ver


class ModelAClient:
    """
    Model A via the official SDK.
    Install: pip install anthropic

    System prompt handled via top-level `system` kwarg (the API does NOT
    accept role="system" inside the messages list).  Markdown fences are
    stripped for consistency.  Returns (text, resp.model) — actual serving
    version confirmed by the API (FIX-2).

    Assistant prefill trick: appends {"role":"assistant","content":"{"}
    to force JSON-only output; re-attaches "{" to completion (see inline
    note below).
    """

    def __init__(self, model_id: str):
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set")
        self.client   = _anthropic.Anthropic(api_key=key)
        self.model_id = model_id
        print(f"    [ModelAClient] model_id='{model_id}'")

    def __call__(self, messages: list[dict]) -> tuple[str, str]:
        # Split system message out — the API requires it as a top-level kwarg,
        # not as a role inside the messages list.
        system_text  = ""
        chat_messages: list[dict] = []
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
            else:
                chat_messages.append({"role": m["role"], "content": m["content"]})

        if not chat_messages:
            raise ValueError(
                "[ModelAClient] messages must contain at least one non-system message"
            )

        # Assistant prefill: appending an assistant turn that starts with "{"
        # forces the model to continue from that point; it cannot output
        # reasoning prose before it.  The prefill brace is re-attached below.
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

            # Extract text from the first text content block
            text = ""
            for block in resp.content:
                if hasattr(block, "text"):
                    text = block.text
                    break

            # Re-attach the opening brace that the API strips from the
            # completion (it echoes the prefill separately but does not
            # include it in resp.content[].text).
            text = "{" + text

            # Strip markdown fences the model sometimes adds
            text = re.sub(r"^```(?:json)?\s*", "", text.strip())
            text = re.sub(r"\s*```$", "", text).strip()

            # resp.model is the actual serving version (FIX-2)
            ver = resp.model or self.model_id
            return text, ver

        except Exception as e:
            print(f"    [ModelAClient] API call failed: {type(e).__name__}: {e}")
            raise


class ModelBClient:
    """
    Model B via OpenAI-compatible endpoint.
    FIX-1: max_tokens=MAX_TOKENS_ALL=256.
    FIX-2: returns (text, actual_model_id_from_api).
    """
    def __init__(self, model_id: str, base_url: str):
        try:
            from openai import OpenAI as _OAI
        except ImportError:
            raise ImportError("pip install openai")
        key = os.environ.get("MODEL_B_API_KEY")
        if not key:
            raise EnvironmentError("MODEL_B_API_KEY not set")
        self.client   = _OAI(api_key=key, base_url=base_url)
        self.model_id = model_id

    def __call__(self, messages: list[dict[str, str]]) -> tuple[str, str]:
        resp = self.client.chat.completions.create(
            model       = self.model_id,
            messages    = messages,
            max_tokens  = MAX_TOKENS_ALL,    # FIX-1
            temperature = 0.0,
        )
        text = resp.choices[0].message.content or ""
        ver  = resp.model                         # FIX-2
        return text, ver


# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────
MODEL_CONFIGS: dict[str, dict] = {
    "model_c": {
        "display": "Model C",
        "api_id":  "[REDACTED]",
        "client":  "model_c",
        "rpm":     30,
    },
    "model_a": {
        "display": "Model A",
        "api_id":  "[REDACTED]",
        "client":  "model_a",
        "rpm":     50,
    },
    "model_b": {
        "display": "Model B",
        "api_id":  "[REDACTED]",
        "client":  "model_b",
        "rpm":     60,
    },
}


def build_client(cfg: dict, model_c_reasoning: str = "none"):
    c, a = cfg["client"], cfg["api_id"]
    if c == "model_c":
        return ModelCClient(a, reasoning_effort=model_c_reasoning)
    if c == "model_a":
        return ModelAClient(a)
    if c == "model_b":
        return ModelBClient(a, base_url="[REDACTED]")
    raise ValueError(f"Unknown client type: {c}")


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation loop (one model × one setting)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_one(
    *,
    model_key:        str,
    setting:          str,
    k_shots:          int,
    test_orbits:      np.ndarray,    # (n, K, T, W)
    test_rule_f:      np.ndarray,
    test_rule_g:      np.ndarray,
    test_taus:        np.ndarray,
    shots_used:       list[dict],
    ckpt_dir:         str,
    raw_dir:          str,
    model_c_reasoning: str = "none",
    retries:          int  = 2,
) -> dict:

    cfg     = MODEL_CONFIGS[model_key]
    display = cfg["display"]
    if model_key == "model_c" and model_c_reasoning != "none":
        display = f"Model C (reasoning={model_c_reasoning})"

    delay   = 60.0 / cfg["rpm"]
    n_total = len(test_orbits)

    reason_sfx = (
        f"_re{model_c_reasoning}"
        if model_key == "model_c" and model_c_reasoning != "none"
        else ""
    )
    ckpt_path = os.path.join(
        ckpt_dir, f"{model_key}{reason_sfx}_{setting}.json"
    )

    print(f"\n  [{display}] [{setting}]  k={k_shots}  N={n_total}")

    try:
        client = build_client(cfg, model_c_reasoning=model_c_reasoning)
    except (ImportError, EnvironmentError) as e:
        print(f"    SKIP: {e}")
        return {
            "model":     display,
            "setting":   setting,
            "k_shots":   k_shots,
            "pair_acc":  None,
            "tau_acc":   None,
            "skipped":   True,
            "error":     str(e),
        }

    results, done_set = load_checkpoint(ckpt_path)
    observed_model_versions: set[str] = set()

    t0 = time.time()

    for i, (orbit_stack, true_f, true_g, true_tau) in enumerate(
        zip(test_orbits, test_rule_f, test_rule_g, test_taus)
    ):
        if i in done_set:
            continue

        # Use orbit[0] — first of K=8 orbits (FIX-T4)
        orbit_single = orbit_stack[0]   # shape (T, W)

        # ── Build messages ────────────────────────────────────────────────
        if k_shots == 0:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": make_user_prompt(orbit_single)},
            ]
        else:
            messages = build_messages(shots_used, orbit_single)

        # ── Call API with retries (FIX-6) ─────────────────────────────────
        raw_output = ""
        api_error  = False
        model_ver  = cfg["api_id"]

        for attempt in range(retries + 1):
            try:
                raw_output, model_ver = client(messages)
                api_error = False
                observed_model_versions.add(model_ver)
                break
            except Exception as e:
                api_error = True
                if attempt < retries:
                    wait = 30 * (attempt + 1)
                    print(
                        f"    Attempt {attempt+1} failed ({e}). "
                        f"Retry in {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    print(f"    Sample {i}: API error after all retries: {e}")

        # ── Save raw response (full — FIX-5) ──────────────────────────────
        raw_fname = f"{model_key}{reason_sfx}_{setting}_{i:04d}.txt"
        with open(os.path.join(raw_dir, raw_fname), "w", encoding="utf-8") as f:
            f.write(
                f"TRUE_F: {int(true_f)}  TRUE_G: {int(true_g)}  "
                f"TRUE_TAU: {float(true_tau):.4f}\n"
                f"MODEL_VERSION: {model_ver}\n"
                f"API_ERROR: {api_error}\n"
            )
            f.write("─" * 60 + "\n")
            f.write(raw_output)

        # ── Parse ─────────────────────────────────────────────────────────
        pred_f, pred_g, pred_tau = parse_response(raw_output)

        pair_ok = (
            rules_match_uo(pred_f, pred_g, int(true_f), int(true_g))
            if pred_f is not None else False
        )

        if pred_tau is not None:
            tau_err = tau_sym_err(pred_tau, float(true_tau))
            tau_ok  = tau_err <= TAU_TOL
        else:
            tau_err = None
            tau_ok  = False

        results.append({
            "sample_idx":    i,
            "true_f":        int(true_f),
            "true_g":        int(true_g),
            "pred_f":        pred_f,
            "pred_g":        pred_g,
            "true_tau":      float(true_tau),
            "pred_tau":      pred_tau,
            "pair_correct":  pair_ok,
            "tau_ok":        tau_ok,
            "tau_mae":       float(tau_err) if tau_err is not None else None,
            "api_error":     api_error,           # FIX-6
            "model_version": model_ver,           # FIX-2
            "raw_output":    raw_output,          # FIX-5: full text
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
            )

        # Progress every 20 samples
        if (i + 1) % 20 == 0 or (i + 1) == n_total:
            nd       = len(results)
            pa       = sum(r["pair_correct"] for r in results) / nd * 100
            ta       = sum(r["tau_ok"]       for r in results) / nd * 100
            pf       = sum(1 for r in results if r["pred_f"] is None)
            ae       = sum(1 for r in results if r.get("api_error", False))
            eta      = (time.time() - t0) / (i + 1) * (n_total - i - 1)
            print(
                f"    [{i+1:3d}/{n_total}]  pair(UO)={pa:5.1f}%  "
                f"tau(±0.05)={ta:5.1f}%  parse_fail={pf}  "
                f"api_err={ae}  ETA={eta:.0f}s"
            )

        time.sleep(delay)

    # ── Final statistics ───────────────────────────────────────────────────
    nd      = len(results)
    pc_vals = [r["pair_correct"] for r in results]
    tk_vals = [r["tau_ok"]       for r in results]
    mae_vals = [r["tau_mae"] for r in results if r["tau_mae"] is not None]

    pair_acc  = sum(pc_vals) / nd * 100 if nd else 0.0
    tau_acc   = sum(tk_vals) / nd * 100 if nd else 0.0
    tau_mae   = float(np.mean(mae_vals)) if mae_vals else 0.0
    pf        = sum(1 for r in results if r["pred_f"] is None)
    ae        = sum(1 for r in results if r.get("api_error", False))

    pair_ci   = bootstrap_ci(pc_vals)
    tau_ci    = bootstrap_ci(tk_vals)

    ver_str = (
        sorted(observed_model_versions)[-1]
        if observed_model_versions else cfg["api_id"]
    )

    print(
        f"  ✓ {display} [{setting}]  "
        f"pair={pair_acc:.2f}% [{pair_ci['lower']*100:.2f},{pair_ci['upper']*100:.2f}]  "
        f"tau={tau_acc:.2f}% [{tau_ci['lower']*100:.2f},{tau_ci['upper']*100:.2f}]  "
        f"tau_mae={tau_mae:.4f}  parse_fail={pf}  api_err={ae}  ver={ver_str}"
    )

    return {
        "model":         display,
        "model_version": ver_str,
        "setting":       setting,
        "k_shots":       k_shots,
        "n":             nd,
        "pair_acc":      round(pair_acc, 4),
        "tau_acc":       round(tau_acc,  4),
        "tau_mae":       round(tau_mae,  6),
        "pair_ci95": {
            "mean":  round(pair_ci["mean"]  * 100, 4),
            "lower": round(pair_ci["lower"] * 100, 4),
            "upper": round(pair_ci["upper"] * 100, 4),
        },
        "tau_ci95": {
            "mean":  round(tau_ci["mean"]  * 100, 4),
            "lower": round(tau_ci["lower"] * 100, 4),
            "upper": round(tau_ci["upper"] * 100, 4),
        },
        "parse_rate":  round((nd - pf) / nd * 100, 4) if nd else 0.0,
        "parse_fail":  pf,
        "api_errors":  ae,
        "skipped":     False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Open-source reference numbers — paper Table 29 (tempECA, N=200)
# ⚑ FIX-8: sha256 fingerprints — populate before camera-ready submission.
# ─────────────────────────────────────────────────────────────────────────────
PAPER_RESULTS: list[dict] = [
    # zero-shot
    {"model": "OS-Model-1",  "setting": "zero_shot", "pair_acc": 0.00, "tau_acc": 0.00,
     "sha256": None, "source_ckpt": "results/zero_shot/OS-Model-1_zero_shot.json"},
    {"model": "OS-Model-2",  "setting": "zero_shot", "pair_acc": 0.00, "tau_acc": 0.00,
     "sha256": None, "source_ckpt": "results/zero_shot/OS-Model-2_zero_shot.json"},
    {"model": "OS-Model-3",  "setting": "zero_shot", "pair_acc": 0.00, "tau_acc": 0.00,
     "sha256": None, "source_ckpt": "results/zero_shot/OS-Model-3_zero_shot.json"},
    {"model": "OS-Model-4",  "setting": "zero_shot", "pair_acc": 0.00, "tau_acc": 0.00,
     "sha256": None, "source_ckpt": "results/zero_shot/OS-Model-4_zero_shot.json"},
    {"model": "OS-Model-5",  "setting": "zero_shot", "pair_acc": 0.00, "tau_acc": 0.00,
     "sha256": None, "source_ckpt": "results/zero_shot/OS-Model-5_zero_shot.json"},
    {"model": "OS-Model-6",  "setting": "zero_shot", "pair_acc": 0.00, "tau_acc": 0.00,
     "sha256": None, "source_ckpt": "results/zero_shot/OS-Model-6_zero_shot.json"},
    # 5-shot
    {"model": "OS-Model-1",  "setting": "5_shot", "pair_acc": 0.00, "tau_acc": 0.00,
     "sha256": None, "source_ckpt": "results/few_shot/OS-Model-1_5shot.json"},
    {"model": "OS-Model-2",  "setting": "5_shot", "pair_acc": 0.00, "tau_acc": 0.00,
     "sha256": None, "source_ckpt": "results/few_shot/OS-Model-2_5shot.json"},
    {"model": "OS-Model-3",  "setting": "5_shot", "pair_acc": 0.00, "tau_acc": 0.00,
     "sha256": None, "source_ckpt": "results/few_shot/OS-Model-3_5shot.json"},
    {"model": "OS-Model-4",  "setting": "5_shot", "pair_acc": 0.00, "tau_acc": 0.00,
     "sha256": None, "source_ckpt": "results/few_shot/OS-Model-4_5shot.json"},
    {"model": "OS-Model-5",  "setting": "5_shot", "pair_acc": 0.00, "tau_acc": 0.00,
     "sha256": None, "source_ckpt": "results/few_shot/OS-Model-5_5shot.json"},
    {"model": "OS-Model-6",  "setting": "5_shot", "pair_acc": 0.00, "tau_acc": 0.00,
     "sha256": None, "source_ckpt": "results/few_shot/OS-Model-6_5shot.json"},
]

TEMPM_REF: dict = {
    "model":    "tempM — signal-matched transformer",
    "setting":  "trained",
    "pair_acc": 89.05,
    "tau_acc":  82.30,
    "tau_mae":  0.0311,
}


def verify_paper_results(paper_results: list[dict]) -> None:
    """Re-read checkpoint files and cross-check hardcoded values. FIX-8."""
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
            fp = round(data.get("pair_accuracy_UO",   data.get("pair_accuracy_UO_pct", 0.0)), 2)
            ft = round(data.get("tau_accuracy_pm005", data.get("tau_accuracy_pct",     0.0)), 2)
            if abs(fp - r["pair_acc"]) > 0.01:
                print(
                    f"  ⚠ pair_acc mismatch for {r['model']} [{r['setting']}]: "
                    f"hardcoded={r['pair_acc']} file={fp}"
                )
                any_warned = True
            if abs(ft - r["tau_acc"]) > 0.01:
                print(
                    f"  ⚠ tau_acc mismatch for {r['model']} [{r['setting']}]: "
                    f"hardcoded={r['tau_acc']} file={ft}"
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
        description="Frontier LLM evaluation on tempECA — anonymous submission"
    )
    parser.add_argument(
        "--data_dir", default="TSCA_Data",
        help="Root data dir containing test/ and train_pairs.npy",
    )
    parser.add_argument(
        "--output_dir", default="frontier_results",
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
    )
    parser.add_argument(
        "--settings", nargs="+",
        default=["zero_shot", "few_shot"],
        choices=["zero_shot", "few_shot"],
    )
    parser.add_argument(
        "--k_shots", nargs="+", type=int,
        default=DEFAULT_K_LIST,
        help="k values for few-shot (default: 1 3 5; nested; max=5)",
    )
    parser.add_argument(
        "--model_c_reasoning",
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
        parser.error(f"Invalid --k_shots values: {bad_k}. Must be in [1, {MAX_K_SHOTS}].")

    # ── Directory layout ──────────────────────────────────────────────────
    ckpt_dir   = os.path.join(args.output_dir, "checkpoints")
    raw_dir    = os.path.join(args.output_dir, "raw")
    index_file = os.path.join(args.output_dir, "fixed_test_indices.npy")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(raw_dir,  exist_ok=True)

    if not args.skip_verify:
        verify_paper_results(PAPER_RESULTS)

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"\n[data] Loading from {args.data_dir}/")
    test_orbits, test_rule_f, test_rule_g, test_taus, all_shots = load_data(
        args.data_dir, args.n_samples, index_file
    )
    print(
        f"[data] {len(test_orbits)} test samples  |  "
        f"{len(all_shots)} shot pool (MAX_K_SHOTS={MAX_K_SHOTS}, real orbits — FIX-T1)"
    )
    print(
        f"[cfg]  max_tokens={MAX_TOKENS_ALL} (uniform — FIX-1)  |  "
        f"bootstrap B={BOOTSTRAP_REPS} (FIX-3)  |  "
        f"SEED={SEED}, shot_seed={SEED+99}"
    )
    if "few_shot" in args.settings:
        k_str = ", ".join(str(k) for k in sorted(args.k_shots))
        print(
            f"[few-shot] k={k_str}  |  nested shots (k=1⊂3⊂5)  |  "
            f"real simulated tempECA orbits (FIX-T1)  |  "
            f"tau avoids τ=0.5 boundary"
        )
    if "model_c" in args.models:
        print(f"[Model C] reasoning_effort={args.model_c_reasoning}")

    # ── Build job list ────────────────────────────────────────────────────
    jobs: list[tuple[str, str, int, list]] = []
    for model_key in args.models:
        if "zero_shot" in args.settings:
            jobs.append((model_key, "zero_shot", 0, []))
        if "few_shot" in args.settings:
            for k in sorted(args.k_shots):
                jobs.append((model_key, f"{k}_shot", k, all_shots[:k]))

    print(f"\n[run] {len(jobs)} conditions queued:")
    for model_key, setting, k, _ in jobs:
        print(f"      {MODEL_CONFIGS[model_key]['display']:<28} {setting}")

    # ── Run ───────────────────────────────────────────────────────────────
    all_metrics: list[dict] = []
    t_total = time.time()

    for model_key, setting, k, shots_k in jobs:
        m = evaluate_one(
            model_key          = model_key,
            setting            = setting,
            k_shots            = k,
            test_orbits        = test_orbits,
            test_rule_f        = test_rule_f,
            test_rule_g        = test_rule_g,
            test_taus          = test_taus,
            shots_used         = shots_k,
            ckpt_dir           = ckpt_dir,
            raw_dir            = raw_dir,
            model_c_reasoning  = args.model_c_reasoning,
        )
        all_metrics.append(m)

    elapsed = (time.time() - t_total) / 60
    print(f"\nTotal wall time: {elapsed:.1f} min")

    # ── Console table ─────────────────────────────────────────────────────
    COL = 96
    print("\n" + "=" * COL)
    print("  tempECA IDENTIFICATION — FRONTIER LLM EVALUATION")
    print(
        f"  Rule pair: unordered-pair (UO) exact match  |  "
        f"Tau: symmetric ±0.05 tolerance  |  N={args.n_samples}"
    )
    print(
        f"  Prompts: verbatim Appendix F  |  Orbit: orbit[0], first 50 rows  "
        f"|  max_tokens={MAX_TOKENS_ALL} (all models)"
    )
    print(
        f"  CIs: 95% bootstrap percentile, B={BOOTSTRAP_REPS}  "
        f"|  Nested shots k=1⊂3⊂5  |  Real simulated orbits (FIX-T1)"
    )
    print("=" * COL)
    print(
        f"  {'Model':<34} {'Setting':<12} {'Pair(UO)%':>10}  "
        f"{'[95% CI]':>16}  {'Tau%':>7}  {'[95% CI]':>16}"
    )
    print("-" * COL)

    print("  — Open-source LLMs (paper Table 29, N=200) —")
    for r in PAPER_RESULTS:
        print(
            f"  {r['model']:<34} {r['setting']:<12} "
            f"{r['pair_acc']:>10.2f}  {'':>16}  "
            f"{r['tau_acc']:>7.2f}"
        )

    print("\n  — Frontier LLMs (this evaluation) —")
    for m in all_metrics:
        if m.get("skipped"):
            print(
                f"  {m['model']:<34} {m['setting']:<12}  "
                f"SKIPPED ({m.get('error', '')})"
            )
        else:
            pc = m["pair_ci95"]
            tc = m["tau_ci95"]
            p_ci = f"[{pc['lower']:.2f},{pc['upper']:.2f}]"
            t_ci = f"[{tc['lower']:.2f},{tc['upper']:.2f}]"
            print(
                f"  {m['model']:<34} {m['setting']:<12} "
                f"{m['pair_acc']:>10.2f}  {p_ci:>16}  "
                f"{m['tau_acc']:>7.2f}  {t_ci:>16}  "
                f"tau_mae={m['tau_mae']:.4f}  ver={m.get('model_version','?')}"
            )

    r = TEMPM_REF
    print(
        f"\n  {r['model']:<34} {r['setting']:<12} "
        f"{r['pair_acc']:>10.2f}  {'':>16}  "
        f"{r['tau_acc']:>7.2f}  tau_mae={r['tau_mae']:.4f}  <- ours"
    )
    print("=" * COL)

    # ── Save results_detail.json ──────────────────────────────────────────
    detail_path = os.path.join(args.output_dir, "results_detail.json")
    with open(detail_path, "w") as f:
        json.dump(
            {
                "frontier_metrics":  all_metrics,
                "paper_reference":   PAPER_RESULTS,
                "tempM_reference":   TEMPM_REF,
                "n_samples":         args.n_samples,
                "k_shots_evaluated": sorted(args.k_shots),

                "methodology": {
                    "token_budget": (
                        f"FIX-1: max_tokens={MAX_TOKENS_ALL} applied uniformly "
                        "to all models."
                    ),
                    "model_versioning": (
                        "FIX-2: Actual model version returned by each API call "
                        "is recorded in every per-sample record."
                    ),
                    "confidence_intervals": (
                        f"FIX-3: 95% bootstrap CIs (percentile, B={BOOTSTRAP_REPS}, "
                        f"seed={BOOTSTRAP_SEED}) for pair_acc and tau_acc."
                    ),
                    "tau_evaluation": (
                        "FIX-4/FIX-T3: pred_tau is snapped to the nearest "
                        f"TAU_SNAP_STEP={TAU_SNAP_STEP} grid point. "
                        "Symmetric error min(|p-t|, |p-(1-t)|) ≤ TAU_TOL=0.05."
                    ),
                    "raw_output_storage": (
                        "FIX-5: Full model output stored in per-sample records "
                        "and in raw/ audit files. No truncation."
                    ),
                    "api_error_tracking": (
                        "FIX-6: 'api_error' boolean per sample distinguishes "
                        "HTTP/SDK failures from model parse failures."
                    ),
                    "reference_verification": (
                        "FIX-8: PAPER_RESULTS entries carry source_ckpt paths and "
                        "sha256 fingerprints.  Run without --skip_verify to check."
                    ),
                    "few_shot_orbits": (
                        "FIX-T1: Few-shot examples use REAL simulated tempECA orbits "
                        "generated on-the-fly from train_pairs using the ECA simulator. "
                        "The open-source few_shot_temp.py used zero-filled placeholder "
                        "orbits, which give the model no orbital signal."
                    ),
                    "shot_design": (
                        "FIX-T2: Nested — all k values drawn from the same "
                        "MAX_K_SHOTS=5 pool (seed=SEED+99): "
                        "k=1→shots[:1], k=3→shots[:3], k=5→shots[:5]. "
                        "tau values in shots avoid the τ=0.5 identifiability boundary."
                    ),
                    "orbit_shown": (
                        "FIX-T4: orbit[0] (first of K=8 orbits) shown to the LLM, "
                        "consistent with open-source zero_shot_temp.py. "
                        "First 50 of T=200 rows are rendered."
                    ),
                    "uo_symmetry": (
                        "Unordered-pair accuracy: prediction counts as correct "
                        "if (pred_f==true_f and pred_g==true_g) OR "
                        "(pred_f==true_g and pred_g==true_f)."
                    ),
                    "determinism": (
                        "temperature=0.0 for all models."
                    ),
                },

                "seed":               SEED,
                "shot_seed":          SEED + 99,
                "bootstrap_seed":     BOOTSTRAP_SEED,
                "bootstrap_reps":     BOOTSTRAP_REPS,
                "max_tokens":         MAX_TOKENS_ALL,
                "model_c_reasoning":  args.model_c_reasoning,
                "paper_table":        "Table 29 (tempECA, N=200)",
                "run_timestamp":      datetime.now(tz=timezone.utc).isoformat(),
            },
            f,
            indent=2,
        )

    # ── Save results_summary.csv ──────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, "results_summary.csv")
    with open(csv_path, "w") as f:
        f.write(
            "Model,Model Version,Setting,k_shots,N,"
            "Pair(UO) Exact Match (%),Pair CI95 Lower,Pair CI95 Upper,"
            "Tau Acc (%),Tau CI95 Lower,Tau CI95 Upper,"
            "Tau MAE (sym),Parse Failures,API Errors\n"
        )
        for r in PAPER_RESULTS:
            k_val = "0" if "zero" in r["setting"] else r["setting"].split("_")[0]
            f.write(
                f"{r['model']},—,{r['setting']},{k_val},{args.n_samples},"
                f"{r['pair_acc']},—,—,{r['tau_acc']},—,—,—,—,—\n"
            )
        for m in all_metrics:
            if not m.get("skipped"):
                pc = m["pair_ci95"]
                tc = m["tau_ci95"]
                f.write(
                    f"{m['model']},{m.get('model_version','?')},"
                    f"{m['setting']},{m['k_shots']},{m['n']},"
                    f"{m['pair_acc']},{pc['lower']},{pc['upper']},"
                    f"{m['tau_acc']},{tc['lower']},{tc['upper']},"
                    f"{m['tau_mae']},{m['parse_fail']},{m['api_errors']}\n"
                )
        r = TEMPM_REF
        f.write(
            f"{r['model']},—,{r['setting']},—,—,"
            f"{r['pair_acc']},—,—,{r['tau_acc']},—,—,"
            f"{r['tau_mae']},—,—\n"
        )

    print(f"\nSaved: {detail_path}")
    print(f"Saved: {csv_path}")
    print(f"Checkpoints: {ckpt_dir}/")
    print(f"Raw responses: {raw_dir}/")


if __name__ == "__main__":
    main()