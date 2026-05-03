"""
frontier_eval_skewECA.py  
=================================================================
Zero-shot and few-shot (k=1,3,5) evaluation of frontier LLMs on the skewECA
identification task (s-skewed noise).

Key configuration:
  - Noise parameter is s (integer 1–20, contiguous block size)
  - Tolerance is ±1 cell (S_TOL=1)
  - s snapped to nearest element of S_VALUES = {1, …, 20} before comparison
  - T=200, W=20, orbit tokens = (T-1)×W = 3,980 per orbit
  - Dataset: ECA_Data_Skew/phase2/{train,test}/  with s_values.npy
  - JSON response format: {"rule": <integer>, "s": <integer>}
  - Paper Table 30 reference numbers embedded (skewECA, N=500)
  - skewM reference: 94.55% rule, 96.97% s±1

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILE STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ECA_s_skewed/
│
├── ECA_Data_Skew/                      ← dataset (read-only)
│   └── phase2/
│       ├── train/  {orbits,rule_ids,s_values}.npy
│       └── test/   {orbits,rule_ids,s_values}.npy
│
├── results/
│   ├── fixed_test_indices.npy          ← SHARED — used by ALL scripts
│   ├── zero_shot/<ModelName>_zero_shot.json
│   └── few_shot/<ModelName>_<k>shot.json
│
├── frontier_results_skew/              ← THIS script writes here
│   ├── fixed_test_indices.npy
│   ├── results_summary.csv
│   ├── results_detail.json
│   ├── checkpoints/
│   │   └── <model>_<setting>.json      ← one per (model × setting)
│   └── raw/
│       └── <model>_<setting>_<idx:04d>.txt
│
└── frontier_eval_skewECA.py            ← THIS FILE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROMPT FIDELITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALL prompts are taken VERBATIM from Appendix F (skewECA section).
Do not modify SYSTEM_PROMPT, _orbit_to_str, make_user_prompt, build_messages,
or parse_response without updating the paper appendix to match.

  SYSTEM_PROMPT   : copied character-for-character from Appendix F §skewECA
  orbit encoding  : raw "0"/"1" strings, no spaces within a row
  user prompt     : "Space-time orbit (200 rows x 20 cells):\n\n{orbit}\n\n..."
  few-shot format : multi-turn chat [system][user][assistant]…[user]
  response format : {"rule": <integer>, "s": <integer>}
  parse logic     : Stage-1 JSON → Stage-2 regex fallback
  test indices    : shared fixed_test_indices.npy (SEED=42)
  shot sampling   : np.random.default_rng(SEED+99), draw MAX_K_SHOTS=5 once;
                    nested slicing: k=1→[:1], k=3→[:3], k=5→[:5]
  s snap          : nearest element of S_VALUES (discrete exact match)
  s tolerance     : ±S_TOL=1 used only as guard after snapping

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Setup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  pip install openai anthropic numpy

  export OPENAI_API_KEY=sk-...
  export ANTHROPIC_API_KEY=sk-ant-...

Usage
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # Full run — all models, zero-shot + k=1,3,5:
  python frontier_eval_skewECA.py --data_dir ECA_Data_Skew --n_samples 200

  # Model A only, zero-shot:
  python frontier_eval_skewECA.py --models model_a --settings zero_shot

  # All shot settings:
  python frontier_eval_skewECA.py --models model_a model_b

  # Resume interrupted run (already-done samples are auto-skipped):
  python frontier_eval_skewECA.py --models model_a --n_samples 200

  # Model B with reasoning (results go to separate checkpoint files):
  python frontier_eval_skewECA.py --models model_b --model_b_reasoning low
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
# Constants — identical to paper Appendix F (skewECA section)
# ─────────────────────────────────────────────────────────────────────────────
SEED           : int       = 42
S_TOL          : int       = 1                          # ±1 cell tolerance (FIX-4)
S_VALUES       : list[int] = list(range(1, 21))         # {1, 2, …, 20}
W              : int       = 20                         # grid width
T              : int       = 200                        # time steps
MAX_K_SHOTS    : int       = 5
DEFAULT_K_LIST : list[int] = [1, 3, 5]

# ⚑ FIX-1: unified token budget — all models get the same ceiling.
MAX_TOKENS_ALL : int       = 256

# Bootstrap parameters (FIX-3)
BOOTSTRAP_REPS : int       = 2000
BOOTSTRAP_SEED : int       = SEED + 7

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — verbatim from Appendix F §skewECA zero-shot
# DO NOT MODIFY — any change breaks comparability with paper Table 30
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT: str = (
    "You are an expert in cellular automata. "
    "You are given a space-time orbit of an Elementary Cellular Automaton (ECA) "
    "perturbed by s-skewed noise.\n\n"
    "In s-skewed noise, at each timestep exactly s contiguous cells are chosen "
    "at random and updated according to the ECA rule. All other cells retain "
    "their current state. The parameter s ranges from 1 to 20 (the full grid "
    "width), where s=20 means all cells update (fully synchronous).\n\n"
    "The orbit is a grid of 0s and 1s. Each row is one timestep. "
    "Grid width is 20 cells.\n\n"
    "Identify:\n"
    "1. The ECA rule number (integer 0\u2013255)\n"
    "2. The value of s (integer from 1 to 20)\n\n"
    "Output ONLY a single JSON object. No reasoning, no explanation, no markdown, "
    "no extra text before or after. Your entire response must be exactly:\n"
    '{"rule": <integer>, "s": <integer>}'
)

# ─────────────────────────────────────────────────────────────────────────────
# Orbit encoding — verbatim from Appendix F _orbit_to_str
# ─────────────────────────────────────────────────────────────────────────────
def _orbit_to_str(orbit: np.ndarray) -> str:
    return "\n".join("".join(str(int(c)) for c in row) for row in orbit)


# ─────────────────────────────────────────────────────────────────────────────
# Prompts — verbatim from Appendix F §skewECA user prompt
# ─────────────────────────────────────────────────────────────────────────────
def make_user_prompt(orbit: np.ndarray) -> str:
    return (
        f"Space-time orbit ({T} rows x {W} cells):\n\n"
        f"{_orbit_to_str(orbit)}\n\n"
        "What is the ECA rule number and s value?"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Few-shot message builder — verbatim from Appendix F §F.2 (skewECA)
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
            "content": json.dumps(
                {"rule": int(s["rule"]), "s": int(s["s"])}
            ),
        })
    messages.append({
        "role": "user",
        "content": make_user_prompt(orbit),
    })
    return messages


# ─────────────────────────────────────────────────────────────────────────────
# Response parser — verbatim from Appendix F §skewECA parse logic
# ─────────────────────────────────────────────────────────────────────────────
def parse_response(response: str) -> tuple[int | None, int | None]:
    # Strip markdown fences that some models add (e.g. ```json ... ```)
    response = re.sub(r"^```(?:json)?\s*", "", response.strip())
    response = re.sub(r"\s*```$", "", response).strip()

    try:
        start = response.find("{")
        end   = response.rfind("}") + 1
        if start != -1 and end > start:
            payload = json.loads(response[start:end])
            rule    = int(payload.get("rule", -1))
            s_raw   = int(round(float(payload.get("s", -1))))
            if 0 <= rule <= 255 and 1 <= s_raw <= 20:
                s_val = min(S_VALUES, key=lambda v: abs(v - s_raw))
                return rule, s_val
    except Exception:
        pass

    rule_m = re.search(r'"rule"\s*:\s*(\d+)', response)
    s_m    = re.search(r'"s"\s*:\s*(\d+)',    response)
    if rule_m and s_m:
        rule  = int(rule_m.group(1))
        s_raw = int(s_m.group(1))
        if 0 <= rule <= 255 and 1 <= s_raw <= 20:
            s_val = min(S_VALUES, key=lambda v: abs(v - s_raw))
            return rule, s_val

    return None, None


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
# SHA-256 helper for checkpoint verification (FIX-8)
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
    canonical = Path(data_dir).parent / "results" / "fixed_test_indices.npy"
    for candidate in [canonical, Path(index_file)]:
        if candidate.exists():
            idx = np.load(str(candidate))
            if len(idx) >= n_samples:
                print(f"    [index] Loaded {len(idx)} indices from {candidate}")
                return idx[:n_samples]

    Path(index_file).parent.mkdir(parents=True, exist_ok=True)
    rng    = np.random.default_rng(SEED)
    orbits = np.load(os.path.join(data_dir, "phase2", "test", "orbits.npy"))
    n      = min(n_samples, len(orbits))
    idx    = rng.choice(len(orbits), size=n, replace=False)
    np.save(index_file, idx)
    print(f"    [index] Generated {n} fresh indices → {index_file}")
    return idx


def load_data(
    data_dir:   str,
    n_samples:  int,
    index_file: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    idx         = get_fixed_test_indices(data_dir, n_samples, index_file)
    test_base   = os.path.join(data_dir, "phase2", "test")
    test_orbits = np.load(os.path.join(test_base, "orbits.npy"))
    test_rules  = np.load(os.path.join(test_base, "rule_ids.npy"))
    test_s_vals = np.load(os.path.join(test_base, "s_values.npy"))

    rng          = np.random.default_rng(SEED + 99)
    train_base   = os.path.join(data_dir, "phase2", "train")
    train_orbits = np.load(os.path.join(train_base, "orbits.npy"))
    train_rules  = np.load(os.path.join(train_base, "rule_ids.npy"))
    train_s_vals = np.load(os.path.join(train_base, "s_values.npy"))
    shot_idx     = rng.choice(len(train_orbits), size=MAX_K_SHOTS, replace=False)
    all_shots: list[dict[str, Any]] = [
        {
            "orbit": train_orbits[i].tolist(),
            "rule":  int(train_rules[i]),
            "s":     int(train_s_vals[i]),
        }
        for i in shot_idx
    ]
    return (
        test_orbits[idx],
        test_rules[idx].astype(int),
        test_s_vals[idx].astype(int),
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
    model_ver:   str,
    setting:     str,
    k_shots:     int,
    n_total:     int,
    shots_used:  list,
    results:     list,
) -> None:
    n         = len(results)
    rc_vals   = [r["rule_correct"] for r in results]
    sk_vals   = [r["s_ok"]         for r in results]
    rule_acc  = sum(rc_vals) / n * 100 if n else 0.0
    s_acc     = sum(sk_vals) / n * 100 if n else 0.0
    rule_ci   = bootstrap_ci(rc_vals)
    s_ci      = bootstrap_ci(sk_vals)
    api_errors = sum(1 for r in results if r.get("api_error", False))

    payload = {
        "model":              model_name,
        "model_version":      model_ver,
        "setting":            setting,
        "k_shots":            k_shots,
        "n_evaluated":        n,
        "n_total":            n_total,
        "rule_accuracy_pct":  round(rule_acc, 4),
        "s_accuracy_pct":     round(s_acc,    4),
        "rule_ci95": {
            "mean":  round(rule_ci["mean"]  * 100, 4),
            "lower": round(rule_ci["lower"] * 100, 4),
            "upper": round(rule_ci["upper"] * 100, 4),
        },
        "s_ci95": {
            "mean":  round(s_ci["mean"]  * 100, 4),
            "lower": round(s_ci["lower"] * 100, 4),
            "upper": round(s_ci["upper"] * 100, 4),
        },
        "parse_failures":     sum(1 for r in results if r["pred_rule"] is None),
        "api_errors":         api_errors,
        "few_shot_examples":  [
            {"rule": s["rule"], "s": s["s"]} for s in shots_used
        ],
        "timestamp":          datetime.now(tz=timezone.utc).isoformat(),
        "samples":            results,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# API clients
# ─────────────────────────────────────────────────────────────────────────────

class ModelAClient:
    """
    LLM-A via OpenAI Chat Completions API.
    Install: pip install openai
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

    def __call__(self, messages: list[dict]) -> tuple[str, str]:
        kw: dict[str, Any] = dict(
            model                 = self.model_id,
            messages              = messages,
            reasoning_effort      = self.reasoning_effort,
            max_completion_tokens = MAX_TOKENS_ALL,
        )
        if self.reasoning_effort == "none":
            kw["temperature"] = 0.0
        resp = self.client.chat.completions.create(**kw)
        return resp.choices[0].message.content or "", resp.model


class ModelBClient:
    """
    LLM-B via Anthropic SDK.
    Install: pip install anthropic

    System prompt extracted and passed as top-level `system` kwarg
    (Anthropic API does NOT accept role="system" inside the messages list).
    Markdown fences stripped. resp.model echoed as model_version (FIX-2).
    Assistant prefill trick: appends {"role":"assistant","content":"{"}
    to force JSON-only output; re-attaches "{" to completion.
    """

    DEFAULT_MODEL = "[REDACTED]"

    def __init__(self, model_id: str = DEFAULT_MODEL):
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set")
        self.client   = _anthropic.Anthropic(api_key=key)
        self.model_id = model_id

    def __call__(self, messages: list[dict]) -> tuple[str, str]:
        system_text   = ""
        chat_messages: list[dict] = []
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
            else:
                chat_messages.append({"role": m["role"], "content": m["content"]})

        if not chat_messages:
            raise ValueError(
                "[ModelBClient] messages must contain at least one non-system message"
            )

        # Assistant prefill — forces JSON-only output.
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

            # Re-attach the opening brace stripped by the Anthropic API
            text = "{" + text

            # Strip markdown fences
            text = re.sub(r"^```(?:json)?\s*", "", text.strip())
            text = re.sub(r"\s*```$", "", text).strip()

            ver = resp.model or self.model_id
            return text, ver

        except Exception as e:
            print(f"    [ModelBClient] API call failed: {type(e).__name__}: {e}")
            raise


# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────
MODEL_CONFIGS: dict[str, dict] = {
    "model_a": {
        "display": "LLM-A",
        "api_id":  "[REDACTED]",
        "client":  "model_a",
        "rpm":     30,
    },
    "model_b": {
        "display": "LLM-B",
        "api_id":  "[REDACTED]",
        "client":  "model_b",
        "rpm":     50,
    },
}


def build_client(cfg: dict, model_b_reasoning: str = "none"):
    c, a = cfg["client"], cfg["api_id"]
    if c == "model_a": return ModelAClient(a, reasoning_effort=model_b_reasoning)
    if c == "model_b": return ModelBClient(a)
    raise ValueError(f"Unknown client type: {c}")


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation loop  (one model × one setting)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_one(
    *,
    model_key:        str,
    setting:          str,
    k_shots:          int,
    test_orbits:      np.ndarray,
    test_rules:       np.ndarray,
    test_s_vals:      np.ndarray,
    shots_used:       list[dict],
    ckpt_dir:         str,
    raw_dir:          str,
    model_b_reasoning: str = "none",
    retries:          int = 2,
) -> dict:

    cfg     = MODEL_CONFIGS[model_key]
    display = cfg["display"]
    if model_key == "model_a" and model_b_reasoning != "none":
        display = f"LLM-A (reasoning={model_b_reasoning})"

    delay   = 60.0 / cfg["rpm"]
    n_total = len(test_orbits)

    reason_sfx = (
        f"_re{model_b_reasoning}"
        if model_key == "model_a" and model_b_reasoning != "none"
        else ""
    )
    ckpt_path = os.path.join(
        ckpt_dir, f"{model_key}{reason_sfx}_{setting}.json"
    )

    print(f"\n  [{display}] [{setting}]  k={k_shots}  N={n_total}")

    try:
        client = build_client(cfg, model_b_reasoning=model_b_reasoning)
    except (ImportError, EnvironmentError) as e:
        print(f"    SKIP: {e}")
        return {
            "model":    display,
            "setting":  setting,
            "k_shots":  k_shots,
            "rule_acc": None,
            "s_acc":    None,
            "skipped":  True,
            "error":    str(e),
        }

    results, done_set = load_checkpoint(ckpt_path)
    observed_model_versions: set[str] = set()
    t0 = time.time()

    for i, (orbit, true_rule, true_s) in enumerate(
        zip(test_orbits, test_rules, test_s_vals)
    ):
        if i in done_set:
            continue

        if k_shots == 0:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": make_user_prompt(orbit)},
            ]
        else:
            messages = build_messages(shots_used, orbit)

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

        raw_fname = f"{model_key}{reason_sfx}_{setting}_{i:04d}.txt"
        with open(os.path.join(raw_dir, raw_fname), "w", encoding="utf-8") as f:
            f.write(
                f"TRUE_RULE: {int(true_rule)}  TRUE_S: {int(true_s)}\n"
                f"MODEL_VERSION: {model_ver}\n"
                f"API_ERROR: {api_error}\n"
            )
            f.write("─" * 60 + "\n")
            f.write(raw_output)

        pred_rule, pred_s = parse_response(raw_output)
        rule_correct = (
            pred_rule == int(true_rule) if pred_rule is not None else False
        )
        s_ok = (
            abs(pred_s - int(true_s)) <= S_TOL if pred_s is not None else False
        )
        s_mae = abs(pred_s - int(true_s)) if pred_s is not None else None

        results.append({
            "sample_idx":    i,
            "true_rule":     int(true_rule),
            "pred_rule":     pred_rule,
            "true_s":        int(true_s),
            "pred_s":        pred_s,
            "rule_correct":  rule_correct,
            "s_ok":          s_ok,
            "s_mae":         s_mae,
            "api_error":     api_error,
            "model_version": model_ver,
            "raw_output":    raw_output,
        })
        done_set.add(i)

        if (i + 1) % 10 == 0 or (i + 1) == n_total:
            ver_str = (
                sorted(observed_model_versions)[-1]
                if observed_model_versions else cfg["api_id"]
            )
            save_checkpoint(
                ckpt_path, display, ver_str, setting,
                k_shots, n_total, shots_used, results,
            )

        if (i + 1) % 20 == 0 or (i + 1) == n_total:
            nd       = len(results)
            rule_acc = sum(r["rule_correct"] for r in results) / nd * 100
            s_acc    = sum(r["s_ok"]         for r in results) / nd * 100
            pf       = sum(1 for r in results if r["pred_rule"] is None)
            ae       = sum(1 for r in results if r.get("api_error", False))
            eta      = (time.time() - t0) / (i + 1) * (n_total - i - 1)
            print(
                f"    [{i+1:3d}/{n_total}]  rule={rule_acc:5.1f}%  "
                f"s(±1)={s_acc:5.1f}%  parse_fail={pf}  "
                f"api_err={ae}  ETA={eta:.0f}s"
            )

        time.sleep(delay)

    nd       = len(results)
    rc_vals  = [r["rule_correct"] for r in results]
    sk_vals  = [r["s_ok"]         for r in results]
    rule_acc = sum(rc_vals) / nd * 100 if nd else 0.0
    s_acc    = sum(sk_vals) / nd * 100 if nd else 0.0
    pf       = sum(1 for r in results if r["pred_rule"] is None)
    ae       = sum(1 for r in results if r.get("api_error", False))
    rule_ci  = bootstrap_ci(rc_vals)
    s_ci     = bootstrap_ci(sk_vals)

    ver_str = (
        sorted(observed_model_versions)[-1]
        if observed_model_versions else cfg["api_id"]
    )

    print(
        f"  ✓ {display} [{setting}]  "
        f"rule={rule_acc:.2f}% [{rule_ci['lower']*100:.2f},{rule_ci['upper']*100:.2f}]  "
        f"s(±1)={s_acc:.2f}% [{s_ci['lower']*100:.2f},{s_ci['upper']*100:.2f}]  "
        f"parse_fail={pf}  api_err={ae}  ver={ver_str}"
    )

    return {
        "model":         display,
        "model_version": ver_str,
        "setting":       setting,
        "k_shots":       k_shots,
        "n":             nd,
        "rule_acc":      round(rule_acc, 4),
        "s_acc":         round(s_acc,   4),
        "rule_ci95": {
            "mean":  round(rule_ci["mean"]  * 100, 4),
            "lower": round(rule_ci["lower"] * 100, 4),
            "upper": round(rule_ci["upper"] * 100, 4),
        },
        "s_ci95": {
            "mean":  round(s_ci["mean"]  * 100, 4),
            "lower": round(s_ci["lower"] * 100, 4),
            "upper": round(s_ci["upper"] * 100, 4),
        },
        "parse_rate": round((nd - pf) / nd * 100, 4) if nd else 0.0,
        "parse_fail": pf,
        "api_errors": ae,
        "skipped":    False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Open-source reference numbers — paper Table 30 (skewECA, N=500)
# ─────────────────────────────────────────────────────────────────────────────
PAPER_RESULTS: list[dict] = [
    {"model": "Qwen2.5-7B",    "setting": "zero_shot", "rule_acc":  0.00, "s_acc": 13.00,
     "sha256": None, "source_ckpt": "results/zero_shot/Qwen2.5-7B_zero_shot.json"},
    {"model": "Qwen2.5-72B",   "setting": "zero_shot", "rule_acc":  0.80, "s_acc": 15.20,
     "sha256": None, "source_ckpt": "results/zero_shot/Qwen2.5-72B_zero_shot.json"},
    {"model": "Llama-3.1-8B",  "setting": "zero_shot", "rule_acc":  0.00, "s_acc": 15.60,
     "sha256": None, "source_ckpt": "results/zero_shot/Llama-3.1-8B_zero_shot.json"},
    {"model": "Llama-3.1-70B", "setting": "zero_shot", "rule_acc":  0.00, "s_acc": 14.80,
     "sha256": None, "source_ckpt": "results/zero_shot/Llama-3.1-70B_zero_shot.json"},
    {"model": "Mistral-7B",    "setting": "zero_shot", "rule_acc":  0.00, "s_acc": 15.80,
     "sha256": None, "source_ckpt": "results/zero_shot/Mistral-7B_zero_shot.json"},
    {"model": "Mixtral-8x7B",  "setting": "zero_shot", "rule_acc":  0.00, "s_acc":  0.20,
     "sha256": None, "source_ckpt": "results/zero_shot/Mixtral-8x7B_zero_shot.json"},
    {"model": "Qwen2.5-7B",    "setting": "5_shot",    "rule_acc":  0.80, "s_acc": 13.60,
     "sha256": None, "source_ckpt": "results/few_shot/Qwen2.5-7B_5shot.json"},
    {"model": "Qwen2.5-72B",   "setting": "5_shot",    "rule_acc":  0.60, "s_acc": 13.20,
     "sha256": None, "source_ckpt": "results/few_shot/Qwen2.5-72B_5shot.json"},
    {"model": "Llama-3.1-8B",  "setting": "5_shot",    "rule_acc":  0.00, "s_acc": 11.60,
     "sha256": None, "source_ckpt": "results/few_shot/Llama-3.1-8B_5shot.json"},
    {"model": "Llama-3.1-70B", "setting": "5_shot",    "rule_acc":  0.00, "s_acc":  6.00,
     "sha256": None, "source_ckpt": "results/few_shot/Llama-3.1-70B_5shot.json"},
    {"model": "Mistral-7B",    "setting": "5_shot",    "rule_acc":  0.00, "s_acc": 19.20,
     "sha256": None, "source_ckpt": "results/few_shot/Mistral-7B_5shot.json"},
    {"model": "Mixtral-8x7B",  "setting": "5_shot",    "rule_acc":  0.00, "s_acc":  3.80,
     "sha256": None, "source_ckpt": "results/few_shot/Mixtral-8x7B_5shot.json"},
    {"model": "Qwen2.5-7B",    "setting": "fine_tuned", "rule_acc":  7.4, "s_acc": 20.2,
     "sha256": None, "source_ckpt": "results/fine_tuned/Qwen2.5-7B_ft.json"},
    {"model": "Qwen2.5-72B",   "setting": "fine_tuned", "rule_acc": 13.2, "s_acc": 24.8,
     "sha256": None, "source_ckpt": "results/fine_tuned/Qwen2.5-72B_ft.json"},
]

SKEWM_REF: dict = {
    "model":    "skewM — spatial contiguity transformer",
    "setting":  "trained",
    "rule_acc": 94.55,
    "s_acc":    96.97,
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
                f"    expected: {r['sha256']}\n    actual:   {sha}"
            )
            any_warned = True
        try:
            with open(ckpt) as f:
                data = json.load(f)
            file_rule = round(
                data.get("rule_accuracy", data.get("rule_accuracy_pct", None)), 2
            )
            file_s = round(
                data.get("s_accuracy_pm1",
                data.get("s_accuracy_pct",
                data.get("s_accuracy", None))), 2
            )
            if file_rule is not None and abs(file_rule - r["rule_acc"]) > 0.01:
                print(
                    f"  ⚠ rule_acc mismatch for {r['model']} [{r['setting']}]: "
                    f"hardcoded={r['rule_acc']} file={file_rule}"
                )
                any_warned = True
            if file_s is not None and abs(file_s - r["s_acc"]) > 0.01:
                print(
                    f"  ⚠ s_acc mismatch for {r['model']} [{r['setting']}]: "
                    f"hardcoded={r['s_acc']} file={file_s}"
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
        description="Frontier LLM evaluation on skewECA — NeurIPS 2026 anonymous submission"
    )
    parser.add_argument("--data_dir",   default="ECA_Data_Skew")
    parser.add_argument("--output_dir", default="frontier_results_skew")
    parser.add_argument("--n_samples",  type=int, default=200)
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
        "--k_shots", nargs="+", type=int, default=DEFAULT_K_LIST,
    )
    parser.add_argument(
        "--model_b_reasoning", default="none",
        choices=["none", "low", "medium", "high"],
    )
    parser.add_argument("--skip_verify", action="store_true")
    args = parser.parse_args()

    bad_k = [k for k in args.k_shots if k > MAX_K_SHOTS or k < 1]
    if bad_k:
        parser.error(f"Invalid --k_shots values: {bad_k}. Must be in [1, {MAX_K_SHOTS}].")

    ckpt_dir   = os.path.join(args.output_dir, "checkpoints")
    raw_dir    = os.path.join(args.output_dir, "raw")
    index_file = os.path.join(args.output_dir, "fixed_test_indices.npy")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(raw_dir,  exist_ok=True)

    if not args.skip_verify:
        verify_paper_results(PAPER_RESULTS)

    print(f"\n[data] Loading from {args.data_dir}/phase2/")
    test_orbits, test_rules, test_s_vals, all_shots = load_data(
        args.data_dir, args.n_samples, index_file
    )
    print(
        f"[data] {len(test_orbits)} test samples  |  "
        f"{len(all_shots)} shot pool (MAX_K_SHOTS={MAX_K_SHOTS})"
    )
    print(
        f"[cfg]  max_tokens={MAX_TOKENS_ALL} (uniform — FIX-1)  |  "
        f"bootstrap B={BOOTSTRAP_REPS} (FIX-3)  |  "
        f"SEED={SEED}, shot_seed={SEED+99}  |  "
        f"T={T}, W={W}"
    )

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

    all_metrics: list[dict] = []
    t_total = time.time()

    for model_key, setting, k, shots_k in jobs:
        m = evaluate_one(
            model_key          = model_key,
            setting            = setting,
            k_shots            = k,
            test_orbits        = test_orbits,
            test_rules         = test_rules,
            test_s_vals        = test_s_vals,
            shots_used         = shots_k,
            ckpt_dir           = ckpt_dir,
            raw_dir            = raw_dir,
            model_b_reasoning  = args.model_b_reasoning,
        )
        all_metrics.append(m)

    elapsed = (time.time() - t_total) / 60
    print(f"\nTotal wall time: {elapsed:.1f} min")

    # ── Console table ─────────────────────────────────────────────────────
    COL = 98
    print("\n" + "=" * COL)
    print("  skewECA IDENTIFICATION — FRONTIER LLM EVALUATION")
    print(
        f"  Rule: exact match  |  s: discrete exact match (snapped, ±{S_TOL} tol)  "
        f"|  N={args.n_samples}"
    )
    print(
        f"  max_tokens={MAX_TOKENS_ALL} (all models, FIX-1)  |  "
        f"CIs: 95% bootstrap percentile B={BOOTSTRAP_REPS}"
    )
    print("=" * COL)
    print(
        f"  {'Model':<34} {'Setting':<12} {'Rule%':>7}  "
        f"{'[95% CI]':>14}  {'s±1%':>7}  {'[95% CI]':>14}"
    )
    print("-" * COL)
    print("  — Open-source LLMs (paper Table 30, N=500) —")
    for r in PAPER_RESULTS:
        print(
            f"  {r['model']:<34} {r['setting']:<12} "
            f"{r['rule_acc']:>7.2f}  {'':>14}  {r['s_acc']:>7.2f}"
        )
    print("\n  — Frontier LLMs (this evaluation) —")
    for m in all_metrics:
        if m.get("skipped"):
            print(
                f"  {m['model']:<34} {m['setting']:<12}  "
                f"SKIPPED ({m.get('error', '')})"
            )
        else:
            rc    = m["rule_ci95"]
            sc    = m["s_ci95"]
            r_ci  = f"[{rc['lower']:.2f},{rc['upper']:.2f}]"
            s_ci  = f"[{sc['lower']:.2f},{sc['upper']:.2f}]"
            print(
                f"  {m['model']:<34} {m['setting']:<12} "
                f"{m['rule_acc']:>7.2f}  {r_ci:>14}  "
                f"{m['s_acc']:>7.2f}  {s_ci:>14}  "
                f"ver={m.get('model_version','?')}"
            )
    r = SKEWM_REF
    print(
        f"\n  {r['model']:<34} {r['setting']:<12} "
        f"{r['rule_acc']:>7.2f}  {'':>14}  {r['s_acc']:>7.2f}  <- ours"
    )
    print("=" * COL)

    # ── Save results_detail.json ──────────────────────────────────────────
    detail_path = os.path.join(args.output_dir, "results_detail.json")
    with open(detail_path, "w") as f:
        json.dump(
            {
                "frontier_metrics":    all_metrics,
                "paper_reference":     PAPER_RESULTS,
                "skewM_reference":     SKEWM_REF,
                "n_samples":           args.n_samples,
                "models_evaluated":    args.models,
                "k_shots_evaluated":   sorted(args.k_shots),
                "noise_type":          "s-skewed",
                "dataset":             "skewECA",
                "methodology": {
                    "token_budget": (
                        f"FIX-1: max_tokens={MAX_TOKENS_ALL} applied uniformly. "
                        "LLM-A→max_completion_tokens; LLM-B→max_tokens."
                    ),
                    "model_versioning": (
                        "FIX-2: actual model version string from API recorded per-sample."
                    ),
                    "confidence_intervals": (
                        f"FIX-3: 95% bootstrap CIs, B={BOOTSTRAP_REPS}, seed={BOOTSTRAP_SEED}."
                    ),
                    "s_evaluation": (
                        f"FIX-4: pred_s snapped to S_VALUES={{1,...,20}} before ±{S_TOL} guard."
                    ),
                    "raw_output_storage": "FIX-5: full output stored, no truncation.",
                    "api_error_tracking": "FIX-6: api_error bool per sample.",
                    "reference_verification": (
                        "FIX-8: sha256 fingerprints in PAPER_RESULTS for artifact verification."
                    ),
                },
                "seed":                SEED,
                "shot_seed":           SEED + 99,
                "bootstrap_seed":      BOOTSTRAP_SEED,
                "bootstrap_reps":      BOOTSTRAP_REPS,
                "max_tokens":          MAX_TOKENS_ALL,
                "s_tol":               S_TOL,
                "T":                   T,
                "W":                   W,
                "paper_table":         "Table 30 (skewECA, N=500)",
                "run_timestamp":       datetime.now(tz=timezone.utc).isoformat(),
            },
            f, indent=2,
        )

    # ── Save results_summary.csv ──────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, "results_summary.csv")
    with open(csv_path, "w") as f:
        f.write(
            "Model,Model Version,Setting,k_shots,N,"
            "Rule Exact Match (%),Rule CI95 Lower,Rule CI95 Upper,"
            "s±1 Accuracy (%),s CI95 Lower,s CI95 Upper,"
            "Parse Failures,API Errors\n"
        )
        for r in PAPER_RESULTS:
            k_val = "0" if "zero" in r["setting"] else r["setting"].split("_")[0]
            f.write(
                f"{r['model']},—,{r['setting']},{k_val},{args.n_samples},"
                f"{r['rule_acc']},—,—,{r['s_acc']},—,—,—,—\n"
            )
        for m in all_metrics:
            if not m.get("skipped"):
                rc = m["rule_ci95"]
                sc = m["s_ci95"]
                f.write(
                    f"{m['model']},{m.get('model_version','?')},"
                    f"{m['setting']},{m['k_shots']},{m['n']},"
                    f"{m['rule_acc']},{rc['lower']},{rc['upper']},"
                    f"{m['s_acc']},{sc['lower']},{sc['upper']},"
                    f"{m['parse_fail']},{m['api_errors']}\n"
                )
        r = SKEWM_REF
        f.write(
            f"{r['model']},—,{r['setting']},—,—,"
            f"{r['rule_acc']},—,—,{r['s_acc']},—,—,—,—\n"
        )

    print(f"\nSaved: {detail_path}")
    print(f"Saved: {csv_path}")
    print(f"Checkpoints: {ckpt_dir}/")
    print(f"Raw responses: {raw_dir}/")


if __name__ == "__main__":
    main()
