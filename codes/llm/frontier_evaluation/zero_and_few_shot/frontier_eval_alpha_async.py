"""
frontier_eval_alphaECA.py  
================================================================
Zero-shot and few-shot (k=1,3,5) evaluation of frontier LLMs on the αECA
identification task.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILE STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


ECA_alpha_async/
│
├── ECA_Data_New/                       ← dataset (read-only)
│   └── phase2/
│       ├── train/  {orbits,rule_ids,alphas}.npy
│       └── test/   {orbits,rule_ids,alphas}.npy
│
├── results/
│   ├── fixed_test_indices.npy          ← SHARED — used by ALL scripts
│   ├── zero_shot/<ModelName>_zero_shot.json
│   └── few_shot/<ModelName>_<k>shot.json
│
├── frontier_results/                   ← THIS script writes here
│   ├── fixed_test_indices.npy
│   ├── results_summary.csv
│   ├── results_detail.json
│   ├── checkpoints/
│   │   └── <model>_<setting>.json      ← one per (model × setting)
│   └── raw/
│       └── <model>_<setting>_<idx:04d>.txt
│
└── frontier_eval_alphaECA.py           ← THIS FILE


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROMPT FIDELITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALL prompts are taken VERBATIM from Appendix F.  Do not modify SYSTEM_PROMPT,
_orbit_to_str, make_user_prompt, build_messages, or parse_response without
updating the paper appendix to match.


 SYSTEM_PROMPT   : copied character-for-character
 orbit encoding  : raw "0"/"1" strings, ASCII "x" separator
 user prompt     : "Space-time orbit (T rows x W cells):\n\n{orbit}\n\n..."
 few-shot format : multi-turn chat [system][user][assistant]…[user]
 response format : {"rule": <integer>, "alpha": <float>}
 parse logic     : Stage-1 JSON → Stage-2 regex fallback
 test indices    : shared fixed_test_indices.npy (SEED=42)
 shot sampling   : np.random.default_rng(SEED+99), draw MAX_K_SHOTS=5 once;
                   nested slicing: k=1→[:1], k=3→[:3], k=5→[:5]
 alpha snap      : nearest element of ALPHA_VALUES (discrete exact match)
 alpha tolerance : ±0.05 used only as float-noise guard after snapping


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Setup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 pip install openai anthropic numpy


 export OPENAI_API_KEY=sk-...
 export ANTHROPIC_API_KEY=sk-ant-...


Usage
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 # Full run — all models (GPT-5.1 + Claude Haiku 4.5), zero-shot + k=1,3,5:
 python frontier_eval_alphaECA.py --data_dir ECA_Data_New --n_samples 200


 # Match paper's 500-sample open-source eval:
 python frontier_eval_alphaECA.py --n_samples 500


 # Zero-shot only, GPT only:
 python frontier_eval_alphaECA.py --settings zero_shot --models gpt51


 # Claude Haiku 4.5 only, zero-shot:
 python frontier_eval_alphaECA.py --models claude_haiku --settings zero_shot


 # Claude few-shot k=5, 100 samples:
 python frontier_eval_alphaECA.py --models claude_haiku --settings few_shot --k_shots 5 --n_samples 100


 # Claude + GPT comparison, all shot settings:
 python frontier_eval_alphaECA.py --models claude_haiku gpt51


 # Resume interrupted run (already-done samples are auto-skipped):
 python frontier_eval_alphaECA.py --models gpt51 --n_samples 200


 # GPT-5.1 with reasoning (results go to separate checkpoint files):
 python frontier_eval_alphaECA.py --models gpt51 --gpt51_reasoning low
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
# Constants — identical to paper Appendix F
# ─────────────────────────────────────────────────────────────────────────────
SEED           : int         = 42
ALPHA_TOL      : float       = 0.05          # float-noise guard only (FIX-4)
ALPHA_VALUES   : list[float] = [round(a * 0.1, 1) for a in range(1, 11)]
W              : int         = 20
T              : int         = 100
MAX_K_SHOTS    : int         = 5
DEFAULT_K_LIST : list[int]   = [1, 3, 5]


# ⚑ FIX-1: unified token budget — all models get the same ceiling.
MAX_TOKENS_ALL : int         = 256


# Bootstrap parameters (FIX-3)
BOOTSTRAP_REPS : int         = 2000
BOOTSTRAP_SEED : int         = SEED + 7


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — verbatim from Appendix F
# DO NOT MODIFY — any change breaks comparability with paper Table 29
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT: str = (
   "You are an expert in cellular automata. "
   "You are given a space-time orbit of an Elementary Cellular Automaton (ECA) "
   "perturbed by alpha-asynchronous noise.\n\n"
   "In alpha-asynchronous noise each cell independently updates with probability "
   "alpha per timestep.\n"
   "Alpha is one of: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0\n\n"
   "The orbit is a grid of 0s and 1s. Each row is one timestep.\n\n"
   "Identify:\n"
   "1. The ECA rule number (integer 0\u2013255)\n"
   "2. The alpha value\n\n"
   'Respond ONLY in this JSON format with no extra text:\n'
   '{"rule": <integer>, "alpha": <float>}'
)


# ─────────────────────────────────────────────────────────────────────────────
# Orbit encoding — verbatim from Appendix F _orbit_to_str
# ─────────────────────────────────────────────────────────────────────────────
def _orbit_to_str(orbit: np.ndarray) -> str:
   return "\n".join("".join(str(int(c)) for c in row) for row in orbit)




# ─────────────────────────────────────────────────────────────────────────────
# Prompts — verbatim from Appendix F
# ASCII "x" separator matches zero_shot_eval.py and Appendix F exactly.
# ─────────────────────────────────────────────────────────────────────────────
def make_user_prompt(orbit: np.ndarray) -> str:
   return (
       f"Space-time orbit ({T} rows x {W} cells):\n\n"
       f"{_orbit_to_str(orbit)}\n\n"
       "What is the ECA rule number and alpha value?"
   )




# ─────────────────────────────────────────────────────────────────────────────
# Few-shot message builder — verbatim from Appendix F Section F.2
# Multi-turn: [system][user][assistant] … [user]
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
               {"rule": int(s["rule"]), "alpha": float(s["alpha"])}
           ),
       })
   messages.append({
       "role": "user",
       "content": make_user_prompt(orbit),
   })
   return messages




# ─────────────────────────────────────────────────────────────────────────────
# Response parser — verbatim from Appendix F parse_response
# Stage 1: JSON parse.  Stage 2: regex fallback.
# ─────────────────────────────────────────────────────────────────────────────
def parse_response(response: str) -> tuple[int | None, float | None]:
   try:
       start = response.find("{")
       end   = response.rfind("}") + 1
       if start != -1 and end > start:
           payload = json.loads(response[start:end])
           rule    = int(payload.get("rule",  -1))
           alpha   = float(payload.get("alpha", -1.0))
           if 0 <= rule <= 255:
               alpha = min(ALPHA_VALUES, key=lambda a: abs(a - alpha))
               return rule, alpha
   except Exception:
       pass


   rule_m  = re.search(r'"rule"\s*:\s*(\d+)',     response)
   alpha_m = re.search(r'"alpha"\s*:\s*([\d.]+)', response)
   if rule_m and alpha_m:
       rule  = int(rule_m.group(1))
       alpha = float(alpha_m.group(1))
       if 0 <= rule <= 255:
           alpha = min(ALPHA_VALUES, key=lambda a: abs(a - alpha))
           return rule, alpha


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
   return {
       "mean":  float(arr.mean()),
       "lower": lo,
       "upper": hi,
   }




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
   test_alphas = np.load(os.path.join(test_base, "alphas.npy"))


   rng          = np.random.default_rng(SEED + 99)
   train_base   = os.path.join(data_dir, "phase2", "train")
   train_orbits = np.load(os.path.join(train_base, "orbits.npy"))
   train_rules  = np.load(os.path.join(train_base, "rule_ids.npy"))
   train_alphas = np.load(os.path.join(train_base, "alphas.npy"))
   shot_idx     = rng.choice(len(train_orbits), size=MAX_K_SHOTS, replace=False)
   all_shots: list[dict[str, Any]] = [
       {
           "orbit": train_orbits[i].tolist(),
           "rule":  int(train_rules[i]),
           "alpha": float(round(float(train_alphas[i]), 1)),
       }
       for i in shot_idx
   ]
   return (
       test_orbits[idx],
       test_rules[idx].astype(int),
       test_alphas[idx].astype(float),
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
   ak_vals   = [r["alpha_ok"]     for r in results]
   rule_acc  = sum(rc_vals) / n * 100 if n else 0.0
   alpha_acc = sum(ak_vals) / n * 100 if n else 0.0


   rule_ci  = bootstrap_ci(rc_vals)
   alpha_ci = bootstrap_ci(ak_vals)


   api_errors = sum(1 for r in results if r.get("api_error", False))


   payload = {
       "model":              model_name,
       "model_version":      model_ver,
       "setting":            setting,
       "k_shots":            k_shots,
       "n_evaluated":        n,
       "n_total":            n_total,
       "rule_accuracy_pct":  round(rule_acc,  4),
       "alpha_accuracy_pct": round(alpha_acc, 4),
       "rule_ci95": {
           "mean":  round(rule_ci["mean"]  * 100, 4),
           "lower": round(rule_ci["lower"] * 100, 4),
           "upper": round(rule_ci["upper"] * 100, 4),
       },
       "alpha_ci95": {
           "mean":  round(alpha_ci["mean"]  * 100, 4),
           "lower": round(alpha_ci["lower"] * 100, 4),
           "upper": round(alpha_ci["upper"] * 100, 4),
       },
       "parse_failures":     sum(1 for r in results if r["pred_rule"] is None),
       "api_errors":         api_errors,
       "few_shot_examples":  [
           {"rule": s["rule"], "alpha": s["alpha"]} for s in shots_used
       ],
       "timestamp":          datetime.now(tz=timezone.utc).isoformat(),
       "samples":            results,
   }
   with open(path, "w") as f:
       json.dump(payload, f, indent=2)




# ─────────────────────────────────────────────────────────────────────────────
# API clients
# ─────────────────────────────────────────────────────────────────────────────


class GPT51Client:
   """
   GPT-5.1 via OpenAI Chat Completions.
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
           model            = self.model_id,
           messages         = messages,
           reasoning_effort = self.reasoning_effort,
           max_completion_tokens       = MAX_TOKENS_ALL,
       )
       if self.reasoning_effort == "none":
           kw["temperature"] = 0.0
       resp = self.client.chat.completions.create(**kw)
       text = resp.choices[0].message.content or ""
       ver  = resp.model
       return text, ver




# ─────────────────────────────────────────────────────────────────────────────
# ClaudeClient — CLAUDE-ADD-1 through CLAUDE-ADD-5, CLAUDE-ADD-10
# ─────────────────────────────────────────────────────────────────────────────
class ClaudeClient:
   """
   Claude Haiku 4.5 via the official Anthropic SDK.
   Install: pip install anthropic


   CLAUDE-ADD-1:  Uses anthropic.Anthropic client (not OpenAI-compatible).
   CLAUDE-ADD-2:  Default model is claude-haiku-4-5-20251001.
                  Swap for claude-sonnet-4-6 or claude-opus-4-6 if needed.
   CLAUDE-ADD-3:  Anthropic API does NOT accept role="system" inside the
                  messages list. This client extracts the system message and
                  passes it as the top-level `system` kwarg automatically,
                  so callers can use the same standard message format as for
                  GPT without any changes.
   CLAUDE-ADD-4:  Markdown fences stripped for consistency with other clients.
   CLAUDE-ADD-5:  Returns (text, resp.model) — actual serving version string
                  confirmed by the API, matching FIX-2 behaviour.
   CLAUDE-ADD-10: Assistant prefill — appends {"role":"assistant","content":"{"}
                  to the message list before the API call. The Anthropic API
                  treats this as already-generated output, so the model is
                  forced to continue from "{" and cannot emit prose first.
                  The opening brace is re-prepended to the returned completion
                  because the API strips the prefill from the response text.
                  This eliminates chain-of-thought leakage that caused 100%
                  parse failures when using the original prompt alone.


   Note: CLAUDE-ADD-9 (system prompt hardening) is not applied here because
   the alphaECA SYSTEM_PROMPT already ends with a strict JSON-only instruction.
   """


   DEFAULT_MODEL = "claude-haiku-4-5-20251001"


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
       print(f"    [ClaudeClient] model_id='{model_id}'")


   def __call__(self, messages: list[dict]) -> tuple[str, str]:
       # CLAUDE-ADD-3: split system message out — Anthropic API requires it
       # as a top-level kwarg, not as a role inside the messages list.
       system_text: str = ""
       chat_messages: list[dict] = []
       for m in messages:
           if m["role"] == "system":
               system_text = m["content"]
           else:
               chat_messages.append({"role": m["role"], "content": m["content"]})


       if not chat_messages:
           raise ValueError(
               "[ClaudeClient] messages must contain at least one non-system message"
           )


       # CLAUDE-ADD-10: assistant prefill — forces JSON-only output.
       # Appending an assistant turn that starts with "{" makes the model
       # continue from that point; it cannot output reasoning prose before it.
       # The prefill brace is re-attached below after the API call.
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


           # CLAUDE-ADD-10 (cont.): re-attach the opening brace that the
           # Anthropic API strips from the completion (it echoes the prefill
           # separately but does not include it in resp.content[].text).
           text = "{" + text


           # CLAUDE-ADD-4: strip markdown fences the model sometimes adds
           text = re.sub(r"^```(?:json)?\s*", "", text.strip())
           text = re.sub(r"\s*```$", "", text).strip()


           # CLAUDE-ADD-5: resp.model is the actual serving version (FIX-2)
           ver = resp.model or self.model_id
           return text, ver


       except Exception as e:
           print(f"    [ClaudeClient] API call failed: {type(e).__name__}: {e}")
           raise




# ─────────────────────────────────────────────────────────────────────────────
# Model registry — GPT-5.1 and Claude Haiku 4.5 only
# ─────────────────────────────────────────────────────────────────────────────
MODEL_CONFIGS: dict[str, dict] = {
   "gpt51": {
       "display": "GPT-5.1",
       "api_id":  "gpt-5.1",
       "client":  "gpt51",
       "rpm":     30,
   },
   # ── CLAUDE-ADD-6 ─────────────────────────────────────────────────────
   "claude_haiku": {
       "display": "Claude Haiku 4.5",
       "api_id":  "claude-haiku-4-5-20251001",
       "client":  "claude",
       "rpm":     50,                 # Anthropic Haiku tier: up to 50 RPM
   },
}




def build_client(cfg: dict, gpt51_reasoning: str = "none"):
   """Instantiates the appropriate API client for the given model config."""
   c, a = cfg["client"], cfg["api_id"]
   if c == "gpt51":   return GPT51Client(a, reasoning_effort=gpt51_reasoning)
   if c == "claude":  return ClaudeClient(a)
   raise ValueError(f"Unknown client type: {c}")




# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation loop  (one model × one setting)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_one(
   *,
   model_key:       str,
   setting:         str,
   k_shots:         int,
   test_orbits:     np.ndarray,
   test_rules:      np.ndarray,
   test_alphas:     np.ndarray,
   shots_used:      list[dict],
   ckpt_dir:        str,
   raw_dir:         str,
   gpt51_reasoning: str = "none",
   retries:         int = 2,
) -> dict:


   cfg     = MODEL_CONFIGS[model_key]
   display = cfg["display"]
   if model_key == "gpt51" and gpt51_reasoning != "none":
       display = f"GPT-5.1 (reasoning={gpt51_reasoning})"


   delay   = 60.0 / cfg["rpm"]
   n_total = len(test_orbits)


   reason_sfx = (
       f"_re{gpt51_reasoning}"
       if model_key == "gpt51" and gpt51_reasoning != "none"
       else ""
   )
   ckpt_path = os.path.join(
       ckpt_dir, f"{model_key}{reason_sfx}_{setting}.json"
   )


   print(f"\n  [{display}] [{setting}]  k={k_shots}  N={n_total}")


   try:
       client = build_client(cfg, gpt51_reasoning=gpt51_reasoning)
   except (ImportError, EnvironmentError) as e:
       print(f"    SKIP: {e}")
       return {
           "model":    display,
           "setting":  setting,
           "k_shots":  k_shots,
           "rule_acc": None,
           "alpha_acc": None,
           "skipped":  True,
           "error":    str(e),
       }


   results, done_set = load_checkpoint(ckpt_path)
   observed_model_versions: set[str] = set()


   t0 = time.time()


   for i, (orbit, true_rule, true_alpha) in enumerate(
       zip(test_orbits, test_rules, test_alphas)
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


       # ── Call API with retries (FIX-6) ─────────────────────────────────
       raw_output  = ""
       api_error   = False
       model_ver   = cfg["api_id"]


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
                   print(
                       f"    Sample {i}: API error after all retries: {e}"
                   )


       # ── Save raw response (full, no truncation — FIX-5) ───────────────
       raw_fname = f"{model_key}{reason_sfx}_{setting}_{i:04d}.txt"
       with open(
           os.path.join(raw_dir, raw_fname), "w", encoding="utf-8"
       ) as f:
           f.write(
               f"TRUE_RULE: {int(true_rule)}  "
               f"TRUE_ALPHA: {float(true_alpha):.1f}\n"
               f"MODEL_VERSION: {model_ver}\n"
               f"API_ERROR: {api_error}\n"
           )
           f.write("─" * 60 + "\n")
           f.write(raw_output)


       # ── Parse ─────────────────────────────────────────────────────────
       pred_rule, pred_alpha = parse_response(raw_output)
       rule_correct = (
           pred_rule == int(true_rule) if pred_rule is not None else False
       )
       alpha_ok = (
           abs(pred_alpha - float(true_alpha)) <= ALPHA_TOL
           if pred_alpha is not None
           else False
       )


       results.append({
           "sample_idx":    i,
           "true_rule":     int(true_rule),
           "pred_rule":     pred_rule,
           "true_alpha":    float(round(float(true_alpha), 1)),
           "pred_alpha":    pred_alpha,
           "rule_correct":  rule_correct,
           "alpha_ok":      alpha_ok,
           "api_error":     api_error,
           "model_version": model_ver,
           "raw_output":    raw_output,
       })
       done_set.add(i)


       # Checkpoint every 10 samples
       if (i + 1) % 10 == 0 or (i + 1) == n_total:
           ver_str = (
               sorted(observed_model_versions)[-1]
               if observed_model_versions
               else cfg["api_id"]
           )
           save_checkpoint(
               ckpt_path, display, ver_str, setting,
               k_shots, n_total, shots_used, results,
           )


       # Progress every 20 samples
       if (i + 1) % 20 == 0 or (i + 1) == n_total:
           nd        = len(results)
           rule_acc  = sum(r["rule_correct"] for r in results) / nd * 100
           alpha_acc = sum(r["alpha_ok"]     for r in results) / nd * 100
           pf        = sum(1 for r in results if r["pred_rule"] is None)
           ae        = sum(1 for r in results if r.get("api_error", False))
           eta       = (time.time() - t0) / (i + 1) * (n_total - i - 1)
           print(
               f"    [{i+1:3d}/{n_total}]  rule={rule_acc:5.1f}%  "
               f"alpha={alpha_acc:5.1f}%  parse_fail={pf}  "
               f"api_err={ae}  ETA={eta:.0f}s"
           )


       time.sleep(delay)


   # ── Final statistics with bootstrap CIs ───────────────────────────────
   nd        = len(results)
   rc_vals   = [r["rule_correct"] for r in results]
   ak_vals   = [r["alpha_ok"]     for r in results]
   rule_acc  = sum(rc_vals) / nd * 100 if nd else 0.0
   alpha_acc = sum(ak_vals) / nd * 100 if nd else 0.0
   pf        = sum(1 for r in results if r["pred_rule"] is None)
   ae        = sum(1 for r in results if r.get("api_error", False))


   rule_ci   = bootstrap_ci(rc_vals)
   alpha_ci  = bootstrap_ci(ak_vals)


   ver_str = (
       sorted(observed_model_versions)[-1]
       if observed_model_versions
       else cfg["api_id"]
   )


   print(
       f"  ✓ {display} [{setting}]  "
       f"rule={rule_acc:.2f}% [{rule_ci['lower']*100:.2f},{rule_ci['upper']*100:.2f}]  "
       f"alpha={alpha_acc:.2f}% [{alpha_ci['lower']*100:.2f},{alpha_ci['upper']*100:.2f}]  "
       f"parse_fail={pf}  api_err={ae}  ver={ver_str}"
   )


   return {
       "model":         display,
       "model_version": ver_str,
       "setting":       setting,
       "k_shots":       k_shots,
       "n":             nd,
       "rule_acc":      round(rule_acc,  4),
       "alpha_acc":     round(alpha_acc, 4),
       "rule_ci95":  {
           "mean":  round(rule_ci["mean"]  * 100, 4),
           "lower": round(rule_ci["lower"] * 100, 4),
           "upper": round(rule_ci["upper"] * 100, 4),
       },
       "alpha_ci95": {
           "mean":  round(alpha_ci["mean"]  * 100, 4),
           "lower": round(alpha_ci["lower"] * 100, 4),
           "upper": round(alpha_ci["upper"] * 100, 4),
       },
       "parse_rate": round((nd - pf) / nd * 100, 4) if nd else 0.0,
       "parse_fail": pf,
       "api_errors": ae,
       "skipped":    False,
   }




# ─────────────────────────────────────────────────────────────────────────────
# Open-source reference numbers — paper Table 29 (αECA, N=500)
# ⚑ FIX-8: sha256 fingerprints — populate before camera-ready submission.
# ─────────────────────────────────────────────────────────────────────────────
PAPER_RESULTS: list[dict] = [
   # zero-shot
   {"model": "Qwen2.5-7B",    "setting": "zero_shot", "rule_acc":  1.00, "alpha_acc": 11.60,
    "sha256": None, "source_ckpt": "results/zero_shot/Qwen2.5-7B_zero_shot.json"},
   {"model": "Qwen2.5-72B",   "setting": "zero_shot", "rule_acc":  1.00, "alpha_acc": 12.20,
    "sha256": None, "source_ckpt": "results/zero_shot/Qwen2.5-72B_zero_shot.json"},
   {"model": "Llama-3.1-8B",  "setting": "zero_shot", "rule_acc":  0.00, "alpha_acc": 11.60,
    "sha256": None, "source_ckpt": "results/zero_shot/Llama-3.1-8B_zero_shot.json"},
   {"model": "Llama-3.1-70B", "setting": "zero_shot", "rule_acc":  0.40, "alpha_acc": 10.00,
    "sha256": None, "source_ckpt": "results/zero_shot/Llama-3.1-70B_zero_shot.json"},
   {"model": "Mistral-7B",    "setting": "zero_shot", "rule_acc":  0.00, "alpha_acc":  0.00,
    "sha256": None, "source_ckpt": "results/zero_shot/Mistral-7B_zero_shot.json"},
   {"model": "Mixtral-8x7B",  "setting": "zero_shot", "rule_acc":  0.00, "alpha_acc":  0.00,
    "sha256": None, "source_ckpt": "results/zero_shot/Mixtral-8x7B_zero_shot.json"},
   # 5-shot
   {"model": "Qwen2.5-7B",    "setting": "5_shot",    "rule_acc":  0.20, "alpha_acc": 11.40,
    "sha256": None, "source_ckpt": "results/few_shot/Qwen2.5-7B_5shot.json"},
   {"model": "Qwen2.5-72B",   "setting": "5_shot",    "rule_acc":  0.60, "alpha_acc": 16.80,
    "sha256": None, "source_ckpt": "results/few_shot/Qwen2.5-72B_5shot.json"},
   {"model": "Llama-3.1-8B",  "setting": "5_shot",    "rule_acc":  0.00, "alpha_acc": 12.00,
    "sha256": None, "source_ckpt": "results/few_shot/Llama-3.1-8B_5shot.json"},
   {"model": "Llama-3.1-70B", "setting": "5_shot",    "rule_acc":  0.00, "alpha_acc":  0.00,
    "sha256": None, "source_ckpt": "results/few_shot/Llama-3.1-70B_5shot.json"},
   {"model": "Mistral-7B",    "setting": "5_shot",    "rule_acc":  0.00, "alpha_acc":  9.80,
    "sha256": None, "source_ckpt": "results/few_shot/Mistral-7B_5shot.json"},
   {"model": "Mixtral-8x7B",  "setting": "5_shot",    "rule_acc":  0.00, "alpha_acc": 11.20,
    "sha256": None, "source_ckpt": "results/few_shot/Mixtral-8x7B_5shot.json"},
   # fine-tuned
   {"model": "Qwen2.5-72B (fine-tuned)", "setting": "fine_tuned",
    "rule_acc": 8.20, "alpha_acc": 22.80,
    "sha256": None, "source_ckpt": "results/fine_tuned/Qwen2.5-72B_ft.json"},
]


ALPHAM_REF: dict = {
   "model":     "αM — signal-matched transformer",
   "setting":   "trained",
   "rule_acc":  99.22,
   "alpha_acc": 95.08,
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
           file_rule  = round(data.get("rule_accuracy",  data.get("rule_accuracy_pct",  None)), 2)
           file_alpha = round(data.get("alpha_accuracy", data.get("alpha_accuracy_pct", None)), 2)
           if file_rule is not None and abs(file_rule - r["rule_acc"]) > 0.01:
               print(
                   f"  ⚠ rule_acc mismatch for {r['model']} [{r['setting']}]: "
                   f"hardcoded={r['rule_acc']} file={file_rule}"
               )
               any_warned = True
           if file_alpha is not None and abs(file_alpha - r["alpha_acc"]) > 0.01:
               print(
                   f"  ⚠ alpha_acc mismatch for {r['model']} [{r['setting']}]: "
                   f"hardcoded={r['alpha_acc']} file={file_alpha}"
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
       description="Frontier LLM evaluation on αECA"
   )
   parser.add_argument(
       "--data_dir", default="ECA_Data_New",
       help="Root data dir containing phase2/test/ and phase2/train/",
   )
   parser.add_argument(
       "--output_dir", default="frontier_results",
       help="Output root (checkpoints/, raw/, summary files)",
   )
   parser.add_argument(
       "--n_samples", type=int, default=200,
       help="Test samples per condition (paper open-source eval uses 500)",
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
       help="'few_shot' runs all --k_shots values",
   )
   parser.add_argument(
       "--k_shots", nargs="+", type=int,
       default=DEFAULT_K_LIST,
       help="k values for few-shot (default: 1 3 5; nested; max=5)",
   )
   parser.add_argument(
       "--gpt51_reasoning",
       default="none",
       choices=["none", "low", "medium", "high"],
       help=(
           "GPT-5.1 reasoning_effort. 'none'=non-reasoning (default). "
           "Non-none results go to separate checkpoint files."
       ),
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


   # ── Load data ─────────────────────────────────────────────────────────
   print(f"\n[data] Loading from {args.data_dir}/phase2/")
   test_orbits, test_rules, test_alphas, all_shots = load_data(
       args.data_dir, args.n_samples, index_file
   )
   print(f"[data] {len(test_orbits)} test samples  |  "
         f"{len(all_shots)} shot pool (MAX_K_SHOTS={MAX_K_SHOTS})")
   print(
       f"[cfg]  max_tokens={MAX_TOKENS_ALL} (uniform — FIX-1)  |  "
       f"bootstrap B={BOOTSTRAP_REPS} (FIX-3)  |  "
       f"SEED={SEED}, shot_seed={SEED+99}"
   )
   if "few_shot" in args.settings:
       k_str = ", ".join(str(k) for k in sorted(args.k_shots))
       print(
           f"[few-shot] k={k_str}  |  nested shots (k=1⊂3⊂5)  |  "
           f"multi-turn chat [sys][usr][ast]…[usr]"
       )
   if "gpt51" in args.models:
       print(f"[GPT-5.1] reasoning_effort={args.gpt51_reasoning}")


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
           model_key       = model_key,
           setting         = setting,
           k_shots         = k,
           test_orbits     = test_orbits,
           test_rules      = test_rules,
           test_alphas     = test_alphas,
           shots_used      = shots_k,
           ckpt_dir        = ckpt_dir,
           raw_dir         = raw_dir,
           gpt51_reasoning = args.gpt51_reasoning,
       )
       all_metrics.append(m)


   elapsed = (time.time() - t_total) / 60
   print(f"\nTotal wall time: {elapsed:.1f} min")


   # ── Console table ─────────────────────────────────────────────────────
   COL = 90
   print("\n" + "=" * COL)
   print("  αECA IDENTIFICATION — FRONTIER LLM EVALUATION")
   print(
       f"  Rule: exact match  |  Alpha: discrete exact match (snapped)  "
       f"|  N={args.n_samples}"
   )
   print(
       f"  Prompts: verbatim Appendix F  |  Orbit: raw 0/1, ASCII 'x'  "
       f"|  max_tokens={MAX_TOKENS_ALL} (all models)"
   )
   print(
       f"  CIs: 95% bootstrap percentile, B={BOOTSTRAP_REPS}  "
       f"|  Nested shots k=1⊂3⊂5"
   )
   print("=" * COL)
   print(
       f"  {'Model':<34} {'Setting':<12} {'Rule%':>7}  "
       f"{'[95% CI]':>14}  {'α%':>7}  {'[95% CI]':>14}"
   )
   print("-" * COL)


   print("  — Open-source LLMs (N=500) —")
   for r in PAPER_RESULTS:
       print(
           f"  {r['model']:<34} {r['setting']:<12} "
           f"{r['rule_acc']:>7.2f}  {'':>14}  "
           f"{r['alpha_acc']:>7.2f}"
       )


   print("\n  — Frontier LLMs (this evaluation) —")
   for m in all_metrics:
       if m.get("skipped"):
           print(
               f"  {m['model']:<34} {m['setting']:<12}  "
               f"SKIPPED ({m.get('error', '')})"
           )
       else:
           rc = m["rule_ci95"]
           ac = m["alpha_ci95"]
           r_ci = f"[{rc['lower']:.2f},{rc['upper']:.2f}]"
           a_ci = f"[{ac['lower']:.2f},{ac['upper']:.2f}]"
           print(
               f"  {m['model']:<34} {m['setting']:<12} "
               f"{m['rule_acc']:>7.2f}  {r_ci:>14}  "
               f"{m['alpha_acc']:>7.2f}  {a_ci:>14}  "
               f"ver={m.get('model_version','?')}"
           )


   r = ALPHAM_REF
   print(
       f"\n  {r['model']:<34} {r['setting']:<12} "
       f"{r['rule_acc']:>7.2f}  {'':>14}  "
       f"{r['alpha_acc']:>7.2f}  <- ours"
   )
   print("=" * COL)


   # ── Save results_detail.json ──────────────────────────────────────────
   detail_path = os.path.join(args.output_dir, "results_detail.json")
   with open(detail_path, "w") as f:
       json.dump(
           {
               "frontier_metrics":  all_metrics,
               "paper_reference":   PAPER_RESULTS,
               "alphaM_reference":  ALPHAM_REF,
               "n_samples":         args.n_samples,
               "k_shots_evaluated": sorted(args.k_shots),


               "methodology": {
                   "token_budget": (
                       f"FIX-1: max_tokens={MAX_TOKENS_ALL} applied uniformly "
                       "to all models (GPT-5.1, Claude Haiku 4.5)."
                   ),
                   "claude_integration": (
                       "CLAUDE-ADD: ClaudeClient uses anthropic SDK. "
                       f"Model: {MODEL_CONFIGS['claude_haiku']['api_id']}. "
                       "System prompt extracted and passed as top-level `system` kwarg "
                       "(Anthropic API requirement — role='system' not accepted in messages). "
                       "Markdown fences stripped. resp.model echoed as model_version (FIX-2). "
                       "CLAUDE-ADD-10: Assistant prefill '{' appended before API call to "
                       "force JSON-only output; opening brace re-attached post-call. "
                       "CLAUDE-ADD-9 not applied: alphaECA SYSTEM_PROMPT already ends "
                       "with a strict JSON-only instruction."
                   ),
                   "model_versioning": (
                       "FIX-2: Actual model version returned by each API call "
                       "is recorded in every per-sample record."
                   ),
                   "confidence_intervals": (
                       f"FIX-3: 95% bootstrap CIs (percentile, B={BOOTSTRAP_REPS}, "
                       f"seed={BOOTSTRAP_SEED}) for rule_acc and alpha_acc."
                   ),
                   "alpha_evaluation": (
                       "FIX-4: pred_alpha is snapped to the nearest element of "
                       "ALPHA_VALUES={0.1,…,1.0} before comparison. Because the "
                       "set is evenly spaced at 0.1, this makes the metric an "
                       "exact match on the discrete set. The ±0.05 guard absorbs "
                       "float representation noise only and never changes the outcome "
                       "for a well-formed prediction."
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
                   "shot_design": (
                       "Nested — all k values drawn from the same MAX_K_SHOTS=5 pool: "
                       "k=1→shots[:1], k=3→shots[:3], k=5→shots[:5]. "
                       "Pool drawn with np.random.default_rng(SEED+99)."
                   ),
                   "few_shot_format": (
                       "Multi-turn chat [system][user][assistant]...[user] — "
                       "verbatim Appendix F Section F.2 build_messages(). "
                       "All frontier APIs support multi-turn natively."
                   ),
                   "determinism": (
                       "temperature=0.0 for all models."
                   ),
               },


               "seed":              SEED,
               "shot_seed":         SEED + 99,
               "bootstrap_seed":    BOOTSTRAP_SEED,
               "bootstrap_reps":    BOOTSTRAP_REPS,
               "max_tokens":        MAX_TOKENS_ALL,
               "gpt51_reasoning":   args.gpt51_reasoning,
               "paper_table":       "Table 29 (alphaECA, N=500)",
               "run_timestamp":     datetime.now(tz=timezone.utc).isoformat(),
           },
           f,
           indent=2,
       )


   # ── Save results_summary.csv ──────────────────────────────────────────
   csv_path = os.path.join(args.output_dir, "results_summary.csv")
   with open(csv_path, "w") as f:
       f.write(
           "Model,Model Version,Setting,k_shots,N,"
           "Rule Exact Match (%),Rule CI95 Lower,Rule CI95 Upper,"
           "Alpha Exact Match (%),Alpha CI95 Lower,Alpha CI95 Upper,"
           "Parse Failures,API Errors\n"
       )
       for r in PAPER_RESULTS:
           k_val = "0" if "zero" in r["setting"] else r["setting"].split("_")[0]
           f.write(
               f"{r['model']},—,{r['setting']},{k_val},{args.n_samples},"
               f"{r['rule_acc']},—,—,{r['alpha_acc']},—,—,—,—\n"
           )
       for m in all_metrics:
           if not m.get("skipped"):
               rc = m["rule_ci95"]
               ac = m["alpha_ci95"]
               f.write(
                   f"{m['model']},{m.get('model_version','?')},"
                   f"{m['setting']},{m['k_shots']},{m['n']},"
                   f"{m['rule_acc']},{rc['lower']},{rc['upper']},"
                   f"{m['alpha_acc']},{ac['lower']},{ac['upper']},"
                   f"{m['parse_fail']},{m['api_errors']}\n"
               )
       r = ALPHAM_REF
       f.write(
           f"{r['model']},—,{r['setting']},—,—,"
           f"{r['rule_acc']},—,—,{r['alpha_acc']},—,—,—,—\n"
       )


   print(f"\nSaved: {detail_path}")
   print(f"Saved: {csv_path}")
   print(f"Checkpoints: {ckpt_dir}/")
   print(f"Raw responses: {raw_dir}/")




if __name__ == "__main__":
   main()
