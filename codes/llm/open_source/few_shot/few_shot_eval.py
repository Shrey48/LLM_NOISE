"""
few_shot_eval.py  --  Few-shot LLM evaluation on alpha-asynchronous ECA inverse problem.

Evaluates on phase2/test using SAME fixed indices as zero_shot_eval.py
Few-shot examples drawn from phase2/train (no test leakage)
Resume support: saves checkpoint every 10 samples

Usage:
  python few_shot_eval.py --model Qwen2.5-7B-Instruct --n_samples 500 --k_shots 3
  python few_shot_eval.py --model all --n_samples 500 --k_shots 5
"""

import os, json, time, argparse, re
import numpy as np
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR    = "/home/user/ECA_alpha_async"
MODELS_DIR  = "/home/user/models"
DATA_DIR    = os.path.join(BASE_DIR, "ECA_Data_New")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "few_shot")
INDEX_FILE  = os.path.join(BASE_DIR, "results", "fixed_test_indices.npy")
SEED        = 42

ALPHA_VALUES = [round(a * 0.1, 1) for a in range(1, 11)]
ALPHA_TOL    = 0.05
W, T         = 20, 100

ALL_MODELS = [
    "Llama-3.1-8B-Instruct",
    "Llama-3.1-70B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "Mixtral-8x7B-Instruct-v0.1",
    "Qwen2.5-7B-Instruct",
    "Qwen2.5-72B-Instruct",
]

SYSTEM_PROMPT = """You are an expert in cellular automata. You are given a space-time orbit of an Elementary Cellular Automaton (ECA) perturbed by alpha-asynchronous noise.

In alpha-asynchronous noise each cell independently updates with probability alpha per timestep.
Alpha is one of: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0

The orbit is a grid of 0s and 1s. Each row is one timestep.

Identify:
1. The ECA rule number (integer 0-255)
2. The alpha value

Respond ONLY in this JSON format with no extra text:
{"rule": <integer>, "alpha": <float>}"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def orbit_to_str(orbit):
    return "\n".join("".join(str(int(c)) for c in row) for row in orbit)

def make_user_prompt(orbit):
    return (f"Space-time orbit ({T} rows x {W} cells):\n\n"
            f"{orbit_to_str(orbit)}\n\nWhat is the ECA rule number and alpha value?")

def parse_response(response):
    try:
        s = response.find("{"); e = response.rfind("}") + 1
        if s != -1 and e > s:
            d     = json.loads(response[s:e])
            rule  = int(d.get("rule", -1))
            alpha = float(d.get("alpha", -1))
            if 0 <= rule <= 255:
                alpha = min(ALPHA_VALUES, key=lambda a: abs(a - alpha))
                return rule, alpha
    except Exception:
        pass
    rm = re.search(r'"rule"\s*:\s*(\d+)',     response)
    am = re.search(r'"alpha"\s*:\s*([\d.]+)', response)
    if rm and am:
        rule  = int(rm.group(1))
        alpha = float(am.group(1))
        if 0 <= rule <= 255:
            alpha = min(ALPHA_VALUES, key=lambda a: abs(a - alpha))
            return rule, alpha
    return None, None

# ── Data ──────────────────────────────────────────────────────────────────────

def get_fixed_test_indices(n_samples):
    """Load shared fixed test indices (created by zero_shot_eval.py or here)."""
    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
    if os.path.exists(INDEX_FILE):
        idx = np.load(INDEX_FILE)
        return idx[:n_samples]
    rng    = np.random.default_rng(SEED)
    orbits = np.load(os.path.join(DATA_DIR, "phase2", "test", "orbits.npy"))
    n      = min(n_samples, len(orbits))
    idx    = rng.choice(len(orbits), size=n, replace=False)
    np.save(INDEX_FILE, idx)
    return idx

def load_data(n_samples, k_shots):
    # Test data: same fixed indices as zero_shot
    idx      = get_fixed_test_indices(n_samples)
    t_orbits   = np.load(os.path.join(DATA_DIR, "phase2", "test", "orbits.npy"))
    t_rule_ids = np.load(os.path.join(DATA_DIR, "phase2", "test", "rule_ids.npy"))
    t_alphas   = np.load(os.path.join(DATA_DIR, "phase2", "test", "alphas.npy"))

    # Few-shot examples from TRAIN split (179 training rules, no leakage)
    rng        = np.random.default_rng(SEED + 99)  # different seed for shots
    s_orbits   = np.load(os.path.join(DATA_DIR, "phase2", "train", "orbits.npy"))
    s_rule_ids = np.load(os.path.join(DATA_DIR, "phase2", "train", "rule_ids.npy"))
    s_alphas   = np.load(os.path.join(DATA_DIR, "phase2", "train", "alphas.npy"))
    s_idx      = rng.choice(len(s_orbits), size=k_shots, replace=False)
    shots = [{"orbit": s_orbits[i], "rule": int(s_rule_ids[i]),
               "alpha": float(s_alphas[i])} for i in s_idx]

    return t_orbits[idx], t_rule_ids[idx], t_alphas[idx], shots

# ── Checkpoint ────────────────────────────────────────────────────────────────

def load_checkpoint(path):
    if not os.path.exists(path):
        return [], set()
    try:
        with open(path) as f:
            data = json.load(f)
        samples = data.get("samples", [])
        done    = {s["sample_idx"] for s in samples}
        print(f"  Resume: {len(done)} samples already done.")
        return samples, done
    except Exception:
        return [], set()

def save_checkpoint(path, model_name, mode, k, n_total, shots, results):
    n         = len(results)
    rule_acc  = sum(r["rule_correct"] for r in results) / n * 100 if n else 0
    alpha_acc = sum(r["alpha_ok"]     for r in results) / n * 100 if n else 0
    with open(path, "w") as f:
        json.dump({
            "model":             model_name,
            "mode":              mode,
            "k_shots":           k,
            "n_evaluated":       n,
            "n_total":           n_total,
            "rule_accuracy":     rule_acc,
            "alpha_accuracy":    alpha_acc,
            "parse_failures":    sum(1 for r in results if r["pred_rule"] is None),
            "few_shot_examples": [{"rule": s["rule"], "alpha": s["alpha"]} for s in shots],
            "timestamp":         datetime.now().isoformat(),
            "samples":           results,
        }, f, indent=2)

# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(model_name):
    path = os.path.join(MODELS_DIR, model_name)
    print(f"  Loading {model_name} ...")
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    is_70b = any(x in model_name for x in ["70B", "72B"])
    if is_70b:
        print(f"  70B model -- loading in 8-bit")
        model = AutoModelForCausalLM.from_pretrained(
            path, load_in_8bit=True,
            device_map="auto", trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16,
            device_map="auto", trust_remote_code=True)
    model.eval()
    return tok, model

def build_messages(shots, orbit):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for s in shots:
        messages.append({"role": "user",
                          "content": make_user_prompt(s["orbit"])})
        messages.append({"role": "assistant",
                          "content": json.dumps({"rule": s["rule"],
                                                 "alpha": s["alpha"]})})
    messages.append({"role": "user", "content": make_user_prompt(orbit)})
    return messages

def run_inference(tok, model, shots, orbit, max_new_tokens=64):
    messages = build_messages(shots, orbit)
    try:
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        parts = []
        for m in messages:
            if m["role"] == "system":      parts.append(f"System: {m['content']}")
            elif m["role"] == "user":      parts.append(f"User: {m['content']}")
            elif m["role"] == "assistant": parts.append(f"Assistant: {m['content']}")
        parts.append("Assistant:")
        text = "\n\n".join(parts)
    inputs = tok(text, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=1.0,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()

# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate_model(model_name, n_samples, k_shots):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_path = os.path.join(RESULTS_DIR, f"{model_name}_{k_shots}shot.json")

    print(f"\n{'='*65}\n  {model_name}  |  {k_shots}-shot  |  N={n_samples}\n{'='*65}")

    orbits, rule_ids, alphas, shots = load_data(n_samples, k_shots)
    print(f"  Few-shot examples: " +
          ", ".join(f"rule={s['rule']} alpha={s['alpha']}" for s in shots))

    results, done_set = load_checkpoint(result_path)
    remaining = [(i, orbits[i], rule_ids[i], alphas[i])
                 for i in range(len(orbits)) if i not in done_set]

    if not remaining:
        print("  All samples done. Nothing to run.")
        return

    tok, model = load_model(model_name)
    t0 = time.time()

    for pos, (i, orbit, true_rule, true_alpha) in enumerate(remaining):
        raw               = run_inference(tok, model, shots, orbit)
        pred_rule, pred_alpha = parse_response(raw)
        r_ok = pred_rule == int(true_rule) if pred_rule is not None else False
        a_ok = abs(pred_alpha - float(true_alpha)) <= ALPHA_TOL if pred_alpha is not None else False

        results.append({
            "sample_idx":   i,
            "true_rule":    int(true_rule),
            "pred_rule":    pred_rule,
            "true_alpha":   float(true_alpha),
            "pred_alpha":   pred_alpha,
            "rule_correct": r_ok,
            "alpha_ok":     a_ok,
            "raw_output":   raw,
        })

        if (pos + 1) % 10 == 0 or pos + 1 == len(remaining):
            save_checkpoint(result_path, model_name, f"{k_shots}_shot",
                            k_shots, n_samples, shots, results)
            done      = len(results)
            eta       = (time.time() - t0) / (pos + 1) * (len(remaining) - pos - 1)
            rule_acc  = sum(r["rule_correct"] for r in results) / done * 100
            alpha_acc = sum(r["alpha_ok"]     for r in results) / done * 100
            print(f"  [{done:>4}/{n_samples}]  rule={rule_acc:.1f}%  "
                  f"alpha={alpha_acc:.1f}%  ETA={eta:.0f}s", flush=True)

    n         = len(results)
    rule_acc  = sum(r["rule_correct"] for r in results) / n * 100
    alpha_acc = sum(r["alpha_ok"]     for r in results) / n * 100
    print(f"\n  FINAL  rule={rule_acc:.2f}%  alpha={alpha_acc:.2f}%")
    print(f"  Saved: {result_path}")
    del model; torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default="Qwen2.5-7B-Instruct")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--k_shots",   type=int, default=3)
    args   = parser.parse_args()
    models = ALL_MODELS if args.model == "all" else [args.model]
    for m in models:
        evaluate_model(m, args.n_samples, args.k_shots)

if __name__ == "__main__":
    main()
