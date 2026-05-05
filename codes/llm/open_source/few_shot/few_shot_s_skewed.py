"""
few_shot_skew.py  --  Few-shot LLM evaluation on s-skewed ECA inverse problem.

Usage:
  python few_shot_skew.py --model Qwen2.5-7B-Instruct --n_samples 500 --k_shots 3
  python few_shot_skew.py --model all --n_samples 500 --k_shots 5
"""

import os, json, time, argparse, re
import numpy as np
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR   = os.environ.get("BASE_DIR",   os.path.join(os.getcwd(), "ECA_s_skewed"))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(os.getcwd(), "models"))
DATA_DIR    = os.path.join(BASE_DIR, "ECA_Data_Skew")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "few_shot")
INDEX_FILE  = os.path.join(BASE_DIR, "results", "fixed_test_indices.npy")
SEED        = 42

S_VALUES = list(range(1, 21))
S_TOL    = 1
W, T     = 20, 200

ALL_MODELS = [
    "Llama-3.1-8B-Instruct",
    "Llama-3.1-70B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "Mixtral-8x7B-Instruct-v0.1",
    "Qwen2.5-7B-Instruct",
    "Qwen2.5-72B-Instruct",
]

SYSTEM_PROMPT = """You are an expert in cellular automata. You are given a space-time orbit of an Elementary Cellular Automaton (ECA) perturbed by s-skewed noise.

In s-skewed noise, at each timestep exactly s contiguous cells are chosen at random and updated according to the ECA rule. All other cells retain their current state. The parameter s ranges from 1 to 20 (the full grid width), where s=20 means all cells update (fully synchronous).

The orbit is a grid of 0s and 1s. Each row is one timestep. Grid width is 20 cells.

Identify:
1. The ECA rule number (integer 0-255)
2. The value of s (integer from 1 to 20)

Respond ONLY in this JSON format with no extra text:
{"rule": <integer>, "s": <integer>}"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def orbit_to_str(orbit):
    return "\n".join("".join(str(int(c)) for c in row) for row in orbit)

def make_user_prompt(orbit):
    return (f"Space-time orbit ({T} rows x {W} cells):\n\n"
            f"{orbit_to_str(orbit)}\n\nWhat is the ECA rule number and s value?")

def parse_response(response):
    try:
        s = response.find("{"); e = response.rfind("}") + 1
        if s != -1 and e > s:
            d    = json.loads(response[s:e])
            rule = int(d.get("rule", -1))
            sv   = int(round(float(d.get("s", -1))))
            if 0 <= rule <= 255 and 1 <= sv <= 20:
                return rule, sv
    except Exception:
        pass
    rm = re.search(r'"rule"\s*:\s*(\d+)', response)
    sm = re.search(r'"s"\s*:\s*(\d+)',    response)
    if rm and sm:
        rule = int(rm.group(1))
        sv   = int(sm.group(1))
        if 0 <= rule <= 255 and 1 <= sv <= 20:
            return rule, sv
    return None, None

# ── Data ──────────────────────────────────────────────────────────────────────

def get_fixed_test_indices(n_samples):
    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
    if os.path.exists(INDEX_FILE):
        return np.load(INDEX_FILE)[:n_samples]
    rng    = np.random.default_rng(SEED)
    orbits = np.load(os.path.join(DATA_DIR, "phase2", "test", "orbits.npy"))
    n      = min(n_samples, len(orbits))
    idx    = rng.choice(len(orbits), size=n, replace=False)
    np.save(INDEX_FILE, idx)
    return idx

def load_data(n_samples, k_shots):
    idx      = get_fixed_test_indices(n_samples)
    t_orbits   = np.load(os.path.join(DATA_DIR, "phase2", "test", "orbits.npy"))
    t_rule_ids = np.load(os.path.join(DATA_DIR, "phase2", "test", "rule_ids.npy"))
    t_s_vals   = np.load(os.path.join(DATA_DIR, "phase2", "test", "s_values.npy"))

    rng        = np.random.default_rng(SEED + 99)
    s_orbits   = np.load(os.path.join(DATA_DIR, "phase2", "train", "orbits.npy"))
    s_rule_ids = np.load(os.path.join(DATA_DIR, "phase2", "train", "rule_ids.npy"))
    s_s_vals   = np.load(os.path.join(DATA_DIR, "phase2", "train", "s_values.npy"))
    s_idx      = rng.choice(len(s_orbits), size=k_shots, replace=False)
    shots = [{"orbit": s_orbits[i], "rule": int(s_rule_ids[i]),
               "s": int(s_s_vals[i])} for i in s_idx]

    return t_orbits[idx], t_rule_ids[idx], t_s_vals[idx], shots

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
    n        = len(results)
    rule_acc = sum(r["rule_correct"] for r in results) / n * 100 if n else 0
    s_acc    = sum(r["s_ok"]         for r in results) / n * 100 if n else 0
    s_mae    = float(np.mean([r["s_mae"] for r in results if r["s_mae"] is not None])) if n else 0
    with open(path, "w") as f:
        json.dump({
            "model":             model_name,
            "mode":              mode,
            "k_shots":           k,
            "n_evaluated":       n,
            "n_total":           n_total,
            "rule_accuracy":     rule_acc,
            "s_accuracy_pm1":    s_acc,
            "s_mae":             s_mae,
            "parse_failures":    sum(1 for r in results if r["pred_rule"] is None),
            "few_shot_examples": [{"rule": s["rule"], "s": s["s"]} for s in shots],
            "timestamp":         datetime.now().isoformat(),
            "samples":           results,
        }, f, indent=2)

# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(model_name):
    path = os.path.join(MODELS_DIR, model_name)
    tok  = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    is_70b = any(x in model_name for x in ["70B", "72B"])
    if is_70b:
        model = AutoModelForCausalLM.from_pretrained(
            path, load_in_8bit=True, device_map="auto", trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()
    return tok, model

def build_messages(shots, orbit):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for sh in shots:
        messages.append({"role": "user",      "content": make_user_prompt(sh["orbit"])})
        messages.append({"role": "assistant", "content": json.dumps({"rule": sh["rule"], "s": sh["s"]})})
    messages.append({"role": "user", "content": make_user_prompt(orbit)})
    return messages

def run_inference(tok, model, shots, orbit, max_new_tokens=32):
    messages = build_messages(shots, orbit)
    try:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=1.0,
                             pad_token_id=tok.pad_token_id,
                             eos_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate_model(model_name, n_samples, k_shots):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_path = os.path.join(RESULTS_DIR, f"{model_name}_{k_shots}shot.json")

    print(f"\n{'='*65}\n  {model_name}  |  {k_shots}-shot (s-skewed)  |  N={n_samples}\n{'='*65}")

    orbits, rule_ids, s_vals, shots = load_data(n_samples, k_shots)
    print(f"  Shots: " + ", ".join(f"rule={s['rule']} s={s['s']}" for s in shots))

    results, done_set = load_checkpoint(result_path)
    remaining = [(i, orbits[i], rule_ids[i], s_vals[i])
                 for i in range(len(orbits)) if i not in done_set]

    if not remaining:
        print("  All samples done.")
        return

    tok, model = load_model(model_name)
    t0 = time.time()

    for pos, (i, orbit, true_rule, true_s) in enumerate(remaining):
        raw             = run_inference(tok, model, shots, orbit)
        pred_rule, pred_s = parse_response(raw)
        r_ok  = pred_rule == int(true_rule) if pred_rule is not None else False
        s_ok  = abs(pred_s - int(true_s)) <= S_TOL if pred_s is not None else False
        s_mae = abs(pred_s - int(true_s)) if pred_s is not None else None

        results.append({
            "sample_idx":   i,
            "true_rule":    int(true_rule),
            "pred_rule":    pred_rule,
            "true_s":       int(true_s),
            "pred_s":       pred_s,
            "rule_correct": r_ok,
            "s_ok":         s_ok,
            "s_mae":        s_mae,
            "raw_output":   raw,
        })

        if (pos + 1) % 10 == 0 or pos + 1 == len(remaining):
            save_checkpoint(result_path, model_name, f"{k_shots}_shot",
                            k_shots, n_samples, shots, results)
            done     = len(results)
            eta      = (time.time() - t0) / (pos + 1) * (len(remaining) - pos - 1)
            rule_acc = sum(r["rule_correct"] for r in results) / done * 100
            s_acc    = sum(r["s_ok"]         for r in results) / done * 100
            print(f"  [{done:>4}/{n_samples}]  rule={rule_acc:.1f}%  "
                  f"s_acc(+-1)={s_acc:.1f}%  ETA={eta:.0f}s", flush=True)

    n        = len(results)
    rule_acc = sum(r["rule_correct"] for r in results) / n * 100
    s_acc    = sum(r["s_ok"]         for r in results) / n * 100
    print(f"\n  FINAL  rule={rule_acc:.2f}%  s_acc(+-1)={s_acc:.2f}%")
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
