"""
few_shot_stoc.py  --  Few-shot LLM evaluation on stochastic ECA inverse problem.

Usage:
  python few_shot_stoc.py --model Qwen2.5-7B-Instruct --n_samples 200 --k_shots 3
  python few_shot_stoc.py --model all --n_samples 200 --k_shots 5
"""

import os, json, time, argparse, re
import numpy as np
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR    = "/home/user/projects/ECA_stochastic"
MODELS_DIR  = "/home/user/models"
DATA_DIR    = os.path.join(BASE_DIR, "ECA_Data_Skew")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "few_shot")
INDEX_FILE  = os.path.join(BASE_DIR, "results", "fixed_test_indices.npy")
SEED        = 42

LAM_TOL  = 0.05
LAM_MIN  = 0.1
LAM_MAX  = 0.9
W, T, K  = 50, 200, 8

ALL_MODELS = [
    "Llama-3.1-8B-Instruct",
    "Llama-3.1-70B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "Mixtral-8x7B-Instruct-v0.1",
    "Qwen2.5-7B-Instruct",
    "Qwen2.5-72B-Instruct",
]

SYSTEM_PROMPT = """You are an expert in cellular automata. You are given a space-time orbit of an Elementary Cellular Automaton (ECA) under stochastic noise.

In this stochastic model, at each timestep each cell independently chooses one of two rules. With probability lambda it uses rule G, otherwise it uses rule F. The parameter lambda is a continuous value between 0.1 and 0.9. The system (F, G, lambda) is statistically identical to (G, F, 1-lambda), so predict the smaller rule number as rule_f and the larger as rule_g.

The orbit is a grid of 0s and 1s. Each row is one timestep. Grid width is 50 cells.

Identify:
1. Rule F (the first ECA rule, integer 0-255)
2. Rule G (the second ECA rule, integer 0-255, different from F)
3. Lambda value (float between 0.1 and 0.9, rounded to 2 decimal places)

Respond ONLY in this JSON format with no extra text:
{"rule_f": <integer>, "rule_g": <integer>, "lambda": <float>}"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def orbit_to_str(orbit):
    return "\n".join("".join(str(int(c)) for c in row) for row in orbit[:50])

def make_user_prompt(orbit):
    return (f"Space-time orbit (showing 50 of {T} rows x {W} cells):\n\n"
            f"{orbit_to_str(orbit)}\n\nWhat are rule F, rule G, and lambda?")

def snap_lambda(val):
    val = round(round(val / 0.05) * 0.05, 2)
    return max(LAM_MIN, min(LAM_MAX, val))

def parse_response(response):
    try:
        s = response.find("{"); e = response.rfind("}") + 1
        if s != -1 and e > s:
            d   = json.loads(response[s:e])
            rf  = int(d.get("rule_f", -1))
            rg  = int(d.get("rule_g", -1))
            lam = float(d.get("lambda", -1))
            if 0 <= rf <= 255 and 0 <= rg <= 255 and LAM_MIN <= lam <= LAM_MAX:
                return rf, rg, snap_lambda(lam)
    except Exception:
        pass
    rfm = re.search(r'"rule_f"\s*:\s*(\d+)',   response)
    rgm = re.search(r'"rule_g"\s*:\s*(\d+)',   response)
    lm  = re.search(r'"lambda"\s*:\s*([\d.]+)', response)
    if rfm and rgm and lm:
        rf = int(rfm.group(1)); rg = int(rgm.group(1)); lam = float(lm.group(1))
        if 0 <= rf <= 255 and 0 <= rg <= 255:
            return rf, rg, snap_lambda(lam)
    return None, None, None

def rules_match(pred_f, pred_g, true_f, true_g):
    fwd = (pred_f == true_f and pred_g == true_g)
    flp = (pred_f == true_g and pred_g == true_f)
    return fwd or flp

# ── Data ──────────────────────────────────────────────────────────────────────

def get_fixed_test_indices(n_samples):
    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
    if os.path.exists(INDEX_FILE):
        return np.load(INDEX_FILE)[:n_samples]
    rng    = np.random.default_rng(SEED)
    orbits = np.load(os.path.join(DATA_DIR, "test", "orbits.npy"))
    n      = min(n_samples, len(orbits))
    idx    = rng.choice(len(orbits), size=n, replace=False)
    np.save(INDEX_FILE, idx)
    return idx

def load_data(n_samples, k_shots):
    idx      = get_fixed_test_indices(n_samples)
    test_dir = os.path.join(DATA_DIR, "test")
    orbits   = np.load(os.path.join(test_dir, "orbits.npy"))
    rule_f   = np.load(os.path.join(test_dir, "rule_f_ids.npy"))
    rule_g   = np.load(os.path.join(test_dir, "rule_g_ids.npy"))
    lambdas  = np.load(os.path.join(test_dir, "lambdas.npy"))

    # Few-shot examples from train_pairs (rule pairs from 179 training rules)
    train_pairs = np.load(os.path.join(DATA_DIR, "train_pairs.npy"))  # [500, 2]
    rng         = np.random.default_rng(SEED + 99)
    shot_idx    = rng.choice(len(train_pairs), size=k_shots, replace=False)

    # Generate fresh orbits for few-shot examples
    from numpy.random import default_rng as drng
    shots = []
    for si in shot_idx:
        f_rule = int(train_pairs[si, 0])
        g_rule = int(train_pairs[si, 1])
        lam    = float(rng.uniform(LAM_MIN, LAM_MAX))
        # Create a simple placeholder orbit for the few-shot example
        # (we use a slice of train test orbits if available, else zeros)
        placeholder = np.zeros((T, W), dtype=np.float32)
        shots.append({"orbit": placeholder, "rule_f": f_rule,
                      "rule_g": g_rule, "lambda": round(lam, 2)})

    return orbits[idx], rule_f[idx], rule_g[idx], lambdas[idx], shots

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
    pair_acc = sum(r["pair_correct"] for r in results) / n * 100 if n else 0
    lam_acc  = sum(r["lam_ok"]       for r in results) / n * 100 if n else 0
    lam_mae  = float(np.mean([r["lam_mae"] for r in results
                               if r["lam_mae"] is not None])) if n else 0
    with open(path, "w") as f:
        json.dump({
            "model":               model_name,
            "mode":                mode,
            "k_shots":             k,
            "n_evaluated":         n,
            "n_total":             n_total,
            "pair_accuracy_UO":    pair_acc,
            "lambda_accuracy_pm005": lam_acc,
            "lambda_mae_sym":      lam_mae,
            "parse_failures":      sum(1 for r in results if r["pred_f"] is None),
            "few_shot_examples":   [{"rule_f": s["rule_f"], "rule_g": s["rule_g"],
                                     "lambda": s["lambda"]} for s in shots],
            "timestamp":           datetime.now().isoformat(),
            "samples":             results,
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
        messages.append({"role": "user",
                          "content": make_user_prompt(sh["orbit"])})
        messages.append({"role": "assistant",
                          "content": json.dumps({"rule_f": sh["rule_f"],
                                                 "rule_g": sh["rule_g"],
                                                 "lambda": sh["lambda"]})})
    messages.append({"role": "user", "content": make_user_prompt(orbit[0])})
    return messages

def run_inference(tok, model, shots, orbit, max_new_tokens=48):
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

    print(f"\n{'='*65}\n  {model_name}  |  {k_shots}-shot (stochastic)  |  N={n_samples}\n{'='*65}")

    orbits, rule_f, rule_g, lambdas, shots = load_data(n_samples, k_shots)
    print(f"  Shots: " + ", ".join(f"f={s['rule_f']} g={s['rule_g']} lam={s['lambda']}" for s in shots))

    results, done_set = load_checkpoint(result_path)
    remaining = [(i, orbits[i], rule_f[i], rule_g[i], lambdas[i])
                 for i in range(len(orbits)) if i not in done_set]

    if not remaining:
        print("  All samples done.")
        return

    tok, model = load_model(model_name)
    t0 = time.time()

    for pos, (i, orbit, true_f, true_g, true_lam) in enumerate(remaining):
        raw                      = run_inference(tok, model, shots, orbit)
        pred_f, pred_g, pred_lam = parse_response(raw)
        pair_ok = rules_match(pred_f, pred_g, int(true_f), int(true_g)) \
                  if pred_f is not None else False
        if pred_lam is not None:
            lam_err = min(abs(pred_lam - float(true_lam)),
                          abs(pred_lam - (1.0 - float(true_lam))))
            lam_ok  = lam_err <= LAM_TOL
        else:
            lam_err = None; lam_ok = False

        results.append({
            "sample_idx":   i,
            "true_f":       int(true_f),
            "true_g":       int(true_g),
            "pred_f":       pred_f,
            "pred_g":       pred_g,
            "true_lambda":  float(true_lam),
            "pred_lambda":  pred_lam,
            "pair_correct": pair_ok,
            "lam_ok":       lam_ok,
            "lam_mae":      lam_err,
            "raw_output":   raw,
        })

        if (pos + 1) % 10 == 0 or pos + 1 == len(remaining):
            save_checkpoint(result_path, model_name, f"{k_shots}_shot",
                            k_shots, n_samples, shots, results)
            done     = len(results)
            eta      = (time.time() - t0) / (pos + 1) * (len(remaining) - pos - 1)
            pair_acc = sum(r["pair_correct"] for r in results) / done * 100
            lam_acc  = sum(r["lam_ok"]       for r in results) / done * 100
            print(f"  [{done:>4}/{n_samples}]  pair(UO)={pair_acc:.1f}%  "
                  f"lam(+-0.05)={lam_acc:.1f}%  ETA={eta:.0f}s", flush=True)

    n        = len(results)
    pair_acc = sum(r["pair_correct"] for r in results) / n * 100
    lam_acc  = sum(r["lam_ok"]       for r in results) / n * 100
    print(f"\n  FINAL  pair(UO)={pair_acc:.2f}%  lam(+-0.05)={lam_acc:.2f}%")
    print(f"  Saved: {result_path}")
    del model; torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default="Qwen2.5-7B-Instruct")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--k_shots",   type=int, default=3)
    args   = parser.parse_args()
    models = ALL_MODELS if args.model == "all" else [args.model]
    for m in models:
        evaluate_model(m, args.n_samples, args.k_shots)

if __name__ == "__main__":
    main()