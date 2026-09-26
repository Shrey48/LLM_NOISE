"""
zero_shot_alpha_async_cot_tools.py -- reasoning + Python-tool zero-shot evaluation
====================================================================================
Combined condition (reasoning permitted and tool access) for the open-source
LLMs on alphaECA. The factorial version, which also runs reasoning-only and
tools-only, is zero_shot_alpha_async_reasoning_eval.py.

Changes from the main zero-shot protocol (data, fixed test indices, models,
checkpointing and metrics are unchanged):
  1. The system prompt permits step-by-step reasoning.
  2. The model has a Python tool: it can emit a ```python ... ``` block, which
     is executed with the orbit available as a numpy array; the printed output
     is returned to the model.
  3. Up to 2048 new tokens per turn and up to 5 tool calls, in a multi-turn
     loop instead of a single 64-token generation.

Usage:
  python zero_shot_alpha_async_cot_tools.py --model Qwen2.5-7B-Instruct --n_samples 50
  python zero_shot_alpha_async_cot_tools.py --model all --n_samples 50

Environment variables:
  BASE_DIR    directory containing ECA_Data_New/ and results/ (default: ./ECA_alpha_async)
  MODELS_DIR  directory containing the model checkpoints (default: ./models)
"""

import os, json, time, argparse, re, io, contextlib, signal, math, collections
import numpy as np
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -- Config ---------------------------------------------------------------------

BASE_DIR    = os.environ.get("BASE_DIR",   os.path.join(os.getcwd(), "ECA_alpha_async"))
MODELS_DIR  = os.environ.get("MODELS_DIR", os.path.join(os.getcwd(), "models"))
DATA_DIR    = os.path.join(BASE_DIR, "ECA_Data_New")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "zero_shot_cot_tools")
INDEX_FILE  = os.path.join(BASE_DIR, "results", "fixed_test_indices.npy")  # same indices as the main protocol
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

# -- Reasoning + tool-use system prompt ---------------------------------------

SYSTEM_PROMPT = """You are an expert in cellular automata. You are given a space-time orbit of an Elementary Cellular Automaton (ECA) perturbed by alpha-asynchronous noise.

In alpha-asynchronous noise each cell independently updates with probability alpha per timestep.
Alpha is one of: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0

The orbit is a grid of 0s and 1s. Each row is one timestep.

Identify:
1. The ECA rule number (integer 0-255)
2. The alpha value

You may reason step by step. Take as much space as you need to think through the problem.

You also have access to a Python tool. If you want to compute something (e.g. count how
often each 3-cell neighbourhood pattern maps to each output bit, estimate per-cell change
rates, or any other statistic), write a code block like this:

```python
# your code here
```

The orbit is available to your code as a numpy array called `orbit`, with shape (100, 20),
dtype int (0s and 1s), where orbit[t, i] is the state of cell i at timestep t. The following
are already available to your code, pre-imported -- do not use import statements, they will
not work in this sandbox: np (numpy), math, collections, Counter, defaultdict. Anything you
print() will be shown back to you. You may use the tool as many times as you find helpful
before answering.

When you are ready to answer, end your response with this exact JSON on its own line:
{"rule": <integer 0-255>, "alpha": <float>}
"""

CODE_BLOCK_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
FINAL_JSON_RE = re.compile(r'\{\s*"rule"\s*:\s*(\d+)\s*,\s*"alpha"\s*:\s*([\d.]+)\s*\}')

MAX_TOOL_CALLS   = 5
MAX_NEW_TOKENS   = 2048     # per generation turn
TOOL_TIMEOUT_SEC = 5


# -- Sandboxed tool execution ---------------------------------------------------

class _ToolTimeout(Exception):
    pass

def _timeout_handler(signum, frame):
    raise _ToolTimeout()

def run_tool_code(code, orbit):
    """Execute model-emitted code with the orbit available, capturing stdout.
    Restricted builtins (no file/network/process access) and a wall-clock
    timeout."""
    safe_builtins = {
        "print": print, "range": range, "len": len, "sum": sum,
        "min": min, "max": max, "abs": abs, "round": round,
        "enumerate": enumerate, "zip": zip, "list": list, "dict": dict,
        "set": set, "tuple": tuple, "float": float, "int": int, "str": str,
        "bool": bool, "sorted": sorted, "any": any, "all": all,
    }
    local_ns = {
        "orbit": orbit, "np": np,
        "math": math, "collections": collections,
        "Counter": collections.Counter, "defaultdict": collections.defaultdict,
    }
    buf = io.StringIO()
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(TOOL_TIMEOUT_SEC)
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, {"__builtins__": safe_builtins}, local_ns)
        out = buf.getvalue()
        if not out.strip():
            out = "(no output -- did you forget to print()?)"
        return out[:4000]   # cap print output
    except _ToolTimeout:
        return f"[Tool error: execution exceeded {TOOL_TIMEOUT_SEC}s and was stopped]"
    except Exception as e:
        return f"[Tool error: {type(e).__name__}: {e}]"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# -- Helpers --------------------------------------------------------------------

def orbit_to_str(orbit):
    return "\n".join("".join(str(int(c)) for c in row) for row in orbit)

def make_user_prompt(orbit):
    return (f"Space-time orbit ({T} rows x {W} cells):\n\n"
            f"{orbit_to_str(orbit)}\n\nWhat is the ECA rule number and alpha value?")

def parse_response(response):
    m = FINAL_JSON_RE.search(response)
    if m:
        try:
            rule  = int(m.group(1))
            alpha = float(m.group(2))
            if 0 <= rule <= 255:
                alpha = min(ALPHA_VALUES, key=lambda a: abs(a - alpha))
                return rule, alpha
        except Exception:
            pass
    return None, None

# -- Data / checkpoint ----------------------------------------------------------

def get_fixed_test_indices(n_samples):
    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
    if os.path.exists(INDEX_FILE):
        idx = np.load(INDEX_FILE)
        return idx[:n_samples]
    rng      = np.random.default_rng(SEED)
    orbits   = np.load(os.path.join(DATA_DIR, "phase2", "test", "orbits.npy"))
    n        = min(n_samples, len(orbits))
    idx      = rng.choice(len(orbits), size=n, replace=False)
    np.save(INDEX_FILE, idx)
    return idx

def load_test_data(n_samples):
    idx      = get_fixed_test_indices(n_samples)
    orbits   = np.load(os.path.join(DATA_DIR, "phase2", "test", "orbits.npy"))
    rule_ids = np.load(os.path.join(DATA_DIR, "phase2", "test", "rule_ids.npy"))
    alphas   = np.load(os.path.join(DATA_DIR, "phase2", "test", "alphas.npy"))
    return orbits[idx], rule_ids[idx], alphas[idx]

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

def save_checkpoint(path, model_name, mode, n_total, results):
    n         = len(results)
    rule_acc  = sum(r["rule_correct"] for r in results) / n * 100 if n else 0
    alpha_acc = sum(r["alpha_ok"]     for r in results) / n * 100 if n else 0
    with open(path, "w") as f:
        json.dump({
            "model":          model_name,
            "mode":           mode,
            "n_evaluated":    n,
            "n_total":        n_total,
            "rule_accuracy":  rule_acc,
            "alpha_accuracy": alpha_acc,
            "parse_failures": sum(1 for r in results if r["pred_rule"] is None),
            "avg_tool_calls": sum(r.get("n_tool_calls", 0) for r in results) / n if n else 0,
            "timestamp":      datetime.now().isoformat(),
            "samples":        results,
        }, f, indent=2)

# -- Model ----------------------------------------------------------------------

def load_model(model_name):
    path = os.path.join(MODELS_DIR, model_name)
    print(f"  Loading {model_name} ...")
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    is_70b = any(x in model_name for x in ["70B", "72B"])
    if is_70b:
        print(f"  70B model detected -- loading in 8-bit quantisation")
        model = AutoModelForCausalLM.from_pretrained(
            path, load_in_8bit=True,
            device_map="auto", trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16,
            device_map="auto", trust_remote_code=True)
    model.eval()
    return tok, model


def generate_turn(tok, model, messages, max_new_tokens=MAX_NEW_TOKENS):
    try:
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\n\nASSISTANT:"
    inputs = tok(text, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=1.0,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()


def run_inference_with_tools(tok, model, orbit):
    """Multi-turn reasoning + tool-use loop. Returns (final_text, n_tool_calls,
    full_transcript)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": make_user_prompt(orbit)},
    ]
    transcript = []
    n_tool_calls = 0

    for turn in range(MAX_TOOL_CALLS + 1):
        response = generate_turn(tok, model, messages)
        transcript.append({"role": "assistant", "content": response})

        if FINAL_JSON_RE.search(response):
            return response, n_tool_calls, transcript

        code_match = CODE_BLOCK_RE.search(response)
        if code_match and turn < MAX_TOOL_CALLS:
            code = code_match.group(1)
            tool_output = run_tool_code(code, orbit)
            n_tool_calls += 1
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"[Tool output]\n{tool_output}"})
            transcript.append({"role": "tool", "content": tool_output})
            continue

        # No code block and no final JSON: ask once more for the final answer.
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content":
            "Please provide your final answer now as JSON: "
            '{"rule": <integer 0-255>, "alpha": <float>}'})

    # Final forced-answer turn
    response = generate_turn(tok, model, messages, max_new_tokens=128)
    transcript.append({"role": "assistant", "content": response})
    return response, n_tool_calls, transcript


# -- Evaluate -------------------------------------------------------------------

def evaluate_model(model_name, n_samples):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_path = os.path.join(RESULTS_DIR, f"{model_name}_zero_shot_cot_tools.json")

    print(f"\n{'='*65}\n  {model_name}  |  zero-shot + reasoning + tools  |  N={n_samples}\n{'='*65}")

    orbits, rule_ids, alphas = load_test_data(n_samples)
    results, done_set        = load_checkpoint(result_path)
    remaining = [(i, orbits[i], rule_ids[i], alphas[i])
                 for i in range(len(orbits)) if i not in done_set]

    if not remaining:
        print("  All samples done. Nothing to run.")
        return

    tok, model = load_model(model_name)
    t0 = time.time()

    for pos, (i, orbit, true_rule, true_alpha) in enumerate(remaining):
        raw, n_tools, transcript = run_inference_with_tools(tok, model, orbit)
        pred_rule, pred_alpha = parse_response(raw)
        r_ok = pred_rule == int(true_rule) if pred_rule is not None else False
        a_ok = abs(pred_alpha - float(true_alpha)) <= ALPHA_TOL if pred_alpha is not None else False

        results.append({
            "sample_idx":    i,
            "true_rule":     int(true_rule),
            "pred_rule":     pred_rule,
            "true_alpha":    float(true_alpha),
            "pred_alpha":    pred_alpha,
            "rule_correct":  r_ok,
            "alpha_ok":      a_ok,
            "raw_output":    raw,
            "n_tool_calls":  n_tools,
            "transcript":    transcript,
        })

        if (pos + 1) % 5 == 0 or pos + 1 == len(remaining):
            save_checkpoint(result_path, model_name,
                            "zero_shot_cot_tools", n_samples, results)
            done      = len(results)
            eta       = (time.time() - t0) / (pos + 1) * (len(remaining) - pos - 1)
            rule_acc  = sum(r["rule_correct"] for r in results) / done * 100
            alpha_acc = sum(r["alpha_ok"]     for r in results) / done * 100
            avg_tools = sum(r["n_tool_calls"] for r in results) / done
            print(f"  [{done:>4}/{n_samples}]  rule={rule_acc:.1f}%  "
                  f"alpha={alpha_acc:.1f}%  avg_tools={avg_tools:.1f}  ETA={eta:.0f}s", flush=True)

    n         = len(results)
    rule_acc  = sum(r["rule_correct"] for r in results) / n * 100
    alpha_acc = sum(r["alpha_ok"]     for r in results) / n * 100
    print(f"\n  FINAL  rule={rule_acc:.2f}%  alpha={alpha_acc:.2f}%")
    print(f"  Saved: {result_path}")
    del model; torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default="Qwen2.5-7B-Instruct")
    parser.add_argument("--n_samples", type=int, default=50)
    args    = parser.parse_args()
    models  = ALL_MODELS if args.model == "all" else [args.model]
    for m in models:
        evaluate_model(m, args.n_samples)

if __name__ == "__main__":
    main()
