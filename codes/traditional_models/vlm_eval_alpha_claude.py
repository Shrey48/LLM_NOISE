"""
vlm_eval_alpha_claude.py -- image-input evaluation of Claude Haiku 4.5 on alphaECA
====================================================================================
Same image-input condition as vlm_eval_alpha_gpt51.py, applied to Claude
Haiku 4.5 through the Anthropic API's native image input. No assistant-turn
prefill.

Usage:
  export ANTHROPIC_API_KEY=<your key>
  python vlm_eval_alpha_claude.py --data_dir ECA_Data_New --n_samples 30
"""

from __future__ import annotations
import argparse, json, os, re, time, base64, io
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

SEED = 42
ALPHA_TOL = 0.05
ALPHA_VALUES = [round(a * 0.1, 1) for a in range(1, 11)]
MODEL_ID = "claude-haiku-4-5-20251001"
MAX_TOKENS = 4000
MAX_RETRIES = 4
RETRY_BASE_DELAY = 5
CELL_SIZE = 8  # pixels per cell

SYSTEM_PROMPT = """You are an expert in cellular automata. You are given an IMAGE of a space-time orbit of an Elementary Cellular Automaton (ECA) perturbed by alpha-asynchronous noise.

In the image, WHITE pixels represent cell state 1, and BLACK pixels represent cell state 0. Each row of the image (top to bottom) is one timestep. Each column is one cell, from left to right. The grid is 100 rows x 20 columns, with each cell rendered as an 8x8 pixel block.

In alpha-asynchronous noise each cell independently updates with probability alpha per timestep.
Alpha is one of: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0

Identify:
1. The ECA rule number (integer 0-255)
2. The alpha value

You may reason step by step. When you are ready to answer, respond with this exact JSON on its own line:
{"rule": <integer 0-255>, "alpha": <float>}
"""

FINAL_JSON_RE = re.compile(r'\{\s*"rule"\s*:\s*(\d+)\s*,\s*"alpha"\s*:\s*([\d.]+)\s*\}')


def orbit_to_image_b64(orbit: np.ndarray) -> str:
    img_array = np.repeat(np.repeat(orbit.astype(np.uint8) * 255, CELL_SIZE, axis=0), CELL_SIZE, axis=1)
    img = Image.fromarray(img_array, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def parse_response(text: str):
    m = FINAL_JSON_RE.search(text or "")
    if m:
        try:
            rule = int(m.group(1)); alpha = float(m.group(2))
            if 0 <= rule <= 255:
                alpha = min(ALPHA_VALUES, key=lambda a: abs(a - alpha))
                return rule, alpha
        except Exception:
            pass
    return None, None


def get_fixed_test_indices(data_dir, n_samples, index_file):
    canonical = Path(data_dir) / "results" / "fixed_test_indices.npy"
    for candidate in [canonical, Path(index_file)]:
        if candidate.exists():
            idx = np.load(str(candidate))
            if len(idx) >= n_samples:
                print(f"    [index] Loaded {len(idx)} indices from {candidate}")
                return idx[:n_samples]
    Path(index_file).parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    orbits = np.load(os.path.join(data_dir, "phase2", "test", "orbits.npy"))
    n = min(n_samples, len(orbits))
    idx = rng.choice(len(orbits), size=n, replace=False)
    np.save(index_file, idx)
    return idx


def load_test_data(data_dir, n_samples, index_file):
    idx = get_fixed_test_indices(data_dir, n_samples, index_file)
    test_base = os.path.join(data_dir, "phase2", "test")
    orbits = np.load(os.path.join(test_base, "orbits.npy"))
    rules = np.load(os.path.join(test_base, "rule_ids.npy"))
    alphas = np.load(os.path.join(test_base, "alphas.npy"))
    return orbits[idx], rules[idx].astype(int), alphas[idx].astype(float)


def load_checkpoint(path):
    if not os.path.exists(path):
        return [], set()
    try:
        with open(path) as f:
            data = json.load(f)
        samples = data.get("samples", [])
        succeeded = [s for s in samples if not s.get("api_error", False)]
        failed_count = len(samples) - len(succeeded)
        done = {s["sample_idx"] for s in succeeded}
        print(f"    Resuming: {len(done)} samples successfully completed.")
        if failed_count > 0:
            print(f"    {failed_count} previously-errored samples will be retried.")
        return succeeded, done
    except Exception:
        return [], set()


def save_checkpoint(path, model_ver, n_total, results):
    n = len(results)
    rule_acc = sum(r["rule_correct"] for r in results) / n * 100 if n else 0.0
    alpha_acc = sum(r["alpha_ok"] for r in results) / n * 100 if n else 0.0
    with open(path, "w") as f:
        json.dump({
            "model": "claude-haiku-4-5", "model_version": model_ver, "condition": "vlm_image_input",
            "prefill_removed": True, "n_evaluated": n, "n_total": n_total,
            "rule_accuracy": rule_acc, "alpha_accuracy": alpha_acc,
            "api_errors": sum(1 for r in results if r.get("api_error", False)),
            "timestamp": datetime.now(timezone.utc).isoformat(), "samples": results,
        }, f, indent=2)


class ClaudeVisionClient:
    def __init__(self, model_id=MODEL_ID):
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set")
        self.client = _anthropic.Anthropic(api_key=key)
        self.model_id = model_id

    def run(self, orbit):
        img_b64 = orbit_to_image_b64(orbit)
        messages = [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
            {"type": "text", "text": "What is the ECA rule number and alpha value shown in this image?"},
        ]}]
        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = self.client.messages.create(
                    model=self.model_id, max_tokens=MAX_TOKENS,
                    system=SYSTEM_PROMPT, messages=messages, temperature=0.0)
                model_ver = resp.model or self.model_id
                text = "\n".join(b.text for b in resp.content if b.type == "text")
                return text, model_ver, False
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"    [retry {attempt+1}/{MAX_RETRIES}] {type(e).__name__}, waiting {delay}s...", flush=True)
                    time.sleep(delay)
        return f"[error after {MAX_RETRIES} retries: {type(last_error).__name__}: {last_error}]", self.model_id, True


def evaluate(data_dir, n_samples, output_dir, sleep_s):
    os.makedirs(output_dir, exist_ok=True)
    index_file = os.path.join(output_dir, "fixed_test_indices.npy")
    result_path = os.path.join(output_dir, "claude_haiku_vlm_image.json")

    print(f"\n{'='*65}\n  Claude Haiku 4.5  |  VLM image input (no prefill)  |  N={n_samples}\n{'='*65}")
    orbits, rule_ids, alphas = load_test_data(data_dir, n_samples, index_file)
    results, done_set = load_checkpoint(result_path)
    remaining = [(i, orbits[i], rule_ids[i], alphas[i]) for i in range(len(orbits)) if i not in done_set]
    if not remaining:
        print("  All samples done."); return

    client = ClaudeVisionClient()
    t0 = time.time()
    for pos, (i, orbit, true_rule, true_alpha) in enumerate(remaining):
        text, model_ver, api_error = client.run(orbit)
        pred_rule, pred_alpha = parse_response(text)
        r_ok = pred_rule == int(true_rule) if pred_rule is not None else False
        a_ok = abs(pred_alpha - float(true_alpha)) <= ALPHA_TOL if pred_alpha is not None else False

        results.append({
            "sample_idx": int(i), "true_rule": int(true_rule), "pred_rule": pred_rule,
            "true_alpha": float(true_alpha), "pred_alpha": pred_alpha,
            "rule_correct": r_ok, "alpha_ok": a_ok, "raw_output": text, "api_error": api_error,
        })

        if (pos + 1) % 5 == 0 or pos + 1 == len(remaining):
            save_checkpoint(result_path, model_ver, n_samples, results)
            done = len(results)
            eta = (time.time() - t0) / (pos + 1) * (len(remaining) - pos - 1)
            rule_acc = sum(r["rule_correct"] for r in results) / done * 100
            alpha_acc = sum(r["alpha_ok"] for r in results) / done * 100
            errors = sum(1 for r in results if r["api_error"])
            print(f"  [{done:>4}/{n_samples}]  rule={rule_acc:.1f}%  alpha={alpha_acc:.1f}%  errors={errors}  ETA={eta:.0f}s", flush=True)
        if sleep_s > 0:
            time.sleep(sleep_s)

    n = len(results)
    rule_acc = sum(r["rule_correct"] for r in results) / n * 100 if n else 0
    alpha_acc = sum(r["alpha_ok"] for r in results) / n * 100 if n else 0
    print(f"\n  FINAL  rule={rule_acc:.2f}%  alpha={alpha_acc:.2f}%")
    print(f"  Saved: {result_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="ECA_Data_New")
    ap.add_argument("--n_samples", type=int, default=30)
    ap.add_argument("--output_dir", default="vlm_results")
    ap.add_argument("--sleep_s", type=float, default=0.5)
    args = ap.parse_args()
    evaluate(args.data_dir, args.n_samples, args.output_dir, args.sleep_s)


if __name__ == "__main__":
    main()
