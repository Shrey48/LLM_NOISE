"""
FINAL_TEST_SKEW.py  --  Comprehensive Final Test (Enhanced)
=============================================================

Tests Phase 2 model on ALL 256 rules x ALL 20 s-values.
Uses 5-feature tokens and T=200.
Model outputs 3 values: rule_logits, s_logits, s_reg.

Hybrid s estimator: estimated_s = round((did_change/should_change) * W)

Usage: python FINAL_TEST_SKEW.py
"""

import torch
import numpy as np
import os
import sys
import time
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from MODEL_SKEW import (ECANetSkew, simulate_eca_skewed, orbit_to_tokens,
                        rule_to_bits, bits_to_rule, build_rule_table,
                        W, T, N_TOK, N_BITS, TOKEN_DIM, S_VALUES, N_S_CLASSES,
                        s_logits_to_value)

ORBITS_PER_PAIR  = 20
RANDOM_SEED      = 42
ALL_RULES        = list(range(256))
CHECKPOINT_PATH  = os.path.join(SCRIPT_DIR, "checkpoints_skew", "phase2_best.pt")
LOG_PATH         = os.path.join(SCRIPT_DIR, "final_test_skew.log")

# ── Tee: write to both terminal and log file ──────────────────────────────────
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

_log_file = open(LOG_PATH, "w", buffering=1, encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, _log_file)
sys.stderr = Tee(sys.__stderr__, _log_file)
# ─────────────────────────────────────────────────────────────────────────────

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Device: CUDA -- {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Device: Apple MPS")
else:
    device = torch.device("cpu")
    print("Device: CPU")

print(f"Checkpoint : {CHECKPOINT_PATH}")
print(f"Log file   : {LOG_PATH}")
print(f"Orbits/pair: {ORBITS_PER_PAIR}")
print(f"Total preds: {256 * len(S_VALUES) * ORBITS_PER_PAIR:,}\n")


def hybrid_s_estimate(orbit, predicted_rule_num):
    rule_table    = build_rule_table(predicted_rule_num)
    should_change = 0
    did_change    = 0
    for t in range(len(orbit) - 1):
        state = orbit[t].astype(np.uint8)
        left  = np.roll(state,  1)
        right = np.roll(state, -1)
        idx   = (4 * left + 2 * state + right).astype(np.uint8)
        rule_output      = rule_table[idx]
        wants_change     = (rule_output != state)
        actually_changed = (orbit[t+1].astype(np.uint8) != state)
        should_change   += int(wants_change.sum())
        did_change      += int((wants_change & actually_changed).sum())
    if should_change == 0:
        return None
    return max(1, min(W, round((did_change / should_change) * W)))


def estimate_s_from_orbits(orbits, predicted_rule_num):
    estimates = []
    for orbit in orbits:
        est = hybrid_s_estimate(orbit, predicted_rule_num)
        if est is not None:
            estimates.append(est)
    if not estimates:
        return None, True
    return max(1, min(W, int(round(float(np.median(estimates)))))), False


def load_model():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: {CHECKPOINT_PATH} not found. Run TRAIN_SKEW.py first.")
        sys.exit(1)
    model = ECANetSkew().to(device)
    ckpt  = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded: phase={ckpt.get('phase','?')}  "
          f"epoch={ckpt.get('epoch','?')}  "
          f"best_exact={ckpt.get('best_exact',0):.2f}%\n")
    return model


def run_final_tests(model):
    rng = np.random.default_rng(seed=RANDOM_SEED)
    rules_order = rng.permutation(ALL_RULES).tolist()
    s_orders    = {r: rng.permutation(S_VALUES).tolist() for r in rules_order}

    print("=" * 65)
    print(f"  FINAL TEST  --  256 Rules x {len(S_VALUES)} S-values x {ORBITS_PER_PAIR} orbits")
    print("=" * 65)

    all_results   = []
    rule_summary  = defaultdict(lambda: {"rule_pass":0,"total":0,"s_errs":[],"s_ok":0})
    s_summary     = defaultdict(lambda: {"rule_pass":0,"total":0,"s_errs":[],"s_ok":0})

    t_start    = time.time()
    pair_count = 0
    total_pairs = 256 * len(S_VALUES)

    print(f"  {'Rule':>4} | {'s':>3} | {'Voted':>5} | {'ROK':>3} | "
          f"{'BitAcc':>6} | {'NeuS':>4} | {'FinS':>4} | "
          f"{'Err':>3} | {'OK':>2} | Meth")
    print(f"  {'-'*65}")

    for rule_num in rules_order:
        for s_val in s_orders[rule_num]:
            pair_count += 1

            orbits = []
            for _ in range(ORBITS_PER_PAIR):
                init = rng.integers(0, 2, size=W).astype(np.int8)
                while init.sum() == 0 or init.sum() == W:
                    init = rng.integers(0, 2, size=W).astype(np.int8)
                orbits.append(simulate_eca_skewed(rule_num, init, T, s_val, rng))

            rule_preds, neural_s_vals, bit_accs = [], [], []

            with torch.no_grad():
                for orbit in orbits:
                    tokens = orbit_to_tokens(orbit)
                    x = torch.tensor(tokens, dtype=torch.float32).unsqueeze(0).to(device)
                    rule_logits, s_logits, _ = model(x)

                    pred_bits = (torch.sigmoid(rule_logits) >= 0.5).float()
                    true_bits = torch.tensor(
                        rule_to_bits(rule_num), dtype=torch.float32
                    ).unsqueeze(0).to(device)
                    bit_accs.append((pred_bits == true_bits).float().mean().item())
                    rule_preds.append(bits_to_rule(pred_bits[0]))
                    neural_s_vals.append(s_logits_to_value(s_logits).item())

            voted_rule    = max(set(rule_preds), key=rule_preds.count)
            avg_bit_acc   = float(np.mean(bit_accs))
            neural_s_pred = max(1, min(W, int(round(float(np.median(neural_s_vals))))))
            rule_correct  = (voted_rule == rule_num)

            hybrid_est, fallback = estimate_s_from_orbits(orbits, voted_rule)
            if fallback or hybrid_est is None:
                final_s, method = neural_s_pred, "neural"
            else:
                final_s, method = hybrid_est, "hybrid"

            s_err = abs(final_s - s_val)
            s_ok  = (s_err == 0)

            print(f"  {rule_num:>4} | {s_val:>3} | {voted_rule:>5} | "
                  f"{'Y' if rule_correct else 'N':>3} | "
                  f"{avg_bit_acc*100:5.1f}% | {neural_s_pred:>4} | {final_s:>4} | "
                  f"{s_err:>3} | {'Y' if s_ok else 'N':>2} | {method}")

            all_results.append({
                "rule": rule_num, "s": s_val, "voted_rule": voted_rule,
                "rule_correct": rule_correct, "bit_acc": avg_bit_acc,
                "neural_s": neural_s_pred, "final_s": final_s,
                "s_err": s_err, "s_ok": s_ok, "method": method,
            })

            for summary, key in [(rule_summary, rule_num), (s_summary, s_val)]:
                summary[key]["total"]     += 1
                summary[key]["rule_pass"] += int(rule_correct)
                summary[key]["s_errs"].append(s_err)
                summary[key]["s_ok"]      += int(s_ok)

            if pair_count % 200 == 0 or pair_count == total_pairs:
                elapsed = time.time() - t_start
                eta     = (elapsed / pair_count) * (total_pairs - pair_count)
                r_acc   = sum(r["rule_correct"] for r in all_results) / len(all_results) * 100
                s_acc   = sum(r["s_ok"]         for r in all_results) / len(all_results) * 100
                print(f"\n  [{pair_count}/{total_pairs}  "
                      f"rule={r_acc:.1f}%  s_exact={s_acc:.1f}%  "
                      f"ETA={int(eta)}s]\n")

    elapsed = time.time() - t_start
    total_n = len(all_results)

    # Overall
    rule_pass = sum(r["rule_correct"] for r in all_results)
    s_ok_n    = sum(r["s_ok"]         for r in all_results)
    hybrid_n  = sum(1 for r in all_results if r["method"] == "hybrid")
    avg_mae   = float(np.mean([r["s_err"] for r in all_results]))
    s_off1    = sum(1 for r in all_results if r["s_err"] <= 1) / total_n * 100
    s_off2    = sum(1 for r in all_results if r["s_err"] <= 2) / total_n * 100

    print(f"\n{'='*65}")
    print(f"  OVERALL: {total_n} pairs  {elapsed:.0f}s")
    print(f"{'='*65}")
    print(f"  Rule exact : {rule_pass}/{total_n} ({rule_pass/total_n*100:.2f}%)")
    print(f"  S exact    : {s_ok_n}/{total_n} ({s_ok_n/total_n*100:.2f}%)")
    print(f"  S MAE      : {avg_mae:.3f}")
    print(f"  S +-1      : {s_off1:.2f}%   S +-2: {s_off2:.2f}%")
    print(f"  Hybrid/Neural: {hybrid_n}/{total_n-hybrid_n}")

    # Per-S breakdown
    print(f"\n  {'S':>3} | {'N':>4} | {'Rule%':>6} | {'MAE':>5} | {'Exact%':>6} | {'+-1%':>5}")
    print(f"  {'-'*45}")
    for sv in S_VALUES:
        d = s_summary[sv]
        n = d["total"]
        if n == 0: continue
        print(f"  {sv:>3} | {n:>4} | {d['rule_pass']/n*100:5.1f}% | "
              f"{np.mean(d['s_errs']):5.2f} | {d['s_ok']/n*100:5.1f}% | "
              f"{sum(1 for e in d['s_errs'] if e<=1)/n*100:4.1f}%")

    # Problem rules
    problem = [r for r in sorted(rule_summary) if
               rule_summary[r]["rule_pass"] < rule_summary[r]["total"] or
               rule_summary[r]["s_ok"] / rule_summary[r]["total"] < 0.8]
    if problem:
        print(f"\n  {len(problem)} problem rules: {problem[:30]}")
        if len(problem) > 30:
            print(f"    ... and {len(problem)-30} more")
    else:
        print(f"\n  All 256 rules perfect!")

    # By range
    print(f"\n  S accuracy by range:")
    for label, filt in [
        ("Low  (1-5) ", lambda r: r["s"] <= 5),
        ("Mid  (6-15)", lambda r: 6 <= r["s"] <= 15),
        ("High (16-20)", lambda r: r["s"] >= 16),
    ]:
        items = [r for r in all_results if filt(r)]
        if not items: continue
        ok  = sum(r["s_ok"] for r in items)
        mae = np.mean([r["s_err"] for r in items])
        rp  = sum(r["rule_correct"] for r in items)
        print(f"    {label}  N={len(items):>4}  rule={rp/len(items)*100:.1f}%  "
              f"sExact={ok/len(items)*100:.1f}%  MAE={mae:.3f}")

    print(f"\n{'='*65}")
    print(f"  FINAL TEST COMPLETE")
    print(f"{'='*65}")

    _log_file.close()
    print(f"\nFull log saved to: {LOG_PATH}", file=sys.__stdout__)


if __name__ == "__main__":
    print("=" * 65)
    print(f"  ECANet s-Skewed -- Final Test (Enhanced, T={T})")
    print("=" * 65)
    model = load_model()
    run_final_tests(model)