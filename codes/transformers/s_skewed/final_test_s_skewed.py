"""
FINAL_TEST_SKEW.py  --  Split-Correct Final Test
==================================================

Tests Phase 2 model on ONLY the 77 held-out TEST rules
(seed=42 split, paper-defined) x ALL 20 s-values x 20 orbits.

KEY FIX vs original:
  Original: ALL_RULES = list(range(256))  -> 179 train rules included (wrong)
  This    : TEST_RULES = 77 held-out rules only              (correct)

Split is reproduced identically to DATAGEN_SKEW:
    rng_split = np.random.default_rng(42)
    shuffled  = rng_split.permutation(256).tolist()
    test_rules = sorted(shuffled[179:])   # last 77

If ECA_Data_Skew/test_rules.npy exists it is cross-checked for safety.

TOTAL PREDICTIONS: 77 rules x 20 s-values x 20 orbits = 30,800

Results saved to:
    split test result.txt   -- human-readable full report
    split test result.json  -- machine-readable structured results

Usage: python FINAL_TEST_SKEW.py
"""

import torch
import numpy as np
import os
import sys
import json
import time
from collections import defaultdict
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from MODEL_SKEW import (ECANetSkew, simulate_eca_skewed, orbit_to_tokens,
                        rule_to_bits, bits_to_rule, build_rule_table,
                        W, T, N_TOK, N_BITS, TOKEN_DIM, S_VALUES, N_S_CLASSES,
                        s_logits_to_value)

# ── Config ────────────────────────────────────────────────────────────────────
ORBITS_PER_PAIR = 20
RANDOM_SEED     = 42
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "checkpoints_skew", "phase2_best.pt")
DATA_DIR        = os.path.join(SCRIPT_DIR, "ECA_Data_Skew")
PRINT_FULL_ROWS = True


# ── Paper-defined 77 test rules (seed=42, identical to DATAGEN_SKEW) ─────────
def get_test_rules():
    """
    Reproduce exact split from DATAGEN_SKEW:
        rng_split = np.random.default_rng(42)
        shuffled  = rng_split.permutation(256).tolist()
        test_rules = sorted(shuffled[179:])

    Cross-checks against ECA_Data_Skew/test_rules.npy if it exists.
    """
    rng_split  = np.random.default_rng(42)
    shuffled   = rng_split.permutation(256).tolist()
    test_rules = sorted(shuffled[179:])
    assert len(test_rules) == 77, f"Expected 77 test rules, got {len(test_rules)}"

    saved_path = os.path.join(DATA_DIR, "test_rules.npy")
    if os.path.exists(saved_path):
        saved = sorted(np.load(saved_path).tolist())
        assert saved == test_rules, (
            f"Mismatch between recomputed and saved test_rules!\n"
            f"  Recomputed: {test_rules[:10]}...\n"
            f"  Saved     : {saved[:10]}..."
        )
        print(f"  ECA_Data_Skew/test_rules.npy cross-check: PASSED")
    else:
        print(f"  ECA_Data_Skew/test_rules.npy not found -- using recomputed split.")

    return test_rules

TEST_RULES = get_test_rules()   # 77 rules — never seen during training


# ── Device ────────────────────────────────────────────────────────────────────
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
print(f"Test rules : {len(TEST_RULES)} (held-out, seed=42 split)")
print(f"Orbits/pair: {ORBITS_PER_PAIR}")
print(f"Total preds: {len(TEST_RULES) * len(S_VALUES) * ORBITS_PER_PAIR:,}\n")


# ── Hybrid S Estimator ────────────────────────────────────────────────────────

def hybrid_s_estimate(orbit, predicted_rule_num):
    """
    Direct s estimation from orbit + predicted rule.

    Under s-skewed ECA: exactly s cells update per step.
    alpha_effective = s/W  =>  s = round(did_change/should_change * W)

    Returns None if rule never produces cell changes (degenerate rule).
    """
    rule_table    = build_rule_table(predicted_rule_num)
    should_change = 0
    did_change    = 0
    for t in range(len(orbit) - 1):
        state            = orbit[t].astype(np.uint8)
        left             = np.roll(state,  1)
        right            = np.roll(state, -1)
        idx              = (4 * left + 2 * state + right).astype(np.uint8)
        rule_output      = rule_table[idx]
        wants_change     = (rule_output != state)
        actually_changed = (orbit[t + 1].astype(np.uint8) != state)
        should_change   += int(wants_change.sum())
        did_change      += int((wants_change & actually_changed).sum())
    if should_change == 0:
        return None
    return max(1, min(W, round((did_change / should_change) * W)))


def estimate_s_from_orbits(orbits, predicted_rule_num):
    """Median of hybrid estimates across multiple orbits."""
    estimates = []
    for orbit in orbits:
        est = hybrid_s_estimate(orbit, predicted_rule_num)
        if est is not None:
            estimates.append(est)
    if not estimates:
        return None, True   # all degenerate -- fallback to neural
    return max(1, min(W, int(round(float(np.median(estimates)))))), False


# ── Load Model ────────────────────────────────────────────────────────────────

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


# ── Run Split Test ────────────────────────────────────────────────────────────

def run_split_test(model):
    rng = np.random.default_rng(seed=RANDOM_SEED)

    rules_order = rng.permutation(TEST_RULES).tolist()
    s_orders    = {r: rng.permutation(S_VALUES).tolist() for r in rules_order}

    total_pairs = len(TEST_RULES) * len(S_VALUES)

    print("=" * 65)
    print(f"  SPLIT FINAL TEST  --  77 Test Rules x {len(S_VALUES)} S-values x {ORBITS_PER_PAIR} orbits")
    print("=" * 65)
    print(f"  Rules tested  : {len(TEST_RULES)}  (held-out test split, seed=42)")
    print(f"  S values/rule : {len(S_VALUES)}  (s=1..{W})")
    print(f"  Orbits/pair   : {ORBITS_PER_PAIR}")
    print(f"  Total pairs   : {total_pairs}")
    print(f"  Total orbits  : {total_pairs * ORBITS_PER_PAIR:,}")
    print(f"  Test rules    : {TEST_RULES[:10]}... (77 total)")
    print()

    if PRINT_FULL_ROWS:
        print(f"  {'Rule':>4} | {'s':>3} | {'Voted':>5} | {'ROK':>3} | "
              f"{'BitAcc':>6} | {'NeuS':>4} | {'FinS':>4} | "
              f"{'Err':>3} | {'OK':>2} | Meth")
        print(f"  {'-'*65}")

    all_results  = []
    rule_summary = defaultdict(lambda: {"rule_pass":0,"total":0,"s_errs":[],"s_ok":0})
    s_summary    = defaultdict(lambda: {"rule_pass":0,"total":0,"s_errs":[],"s_ok":0})

    t_start    = time.time()
    pair_count = 0

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

            if PRINT_FULL_ROWS:
                print(f"  {rule_num:>4} | {s_val:>3} | {voted_rule:>5} | "
                      f"{'Y' if rule_correct else 'N':>3} | "
                      f"{avg_bit_acc*100:5.1f}% | {neural_s_pred:>4} | {final_s:>4} | "
                      f"{s_err:>3} | {'Y' if s_ok else 'N':>2} | {method}")

            result = {
                "rule": rule_num, "s": s_val, "voted_rule": voted_rule,
                "rule_correct": rule_correct, "bit_acc": avg_bit_acc,
                "neural_s": neural_s_pred, "final_s": final_s,
                "s_err": s_err, "s_ok": s_ok, "method": method,
            }
            all_results.append(result)

            for summary, key in [(rule_summary, rule_num), (s_summary, s_val)]:
                summary[key]["total"]     += 1
                summary[key]["rule_pass"] += int(rule_correct)
                summary[key]["s_errs"].append(s_err)
                summary[key]["s_ok"]      += int(s_ok)

            if pair_count % 100 == 0 or pair_count == total_pairs:
                elapsed = time.time() - t_start
                eta     = (elapsed / pair_count) * (total_pairs - pair_count) \
                          if pair_count < total_pairs else 0
                r_acc   = sum(r["rule_correct"] for r in all_results) / len(all_results) * 100
                s_acc   = sum(r["s_ok"]         for r in all_results) / len(all_results) * 100
                print(f"\n  [{pair_count}/{total_pairs}  "
                      f"rule={r_acc:.1f}%  s_exact={s_acc:.1f}%  "
                      f"ETA={int(eta)}s]\n")

    elapsed = time.time() - t_start
    total_n = len(all_results)

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    rule_pass = sum(r["rule_correct"] for r in all_results)
    s_ok_n    = sum(r["s_ok"]         for r in all_results)
    hybrid_n  = sum(1 for r in all_results if r["method"] == "hybrid")
    avg_mae   = float(np.mean([r["s_err"] for r in all_results]))
    s_off1    = sum(1 for r in all_results if r["s_err"] <= 1) / total_n * 100
    s_off2    = sum(1 for r in all_results if r["s_err"] <= 2) / total_n * 100

    # ── Build summary lines ───────────────────────────────────────────────────
    summary_lines = []
    summary_lines.append("=" * 65)
    summary_lines.append("  SPLIT TEST RESULT  --  77 Held-Out Test Rules x 20 S-values")
    summary_lines.append("=" * 65)
    summary_lines.append(f"  Evaluated at        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"  Test rules (77)     : {TEST_RULES}")
    summary_lines.append(f"  Total (rule,s) pairs: {total_n}")
    summary_lines.append(f"  Orbits per pair     : {ORBITS_PER_PAIR}")
    summary_lines.append(f"  Total orbits        : {total_n * ORBITS_PER_PAIR:,}")
    summary_lines.append(f"  Time elapsed        : {elapsed:.1f}s")
    summary_lines.append("")
    summary_lines.append(f"  Rule exact match : {rule_pass}/{total_n} ({rule_pass/total_n*100:.2f}%)")
    summary_lines.append(f"  S exact (err=0)  : {s_ok_n}/{total_n} ({s_ok_n/total_n*100:.2f}%)")
    summary_lines.append(f"  S MAE            : {avg_mae:.3f}")
    summary_lines.append(f"  S ±1 acc         : {s_off1:.2f}%")
    summary_lines.append(f"  S ±2 acc         : {s_off2:.2f}%")
    summary_lines.append(f"  Hybrid/Neural    : {hybrid_n}/{total_n - hybrid_n}")

    # ── Per-S breakdown ───────────────────────────────────────────────────────
    summary_lines.append("")
    summary_lines.append(f"  {'='*60}")
    summary_lines.append(f"  PER-S BREAKDOWN  (across 77 test rules)")
    summary_lines.append(f"  {'='*60}")
    summary_lines.append(f"  {'S':>3} | {'N':>4} | {'Rule%':>6} | {'MAE':>5} | "
                         f"{'Exact%':>6} | {'±1%':>5}")
    summary_lines.append(f"  {'-'*45}")
    for sv in S_VALUES:
        d = s_summary[sv]
        n = d["total"]
        if n == 0:
            continue
        summary_lines.append(
            f"  {sv:>3} | {n:>4} | {d['rule_pass']/n*100:5.1f}% | "
            f"{float(np.mean(d['s_errs'])):5.2f} | {d['s_ok']/n*100:5.1f}% | "
            f"{sum(1 for e in d['s_errs'] if e<=1)/n*100:4.1f}%"
        )

    # ── Per-rule breakdown ────────────────────────────────────────────────────
    summary_lines.append("")
    summary_lines.append(f"  {'='*65}")
    summary_lines.append(f"  PER-RULE BREAKDOWN  (all 77 test rules)")
    summary_lines.append(f"  {'='*65}")
    summary_lines.append(f"  {'Rule':>4} | {'Pairs':>5} | {'RulePass':>9} | "
                         f"{'MAE':>5} | {'Exact%':>6} | {'±1%':>5} | Status")
    summary_lines.append(f"  {'-'*60}")

    problem_rules = []
    for rule_num in sorted(rule_summary.keys()):
        d      = rule_summary[rule_num]
        n      = d["total"]
        rp     = d["rule_pass"]
        mae    = float(np.mean(d["s_errs"]))
        ex_pct = d["s_ok"] / n * 100
        off1   = sum(1 for e in d["s_errs"] if e <= 1) / n * 100
        status = "OK" if (rp == n and ex_pct >= 80) else "PROBLEM"
        if status == "PROBLEM":
            problem_rules.append(rule_num)
        summary_lines.append(
            f"  {rule_num:>4} | {n:>5} | {rp:>4}/{n:<4} | "
            f"{mae:5.2f} | {ex_pct:5.1f}% | {off1:4.1f}% | {status}"
        )

    summary_lines.append("")
    if not problem_rules:
        summary_lines.append("  All 77 test rules performed perfectly!")
    else:
        summary_lines.append(f"  {len(problem_rules)} rules with issues: {problem_rules}")

    # ── S accuracy by range ───────────────────────────────────────────────────
    summary_lines.append("")
    summary_lines.append("  S accuracy by range:")
    for label, filt in [
        ("Low  (s=1-5) ", lambda r: r["s"] <= 5),
        ("Mid  (s=6-15)", lambda r: 6 <= r["s"] <= 15),
        ("High (s=16-20)", lambda r: r["s"] >= 16),
    ]:
        items = [r for r in all_results if filt(r)]
        if not items:
            continue
        n    = len(items)
        ok   = sum(r["s_ok"]        for r in items)
        mae  = float(np.mean([r["s_err"]      for r in items]))
        off1 = sum(1 for r in items if r["s_err"] <= 1)
        rp   = sum(r["rule_correct"] for r in items)
        summary_lines.append(
            f"    {label}  N={n:>4}  rule={rp/n*100:.1f}%  "
            f"sExact={ok/n*100:.1f}%  s±1={off1/n*100:.1f}%  MAE={mae:.3f}"
        )

    summary_lines.append("")
    summary_lines.append("=" * 65)
    summary_lines.append("  SPLIT TEST COMPLETE")
    summary_lines.append("=" * 65)

    # Print to console
    for line in summary_lines:
        print(line)

    # ── Save results ──────────────────────────────────────────────────────────
    txt_path  = os.path.join(SCRIPT_DIR, "split test result.txt")
    json_path = os.path.join(SCRIPT_DIR, "split test result.json")

    with open(txt_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\n  [Saved] {txt_path}")

    json_out = {
        "meta": {
            "evaluated_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model":           "ECANetSkew (s-skewed)",
            "checkpoint":      CHECKPOINT_PATH,
            "test_rules":      TEST_RULES,
            "num_test_rules":  len(TEST_RULES),
            "orbits_per_pair": ORBITS_PER_PAIR,
            "total_pairs":     total_n,
            "seed":            RANDOM_SEED,
            "split_seed":      42,
        },
        "aggregate": {
            "rule_exact_match_pct": round(rule_pass / total_n * 100, 4),
            "s_exact_pct":          round(s_ok_n   / total_n * 100, 4),
            "s_mae":                round(avg_mae, 4),
            "s_off1_pct":           round(s_off1, 4),
            "s_off2_pct":           round(s_off2, 4),
            "hybrid_used":          hybrid_n,
            "neural_fallback_used": total_n - hybrid_n,
            "time_elapsed_s":       round(elapsed, 1),
        },
        "per_s": {
            str(sv): {
                "n":            s_summary[sv]["total"],
                "rule_pass":    s_summary[sv]["rule_pass"],
                "rule_acc_pct": round(s_summary[sv]["rule_pass"] /
                                      max(1, s_summary[sv]["total"]) * 100, 2),
                "mae":          round(float(np.mean(s_summary[sv]["s_errs"])), 4)
                                if s_summary[sv]["s_errs"] else 0,
                "exact_pct":    round(s_summary[sv]["s_ok"] /
                                      max(1, s_summary[sv]["total"]) * 100, 2),
                "off1_pct":     round(sum(1 for e in s_summary[sv]["s_errs"] if e <= 1) /
                                      max(1, s_summary[sv]["total"]) * 100, 2),
            }
            for sv in S_VALUES
        },
        "per_rule": {
            str(rule_num): {
                "n":            rule_summary[rule_num]["total"],
                "rule_pass":    rule_summary[rule_num]["rule_pass"],
                "rule_acc_pct": round(rule_summary[rule_num]["rule_pass"] /
                                      max(1, rule_summary[rule_num]["total"]) * 100, 2),
                "mae":          round(float(np.mean(rule_summary[rule_num]["s_errs"])), 4)
                                if rule_summary[rule_num]["s_errs"] else 0,
                "exact_pct":    round(rule_summary[rule_num]["s_ok"] /
                                      max(1, rule_summary[rule_num]["total"]) * 100, 2),
            }
            for rule_num in sorted(rule_summary.keys())
        },
        "all_results": all_results,
    }

    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"  [Saved] {json_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print(f"  ECANetSkew  --  Split-Correct Final Test (T={T})")
    print("=" * 65)
    print(f"  77 held-out test rules (seed=42) x {len(S_VALUES)} s-values x {ORBITS_PER_PAIR} orbits")
    print(f"  Fresh data generated at runtime")
    print(f"  S: hybrid estimator primary; neural fallback for degenerate rules")
    print(f"  Results saved to: 'split test result.txt' and 'split test result.json'\n")

    model = load_model()
    run_split_test(model)
