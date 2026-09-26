"""
algorithmic_baseline_alpha_skew.py -- zero-training CA-identification baseline
=================================================================================
A zero-parameter, zero-training identification method for alphaECA and
skewECA (Appendix, zero-training algorithmic baseline).

RULE IDENTIFICATION (identical logic for both datasets):
  Under alpha-asynchronous and s-skewed noise, a cell either keeps its state
  or takes the rule output. For each of the 8 neighbourhood patterns, a
  transition 0 -> 1 therefore implies that the rule outputs 1 for that
  pattern, and a transition 1 -> 0 implies that it outputs 0. This
  direction-of-flip evidence is unaffected by how many cells the noise left
  un-updated. Patterns with tied or no evidence fall back to a majority vote
  over all observed outcomes for that pattern.

NOISE-PARAMETER ESTIMATION (dataset-specific):
  alphaECA: alpha = fraction of (cell, timestep) transitions that match the
            identified rule's prediction. This naive statistic does not
            correct for the rule's own dynamics, which conflate with the noise.
  skewECA:  s = the maximum, over all timesteps, of the circular spatial span
            of cells that changed value. Only cells inside the s-cell update
            window can change, so the observed span at any timestep is <= s,
            and its maximum over many timesteps approaches s.

Usage:
  python3 algorithmic_baseline_alpha_skew.py --dataset alpha --data_dir ECA_Data_New
  python3 algorithmic_baseline_alpha_skew.py --dataset skew  --data_dir ECA_Data_Skew
"""

import argparse, json, os, time
import numpy as np


# -- Rule identification (shared logic, both datasets) ------------------------

def identify_rule_by_lookup(orbit: np.ndarray) -> int:
    """orbit: [T, W] array of 0/1 states. Returns the identified rule number."""
    before, after = orbit[:-1].astype(np.int64), orbit[1:].astype(np.int64)
    left  = np.roll(before, 1, axis=1)
    right = np.roll(before, -1, axis=1)
    pattern_idx = left * 4 + before * 2 + right

    bits = []
    for p in range(8):
        mask0 = (pattern_idx == p) & (before == 0)
        mask1 = (pattern_idx == p) & (before == 1)
        up_from_0   = np.sum(after[mask0] == 1) if mask0.sum() > 0 else 0
        down_from_1 = np.sum(after[mask1] == 0) if mask1.sum() > 0 else 0
        if up_from_0 > down_from_1:
            bits.append(1)
        elif down_from_1 > up_from_0:
            bits.append(0)
        else:
            mask_all = (pattern_idx == p)
            bits.append(1 if mask_all.sum() > 0 and after[mask_all].mean() > 0.5 else 0)
    return sum(b << i for i, b in enumerate(bits))


def rule_to_table(rule_number: int):
    return [(rule_number >> i) & 1 for i in range(8)]


def apply_rule(state: np.ndarray, table):
    state = state.astype(np.int64)
    left = np.roll(state, 1, axis=-1)
    right = np.roll(state, -1, axis=-1)
    idx = (left * 4 + state * 2 + right).astype(np.int64)
    return np.array(table)[idx]


# -- alphaECA: naive match-rate noise estimator --------------------------------

def estimate_alpha_naive(orbit: np.ndarray, identified_rule: int) -> float:
    table = rule_to_table(identified_rule)
    before, after = orbit[:-1], orbit[1:]
    predicted = apply_rule(before, table)
    return float(np.mean(predicted == after))


ALPHA_VALUES = [round(a * 0.1, 1) for a in range(1, 11)]

def snap_alpha(x):
    return min(ALPHA_VALUES, key=lambda a: abs(a - x))


# -- skewECA: max-span noise estimator -----------------------------------------

def estimate_s_max_span(orbit: np.ndarray, w: int) -> int:
    before, after = orbit[:-1], orbit[1:]
    changed = (before != after)
    max_span = 0
    for t in range(changed.shape[0]):
        idx = np.where(changed[t])[0]
        if len(idx) == 0:
            continue
        best_span = w
        for start_candidate in idx:
            shifted = sorted(((i - start_candidate) % w) for i in idx)
            span = shifted[-1] + 1
            best_span = min(best_span, span)
        max_span = max(max_span, best_span)
    return max(1, max_span)


# -- Evaluation driver ----------------------------------------------------------

def evaluate_alpha(data_dir, output_dir):
    test_base = os.path.join(data_dir, "phase2", "test")
    orbits   = np.load(os.path.join(test_base, "orbits.npy"))
    rule_ids = np.load(os.path.join(test_base, "rule_ids.npy"))
    alphas   = np.load(os.path.join(test_base, "alphas.npy"))

    n = len(orbits)
    print(f"Evaluating algorithmic baseline on {n} alphaECA test samples...")
    t0 = time.time()

    rule_correct = 0
    alpha_errs = []
    results = []
    for i in range(n):
        orbit = orbits[i]
        true_rule = int(rule_ids[i])
        true_alpha = float(alphas[i])

        identified = identify_rule_by_lookup(orbit)
        r_ok = bool(identified == true_rule)
        rule_correct += int(r_ok)

        est_alpha_raw = estimate_alpha_naive(orbit, identified)
        est_alpha = snap_alpha(est_alpha_raw)
        a_err = float(abs(est_alpha - true_alpha))
        alpha_errs.append(a_err)

        results.append({
            "sample_idx": int(i), "true_rule": true_rule, "identified_rule": int(identified),
            "rule_correct": r_ok, "true_alpha": true_alpha,
            "est_alpha_raw": est_alpha_raw, "est_alpha": est_alpha,
            "alpha_err": a_err, "alpha_tol_ok": bool(a_err <= 0.05),
        })

        if (i + 1) % 500 == 0 or i + 1 == n:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{n}] rule={rule_correct/(i+1)*100:.1f}% "
                  f"alpha_tol={sum(r['alpha_tol_ok'] for r in results)/(i+1)*100:.1f}% "
                  f"elapsed={elapsed:.0f}s")

    rule_acc = rule_correct / n * 100
    alpha_tol_acc = sum(1 for e in alpha_errs if e <= 0.05) / n * 100
    alpha_mae = float(np.mean(alpha_errs))

    print(f"\nFINAL (alphaECA, algorithmic baseline): rule={rule_acc:.2f}% "
          f"alpha_tol={alpha_tol_acc:.2f}% alpha_MAE={alpha_mae:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "algorithmic_baseline_alpha_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "dataset": "alphaECA", "n": n, "rule_accuracy": rule_acc,
            "alpha_tolerance_accuracy": alpha_tol_acc, "alpha_mae": alpha_mae,
            "samples": results,
        }, f, indent=2)
    print(f"Saved: {out_path}")


def evaluate_skew(data_dir, output_dir, w=20):
    test_base = os.path.join(data_dir, "phase2", "test")
    orbits   = np.load(os.path.join(test_base, "orbits.npy"))
    rule_ids = np.load(os.path.join(test_base, "rule_ids.npy"))
    s_values = np.load(os.path.join(test_base, "s_values.npy"))

    n = len(orbits)
    print(f"Evaluating algorithmic baseline on {n} skewECA test samples...")
    t0 = time.time()

    rule_correct = 0
    s_errs = []
    results = []
    for i in range(n):
        orbit = orbits[i]
        true_rule = int(rule_ids[i])
        true_s = int(s_values[i])

        identified = identify_rule_by_lookup(orbit)
        r_ok = bool(identified == true_rule)
        rule_correct += int(r_ok)

        est_s = int(estimate_s_max_span(orbit, w))
        s_err = int(abs(est_s - true_s))
        s_errs.append(s_err)

        results.append({
            "sample_idx": int(i), "true_rule": true_rule, "identified_rule": int(identified),
            "rule_correct": r_ok, "true_s": true_s, "est_s": est_s,
            "s_err": s_err, "s_exact": bool(s_err == 0), "s_tol1": bool(s_err <= 1),
        })

        if (i + 1) % 500 == 0 or i + 1 == n:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{n}] rule={rule_correct/(i+1)*100:.1f}% "
                  f"s_exact={sum(r['s_exact'] for r in results)/(i+1)*100:.1f}% "
                  f"elapsed={elapsed:.0f}s")

    rule_acc = rule_correct / n * 100
    s_exact_acc = sum(1 for e in s_errs if e == 0) / n * 100
    s_tol1_acc  = sum(1 for e in s_errs if e <= 1) / n * 100
    s_mae = float(np.mean(s_errs))

    print(f"\nFINAL (skewECA, algorithmic baseline): rule={rule_acc:.2f}% "
          f"s_exact={s_exact_acc:.2f}% s_tol1={s_tol1_acc:.2f}% s_MAE={s_mae:.3f}")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "algorithmic_baseline_skew_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "dataset": "skewECA", "n": n, "rule_accuracy": rule_acc,
            "s_exact_accuracy": s_exact_acc, "s_tol1_accuracy": s_tol1_acc,
            "s_mae": s_mae, "samples": results,
        }, f, indent=2)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["alpha", "skew"], required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output_dir", default="algorithmic_baseline_results")
    args = ap.parse_args()

    if args.dataset == "alpha":
        evaluate_alpha(args.data_dir, args.output_dir)
    else:
        evaluate_skew(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
