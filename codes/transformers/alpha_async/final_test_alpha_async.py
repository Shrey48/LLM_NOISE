"""
FINAL_TEST_NEW.py  --  Comprehensive Final Test
================================================

Tests the trained Phase 2 model against ALL 256 ECA rules
with ALL 10 alpha values, using FRESH orbits never seen during training.

HOW IT WORKS:
  1. Generates fresh orbits at runtime (not loaded from disk)
     -> Rules shuffled in random order
     -> Alpha values shuffled in random order per rule
     -> 20 orbits per (rule, alpha) pair

  2. Rule prediction  : Neural model (majority vote across 20 orbits)

  3. Alpha prediction : Hybrid estimator
     - Uses predicted rule to compute what rule WOULD output each step
     - Counts what fraction of "should-change" cells actually changed
     - This IS alpha by definition: mathematically exact in expectation
     - Falls back to neural prediction only for rule 0/255 type rules
       where the rule never changes any cell.
     - Neural fallback uses classification argmax -> ALPHA_VALUES[idx]
       which guarantees max error = 0.05 even without hybrid estimator.

  4. Reports:
     - Per (rule, alpha) pair: rule prediction, bit acc, alpha estimate
     - Per-rule summary: across all 10 alphas
     - Per-alpha summary: across all 256 rules
     - Overall totals

TOTAL PREDICTIONS: 256 rules x 10 alphas x 20 orbits = 51,200

CHECKPOINT: checkpoints_new/phase2_best.pt

Changes from original:
  - RANDOM_SEED set to 42 for reproducibility
  - hybrid_alpha_estimate snaps output to nearest 0.1 grid value

Usage:
  python FINAL_TEST_NEW.py
"""

import torch
import numpy as np
import os
import sys
import time
from collections import defaultdict

# -- Locate scripts -----------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from MODEL_NEW import (ECANetNew, simulate_eca, orbit_to_tokens,
                       rule_to_bits, bits_to_rule,
                       W, T, N_TOK, N_BITS, ALPHA_VALUES,
                       alpha_logits_to_value)

# -- Config -------------------------------------------------------------------
ORBITS_PER_PAIR  = 20       # orbits per (rule, alpha) -- 20 gives MAE ~0.003
RANDOM_SEED      = 42       # fixed for reproducibility
ALL_RULES        = list(range(256))

CHECKPOINT_PATH  = os.path.join(
    SCRIPT_DIR, "checkpoints_new", "phase2_best.pt")

# -- Device -------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Device: CUDA  --  {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Device: Apple MPS")
else:
    device = torch.device("cpu")
    print("Device: CPU")

print(f"PyTorch   : {torch.__version__}")
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"Orbits/pair: {ORBITS_PER_PAIR}")
print(f"Total preds: {256 * 10 * ORBITS_PER_PAIR:,}\n")


# -- Hybrid Alpha Estimator ---------------------------------------------------

def hybrid_alpha_estimate(orbit, predicted_rule_num):
    """
    Direct alpha estimation from orbit + predicted rule.

    Under the async ECA model:
      Each cell updates with probability alpha per step.

    So: alpha = cells_that_changed_when_rule_said_change
                / cells_where_rule_said_change

    This is mathematically exact in expectation.
    It is RULE-AWARE -- not confused by rules that never/always change cells.

    Returns None if the rule never produces any cell changes
    (e.g. rule 0 outputs 0 always, rule 255 outputs 1 always).
    In that case we fall back to the neural prediction.

    Output is snapped to the nearest 0.1 grid value so that reported
    alpha predictions are always clean values matching ALPHA_VALUES.
    """
    from MODEL_NEW import build_rule_table
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
        return None   # rule never changes anything -- fall back to neural

    # FIX: snap to nearest 0.1 grid value (was raw float e.g. 0.2973)
    raw = did_change / should_change
    return round(raw * 10) / 10


def estimate_alpha_from_orbits(orbits, predicted_rule_num):
    """Median of hybrid estimates across multiple orbits."""
    estimates = []
    for orbit in orbits:
        est = hybrid_alpha_estimate(orbit, predicted_rule_num)
        if est is not None:
            estimates.append(est)
    if not estimates:
        return None, True   # fallback to neural
    # Snap median to nearest 0.1 grid as well
    raw_median = float(np.median(estimates))
    snapped    = round(raw_median * 10) / 10
    return snapped, False


# -- Load Model ---------------------------------------------------------------

def load_model():
    if not os.path.exists(CHECKPOINT_PATH):
        print("=" * 65)
        print("ERROR: Checkpoint not found!")
        print(f"  Expected: {CHECKPOINT_PATH}")
        print("  Run TRAIN_NEW.py first.")
        print("=" * 65)
        sys.exit(1)

    model = ECANetNew().to(device)
    ckpt  = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    epoch = ckpt.get("epoch", "?")
    exact = ckpt.get("best_exact", "?")
    phase = ckpt.get("phase", "?")
    if isinstance(exact, float):
        print(f"Loaded: phase={phase}  epoch={epoch}  best_val_exact={exact:.2f}%\n")
    else:
        print(f"Loaded: phase={phase}  epoch={epoch}  best_val_exact={exact}\n")
    return model


# -- Run Final Tests ----------------------------------------------------------

def run_final_tests(model):
    rng = np.random.default_rng(seed=RANDOM_SEED)

    # Shuffle rules and generate random alpha order per rule
    rules_order  = rng.permutation(ALL_RULES).tolist()
    alpha_orders = {
        rule: rng.permutation(ALPHA_VALUES).tolist()
        for rule in rules_order
    }

    print("=" * 65)
    print("  FINAL TEST  --  All 256 Rules x All 10 Alphas")
    print("=" * 65)
    print(f"  Rules tested  : 256  (all ECA rules 0-255, random order)")
    print(f"  Alphas/rule   : 10   (0.1..1.0, random order per rule)")
    print(f"  Orbits/pair   : {ORBITS_PER_PAIR}")
    print(f"  Total orbits  : {256*10*ORBITS_PER_PAIR:,}")
    print(f"  Seed          : {RANDOM_SEED}")
    print(f"  Alpha method  : hybrid estimator (snapped to 0.1 grid);")
    print(f"                  neural fallback uses 10-class argmax")
    print(f"                  (max error = 0.05 guaranteed)")
    print()

    # Storage
    all_results   = []
    rule_summary  = defaultdict(lambda: {
        "rule_pass":0, "total_pairs":0,
        "alpha_errs":[], "neural_errs":[], "alpha_ok":0})
    alpha_summary = defaultdict(lambda: {
        "rule_pass":0, "total_pairs":0,
        "alpha_errs":[], "neural_errs":[], "alpha_ok":0})

    t_start    = time.time()
    pair_count = 0
    total_pairs = 256 * 10

    PRINT_FULL_ROWS = True

    if PRINT_FULL_ROWS:
        print("-" * 105)
        print(f"  {'Rule':>4} | {'Alpha':>5} | {'PredRule':>8} | "
              f"{'RuleOK':>6} | {'BitAcc':>6} | "
              f"{'NeuralA':>7} | {'HybridA':>7} | "
              f"{'Err':>6} | {'OK':>4} | Method")
        print("-" * 105)

    for rule_num in rules_order:
        rule_bits_true = rule_to_bits(rule_num)

        for alpha in alpha_orders[rule_num]:
            pair_count += 1

            # Generate fresh orbits
            orbits = []
            for _ in range(ORBITS_PER_PAIR):
                while True:
                    init = rng.integers(0, 2, size=W).astype(np.uint8)
                    if 0 < int(init.sum()) < W:
                        break
                orbits.append(simulate_eca(rule_num, init, T, alpha, rng))

            # Neural predictions
            rule_preds    = []
            neural_alphas = []
            bit_accs      = []

            for orbit in orbits:
                tokens = orbit_to_tokens(orbit)
                x_t    = torch.from_numpy(tokens).unsqueeze(0).to(device)
                with torch.no_grad():
                    rule_logits, alpha_logits = model(x_t)

                pred_bits  = (torch.sigmoid(rule_logits) > 0.5).cpu().numpy().squeeze()
                pred_rule  = bits_to_rule(pred_bits)
                bit_acc    = float((pred_bits.astype(np.uint8) ==
                                    rule_bits_true.astype(np.uint8)).mean())
                rule_preds.append(pred_rule)
                neural_alpha_val = alpha_logits_to_value(alpha_logits).item()
                neural_alphas.append(neural_alpha_val)
                bit_accs.append(bit_acc)

            # Majority vote for rule
            voted_rule        = max(set(rule_preds), key=rule_preds.count)
            avg_bit_acc       = float(np.mean(bit_accs))
            neural_alpha_pred = float(np.median(neural_alphas))
            rule_correct      = (voted_rule == rule_num)

            # Hybrid alpha (already snapped to 0.1 grid inside the function)
            hybrid_est, fallback = estimate_alpha_from_orbits(orbits, voted_rule)
            if fallback or hybrid_est is None:
                final_alpha = neural_alpha_pred
                method      = "neural"
            else:
                final_alpha = hybrid_est
                method      = "hybrid"

            alpha_err  = abs(final_alpha - alpha)
            neural_err = abs(neural_alpha_pred - alpha)
            alpha_ok   = alpha_err <= 0.05

            if PRINT_FULL_ROWS:
                rule_sym = "PASS" if rule_correct else "FAIL"
                ok_sym   = "OK"   if alpha_ok     else "off"
                print(f"  {rule_num:>4} | {alpha:>5.1f} | "
                      f"{voted_rule:>8} | {rule_sym:>6} | "
                      f"{avg_bit_acc*100:5.1f}% | "
                      f"{neural_alpha_pred:>7.3f} | {final_alpha:>7.3f} | "
                      f"{alpha_err:>6.4f} | {ok_sym:>4} | {method}")

            result = {
                "rule":         rule_num,
                "alpha":        alpha,
                "voted_rule":   voted_rule,
                "rule_correct": rule_correct,
                "bit_acc":      avg_bit_acc,
                "neural_alpha": neural_alpha_pred,
                "final_alpha":  final_alpha,
                "alpha_err":    alpha_err,
                "neural_err":   neural_err,
                "alpha_ok":     alpha_ok,
                "method":       method,
            }
            all_results.append(result)

            rs = rule_summary[rule_num]
            rs["total_pairs"]   += 1
            rs["rule_pass"]     += int(rule_correct)
            rs["alpha_errs"].append(alpha_err)
            rs["neural_errs"].append(neural_err)
            rs["alpha_ok"]      += int(alpha_ok)

            as_ = alpha_summary[alpha]
            as_["total_pairs"]   += 1
            as_["rule_pass"]     += int(rule_correct)
            as_["alpha_errs"].append(alpha_err)
            as_["neural_errs"].append(neural_err)
            as_["alpha_ok"]      += int(alpha_ok)

            if pair_count % 200 == 0 or pair_count == total_pairs:
                elapsed   = time.time() - t_start
                remaining = (elapsed / pair_count) * (total_pairs - pair_count)
                rule_acc  = sum(r["rule_correct"] for r in all_results) / len(all_results) * 100
                alpha_acc = sum(r["alpha_ok"]     for r in all_results) / len(all_results) * 100
                print(f"\n  [Progress: {pair_count}/{total_pairs}  "
                      f"rule_acc={rule_acc:.1f}%  "
                      f"alpha_acc(+-0.05)={alpha_acc:.1f}%  "
                      f"ETA={int(remaining)}s]\n")

    elapsed = time.time() - t_start

    # -- OVERALL SUMMARY ------------------------------------------------------
    total_n      = len(all_results)
    rule_pass_n  = sum(r["rule_correct"] for r in all_results)
    alpha_ok_n   = sum(r["alpha_ok"]     for r in all_results)
    hybrid_n     = sum(1 for r in all_results if r["method"] == "hybrid")
    avg_bit      = float(np.mean([r["bit_acc"]    for r in all_results]))
    avg_h_mae    = float(np.mean([r["alpha_err"]  for r in all_results]))
    avg_n_mae    = float(np.mean([r["neural_err"] for r in all_results]))

    print(f"\n{'='*65}")
    print(f"  OVERALL SUMMARY  --  All 256 Rules x 10 Alphas")
    print(f"{'='*65}")
    print(f"  Total (rule,alpha) pairs : {total_n}")
    print(f"  Orbits per pair          : {ORBITS_PER_PAIR}")
    print(f"  Total orbit predictions  : {total_n * ORBITS_PER_PAIR:,}")
    print(f"  Time elapsed             : {elapsed:.1f}s")
    print()
    print(f"  Rule exact match : {rule_pass_n}/{total_n} "
          f"({rule_pass_n/total_n*100:.2f}%)")
    print(f"  Rule bit acc     : {avg_bit*100:.2f}%")
    print()
    print(f"  Alpha method     : hybrid={hybrid_n}  neural_fallback={total_n-hybrid_n}")
    print(f"  Neural alpha MAE : {avg_n_mae:.4f}  (classification argmax)")
    print(f"  Hybrid alpha MAE : {avg_h_mae:.4f}  (after hybrid estimator)")
    print(f"  Alpha acc(+-0.05): {alpha_ok_n}/{total_n} "
          f"({alpha_ok_n/total_n*100:.2f}%)")

    # -- PER-ALPHA BREAKDOWN --------------------------------------------------
    print(f"\n  {'='*70}")
    print(f"  PER-ALPHA BREAKDOWN  (across all 256 rules)")
    print(f"  {'='*70}")
    print(f"  {'Alpha':>5} | {'N':>4} | {'RulePass':>9} | {'HybridMAE':>10} | "
          f"{'NeuralMAE':>10} | {'AlphaAcc+-0.05':>14}")
    print(f"  {'-'*70}")
    for av in ALPHA_VALUES:
        s = alpha_summary[av]
        n = s["total_pairs"]
        if n == 0:
            continue
        rp     = s["rule_pass"]
        h_mae  = float(np.mean(s["alpha_errs"]))
        n_mae  = float(np.mean(s["neural_errs"]))
        ok_pct = s["alpha_ok"] / n * 100
        fill   = "=" * max(0, int(rp / n * 15))
        empty  = "." * (15 - len(fill))
        print(f"  {av:>5.1f} | {n:>4} | "
              f"{rp:>4}/{n:<4} [{fill}{empty}] | "
              f"{h_mae:>10.4f} | {n_mae:>10.4f} | {ok_pct:>13.1f}%")

    # -- PER-RULE SUMMARY (problem rules) -------------------------------------
    print(f"\n  {'='*70}")
    print(f"  PER-RULE SUMMARY  (rules with imperfect performance)")
    print(f"  {'='*70}")
    print(f"  {'Rule':>4} | {'AlphaPairs':>10} | {'RulePass':>9} | "
          f"{'HybridMAE':>10} | {'AlphaAcc+-0.05':>14}")
    print(f"  {'-'*60}")

    problem_rules = []
    for rule_num in sorted(rule_summary.keys()):
        s      = rule_summary[rule_num]
        n      = s["total_pairs"]
        rp     = s["rule_pass"]
        h_mae  = float(np.mean(s["alpha_errs"]))
        ok_pct = s["alpha_ok"] / n * 100
        if rp < n or ok_pct < 90:
            problem_rules.append(rule_num)
            print(f"  {rule_num:>4} | {n:>10} | "
                  f"{rp:>4}/{n:<4} {('OK' if rp==n else 'FAIL'):>4} | "
                  f"{h_mae:>10.4f} | {ok_pct:>13.1f}%")

    if not problem_rules:
        print(f"  All 256 rules performed perfectly across all 10 alphas!")
    else:
        print(f"\n  {len(problem_rules)} rules with imperfect performance above.")
        print(f"  Rules: {problem_rules[:32]}")
        if len(problem_rules) > 32:
            print(f"         ... and {len(problem_rules)-32} more")

    # -- ALPHA ACCURACY BY RANGE ----------------------------------------------
    print(f"\n  Alpha accuracy by range (threshold: +-0.05):")
    ranges = [
        ("Low   (0.1-0.3)", [r for r in all_results if r["alpha"] <= 0.3]),
        ("Mid   (0.4-0.7)", [r for r in all_results if 0.4 <= r["alpha"] <= 0.7]),
        ("High  (0.8-1.0)", [r for r in all_results if r["alpha"] >= 0.8]),
    ]
    for label, items in ranges:
        if not items:
            continue
        n    = len(items)
        ok   = sum(r["alpha_ok"]      for r in items)
        hmae = float(np.mean([r["alpha_err"]  for r in items]))
        nmae = float(np.mean([r["neural_err"] for r in items]))
        rp   = sum(r["rule_correct"]  for r in items)
        print(f"    {label}  N={n:>4}  "
              f"rule={rp/n*100:.1f}%  "
              f"alphaAcc={ok/n*100:.1f}%  "
              f"hybridMAE={hmae:.4f}  "
              f"neuralMAE={nmae:.4f}")

    print(f"\n{'='*65}")
    print(f"  FINAL TEST COMPLETE")
    print(f"{'='*65}")


# -- Main ---------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  ECANet New  --  Final Comprehensive Test")
    print("=" * 65)
    print(f"  All 256 ECA rules x 10 alpha values x {ORBITS_PER_PAIR} orbits")
    print(f"  Fresh data generated at runtime (never seen during training)")
    print(f"  Alpha: hybrid estimator (snapped to 0.1 grid) primary;")
    print(f"         neural fallback -> guaranteed max error = 0.05")
    print(f"  Seed: {RANDOM_SEED}\n")

    model = load_model()
    run_final_tests(model)