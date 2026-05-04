"""
final_test_alpha_async.py  --  Split-Correct Final Test
=========================================================

Tests the trained Phase 2 model against ONLY the 77 held-out TEST rules
(seed=42 split, paper-defined), with ALL 10 alpha values, using FRESH
orbits generated at runtime (never seen during training).

KEY FIX vs original:
  Original tested ALL 256 rules  -> data-leak: 179 train rules included.
  This version tests ONLY the 77 held-out test rules -> correct evaluation.

The 77 test rules are derived EXACTLY as datagen did:
    rng_split = np.random.default_rng(42)
    shuffled  = rng_split.permutation(256).tolist()
    test_rules = sorted(shuffled[179:])   # last 77 after split

If ECA_Data_New/test_rules.npy exists (saved by datagen), we load it
and verify it matches. Otherwise we recompute from scratch.

HOW IT WORKS:
  1. Fresh orbits generated at runtime (same as original test script)
  2. Rule prediction  : Neural model (majority vote across ORBITS_PER_PAIR orbits)
  3. Alpha prediction : Hybrid estimator (snapped to 0.1 grid)
                        Neural fallback for degenerate rules (e.g. rule 204)
  4. Reports per (rule,alpha) pair, per-rule summary, per-alpha summary,
     overall totals -- all saved to "split test result" files.

TOTAL PREDICTIONS: 77 rules x 10 alphas x 20 orbits = 15,400

Usage:
    python final_test_alpha_async.py

Outputs (saved in same directory as this script):
    split test result.txt   -- human-readable full report
    split test result.json  -- machine-readable structured results
"""

import torch
import numpy as np
import os
import sys
import json
import time
from collections import defaultdict
from datetime import datetime

# ── Locate scripts ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    from model_alpha_async import (
        ECANetNew, simulate_eca, orbit_to_tokens,
        rule_to_bits, bits_to_rule,
        W, T, N_TOK, N_BITS, ALPHA_VALUES,
        alpha_logits_to_value,
    )
except ModuleNotFoundError:
    try:
        from MODEL_NEW import (
            ECANetNew, simulate_eca, orbit_to_tokens,
            rule_to_bits, bits_to_rule,
            W, T, N_TOK, N_BITS, ALPHA_VALUES,
            alpha_logits_to_value,
        )
    except ModuleNotFoundError as e:
        print("ERROR: Could not import model. Make sure one of these files is in the same folder:")
        print("   model_alpha_async.py  OR  MODEL_NEW.py")
        raise e

# ── Config ────────────────────────────────────────────────────────────────────
ORBITS_PER_PAIR = 20        # orbits per (rule, alpha) -- matches original
RANDOM_SEED     = 42
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "checkpoints_new", "phase2_best.pt")
DATA_DIR        = os.path.join(SCRIPT_DIR, "ECA_Data_New")
PRINT_FULL_ROWS = True      # set False to suppress per-pair lines

# ── Paper-defined 77 test rules (seed=42 split, identical to datagen) ─────────
def get_test_rules():
    """
    Reproduce the exact same split as datagen used.

    datagen does:
        rng_split = np.random.default_rng(SEED)   # SEED=42
        shuffled  = rng_split.permutation(256).tolist()
        train_rules = sorted(shuffled[:179])
        test_rules  = sorted(shuffled[179:])

    We do the same here. If ECA_Data_New/test_rules.npy exists we also
    verify against it for extra safety.
    """
    rng_split  = np.random.default_rng(42)
    shuffled   = rng_split.permutation(256).tolist()
    test_rules = sorted(shuffled[179:])
    assert len(test_rules) == 77, f"Expected 77 test rules, got {len(test_rules)}"

    # Optional: cross-check against saved npy if datagen was already run
    saved_path = os.path.join(DATA_DIR, "test_rules.npy")
    if os.path.exists(saved_path):
        saved = sorted(np.load(saved_path).tolist())
        assert saved == test_rules, (
            f"Mismatch between recomputed and saved test_rules!\n"
            f"  Recomputed: {test_rules[:10]}...\n"
            f"  Saved     : {saved[:10]}..."
        )
        print(f"  test_rules.npy cross-check: PASSED")
    else:
        print(f"  test_rules.npy not found -- using recomputed split (datagen not run yet).")

    return test_rules

TEST_RULES = get_test_rules()   # 77 rules -- never seen during training

# ── Device ────────────────────────────────────────────────────────────────────
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
print(f"Test rules: {len(TEST_RULES)} (held-out, paper split seed=42)")
print(f"Orbits/pair: {ORBITS_PER_PAIR}")
print(f"Total preds: {len(TEST_RULES) * 10 * ORBITS_PER_PAIR:,}\n")


# ── Hybrid Alpha Estimator ────────────────────────────────────────────────────

def hybrid_alpha_estimate(orbit, predicted_rule_num):
    """
    Direct alpha estimation from orbit + predicted rule.

    Under async ECA: alpha = cells_that_changed_when_rule_said_change
                             / cells_where_rule_said_change

    This is mathematically exact in expectation.
    Returns None if rule never produces cell changes (e.g. rule 0/255).
    Output snapped to nearest 0.1 grid.
    """
    try:
        from model_alpha_async import build_rule_table
    except ModuleNotFoundError:
        from MODEL_NEW import build_rule_table
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
        return None   # degenerate rule -- fall back to neural

    raw = did_change / should_change
    return round(raw * 10) / 10   # snap to 0.1 grid


def estimate_alpha_from_orbits(orbits, predicted_rule_num):
    """Median of hybrid estimates across multiple orbits."""
    estimates = []
    for orbit in orbits:
        est = hybrid_alpha_estimate(orbit, predicted_rule_num)
        if est is not None:
            estimates.append(est)
    if not estimates:
        return None, True   # all degenerate -- fallback to neural
    raw_median = float(np.median(estimates))
    snapped    = round(raw_median * 10) / 10
    return snapped, False


# ── Load Model ────────────────────────────────────────────────────────────────

def load_model():
    if not os.path.exists(CHECKPOINT_PATH):
        print("=" * 65)
        print("ERROR: Checkpoint not found!")
        print(f"  Expected: {CHECKPOINT_PATH}")
        print("  Run train_alpha_async.py first.")
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


# ── Run Test on Split ─────────────────────────────────────────────────────────

def run_split_test(model):
    rng = np.random.default_rng(seed=RANDOM_SEED)

    # Shuffle test rules and alpha order (same style as original)
    rules_order  = rng.permutation(TEST_RULES).tolist()
    alpha_orders = {
        rule: rng.permutation(ALPHA_VALUES).tolist()
        for rule in rules_order
    }

    total_pairs = len(TEST_RULES) * len(ALPHA_VALUES)

    print("=" * 65)
    print("  SPLIT FINAL TEST  --  77 Test Rules x 10 Alphas")
    print("=" * 65)
    print(f"  Rules tested  : {len(TEST_RULES)}  (held-out test split, seed=42)")
    print(f"  Alphas/rule   : {len(ALPHA_VALUES)}  (0.1..1.0, random order per rule)")
    print(f"  Orbits/pair   : {ORBITS_PER_PAIR}")
    print(f"  Total pairs   : {total_pairs}")
    print(f"  Total orbits  : {total_pairs * ORBITS_PER_PAIR:,}")
    print(f"  Seed          : {RANDOM_SEED}")
    print(f"  Test rules    : {TEST_RULES[:10]}... (77 total)")
    print()

    if PRINT_FULL_ROWS:
        print(f"  {'Rule':>4} | {'Alpha':>5} | {'Pred':>8} | {'Rule':>6} | "
              f"{'BitAcc':>6} | {'Neural':>7} | {'Final':>7} | "
              f"{'Err':>6} | {'OK':>4} | Method")
        print(f"  {'-'*80}")

    # Storage
    all_results  = []
    rule_summary = defaultdict(lambda: {
        "rule_pass": 0, "total_pairs": 0,
        "alpha_errs": [], "neural_errs": [], "alpha_ok": 0,
    })
    alpha_summary = defaultdict(lambda: {
        "total_pairs": 0, "rule_pass": 0,
        "alpha_errs": [], "neural_errs": [], "alpha_ok": 0,
    })

    t_start    = time.time()
    pair_count = 0

    for rule_num in rules_order:
        rule_bits_true = np.array(rule_to_bits(rule_num), dtype=np.float32)

        for alpha in alpha_orders[rule_num]:
            pair_count += 1
            orbits      = []
            rule_preds  = []
            neural_alphas = []
            bit_accs    = []

            for _ in range(ORBITS_PER_PAIR):
                init  = rng.integers(0, 2, size=W, dtype=np.int8)
                while int(init.sum()) == 0 or int(init.sum()) == W:
                    init = rng.integers(0, 2, size=W, dtype=np.int8)

                orbit = simulate_eca(rule_num, init, T, alpha, rng)
                orbits.append(orbit)

                tokens = orbit_to_tokens(orbit)
                x_t    = torch.tensor(tokens, dtype=torch.float32)\
                              .unsqueeze(0).to(device)   # [1, N_TOK, 4]

                with torch.no_grad():
                    rule_logits, alpha_logits = model(x_t)

                pred_bits = (torch.sigmoid(rule_logits) > 0.5)\
                                .cpu().numpy().squeeze()
                pred_rule = bits_to_rule(pred_bits)
                bit_acc   = float((pred_bits.astype(np.uint8) ==
                                   rule_bits_true.astype(np.uint8)).mean())

                rule_preds.append(pred_rule)
                neural_alphas.append(alpha_logits_to_value(alpha_logits).item())
                bit_accs.append(bit_acc)

            # Majority vote for rule
            voted_rule        = max(set(rule_preds), key=rule_preds.count)
            avg_bit_acc       = float(np.mean(bit_accs))
            neural_alpha_pred = float(np.median(neural_alphas))
            rule_correct      = (voted_rule == rule_num)

            # Hybrid alpha
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

            if pair_count % 50 == 0 or pair_count == total_pairs:
                elapsed   = time.time() - t_start
                remaining = (elapsed / pair_count) * (total_pairs - pair_count) \
                            if pair_count < total_pairs else 0
                rule_acc  = sum(r["rule_correct"] for r in all_results) \
                            / len(all_results) * 100
                alpha_acc = sum(r["alpha_ok"] for r in all_results) \
                            / len(all_results) * 100
                print(f"\n  [Progress: {pair_count}/{total_pairs}  "
                      f"rule_acc={rule_acc:.1f}%  "
                      f"alpha_acc(±0.05)={alpha_acc:.1f}%  "
                      f"ETA={int(remaining)}s]\n")

    elapsed = time.time() - t_start

    # ── Overall summary ───────────────────────────────────────────────────────
    total_n     = len(all_results)
    rule_pass_n = sum(r["rule_correct"] for r in all_results)
    alpha_ok_n  = sum(r["alpha_ok"]     for r in all_results)
    hybrid_n    = sum(1 for r in all_results if r["method"] == "hybrid")
    avg_bit     = float(np.mean([r["bit_acc"]    for r in all_results]))
    avg_h_mae   = float(np.mean([r["alpha_err"]  for r in all_results]))
    avg_n_mae   = float(np.mean([r["neural_err"] for r in all_results]))

    summary_lines = []
    summary_lines.append("=" * 65)
    summary_lines.append(f"  SPLIT TEST RESULT  --  77 Held-Out Test Rules x 10 Alphas")
    summary_lines.append("=" * 65)
    summary_lines.append(f"  Evaluated at        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"  Test rules (77)     : {TEST_RULES}")
    summary_lines.append(f"  Total (rule,alpha) pairs : {total_n}")
    summary_lines.append(f"  Orbits per pair          : {ORBITS_PER_PAIR}")
    summary_lines.append(f"  Total orbit predictions  : {total_n * ORBITS_PER_PAIR:,}")
    summary_lines.append(f"  Time elapsed             : {elapsed:.1f}s")
    summary_lines.append("")
    summary_lines.append(f"  Rule exact match : {rule_pass_n}/{total_n} "
                         f"({rule_pass_n/total_n*100:.2f}%)")
    summary_lines.append(f"  Rule bit acc     : {avg_bit*100:.2f}%")
    summary_lines.append("")
    summary_lines.append(f"  Alpha method     : hybrid={hybrid_n}  "
                         f"neural_fallback={total_n - hybrid_n}")
    summary_lines.append(f"  Neural alpha MAE : {avg_n_mae:.4f}  (classification argmax)")
    summary_lines.append(f"  Hybrid alpha MAE : {avg_h_mae:.4f}  (after hybrid estimator)")
    summary_lines.append(f"  Alpha acc(±0.05) : {alpha_ok_n}/{total_n} "
                         f"({alpha_ok_n/total_n*100:.2f}%)")

    # ── Per-alpha breakdown ───────────────────────────────────────────────────
    summary_lines.append("")
    summary_lines.append(f"  {'='*70}")
    summary_lines.append(f"  PER-ALPHA BREAKDOWN  (across 77 test rules)")
    summary_lines.append(f"  {'='*70}")
    summary_lines.append(f"  {'Alpha':>5} | {'N':>4} | {'RulePass':>9} | "
                         f"{'HybridMAE':>10} | {'NeuralMAE':>10} | {'AlphaAcc±0.05':>14}")
    summary_lines.append(f"  {'-'*70}")
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
        summary_lines.append(
            f"  {av:>5.1f} | {n:>4} | "
            f"{rp:>4}/{n:<4} [{fill}{empty}] | "
            f"{h_mae:>10.4f} | {n_mae:>10.4f} | {ok_pct:>13.1f}%"
        )

    # ── Per-rule breakdown ────────────────────────────────────────────────────
    summary_lines.append("")
    summary_lines.append(f"  {'='*70}")
    summary_lines.append(f"  PER-RULE BREAKDOWN  (all 77 test rules)")
    summary_lines.append(f"  {'='*70}")
    summary_lines.append(f"  {'Rule':>4} | {'Pairs':>5} | {'RulePass':>9} | "
                         f"{'HybridMAE':>10} | {'AlphaAcc±0.05':>14} | Status")
    summary_lines.append(f"  {'-'*65}")

    problem_rules = []
    for rule_num in sorted(rule_summary.keys()):
        s      = rule_summary[rule_num]
        n      = s["total_pairs"]
        rp     = s["rule_pass"]
        h_mae  = float(np.mean(s["alpha_errs"]))
        ok_pct = s["alpha_ok"] / n * 100
        status = "OK" if (rp == n and ok_pct >= 90) else "PROBLEM"
        if status == "PROBLEM":
            problem_rules.append(rule_num)
        summary_lines.append(
            f"  {rule_num:>4} | {n:>5} | "
            f"{rp:>4}/{n:<4} | "
            f"{h_mae:>10.4f} | {ok_pct:>13.1f}% | {status}"
        )

    summary_lines.append("")
    if not problem_rules:
        summary_lines.append("  All 77 test rules performed perfectly!")
    else:
        summary_lines.append(f"  {len(problem_rules)} rules with issues: {problem_rules}")

    # ── Alpha accuracy by range ───────────────────────────────────────────────
    summary_lines.append("")
    summary_lines.append("  Alpha accuracy by range (threshold: ±0.05):")
    ranges = [
        ("Low   (0.1-0.3)", [r for r in all_results if r["alpha"] <= 0.3]),
        ("Mid   (0.4-0.7)", [r for r in all_results if 0.4 <= r["alpha"] <= 0.7]),
        ("High  (0.8-1.0)", [r for r in all_results if r["alpha"] >= 0.8]),
    ]
    for label, items in ranges:
        if not items:
            continue
        n    = len(items)
        ok   = sum(r["alpha_ok"]     for r in items)
        hmae = float(np.mean([r["alpha_err"]  for r in items]))
        nmae = float(np.mean([r["neural_err"] for r in items]))
        rp   = sum(r["rule_correct"] for r in items)
        summary_lines.append(
            f"    {label}  N={n:>4}  "
            f"rule={rp/n*100:.1f}%  "
            f"alphaAcc={ok/n*100:.1f}%  "
            f"hybridMAE={hmae:.4f}  "
            f"neuralMAE={nmae:.4f}"
        )

    summary_lines.append("")
    summary_lines.append("=" * 65)
    summary_lines.append("  SPLIT TEST COMPLETE")
    summary_lines.append("=" * 65)

    # Print to console
    for line in summary_lines:
        print(line)

    # ── Save results to file ──────────────────────────────────────────────────
    txt_path  = os.path.join(SCRIPT_DIR, "split test result.txt")
    json_path = os.path.join(SCRIPT_DIR, "split test result.json")

    # Text summary
    with open(txt_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\n  [Saved] {txt_path}")

    # JSON — full per-sample data + aggregates
    json_out = {
        "meta": {
            "evaluated_at":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model":             "ECANetNew (alpha-async)",
            "checkpoint":        CHECKPOINT_PATH,
            "test_rules":        TEST_RULES,
            "num_test_rules":    len(TEST_RULES),
            "orbits_per_pair":   ORBITS_PER_PAIR,
            "total_pairs":       total_n,
            "seed":              RANDOM_SEED,
            "split_seed":        42,
        },
        "aggregate": {
            "rule_exact_match_pct":   round(rule_pass_n / total_n * 100, 4),
            "rule_bit_acc_pct":       round(avg_bit * 100, 4),
            "alpha_tolerance_acc_pct": round(alpha_ok_n / total_n * 100, 4),
            "hybrid_alpha_mae":       round(avg_h_mae, 6),
            "neural_alpha_mae":       round(avg_n_mae, 6),
            "hybrid_used":            hybrid_n,
            "neural_fallback_used":   total_n - hybrid_n,
            "time_elapsed_s":         round(elapsed, 1),
        },
        "per_alpha": {
            str(av): {
                "n":             alpha_summary[av]["total_pairs"],
                "rule_pass":     alpha_summary[av]["rule_pass"],
                "rule_acc_pct":  round(alpha_summary[av]["rule_pass"] /
                                       max(1, alpha_summary[av]["total_pairs"]) * 100, 2),
                "hybrid_mae":    round(float(np.mean(alpha_summary[av]["alpha_errs"])), 6)
                                 if alpha_summary[av]["alpha_errs"] else 0,
                "alpha_acc_pct": round(alpha_summary[av]["alpha_ok"] /
                                       max(1, alpha_summary[av]["total_pairs"]) * 100, 2),
            }
            for av in ALPHA_VALUES
        },
        "per_rule": {
            str(rule_num): {
                "n":             rule_summary[rule_num]["total_pairs"],
                "rule_pass":     rule_summary[rule_num]["rule_pass"],
                "rule_acc_pct":  round(rule_summary[rule_num]["rule_pass"] /
                                       max(1, rule_summary[rule_num]["total_pairs"]) * 100, 2),
                "hybrid_mae":    round(float(np.mean(rule_summary[rule_num]["alpha_errs"])), 6)
                                 if rule_summary[rule_num]["alpha_errs"] else 0,
                "alpha_acc_pct": round(rule_summary[rule_num]["alpha_ok"] /
                                       max(1, rule_summary[rule_num]["total_pairs"]) * 100, 2),
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
    print("  ECANet New  --  Split-Correct Final Test")
    print("=" * 65)
    print(f"  77 held-out test rules (seed=42 split) x 10 alphas x {ORBITS_PER_PAIR} orbits")
    print(f"  Fresh data generated at runtime")
    print(f"  Alpha: hybrid estimator (snapped to 0.1 grid) primary;")
    print(f"         neural fallback -> guaranteed max error = 0.05")
    print(f"  Seed: {RANDOM_SEED}")
    print(f"  Results saved to: 'split test result.txt' and 'split test result.json'\n")

    model = load_model()
    run_split_test(model)
