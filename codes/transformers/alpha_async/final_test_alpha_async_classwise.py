"""
FINAL_TEST_NEW_CLASSWISE.py  --  ECANet New: Full + Class-Wise Evaluation
==========================================================================
Model : ECANetNew  (single async ECA rule, alpha ∈ {0.1..1.0})
Infra : MODEL_NEW + checkpoints_new/phase2_best.pt

WHAT THIS DOES
--------------
1.  Generates fresh test orbits at runtime (no saved data needed).
2.  Runs the SAME evaluation as FINAL_TEST_NEW.py:
      - Rule majority vote across ORBITS_PER_PAIR orbits
      - Hybrid alpha estimator (snapped to 0.1 grid); neural fallback
3.  ADDITIONALLY reports accuracy broken down by Wolfram class:

    Class 1 (NULL / Uniform)   : rules that converge to a fixed point — all 0s or 1s
    Class 2 (PERIODIC / FP)    : rules with periodic or fixed-point attractors
    Class 3 (CHAOTIC / COMPLEX): chaotic or complex rules
    Class 4 (NAMED COMPLEX)    : the four class-IV rules — explicitly named

    For each class:
      - Lists every rule number in that class (with Wolfram name where known)
      - Reports rule-exact accuracy  (% of predictions correct)
      - Reports alpha-exact accuracy (within ±0.05)
      - Reports alpha MAE
      - Per-alpha breakdown within the class

Usage:
    python FINAL_TEST_NEW_CLASSWISE.py
    (No data files needed — generated on the fly)
"""

import torch
import numpy as np
import os
import sys
import time
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from MODEL_NEW import (ECANetNew, simulate_eca, orbit_to_tokens,
                       rule_to_bits, bits_to_rule, build_rule_table,
                       W, T, N_TOK, N_BITS, ALPHA_VALUES,
                       alpha_logits_to_value)

# ── Config ────────────────────────────────────────────────────────────────────
ORBITS_PER_PAIR = 20
RANDOM_SEED     = 42
ALL_RULES       = list(range(256))
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "checkpoints_new", "phase2_best.pt")
LOG_PATH        = os.path.join(SCRIPT_DIR, "final_test_new_classwise.log")

# ── Tee ───────────────────────────────────────────────────────────────────────
class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, d):
        for s in self.streams: s.write(d); s.flush()
    def flush(self):
        for s in self.streams: s.flush()

_log = open(LOG_PATH, "w", buffering=1, encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, _log)
sys.stderr = Tee(sys.__stderr__, _log)

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Device: CUDA -- {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps"); print("Device: Apple MPS")
else:
    device = torch.device("cpu"); print("Device: CPU")

# ── Wolfram Classification ────────────────────────────────────────────────────
# Minimal representatives from the paper (Table 1 / standard refs)
_CLASS_I  = {0, 8, 32, 40, 128, 136, 160, 168}
_CLASS_III = {18, 22, 30, 45, 60, 90, 105, 122, 126, 146, 150}
_CLASS_IV  = {41, 54, 106, 110}
_CLASS_LC  = {26, 73, 154}   # "locally chaotic" / complex — grouped with C

# Canonical names for well-known rules
RULE_NAMES = {
    0:   "Zero (C1)",       8:   "Copy (C1)",       32:  "C1-32",
    40:  "C1-40",           128: "C1-128",           136: "C1-136",
    160: "C1-160",          168: "C1-168",
    18:  "Rule 18 (C3)",    22:  "Rule 22 (C3)",     30:  "Rule 30 (C3/chaos)",
    45:  "Rule 45 (C3)",    60:  "Rule 60 (C3)",     90:  "Rule 90 (C3/Sierpinski)",
    105: "Rule 105 (C3)",   122: "Rule 122 (C3)",    126: "Rule 126 (C3)",
    146: "Rule 146 (C3)",   150: "Rule 150 (C3)",
    41:  "Rule 41 (C4)",    54:  "Rule 54 (C4)",     106: "Rule 106 (C4)",
    110: "Rule 110 (C4/universal)",
    26:  "Rule 26 (LC)",    73:  "Rule 73 (LC)",     154: "Rule 154 (LC)",
}

def _reflect(r):
    b = [(r >> i) & 1 for i in range(8)]
    p = [0, 4, 2, 6, 1, 5, 3, 7]
    return sum(b[p[i]] * (1 << i) for i in range(8))

def _complement(r):
    b = [(r >> i) & 1 for i in range(8)]
    return sum((1 - b[7 - i]) * (1 << i) for i in range(8))

def _min_rep(r):
    c = _complement(r)
    return min(r, _reflect(r), c, _reflect(c))

def wolfram_class_full(r):
    """Returns '1', '2', '3', '4' based on Wolfram class."""
    m = _min_rep(r)
    if m in _CLASS_I:                return '1'
    if m in _CLASS_IV:               return '4'
    if m in _CLASS_III | _CLASS_LC:  return '3'
    return '2'

# Build class membership for all 256 rules
RULE_CLASSES = {r: wolfram_class_full(r) for r in range(256)}

CLASS_LABELS = {
    '1': 'Class 1 — NULL / Uniform (fixed-point attractors, all-0 or all-1)',
    '2': 'Class 2 — PERIODIC / FP  (periodic or fixed-point attractors)',
    '3': 'Class 3 — CHAOTIC / COMPLEX (chaotic, Sierpinski, etc.)',
    '4': 'Class 4 — NAMED COMPLEX  (universal computation, long transients)',
}
CLASS_SHORT = {'1': 'C1-Null', '2': 'C2-Periodic', '3': 'C3-Chaotic', '4': 'C4-Complex'}

# ── Hybrid Alpha Estimator ────────────────────────────────────────────────────
def hybrid_alpha_estimate(orbit, predicted_rule_num):
    rule_table = build_rule_table(predicted_rule_num)
    should_change = 0; did_change = 0
    for t in range(len(orbit) - 1):
        state = orbit[t].astype(np.uint8)
        left  = np.roll(state,  1); right = np.roll(state, -1)
        idx   = (4 * left + 2 * state + right).astype(np.uint8)
        rule_output      = rule_table[idx]
        wants_change     = (rule_output != state)
        actually_changed = (orbit[t+1].astype(np.uint8) != state)
        should_change   += int(wants_change.sum())
        did_change      += int((wants_change & actually_changed).sum())
    if should_change == 0:
        return None
    raw = did_change / should_change
    return round(raw * 10) / 10

def estimate_alpha_from_orbits(orbits, predicted_rule_num):
    estimates = [hybrid_alpha_estimate(o, predicted_rule_num) for o in orbits]
    estimates = [e for e in estimates if e is not None]
    if not estimates:
        return None, True
    return round(float(np.median(estimates)) * 10) / 10, False

# ── Load Model ────────────────────────────────────────────────────────────────
def load_model():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: {CHECKPOINT_PATH} not found. Run TRAIN_NEW.py first.")
        sys.exit(1)
    model = ECANetNew().to(device)
    ckpt  = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded: phase={ckpt.get('phase','?')}  "
          f"epoch={ckpt.get('epoch','?')}  "
          f"best_exact={ckpt.get('best_exact', ckpt.get('best_val_exact', '?'))}\n")
    return model

# ── Main Test Loop ────────────────────────────────────────────────────────────
def run_final_tests(model):
    rng          = np.random.default_rng(seed=RANDOM_SEED)
    rules_order  = rng.permutation(ALL_RULES).tolist()
    alpha_orders = {r: rng.permutation(ALPHA_VALUES).tolist() for r in rules_order}

    print("=" * 70)
    print(f"  FINAL TEST NEW (Class-Wise)  --  256 Rules x 10 Alphas x {ORBITS_PER_PAIR} Orbits")
    print("=" * 70)

    all_results  = []
    rule_summary = defaultdict(lambda: {"rule_pass": 0, "total": 0,
                                        "alpha_errs": [], "alpha_ok": 0})
    alpha_summary = defaultdict(lambda: {"rule_pass": 0, "total": 0,
                                         "alpha_errs": [], "alpha_ok": 0})
    class_summary = defaultdict(lambda: {"rule_pass": 0, "total": 0,
                                          "alpha_errs": [], "alpha_ok": 0,
                                          "alpha_breakdown": defaultdict(
                                              lambda: {"rule_pass":0,"total":0,
                                                       "alpha_errs":[],"alpha_ok":0})})

    t_start    = time.time()
    pair_count = 0
    total_pairs = 256 * len(ALPHA_VALUES)

    print(f"  {'Rule':>4} | {'Cls':>3} | {'Alpha':>5} | {'Voted':>5} | "
          f"{'ROK':>3} | {'BitAcc':>6} | {'NeuA':>6} | {'FinA':>5} | "
          f"{'Err':>5} | {'OK':>2} | Meth")
    print(f"  {'-'*80}")

    for rule_num in rules_order:
        for alpha in alpha_orders[rule_num]:
            pair_count += 1

            orbits = []
            for _ in range(ORBITS_PER_PAIR):
                init = rng.integers(0, 2, size=W).astype(np.int8)
                while init.sum() == 0 or init.sum() == W:
                    init = rng.integers(0, 2, size=W).astype(np.int8)
                orbits.append(simulate_eca(rule_num, init, T, alpha, rng))

            rule_preds = []; neural_alphas = []; bit_accs = []
            rule_bits_true = np.array(rule_to_bits(rule_num), dtype=np.uint8)

            with torch.no_grad():
                for orbit in orbits:
                    tokens = orbit_to_tokens(orbit)
                    x = torch.tensor(tokens, dtype=torch.float32).unsqueeze(0).to(device)
                    rule_logits, alpha_logits = model(x)
                    pred_bits = (torch.sigmoid(rule_logits) > 0.5).cpu().numpy().squeeze()
                    pred_rule = bits_to_rule(pred_bits)
                    bit_acc   = float((pred_bits.astype(np.uint8) == rule_bits_true).mean())
                    rule_preds.append(pred_rule)
                    neural_alphas.append(alpha_logits_to_value(alpha_logits).item())
                    bit_accs.append(bit_acc)

            voted_rule        = max(set(rule_preds), key=rule_preds.count)
            avg_bit_acc       = float(np.mean(bit_accs))
            neural_alpha_pred = round(float(np.median(neural_alphas)) * 10) / 10
            rule_correct      = (voted_rule == rule_num)

            hybrid_est, fallback = estimate_alpha_from_orbits(orbits, voted_rule)
            if fallback or hybrid_est is None:
                final_alpha, method = neural_alpha_pred, "neural"
            else:
                final_alpha, method = hybrid_est, "hybrid"

            alpha_err = abs(final_alpha - alpha)
            alpha_ok  = alpha_err <= 0.05
            cls       = RULE_CLASSES[rule_num]

            print(f"  {rule_num:>4} | {cls:>3} | {alpha:>5.1f} | {voted_rule:>5} | "
                  f"{'Y' if rule_correct else 'N':>3} | {avg_bit_acc*100:5.1f}% | "
                  f"{neural_alpha_pred:>6.3f} | {final_alpha:>5.3f} | "
                  f"{alpha_err:>5.3f} | {'Y' if alpha_ok else 'N':>2} | {method}")

            res = {"rule": rule_num, "alpha": alpha, "voted_rule": voted_rule,
                   "rule_correct": rule_correct, "bit_acc": avg_bit_acc,
                   "neural_alpha": neural_alpha_pred, "final_alpha": final_alpha,
                   "alpha_err": alpha_err, "alpha_ok": alpha_ok,
                   "method": method, "class": cls}
            all_results.append(res)

            for summ, key in [(rule_summary, rule_num), (alpha_summary, alpha)]:
                summ[key]["total"]      += 1
                summ[key]["rule_pass"]  += int(rule_correct)
                summ[key]["alpha_errs"].append(alpha_err)
                summ[key]["alpha_ok"]   += int(alpha_ok)

            cs = class_summary[cls]
            cs["total"]     += 1
            cs["rule_pass"] += int(rule_correct)
            cs["alpha_errs"].append(alpha_err)
            cs["alpha_ok"]  += int(alpha_ok)
            ab = cs["alpha_breakdown"][alpha]
            ab["total"]     += 1
            ab["rule_pass"] += int(rule_correct)
            ab["alpha_errs"].append(alpha_err)
            ab["alpha_ok"]  += int(alpha_ok)

            if pair_count % 200 == 0 or pair_count == total_pairs:
                elapsed = time.time() - t_start
                eta     = (elapsed / pair_count) * (total_pairs - pair_count)
                r_acc   = sum(r["rule_correct"] for r in all_results) / len(all_results) * 100
                a_acc   = sum(r["alpha_ok"]     for r in all_results) / len(all_results) * 100
                print(f"\n  [{pair_count}/{total_pairs}  "
                      f"rule={r_acc:.1f}%  alpha_exact={a_acc:.1f}%  "
                      f"ETA={int(eta)}s]\n")

    elapsed = time.time() - t_start
    total_n = len(all_results)

    # ── OVERALL ───────────────────────────────────────────────────────────────
    rule_pass  = sum(r["rule_correct"] for r in all_results)
    alpha_ok_n = sum(r["alpha_ok"]     for r in all_results)
    hybrid_n   = sum(1 for r in all_results if r["method"] == "hybrid")
    avg_mae    = float(np.mean([r["alpha_err"] for r in all_results]))
    a_off1     = sum(1 for r in all_results if r["alpha_err"] <= 0.1)  / total_n * 100

    print(f"\n{'='*70}")
    print(f"  OVERALL  ({total_n} pairs, {elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"  Rule exact      : {rule_pass}/{total_n} ({rule_pass/total_n*100:.2f}%)")
    print(f"  Alpha exact±0.05: {alpha_ok_n}/{total_n} ({alpha_ok_n/total_n*100:.2f}%)")
    print(f"  Alpha MAE       : {avg_mae:.4f}")
    print(f"  Alpha ±0.1      : {a_off1:.2f}%")
    print(f"  Hybrid/Neural   : {hybrid_n}/{total_n-hybrid_n}")

    # ── PER-ALPHA ─────────────────────────────────────────────────────────────
    print(f"\n  {'='*60}")
    print(f"  PER-ALPHA BREAKDOWN (all 256 rules)")
    print(f"  {'='*60}")
    print(f"  {'Alpha':>5} | {'N':>4} | {'Rule%':>6} | {'AlphaAcc±0.05':>13} | {'MAE':>6}")
    print(f"  {'-'*50}")
    for av in ALPHA_VALUES:
        s = alpha_summary[av]; n = s["total"]
        if n == 0: continue
        print(f"  {av:>5.1f} | {n:>4} | {s['rule_pass']/n*100:5.1f}% | "
              f"{s['alpha_ok']/n*100:12.1f}% | "
              f"{np.mean(s['alpha_errs']):.4f}")

    # ── CLASS-WISE (NEW SECTION) ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  CLASS-WISE ACCURACY BREAKDOWN")
    print(f"{'='*70}")
    print(f"""
  Wolfram Classification (based on minimal representative):
    Class 1 — NULL / Uniform  : rules whose dynamics collapse to a fixed point
                                 (all zeros or all ones)
    Class 2 — PERIODIC / FP   : rules with periodic orbits or fixed points
                                 (structured, predictable dynamics)
    Class 3 — CHAOTIC / COMPLEX: rules with chaotic / pseudo-random dynamics
                                 (sensitive to initial conditions)
    Class 4 — NAMED COMPLEX   : rules capable of universal computation
                                 (long transients, gliders, complex interactions)
""")

    for cls_id in ['1', '2', '3', '4']:
        label = CLASS_LABELS[cls_id]
        cs    = class_summary[cls_id]
        n     = cs["total"]
        if n == 0: continue

        # Which rules belong to this class?
        cls_rules = sorted(r for r in range(256) if RULE_CLASSES[r] == cls_id)
        rule_strs = []
        for r in cls_rules:
            name = RULE_NAMES.get(r, "")
            if name:
                rule_strs.append(f"{r} ({name})")
            else:
                rule_strs.append(str(r))

        print(f"\n  ── {label}")
        print(f"     Rules ({len(cls_rules)} total):")
        # print 10 per line
        chunk = 10
        for i in range(0, len(rule_strs), chunk):
            print(f"       {', '.join(rule_strs[i:i+chunk])}")

        rp  = cs["rule_pass"]
        aok = cs["alpha_ok"]
        mae = float(np.mean(cs["alpha_errs"]))
        print(f"\n     Accuracy:")
        print(f"       Rule exact          : {rp}/{n}  ({rp/n*100:.2f}%)")
        print(f"       Alpha exact (±0.05) : {aok}/{n} ({aok/n*100:.2f}%)")
        print(f"       Alpha MAE           : {mae:.4f}")

        # Per-alpha within this class
        print(f"\n     Per-alpha breakdown:")
        print(f"       {'Alpha':>5} | {'N':>4} | {'Rule%':>6} | {'Alpha±0.05%':>11} | {'MAE':>6}")
        print(f"       {'-'*45}")
        for av in ALPHA_VALUES:
            ab = cs["alpha_breakdown"][av]; an = ab["total"]
            if an == 0: continue
            print(f"       {av:>5.1f} | {an:>4} | "
                  f"{ab['rule_pass']/an*100:5.1f}% | "
                  f"{ab['alpha_ok']/an*100:10.1f}% | "
                  f"{np.mean(ab['alpha_errs']):.4f}")

    # ── CLASS SUMMARY TABLE ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  CLASS SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"  {'Class':>12} | {'N pairs':>7} | {'N rules':>7} | "
          f"{'Rule%':>6} | {'Alpha±0.05%':>11} | {'Alpha MAE':>9}")
    print(f"  {'-'*65}")
    for cls_id in ['1', '2', '3', '4']:
        cs = class_summary[cls_id]; n = cs["total"]
        if n == 0: continue
        cls_rules_n = len([r for r in range(256) if RULE_CLASSES[r] == cls_id])
        rp  = cs["rule_pass"]
        aok = cs["alpha_ok"]
        mae = float(np.mean(cs["alpha_errs"]))
        print(f"  {CLASS_SHORT[cls_id]:>12} | {n:>7} | {cls_rules_n:>7} | "
              f"{rp/n*100:5.1f}% | {aok/n*100:10.1f}% | {mae:>9.4f}")

    # ── PROBLEM RULES ─────────────────────────────────────────────────────────
    problem = sorted(r for r in rule_summary
                     if (rule_summary[r]["rule_pass"] < rule_summary[r]["total"] or
                         rule_summary[r]["alpha_ok"] / rule_summary[r]["total"] < 0.80))
    if problem:
        print(f"\n  {len(problem)} problem rules (rule miss OR alpha<80%):")
        for r in problem:
            s = rule_summary[r]; n = s["total"]
            print(f"    Rule {r:>3} [{CLASS_SHORT[RULE_CLASSES[r]]}]  "
                  f"rule_ok={s['rule_pass']}/{n}  "
                  f"alpha_ok={s['alpha_ok']}/{n}  "
                  f"MAE={np.mean(s['alpha_errs']):.4f}  "
                  f"name={RULE_NAMES.get(r,'')}")
    else:
        print(f"\n  All 256 rules performed perfectly!")

    print(f"\n{'='*70}")
    print(f"  FINAL TEST COMPLETE  |  Log: {LOG_PATH}")
    print(f"{'='*70}")

    _log.close()

if __name__ == "__main__":
    print("=" * 70)
    print(f"  ECANet New (Class-Wise)  --  Final Test  T={T} W={W}")
    print("=" * 70)
    model = load_model()
    run_final_tests(model)
