"""
FINAL_TEST_SKEW_CLASSWISE.py  --  ECANet Skew: Full + Class-Wise Evaluation
=============================================================================
Model : ECANetSkew  (s-skewed ECA, single rule, s ∈ {1..W=20})
Infra : MODEL_SKEW + checkpoints_skew/phase2_best.pt

WHAT THIS DOES
--------------
1.  Generates fresh test orbits at runtime (no saved data needed).
    Uses simulate_eca_skewed — s cells updated per step from random start.
2.  Runs the SAME evaluation as FINAL_TEST_SKEW.py:
      - Rule majority vote; hybrid s estimator; neural fallback
3.  ADDITIONALLY reports accuracy broken down by Wolfram class:

    Class 1 (NULL / Uniform)   : fixed-point attractor rules
    Class 2 (PERIODIC / FP)    : periodic / fixed-point rules
    Class 3 (CHAOTIC / COMPLEX): chaotic / Sierpinski / complex rules
    Class 4 (NAMED COMPLEX)    : the 4 class-IV rules (41, 54, 106, 110)

    Per class:
      - Rule list with names
      - Rule-exact accuracy
      - s-exact accuracy  (|pred_s - true_s| == 0)
      - s MAE
      - s ±1 accuracy
      - Per-s-value breakdown within class

Usage:
    python FINAL_TEST_SKEW_CLASSWISE.py
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

from MODEL_SKEW import (ECANetSkew, simulate_eca_skewed, orbit_to_tokens,
                        rule_to_bits, bits_to_rule, build_rule_table,
                        W, T, N_TOK, N_BITS, TOKEN_DIM, S_VALUES, N_S_CLASSES,
                        s_logits_to_value)

# ── Config ────────────────────────────────────────────────────────────────────
ORBITS_PER_PAIR = 20
RANDOM_SEED     = 42
ALL_RULES       = list(range(256))
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "checkpoints_skew", "phase2_best.pt")
LOG_PATH        = os.path.join(SCRIPT_DIR, "final_test_skew_classwise.log")

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
_CLASS_I   = {0, 8, 32, 40, 128, 136, 160, 168}
_CLASS_III = {18, 22, 30, 45, 60, 90, 105, 122, 126, 146, 150}
_CLASS_IV  = {41, 54, 106, 110}
_CLASS_LC  = {26, 73, 154}

RULE_NAMES = {
    0:   "Zero (C1)",       8:   "Copy (C1)",      32:  "C1-32",
    40:  "C1-40",           128: "C1-128",          136: "C1-136",
    160: "C1-160",          168: "C1-168",
    18:  "Rule 18 (C3)",    22:  "Rule 22 (C3)",    30:  "Rule 30 (chaos)",
    45:  "Rule 45 (C3)",    60:  "Rule 60 (C3)",    90:  "Rule 90 (Sierpinski)",
    105: "Rule 105 (C3)",   122: "Rule 122 (C3)",   126: "Rule 126 (C3)",
    146: "Rule 146 (C3)",   150: "Rule 150 (C3)",
    41:  "Rule 41 (C4)",    54:  "Rule 54 (C4)",    106: "Rule 106 (C4)",
    110: "Rule 110 (universal)",
    26:  "Rule 26 (LC)",    73:  "Rule 73 (LC)",    154: "Rule 154 (LC)",
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
    m = _min_rep(r)
    if m in _CLASS_I:                return '1'
    if m in _CLASS_IV:               return '4'
    if m in _CLASS_III | _CLASS_LC:  return '3'
    return '2'

RULE_CLASSES = {r: wolfram_class_full(r) for r in range(256)}

CLASS_LABELS = {
    '1': 'Class 1 — NULL / Uniform (fixed-point attractors)',
    '2': 'Class 2 — PERIODIC / FP  (periodic or fixed-point)',
    '3': 'Class 3 — CHAOTIC / COMPLEX (chaotic, Sierpinski, etc.)',
    '4': 'Class 4 — NAMED COMPLEX  (universal, gliders, long transients)',
}
CLASS_SHORT = {'1': 'C1-Null', '2': 'C2-Periodic', '3': 'C3-Chaotic', '4': 'C4-Complex'}

# ── Hybrid s Estimator ────────────────────────────────────────────────────────
def hybrid_s_estimate(orbit, predicted_rule_num):
    rule_table    = build_rule_table(predicted_rule_num)
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
    return max(1, min(W, round((did_change / should_change) * W)))

def estimate_s_from_orbits(orbits, predicted_rule_num):
    estimates = [hybrid_s_estimate(o, predicted_rule_num) for o in orbits]
    estimates = [e for e in estimates if e is not None]
    if not estimates:
        return None, True
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
          f"best_exact={ckpt.get('best_exact', 0):.2f}%\n")
    return model

# ── Main Test Loop ────────────────────────────────────────────────────────────
def run_final_tests(model):
    rng         = np.random.default_rng(seed=RANDOM_SEED)
    rules_order = rng.permutation(ALL_RULES).tolist()
    s_orders    = {r: rng.permutation(S_VALUES).tolist() for r in rules_order}

    print("=" * 70)
    print(f"  FINAL TEST SKEW (Class-Wise)  --  "
          f"256 Rules x {len(S_VALUES)} S-values x {ORBITS_PER_PAIR} Orbits")
    print("=" * 70)

    all_results   = []
    rule_summary  = defaultdict(lambda: {"rule_pass": 0, "total": 0,
                                         "s_errs": [], "s_ok": 0})
    s_summary     = defaultdict(lambda: {"rule_pass": 0, "total": 0,
                                         "s_errs": [], "s_ok": 0})
    class_summary = defaultdict(lambda: {"rule_pass": 0, "total": 0,
                                          "s_errs": [], "s_ok": 0,
                                          "s_breakdown": defaultdict(
                                              lambda: {"rule_pass":0,"total":0,
                                                       "s_errs":[],"s_ok":0})})

    t_start     = time.time()
    pair_count  = 0
    total_pairs = 256 * len(S_VALUES)

    print(f"  {'Rule':>4} | {'Cls':>3} | {'s':>3} | {'Voted':>5} | "
          f"{'ROK':>3} | {'BitAcc':>6} | {'NeuS':>4} | "
          f"{'FinS':>4} | {'Err':>3} | {'OK':>2} | Meth")
    print(f"  {'-'*75}")

    for rule_num in rules_order:
        for s_val in s_orders[rule_num]:
            pair_count += 1

            orbits = []
            for _ in range(ORBITS_PER_PAIR):
                init = rng.integers(0, 2, size=W).astype(np.int8)
                while init.sum() == 0 or init.sum() == W:
                    init = rng.integers(0, 2, size=W).astype(np.int8)
                orbits.append(simulate_eca_skewed(rule_num, init, T, s_val, rng))

            rule_preds = []; neural_s_vals = []; bit_accs = []

            with torch.no_grad():
                for orbit in orbits:
                    tokens = orbit_to_tokens(orbit)
                    x = torch.tensor(tokens, dtype=torch.float32).unsqueeze(0).to(device)
                    rule_logits, s_logits, _ = model(x)
                    pred_bits = (torch.sigmoid(rule_logits) >= 0.5).float()
                    true_bits = torch.tensor(
                        rule_to_bits(rule_num), dtype=torch.float32).unsqueeze(0).to(device)
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
            cls   = RULE_CLASSES[rule_num]

            print(f"  {rule_num:>4} | {cls:>3} | {s_val:>3} | {voted_rule:>5} | "
                  f"{'Y' if rule_correct else 'N':>3} | {avg_bit_acc*100:5.1f}% | "
                  f"{neural_s_pred:>4} | {final_s:>4} | {s_err:>3} | "
                  f"{'Y' if s_ok else 'N':>2} | {method}")

            res = {"rule": rule_num, "s": s_val, "voted_rule": voted_rule,
                   "rule_correct": rule_correct, "bit_acc": avg_bit_acc,
                   "neural_s": neural_s_pred, "final_s": final_s,
                   "s_err": s_err, "s_ok": s_ok, "method": method, "class": cls}
            all_results.append(res)

            for summ, key in [(rule_summary, rule_num), (s_summary, s_val)]:
                summ[key]["total"]     += 1
                summ[key]["rule_pass"] += int(rule_correct)
                summ[key]["s_errs"].append(s_err)
                summ[key]["s_ok"]      += int(s_ok)

            cs = class_summary[cls]
            cs["total"]     += 1
            cs["rule_pass"] += int(rule_correct)
            cs["s_errs"].append(s_err)
            cs["s_ok"]      += int(s_ok)
            sb = cs["s_breakdown"][s_val]
            sb["total"]     += 1
            sb["rule_pass"] += int(rule_correct)
            sb["s_errs"].append(s_err)
            sb["s_ok"]      += int(s_ok)

            if pair_count % 200 == 0 or pair_count == total_pairs:
                elapsed = time.time() - t_start
                eta     = (elapsed / pair_count) * (total_pairs - pair_count)
                r_acc   = sum(r["rule_correct"] for r in all_results) / len(all_results) * 100
                s_acc   = sum(r["s_ok"]         for r in all_results) / len(all_results) * 100
                print(f"\n  [{pair_count}/{total_pairs}  "
                      f"rule={r_acc:.1f}%  s_exact={s_acc:.1f}%  ETA={int(eta)}s]\n")

    elapsed = time.time() - t_start
    total_n = len(all_results)

    # ── OVERALL ───────────────────────────────────────────────────────────────
    rule_pass  = sum(r["rule_correct"] for r in all_results)
    s_ok_n     = sum(r["s_ok"]         for r in all_results)
    hybrid_n   = sum(1 for r in all_results if r["method"] == "hybrid")
    avg_mae    = float(np.mean([r["s_err"] for r in all_results]))
    s_off1     = sum(1 for r in all_results if r["s_err"] <= 1) / total_n * 100
    s_off2     = sum(1 for r in all_results if r["s_err"] <= 2) / total_n * 100

    print(f"\n{'='*70}")
    print(f"  OVERALL  ({total_n} pairs, {elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"  Rule exact   : {rule_pass}/{total_n} ({rule_pass/total_n*100:.2f}%)")
    print(f"  S exact      : {s_ok_n}/{total_n} ({s_ok_n/total_n*100:.2f}%)")
    print(f"  S MAE        : {avg_mae:.3f}")
    print(f"  S ±1         : {s_off1:.2f}%   S ±2: {s_off2:.2f}%")
    print(f"  Hybrid/Neural: {hybrid_n}/{total_n-hybrid_n}")

    # ── PER-S BREAKDOWN ───────────────────────────────────────────────────────
    print(f"\n  {'S':>3} | {'N':>4} | {'Rule%':>6} | {'MAE':>5} | {'Exact%':>6} | {'±1%':>5}")
    print(f"  {'-'*45}")
    for sv in S_VALUES:
        d = s_summary[sv]; n = d["total"]
        if n == 0: continue
        print(f"  {sv:>3} | {n:>4} | {d['rule_pass']/n*100:5.1f}% | "
              f"{np.mean(d['s_errs']):5.2f} | {d['s_ok']/n*100:5.1f}% | "
              f"{sum(1 for e in d['s_errs'] if e<=1)/n*100:4.1f}%")

    # ── CLASS-WISE (NEW SECTION) ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  CLASS-WISE ACCURACY BREAKDOWN")
    print(f"{'='*70}")
    print(f"""
  Wolfram Classification:
    Class 1 — NULL / Uniform  : dynamics collapse to fixed point (all 0 / all 1)
    Class 2 — PERIODIC / FP   : periodic orbits or simple fixed points
    Class 3 — CHAOTIC / COMPLEX: chaotic dynamics, sensitive to ICs
    Class 4 — NAMED COMPLEX   : 41 (C4), 54 (C4), 106 (C4), 110 (universal)
""")

    for cls_id in ['1', '2', '3', '4']:
        label     = CLASS_LABELS[cls_id]
        cs        = class_summary[cls_id]
        n         = cs["total"]
        if n == 0: continue

        cls_rules = sorted(r for r in range(256) if RULE_CLASSES[r] == cls_id)
        rule_strs = []
        for r in cls_rules:
            name = RULE_NAMES.get(r, "")
            rule_strs.append(f"{r}({name})" if name else str(r))

        print(f"\n  ── {label}")
        print(f"     Rules ({len(cls_rules)} total):")
        chunk = 8
        for i in range(0, len(rule_strs), chunk):
            print(f"       {', '.join(rule_strs[i:i+chunk])}")

        rp  = cs["rule_pass"]
        sok = cs["s_ok"]
        mae = float(np.mean(cs["s_errs"]))
        s1  = sum(1 for e in cs["s_errs"] if e <= 1)
        print(f"\n     Accuracy:")
        print(f"       Rule exact    : {rp}/{n}  ({rp/n*100:.2f}%)")
        print(f"       S exact       : {sok}/{n}  ({sok/n*100:.2f}%)")
        print(f"       S MAE         : {mae:.3f}")
        print(f"       S ±1          : {s1}/{n}  ({s1/n*100:.2f}%)")

        print(f"\n     Per-s breakdown:")
        print(f"       {'s':>3} | {'N':>4} | {'Rule%':>6} | {'Exact%':>7} | {'±1%':>5} | {'MAE':>5}")
        print(f"       {'-'*45}")
        for sv in S_VALUES:
            sb = cs["s_breakdown"][sv]; sn = sb["total"]
            if sn == 0: continue
            s1b = sum(1 for e in sb["s_errs"] if e <= 1)
            print(f"       {sv:>3} | {sn:>4} | "
                  f"{sb['rule_pass']/sn*100:5.1f}% | "
                  f"{sb['s_ok']/sn*100:6.1f}% | "
                  f"{s1b/sn*100:4.1f}% | "
                  f"{np.mean(sb['s_errs']):.3f}")

    # ── CLASS SUMMARY TABLE ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  CLASS SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"  {'Class':>12} | {'N pairs':>7} | {'N rules':>7} | "
          f"{'Rule%':>6} | {'S Exact%':>8} | {'S MAE':>6} | {'S ±1%':>6}")
    print(f"  {'-'*70}")
    for cls_id in ['1', '2', '3', '4']:
        cs = class_summary[cls_id]; n = cs["total"]
        if n == 0: continue
        cls_rules_n = len([r for r in range(256) if RULE_CLASSES[r] == cls_id])
        rp  = cs["rule_pass"]; sok = cs["s_ok"]
        mae = float(np.mean(cs["s_errs"]))
        s1  = sum(1 for e in cs["s_errs"] if e <= 1)
        print(f"  {CLASS_SHORT[cls_id]:>12} | {n:>7} | {cls_rules_n:>7} | "
              f"{rp/n*100:5.1f}% | {sok/n*100:7.1f}% | {mae:>6.3f} | {s1/n*100:5.1f}%")

    # ── S ACCURACY BY RANGE ────────────────────────────────────────────────────
    print(f"\n  S accuracy by range:")
    for label, filt in [
        ("Low  (1-5) ", lambda r: r["s"] <= 5),
        ("Mid  (6-15)", lambda r: 6 <= r["s"] <= 15),
        ("High (16-20)", lambda r: r["s"] >= 16),
    ]:
        items = [r for r in all_results if filt(r)]
        if not items: continue
        ok  = sum(r["s_ok"] for r in items)
        mae = float(np.mean([r["s_err"] for r in items]))
        rp  = sum(r["rule_correct"] for r in items)
        s1  = sum(1 for r in items if r["s_err"] <= 1)
        print(f"    {label}  N={len(items):>4}  rule={rp/len(items)*100:.1f}%  "
              f"sExact={ok/len(items)*100:.1f}%  s±1={s1/len(items)*100:.1f}%  MAE={mae:.3f}")

    # ── PROBLEM RULES ─────────────────────────────────────────────────────────
    problem = sorted(r for r in rule_summary
                     if (rule_summary[r]["rule_pass"] < rule_summary[r]["total"] or
                         rule_summary[r]["s_ok"] / rule_summary[r]["total"] < 0.80))
    if problem:
        print(f"\n  {len(problem)} problem rules:")
        for r in problem:
            s = rule_summary[r]; n = s["total"]
            print(f"    Rule {r:>3} [{CLASS_SHORT[RULE_CLASSES[r]]}]  "
                  f"rule_ok={s['rule_pass']}/{n}  "
                  f"s_ok={s['s_ok']}/{n}  MAE={np.mean(s['s_errs']):.3f}  "
                  f"name={RULE_NAMES.get(r,'')}")
    else:
        print(f"\n  All 256 rules performed perfectly!")

    print(f"\n{'='*70}")
    print(f"  FINAL TEST COMPLETE  |  Log: {LOG_PATH}")
    print(f"{'='*70}")

    _log.close()

if __name__ == "__main__":
    print("=" * 70)
    print(f"  ECANet Skew (Class-Wise)  --  Final Test  T={T} W={W}")
    print("=" * 70)
    model = load_model()
    run_final_tests(model)
