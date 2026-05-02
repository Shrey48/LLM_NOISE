"""
FINAL_TEST_SCA_CLASSWISE.py  --  ECANet SCA: Full + Class-Wise Evaluation
==========================================================================
Model : ECANetSCA  (stochastic CA, two rules, lambda ∈ [LAM_MIN, LAM_MAX])
Infra : MODEL_SCA + checkpoints_sca_v1/stage2.pt (or stage1.pt)
Dir   : stochastic_v1/

WHAT THIS DOES
--------------
1.  Generates fresh test data at runtime using simulate_sca + compute_frac_stats
    + compute_within_var_stats.
    Matches DATAGEN_SCA.py exactly: 500 pairs × 20 samples = 10,000 samples.
2.  Runs the SAME evaluation as FINAL_TEST_SCA.py:
      - Both-rules-exact (symmetric: forward or flipped)
      - Symmetric lambda MAE: min(|pred-lam|, |pred-(1-lam)|)
      - Analytical lambda baseline
3.  ADDITIONALLY reports accuracy broken down by Wolfram class combo:

    Class 1 (NULL / Uniform)    : rules converging to fixed point
    Class 2 (PERIODIC / FP)     : periodic / fixed-point rules
    Class 3 (CHAOTIC / COMPLEX) : chaotic rules
    Class 4 (NAMED COMPLEX)     : 41, 54, 106, 110

    For each pair-combo (C1×C1, C1×C2, … C4×C4):
      - Both-exact accuracy
      - Lambda MAE
    For each individual class:
      - Rule list with names

Usage:
    python FINAL_TEST_SCA_CLASSWISE.py
    (No pre-saved data needed; directory must contain MODEL_SCA.py and checkpoint)
"""

import torch
import numpy as np
import os
import sys
import time
from collections import defaultdict, Counter
from torch.utils.data import Dataset, DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from MODEL_SCA import (
    ECANetSCA, orbit_to_tokens,
    simulate_sca, compute_frac_stats, compute_within_var_stats,
    rule_to_bits, random_init,
    W, T, N_ORBITS, N_TOK, N_TRANS, FRAC_DIM, WVAR_DIM, LAM_MIN, LAM_MAX
)

# ── Config ────────────────────────────────────────────────────────────────────
CKPT_DIR     = os.path.join(SCRIPT_DIR, "checkpoints_sca_v1")
LOG_PATH     = os.path.join(SCRIPT_DIR, "final_test_sca_classwise.log")
BATCH        = 8
RANDOM_SEED  = 42
N_TRAIN      = 179
N_TEST       = 77
TEST_PAIRS   = 500
TEST_SAMPLES = 20   # 500 × 20 = 10,000

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

def wolfram_class_abc(r):
    m = _min_rep(r)
    if m in _CLASS_I:                           return 'A'
    if m in _CLASS_III | _CLASS_IV | _CLASS_LC: return 'C'
    return 'B'

RULE_CLASSES = {r: wolfram_class_full(r) for r in range(256)}

CLASS_LABELS = {
    '1': 'Class 1 — NULL / Uniform (fixed-point attractors)',
    '2': 'Class 2 — PERIODIC / FP  (periodic or fixed-point)',
    '3': 'Class 3 — CHAOTIC / COMPLEX (chaotic, Sierpinski, etc.)',
    '4': 'Class 4 — NAMED COMPLEX  (41, 54, 106, 110)',
}
CLASS_SHORT = {'1': 'C1-Null', '2': 'C2-Periodic', '3': 'C3-Chaotic', '4': 'C4-Complex'}

def bits_to_rule(bits):
    b = (np.array(bits) > 0.5).astype(int)
    return int(sum(b[i] * (2 ** i) for i in range(8)))

# ── Pair Sampling (mirrors DATAGEN_SCA.py) ────────────────────────────────────
def sample_pairs(rules, n_pairs, rng, label):
    all_pairs = [(f, g) for f in rules for g in rules if f != g]
    perm      = rng.permutation(len(all_pairs))
    all_pairs = [all_pairs[i] for i in perm]

    combo_pools = {}
    for f, g in all_pairs:
        key = tuple(sorted([wolfram_class_abc(f), wolfram_class_abc(g)]))
        combo_pools.setdefault(key, []).append((f, g))

    selected     = list(all_pairs[:min(n_pairs, len(all_pairs))])
    selected_set = set(selected)
    sel_combos   = set(tuple(sorted([wolfram_class_abc(f), wolfram_class_abc(g)]))
                       for f, g in selected)
    for combo in [('A','A'),('A','B'),('A','C'),('B','B'),('B','C'),('C','C')]:
        if combo not in sel_combos and combo in combo_pools:
            for p in combo_pools[combo][:10]:
                if p not in selected_set:
                    selected.append(p); selected_set.add(p)

    final = selected[:n_pairs]
    assert len(final) == n_pairs
    cov = Counter(tuple(sorted([wolfram_class_abc(f), wolfram_class_abc(g)]))
                  for f, g in final)
    print(f"  {label}: {len(final)} pairs  " +
          " ".join(f"{k[0]}x{k[1]}:{v}" for k, v in sorted(cov.items())))
    return final

# ── Fresh Data Generation ─────────────────────────────────────────────────────
def generate_test_data(pairs, n_samples, rng):
    """Generate test samples on the fly — mirrors DATAGEN_SCA.py."""
    orbits_l=[]; frac_l=[]; wvar_l=[]
    rfb_l=[]; rgb_l=[]; lam_l=[]; rfi_l=[]; rgi_l=[]

    print(f"  Generating {len(pairs)} pairs × {n_samples} samples ...")
    for pi, (f, g) in enumerate(pairs):
        bf = rule_to_bits(int(f)); bg = rule_to_bits(int(g))
        for _ in range(n_samples):
            lam    = float(rng.uniform(LAM_MIN, LAM_MAX))
            orbits = []
            for _ in range(N_ORBITS):
                init = random_init(rng)
                orb  = simulate_sca(int(f), int(g), init, lam, rng)
                orbits.append(orb)
            orbs_k = np.stack(orbits, axis=0)
            fs     = compute_frac_stats(orbs_k)
            wv     = compute_within_var_stats(orbs_k)
            orbits_l.append(orbs_k); frac_l.append(fs); wvar_l.append(wv)
            rfb_l.append(bf); rgb_l.append(bg); lam_l.append(lam)
            rfi_l.append(int(f)); rgi_l.append(int(g))

        if (pi + 1) % 100 == 0 or (pi + 1) == len(pairs):
            print(f"    {pi+1}/{len(pairs)} pairs done ...", end="\r")
    print()

    return (np.array(orbits_l, dtype=np.float32),
            np.array(frac_l,   dtype=np.float32),
            np.array(wvar_l,   dtype=np.float32),
            np.array(rfb_l,    dtype=np.float32),
            np.array(rgb_l,    dtype=np.float32),
            np.array(lam_l,    dtype=np.float32),
            np.array(rfi_l,    dtype=np.int32),
            np.array(rgi_l,    dtype=np.int32))

# ── Dataset ───────────────────────────────────────────────────────────────────
class SCAInMemDataset(Dataset):
    def __init__(self, orbits, fracs, wvars, rfb, rgb, lams, rfi, rgi):
        self.orbits = orbits; self.fracs = fracs; self.wvars = wvars
        self.rfb    = rfb;    self.rgb   = rgb;   self.lams  = lams
        self.rfi    = rfi;    self.rgi   = rgi

    def __len__(self): return len(self.orbits)

    def __getitem__(self, idx):
        toks = np.stack([orbit_to_tokens(self.orbits[idx, k])
                         for k in range(N_ORBITS)], axis=0)
        return (torch.tensor(toks,              dtype=torch.float32),
                torch.tensor(self.fracs[idx],   dtype=torch.float32),
                torch.tensor(self.wvars[idx],   dtype=torch.float32),
                torch.tensor(self.rfb[idx],     dtype=torch.float32),
                torch.tensor(self.rgb[idx],     dtype=torch.float32),
                torch.tensor([self.lams[idx]],  dtype=torch.float32),
                int(self.rfi[idx]),
                int(self.rgi[idx]))

# ── Load Model ────────────────────────────────────────────────────────────────
def load_model():
    s2 = os.path.join(CKPT_DIR, "stage2.pt")
    s1 = os.path.join(CKPT_DIR, "stage1.pt")
    ckpt_path = s2 if os.path.exists(s2) else s1
    if not os.path.exists(ckpt_path):
        print(f"ERROR: No checkpoint found in {CKPT_DIR}")
        print("Run TRAIN_SCA.py first."); sys.exit(1)
    if ckpt_path == s1:
        print(f"Note: stage2.pt not found — evaluating stage1.pt")
    model = ECANetSCA().to(device)
    ck    = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"])
    model.eval()
    print(f"Loaded: stage={ck.get('stage','?')}  "
          f"epoch={ck.get('epoch','?')}  "
          f"best_both={ck.get('best_both', 0):.2f}%\n")
    return model

# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_evaluation(model, loader):
    all_results   = []
    pair_summary  = defaultdict(lambda: {"both_exact":0,"total":0,"lam_errs":[]})
    lam_buckets   = defaultdict(lambda: {"both_exact":0,"total":0,"lam_errs":[]})
    class_summary = defaultdict(lambda: {"both_exact":0,"total":0,"lam_errs":[]})
    combo_summary = defaultdict(lambda: {"both_exact":0,"total":0,"lam_errs":[]})

    anal_preds=[]; anal_trues=[]

    for x, fracs, wvars, rf, rg, lam, idf, idg in loader:
        x_d  = x.to(device); fr_d = fracs.to(device); wv_d = wvars.to(device)
        prf, prg, lp, _, _ = model(x_d, fr_d, wv_d)
        la = model.forward_lambda_analytical(fr_d, wv_d)
        anal_preds.append(la.cpu()); anal_trues.append(lam)

        prf_np = torch.sigmoid(prf).cpu().numpy()
        prg_np = torch.sigmoid(prg).cpu().numpy()
        lp_np  = lp.cpu().numpy().reshape(-1)
        tf_np  = rf.numpy(); tg_np = rg.numpy()
        tl_np  = lam.numpy().reshape(-1)
        idf_np = idf.numpy() if isinstance(idf, torch.Tensor) else np.array(idf)
        idg_np = idg.numpy() if isinstance(idg, torch.Tensor) else np.array(idg)

        for i in range(len(prf_np)):
            pf  = prf_np[i]; pg_ = prg_np[i]; pl = float(lp_np[i])
            tf  = tf_np[i];  tg  = tg_np[i];  tl = float(tl_np[i])
            true_f = int(idf_np[i]); true_g = int(idg_np[i])

            both_fwd = (((pf>0.5)==(tf>0.5)).all() and ((pg_>0.5)==(tg>0.5)).all())
            both_flp = (((pf>0.5)==(tg>0.5)).all() and ((pg_>0.5)==(tf>0.5)).all())
            both_exact = both_fwd or both_flp
            lam_err    = min(abs(pl - tl), abs(pl - (1.0 - tl)))

            cf = RULE_CLASSES[true_f]; cg = RULE_CLASSES[true_g]
            combo_key = tuple(sorted([cf, cg]))

            all_results.append({
                "true_f": true_f, "true_g": true_g,
                "pred_f": bits_to_rule(pf), "pred_g": bits_to_rule(pg_),
                "true_lam": tl, "pred_lam": pl,
                "both_exact": both_exact, "lam_err": lam_err,
                "cls_f": cf, "cls_g": cg, "combo": combo_key,
            })

            pk = (min(true_f, true_g), max(true_f, true_g))
            pair_summary[pk]["both_exact"] += int(both_exact)
            pair_summary[pk]["total"]      += 1
            pair_summary[pk]["lam_errs"].append(lam_err)

            tb = round(round(tl * 10) / 10, 1)
            lam_buckets[tb]["both_exact"] += int(both_exact)
            lam_buckets[tb]["total"]      += 1
            lam_buckets[tb]["lam_errs"].append(lam_err)

            for cls_r in [cf, cg]:
                class_summary[cls_r]["both_exact"] += int(both_exact)
                class_summary[cls_r]["total"]       += 1
                class_summary[cls_r]["lam_errs"].append(lam_err)

            combo_summary[combo_key]["both_exact"] += int(both_exact)
            combo_summary[combo_key]["total"]      += 1
            combo_summary[combo_key]["lam_errs"].append(lam_err)

    ap = torch.cat(anal_preds).squeeze()
    at = torch.cat(anal_trues).squeeze()
    anal_sym_mae = float(torch.minimum(
        (ap - at).abs(), (ap - (1.0 - at)).abs()).mean())

    return (all_results, dict(pair_summary), dict(lam_buckets),
            dict(class_summary), dict(combo_summary), anal_sym_mae)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    rng     = np.random.default_rng(RANDOM_SEED)

    print("=" * 70)
    print("  FINAL TEST SCA (Class-Wise)  --  Full Evaluation")
    print("=" * 70)
    print(f"  Ckpt dir  : {CKPT_DIR}")
    print(f"  N_ORBITS  : {N_ORBITS}   W={W}   T={T}")
    print(f"  Lambda    : continuous uniform [{LAM_MIN}, {LAM_MAX}]")
    print(f"  lam_pred  : always ≤ 0.5 by model design")
    print(f"  Pairs     : {TEST_PAIRS}   Samples/pair: {TEST_SAMPLES}")
    print(f"  Total     : {TEST_PAIRS * TEST_SAMPLES:,} test samples (generated fresh)")
    print("=" * 70 + "\n")

    # Rule split: same SEED=42 as training
    rng_split   = np.random.default_rng(42)
    shuffled    = rng_split.permutation(256).tolist()
    train_rules = sorted(shuffled[:N_TRAIN])
    test_rules  = sorted(shuffled[N_TRAIN:])
    assert not set(train_rules) & set(test_rules)

    print("  Sampling test pairs ...")
    test_pairs = sample_pairs(test_rules, TEST_PAIRS, rng, "Test")

    print(f"\n  Generating test data ...")
    t_gen = time.time()
    data  = generate_test_data(test_pairs, TEST_SAMPLES, rng)
    (orbits_arr, fracs_arr, wvars_arr,
     rfb, rgb, lams_arr, rfi_arr, rgi_arr) = data
    print(f"  Generated {len(orbits_arr):,} samples in {time.time()-t_gen:.1f}s\n")

    model  = load_model()
    ds     = SCAInMemDataset(*data)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False,
                        num_workers=0, pin_memory=(device.type == "cuda"))

    print(f"  Running evaluation ...")
    (results, pair_summary, lam_buckets,
     class_summary, combo_summary, anal_sym_mae) = run_evaluation(model, loader)

    total_n = len(results)
    ok      = sum(r["both_exact"] for r in results)
    mae     = float(np.mean([r["lam_err"] for r in results]))

    # ── OVERALL ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  OVERALL RESULTS")
    print(f"{'='*70}")
    print(f"  Test samples         : {total_n:,}")
    print(f"  Both exact (sym)     : {ok}/{total_n}  ({ok/total_n*100:.2f}%)")
    print(f"  Lambda MAE (sym)     : {mae:.4f}")
    print(f"  Analytical lam MAE   : {anal_sym_mae:.4f}")
    print(f"  (symmetric = min(|pred-lam|, |pred-(1-lam)|))")

    # ── PER-LAMBDA BREAKDOWN ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  PER-LAMBDA BREAKDOWN")
    print(f"{'='*70}")
    print(f"  {'Lambda':>7} | {'BothExact':>12} | {'LamMAE':>7} | Visual")
    print(f"  {'-'*55}")
    for lam in sorted(lam_buckets.keys()):
        s    = lam_buckets[lam]
        bk   = s["both_exact"]; tot = s["total"]
        pct  = bk / tot * 100 if tot > 0 else 0.0
        lmae = float(np.mean(s["lam_errs"])) if s["lam_errs"] else 0.0
        bar  = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        sym  = " ←sym" if abs(lam - 0.5) < 0.01 else ""
        print(f"  {lam:>7.1f} | {bk:>4}/{tot:<6} {pct:5.1f}% | "
              f"{lmae:>7.4f} | [{bar}]{sym}")

    # ── LAMBDA ACCURACY ────────────────────────────────────────────────────────
    errs = [r["lam_err"] for r in results]
    print(f"\n{'='*70}")
    print(f"  LAMBDA ESTIMATION ACCURACY (symmetric)")
    print(f"{'='*70}")
    for t in [0.02, 0.05, 0.10, 0.15, 0.20]:
        frac = float(np.mean([e <= t for e in errs])) * 100
        bar  = "█" * int(frac / 5)
        print(f"  Error <= {t:.2f}  :  {frac:5.1f}%  {bar}")
    print(f"\n  Mean={float(np.mean(errs)):.4f}  "
          f"Median={float(np.median(errs)):.4f}  "
          f"Max={float(np.max(errs)):.4f}")

    # ── CLASS-WISE (NEW SECTION) ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  CLASS-WISE ACCURACY BREAKDOWN")
    print(f"{'='*70}")
    print(f"""
  Wolfram Classification:
    Class 1 — NULL / Uniform  : dynamics collapse to fixed point (all-0 / all-1)
    Class 2 — PERIODIC / FP   : periodic orbits or fixed points
    Class 3 — CHAOTIC / COMPLEX: chaotic, Sierpinski, sensitive to ICs
    Class 4 — NAMED COMPLEX   : 41 (C4), 54 (C4), 106 (C4), 110 (universal)

  SCA: at each timestep, each cell picks rule_f (prob lam) or rule_g (prob 1-lam)
  and applies it. W=50 cells per step gives strong within-var signal.

  NOTE: Each sample has TWO rules, so class_summary counts both appearances.
""")

    for cls_id in ['1', '2', '3', '4']:
        label = CLASS_LABELS[cls_id]
        cs    = class_summary.get(cls_id, None)
        if cs is None or cs["total"] == 0:
            print(f"\n  ── {label}\n     (No samples)")
            continue

        cls_rules       = sorted(r for r in range(256) if RULE_CLASSES[r] == cls_id)
        test_cls_rules  = [r for r in cls_rules if r in test_rules]
        rule_strs = []
        for r in cls_rules:
            name = RULE_NAMES.get(r, "")
            rule_strs.append(f"{r}({name})" if name else str(r))

        print(f"\n  ── {label}")
        print(f"     All rules in class ({len(cls_rules)} total, "
              f"{len(test_cls_rules)} in test split):")
        chunk = 8
        for i in range(0, len(rule_strs), chunk):
            print(f"       {', '.join(rule_strs[i:i+chunk])}")

        n   = cs["total"]
        ok_ = cs["both_exact"]
        m   = float(np.mean(cs["lam_errs"]))
        print(f"\n     Accuracy (samples where at least one rule ∈ this class):")
        print(f"       Both exact    : {ok_}/{n}  ({ok_/n*100:.2f}%)")
        print(f"       Lambda MAE    : {m:.4f}")

    # ── COMBO BREAKDOWN ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  CLASS COMBO BREAKDOWN  (each sample = one combo)")
    print(f"{'='*70}")
    print(f"  {'Combo':>6} | {'BothExact':>14} | {'LamMAE':>7} | {'N':>6}")
    print(f"  {'-'*50}")

    all_combos = []
    for a in ['1','2','3','4']:
        for b in ['1','2','3','4']:
            if (a, b) not in all_combos and (b, a) not in all_combos:
                all_combos.append((a, b))

    for combo in sorted(all_combos):
        cs = combo_summary.get(combo, None)
        if cs is None or cs["total"] == 0: continue
        ok_ = cs["both_exact"]; tot = cs["total"]
        m   = float(np.mean(cs["lam_errs"]))
        print(f"  C{combo[0]}×C{combo[1]:>1}  | {ok_:>4}/{tot:<8} {ok_/tot*100:5.1f}% | "
              f"{m:>7.4f} | {tot:>6}")

    # ── CLASS SUMMARY TABLE ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  CLASS SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"  {'Class':>12} | {'Appearances':>11} | "
          f"{'BothExact%':>10} | {'Lam MAE':>7}")
    print(f"  {'-'*55}")
    for cls_id in ['1', '2', '3', '4']:
        cs = class_summary.get(cls_id, None)
        if cs is None or cs["total"] == 0: continue
        n = cs["total"]; ok_ = cs["both_exact"]
        m = float(np.mean(cs["lam_errs"]))
        print(f"  {CLASS_SHORT[cls_id]:>12} | {n:>11} | "
              f"{ok_/n*100:>10.2f}% | {m:>7.4f}")

    # ── HARDEST PAIRS ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  HARDEST RULE PAIRS (top 20)")
    print(f"{'='*70}")
    print(f"  {'RF':>4} | {'RG':>4} | {'Cls':>5} | {'BothExact':>12} | "
          f"{'LamMAE':>7} | {'Disagree':>8}")
    print(f"  {'-'*60}")
    sorted_pairs = sorted(pair_summary.items(),
                          key=lambda x: (x[1]["both_exact"] / x[1]["total"],
                                         np.mean(x[1]["lam_errs"])))
    for (rf, rg), s in sorted_pairs[:20]:
        tot  = s["total"]; ok_ = s["both_exact"]
        m    = float(np.mean(s["lam_errs"]))
        cf   = wolfram_class_full(rf); cg = wolfram_class_full(rg)
        n_dis = bin(rf ^ rg).count('1')
        print(f"  {rf:>4} | {rg:>4} | C{cf}xC{cg:>1} | "
              f"{ok_:>4}/{tot:<6} {ok_/tot*100:5.1f}% | "
              f"{m:>7.4f} | {n_dis:>8}")

    # ── CEILING ───────────────────────────────────────────────────────────────
    var_   = fracs_arr[:, 8:16]
    wvar_  = wvars_arr
    total  = len(fracs_arr)
    no_frac  = var_.max(axis=1) < 0.01
    no_wvar  = wvar_.max(axis=1) < 0.002
    no_sig   = int((no_frac & no_wvar).sum())
    weak     = int(((var_.max(axis=1) >= 0.01) & (var_.max(axis=1) < 0.05)).sum())
    strong   = int((var_.max(axis=1) >= 0.05).sum())
    real     = (strong * 0.95 + weak * 0.65) / total * 100
    print(f"\n{'='*70}")
    print(f"  CEILING ANALYSIS")
    print(f"{'='*70}")
    print(f"  No signal    : {no_sig:>5,} ({no_sig/total*100:4.1f}%)  ← both signals absent")
    print(f"  Weak signal  : {weak:>5,} ({weak/total*100:4.1f}%)  ← hard")
    print(f"  Strong signal: {strong:>5,} ({strong/total*100:4.1f}%)  ← solvable")
    print(f"  Realistic ceiling: ~{real:.0f}%")
    print(f"  SCA advantage: W=50 cells + within_var as secondary lambda estimator")

    # ── VERDICT ───────────────────────────────────────────────────────────────
    r_pct = ok / total_n * 100
    print(f"\n{'='*70}")
    print(f"  VERDICT")
    print(f"{'='*70}")
    print(f"  Both exact (symmetric) : {r_pct:.2f}%")
    print(f"  Lambda MAE (symmetric) : {mae:.4f}")
    if   r_pct >= 95: v = "EXCELLENT (95%+)"
    elif r_pct >= 90: v = "EXCELLENT"
    elif r_pct >= 85: v = "VERY GOOD"
    elif r_pct >= 80: v = "GOOD"
    elif r_pct >= 70: v = "MODERATE"
    else:             v = "FAIR"
    print(f"\n  {v}")

    print(f"\n{'='*70}")
    print(f"  FINAL TEST COMPLETE  |  "
          f"Time: {time.time()-t_start:.1f}s  |  Log: {LOG_PATH}")
    print(f"{'='*70}")

    _log.close()

if __name__ == "__main__":
    main()
