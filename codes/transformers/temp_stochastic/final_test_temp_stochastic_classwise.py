"""
FINAL_TEST_TSCA_CLASSWISE.py  --  ECANet TSCA: Full + Class-Wise Evaluation
=============================================================================
Model : ECANetTSCA  (two-rule temporal-switching ECA, tau ∈ [TAU_MIN, TAU_MAX])
Infra : MODEL_TSCA + checkpoints_tsca_v9/stage2.pt (or stage1.pt)
Dir   : temporal_v9/

WHAT THIS DOES
--------------
1.  Generates fresh test data at runtime using simulate_labeled + compute_frac_stats.
    Matches DATAGEN_TSCA.py exactly: 500 pairs × 20 samples = 10,000 samples.
2.  Runs the SAME evaluation as FINAL_TEST_TSCA.py:
      - Both-rules-exact (symmetric: forward or flipped orientation)
      - Symmetric tau MAE: min(|pred-tau|, |pred-(1-tau)|)
      - Timestep assignment accuracy
      - Analytical tau baseline
3.  ADDITIONALLY reports accuracy broken down by Wolfram class combo:

    Class 1 (NULL / Uniform)    : rules converging to fixed point
    Class 2 (PERIODIC / FP)     : periodic / fixed-point rules
    Class 3 (CHAOTIC / COMPLEX) : chaotic rules
    Class 4 (NAMED COMPLEX)     : 41, 54, 106, 110

    For each pair-combo (e.g. C1×C2, C3×C4 etc.):
      - Both-exact accuracy
      - Tau MAE
    For each individual class:
      - List of rules with names
      - Participation in pairs from this class

Usage:
    python FINAL_TEST_TSCA_CLASSWISE.py
    (No pre-saved data needed; BASE_DIR must contain MODEL_TSCA.py and checkpoint)

BASE_DIR is auto-set to the directory containing this script.
"""

import torch
import numpy as np
import os
import sys
import time
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from MODEL_TSCA import (
    ECANetTSCA, orbit_to_tokens, simulate_labeled, compute_frac_stats,
    rule_to_bits, random_init,
    W, T, N_ORBITS, N_TOK, N_TRANS, FRAC_DIM, TAU_MIN, TAU_MAX
)

# ── Config ────────────────────────────────────────────────────────────────────
CKPT_DIR    = os.path.join(SCRIPT_DIR, "checkpoints_tsca_v9")
LOG_PATH    = os.path.join(SCRIPT_DIR, "final_test_tsca_classwise.log")
BATCH       = 8
RANDOM_SEED = 42
N_TRAIN     = 179
N_TEST      = 77
TEST_PAIRS  = 500
TEST_SAMPLES = 20      # 500 × 20 = 10,000

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

# Old 3-level classification used by pair sampling
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

# ── Pair Sampling (mirrors DATAGEN_TSCA.py) ───────────────────────────────────
from collections import Counter

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
    """Generate test samples on the fly — same as DATAGEN_TSCA.py."""
    orbits_l=[]; frac_l=[]; rfb_l=[]; rgb_l=[]
    taus_l=[]; rfi_l=[]; rgi_l=[]; slabels_l=[]

    print(f"  Generating {len(pairs)} pairs × {n_samples} samples ...")
    for pi, (f, g) in enumerate(pairs):
        bf = rule_to_bits(int(f)); bg = rule_to_bits(int(g))
        for _ in range(n_samples):
            tau = float(rng.uniform(TAU_MIN, TAU_MAX))
            orbits = []; slbls = []
            for _ in range(N_ORBITS):
                init      = random_init(rng)
                orb, slbl = simulate_labeled(int(f), int(g), init, tau, rng)
                orbits.append(orb); slbls.append(slbl)
            orbs_k = np.stack(orbits, axis=0)    # [K, T, W]
            slbs_k = np.stack(slbls,  axis=0)    # [K, T-1]
            fs     = compute_frac_stats(orbs_k)
            orbits_l.append(orbs_k); frac_l.append(fs)
            slabels_l.append(slbs_k); rfb_l.append(bf); rgb_l.append(bg)
            taus_l.append(tau); rfi_l.append(int(f)); rgi_l.append(int(g))

        if (pi + 1) % 100 == 0 or (pi + 1) == len(pairs):
            print(f"    {pi+1}/{len(pairs)} pairs done ...", end="\r")
    print()

    return (np.array(orbits_l,  dtype=np.float32),
            np.array(frac_l,    dtype=np.float32),
            np.array(rfb_l,     dtype=np.float32),
            np.array(rgb_l,     dtype=np.float32),
            np.array(taus_l,    dtype=np.float32),
            np.array(rfi_l,     dtype=np.int32),
            np.array(rgi_l,     dtype=np.int32),
            np.array(slabels_l, dtype=np.int8))

# ── Dataset ───────────────────────────────────────────────────────────────────
class TSCAInMemDataset(Dataset):
    def __init__(self, orbits, fracs, rfb, rgb, taus, rfi, rgi, slabels):
        self.orbits  = orbits; self.fracs   = fracs
        self.rfb     = rfb;    self.rgb     = rgb
        self.taus    = taus;   self.rfi     = rfi
        self.rgi     = rgi;    self.slabels = slabels

    def __len__(self): return len(self.orbits)

    def __getitem__(self, idx):
        toks = np.stack([orbit_to_tokens(self.orbits[idx, k])
                         for k in range(N_ORBITS)], axis=0)
        return (torch.tensor(toks,               dtype=torch.float32),
                torch.tensor(self.fracs[idx],    dtype=torch.float32),
                torch.tensor(self.rfb[idx],      dtype=torch.float32),
                torch.tensor(self.rgb[idx],      dtype=torch.float32),
                torch.tensor([self.taus[idx]],   dtype=torch.float32),
                int(self.rfi[idx]),
                int(self.rgi[idx]),
                torch.tensor(self.slabels[idx].astype(np.float32), dtype=torch.float32))

# ── Load Model ────────────────────────────────────────────────────────────────
def load_model():
    s2 = os.path.join(CKPT_DIR, "stage2.pt")
    s1 = os.path.join(CKPT_DIR, "stage1.pt")
    ckpt_path = s2 if os.path.exists(s2) else s1
    if not os.path.exists(ckpt_path):
        print(f"ERROR: No checkpoint found in {CKPT_DIR}")
        print("Run TRAIN_TSCA.py first."); sys.exit(1)
    if ckpt_path == s1:
        print(f"Note: stage2.pt not found — evaluating stage1.pt")
    model = ECANetTSCA().to(device)
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
    all_results    = []
    pair_summary   = defaultdict(lambda: {"both_exact":0,"total":0,"tau_errs":[]})
    tau_buckets    = defaultdict(lambda: {"both_exact":0,"total":0,"tau_errs":[]})
    class_summary  = defaultdict(lambda: {"both_exact":0,"total":0,"tau_errs":[]})
    combo_summary  = defaultdict(lambda: {"both_exact":0,"total":0,"tau_errs":[]})

    assign_fwd = 0; assign_flp = 0; assign_total = 0
    anal_preds = []; anal_trues = []

    for x, fracs, rf, rg, tau, idf, idg, slabels in loader:
        x_d = x.to(device); fr_d = fracs.to(device)
        prf, prg, pt, prob_g, _, _ = model(x_d, fr_d)
        pt_anal = model.forward_tau_analytical(fr_d)
        anal_preds.append(pt_anal.cpu()); anal_trues.append(tau)

        mean_labels = slabels.mean(dim=1).numpy()
        pg_np       = prob_g.cpu().numpy()
        pred_assign = (pg_np > 0.5).astype(np.float32)
        true_fwd    = (mean_labels > 0.5).astype(np.float32)
        assign_fwd   += (pred_assign == true_fwd).sum()
        assign_flp   += (pred_assign == (1.0 - true_fwd)).sum()
        assign_total += pred_assign.size

        prf_np = torch.sigmoid(prf).cpu().numpy()
        prg_np = torch.sigmoid(prg).cpu().numpy()
        pt_np  = pt.cpu().numpy().reshape(-1)
        trf_np = rf.numpy(); trg_np = rg.numpy()
        tt_np  = tau.numpy().reshape(-1)
        idf_np = idf.numpy() if isinstance(idf, torch.Tensor) else np.array(idf)
        idg_np = idg.numpy() if isinstance(idg, torch.Tensor) else np.array(idg)

        for i in range(len(prf_np)):
            pf = prf_np[i]; pg_ = prg_np[i]; pl = float(pt_np[i])
            tf = trf_np[i]; tg  = trg_np[i]; tl = float(tt_np[i])
            true_f = int(idf_np[i]); true_g = int(idg_np[i])

            both_fwd = (((pf > 0.5) == (tf > 0.5)).all() and
                        ((pg_ > 0.5) == (tg > 0.5)).all())
            both_flp = (((pf > 0.5) == (tg > 0.5)).all() and
                        ((pg_ > 0.5) == (tf > 0.5)).all())
            both_exact = both_fwd or both_flp
            tau_err    = min(abs(pl - tl), abs(pl - (1.0 - tl)))

            cf = RULE_CLASSES[true_f]; cg = RULE_CLASSES[true_g]
            combo_key = tuple(sorted([cf, cg]))

            all_results.append({
                "true_f": true_f, "true_g": true_g,
                "pred_f": bits_to_rule(pf), "pred_g": bits_to_rule(pg_),
                "true_tau": tl, "pred_tau": pl,
                "both_exact": both_exact, "tau_err": tau_err,
                "cls_f": cf, "cls_g": cg, "combo": combo_key,
            })

            pk = (min(true_f, true_g), max(true_f, true_g))
            pair_summary[pk]["both_exact"] += int(both_exact)
            pair_summary[pk]["total"]      += 1
            pair_summary[pk]["tau_errs"].append(tau_err)

            tb = round(round(tl * 10) / 10, 1)
            tau_buckets[tb]["both_exact"] += int(both_exact)
            tau_buckets[tb]["total"]      += 1
            tau_buckets[tb]["tau_errs"].append(tau_err)

            for cls_r in [cf, cg]:
                class_summary[cls_r]["both_exact"] += int(both_exact)
                class_summary[cls_r]["total"]       += 1
                class_summary[cls_r]["tau_errs"].append(tau_err)

            combo_summary[combo_key]["both_exact"] += int(both_exact)
            combo_summary[combo_key]["total"]      += 1
            combo_summary[combo_key]["tau_errs"].append(tau_err)

    assign_acc = max(assign_fwd, assign_flp) / assign_total * 100
    ap = torch.cat(anal_preds).squeeze()
    at = torch.cat(anal_trues).squeeze()
    anal_sym_mae = float(torch.minimum(
        (ap - at).abs(), (ap - (1.0 - at)).abs()).mean())

    return (all_results, dict(pair_summary), dict(tau_buckets),
            dict(class_summary), dict(combo_summary), assign_acc, anal_sym_mae)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    rng     = np.random.default_rng(RANDOM_SEED)

    print("=" * 70)
    print("  FINAL TEST TSCA (Class-Wise)  --  Full Evaluation")
    print("=" * 70)
    print(f"  Ckpt dir  : {CKPT_DIR}")
    print(f"  N_ORBITS  : {N_ORBITS}   W={W}   T={T}")
    print(f"  tau range : [{TAU_MIN}, {TAU_MAX}]  (continuous uniform)")
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
    orbits_arr, fracs_arr, rfb, rgb, taus_arr, rfi_arr, rgi_arr, slabels_arr = data
    print(f"  Generated {len(orbits_arr):,} samples in {time.time()-t_gen:.1f}s\n")

    model  = load_model()
    ds     = TSCAInMemDataset(*data)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False,
                        num_workers=0, pin_memory=(device.type == "cuda"))

    print(f"  Running evaluation ...")
    (results, pair_summary, tau_buckets,
     class_summary, combo_summary,
     assign_acc, anal_sym_mae) = run_evaluation(model, loader)

    total_n = len(results)
    ok      = sum(r["both_exact"] for r in results)
    mae     = float(np.mean([r["tau_err"] for r in results]))

    # ── OVERALL ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  OVERALL RESULTS")
    print(f"{'='*70}")
    print(f"  Test samples         : {total_n:,}")
    print(f"  Both exact (sym)     : {ok}/{total_n}  ({ok/total_n*100:.2f}%)")
    print(f"  Tau MAE (sym)        : {mae:.4f}")
    print(f"  Analytical tau MAE   : {anal_sym_mae:.4f}")
    print(f"  Assignment accuracy  : {assign_acc:.2f}%")

    # ── PER-TAU BREAKDOWN ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  PER-TAU BREAKDOWN")
    print(f"{'='*70}")
    print(f"  {'Tau':>5} | {'BothExact':>12} | {'TauMAE':>7} | Visual")
    print(f"  {'-'*55}")
    for tau in sorted(tau_buckets.keys()):
        s    = tau_buckets[tau]
        bk   = s["both_exact"]; tot = s["total"]
        pct  = bk / tot * 100
        tmae = float(np.mean(s["tau_errs"]))
        bar  = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        print(f"  {tau:>5.1f} | {bk:>4}/{tot:<6} {pct:5.1f}% | "
              f"{tmae:>7.4f} | [{bar}]")

    # ── TAU ESTIMATION ACCURACY ────────────────────────────────────────────────
    errs = [r["tau_err"] for r in results]
    print(f"\n{'='*70}")
    print(f"  TAU ESTIMATION ACCURACY (symmetric)")
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
    Class 4 — NAMED COMPLEX   : 41, 54, 106, 110 (universal computation / gliders)

  NOTE: Each sample involves TWO rules, so a sample appears in BOTH its
        constituent class entries below. Combo breakdown is more granular.
""")

    for cls_id in ['1', '2', '3', '4']:
        label = CLASS_LABELS[cls_id]
        cs    = class_summary.get(cls_id, None)
        if cs is None or cs["total"] == 0:
            print(f"\n  ── {label}\n     (No samples)")
            continue

        cls_rules = sorted(r for r in range(256) if RULE_CLASSES[r] == cls_id)
        test_cls_rules = [r for r in cls_rules if r in test_rules]
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
        m   = float(np.mean(cs["tau_errs"]))
        print(f"\n     Accuracy (samples where at least one rule ∈ this class):")
        print(f"       Both exact    : {ok_}/{n}  ({ok_/n*100:.2f}%)")
        print(f"       Tau MAE (sym) : {m:.4f}")

    # ── COMBO BREAKDOWN ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  CLASS COMBO BREAKDOWN  (each sample = one combo)")
    print(f"{'='*70}")
    print(f"  {'Combo':>6} | {'BothExact':>14} | {'TauMAE':>7} | {'N':>6}")
    print(f"  {'-'*50}")

    # All possible combos
    all_combos = []
    for a in ['1','2','3','4']:
        for b in ['1','2','3','4']:
            if (a, b) not in all_combos and (b, a) not in all_combos:
                all_combos.append((a, b))

    for combo in sorted(all_combos):
        cs = combo_summary.get(combo, None)
        if cs is None or cs["total"] == 0: continue
        ok_ = cs["both_exact"]; tot = cs["total"]
        m   = float(np.mean(cs["tau_errs"]))
        print(f"  C{combo[0]}×C{combo[1]:>1}  | {ok_:>4}/{tot:<8} {ok_/tot*100:5.1f}% | "
              f"{m:>7.4f} | {tot:>6}")

    # ── CLASS SUMMARY TABLE ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  CLASS SUMMARY TABLE (single-rule perspective)")
    print(f"{'='*70}")
    print(f"  {'Class':>12} | {'Appearances':>11} | "
          f"{'BothExact%':>10} | {'Tau MAE':>7}")
    print(f"  {'-'*55}")
    for cls_id in ['1', '2', '3', '4']:
        cs = class_summary.get(cls_id, None)
        if cs is None or cs["total"] == 0: continue
        n = cs["total"]; ok_ = cs["both_exact"]
        m = float(np.mean(cs["tau_errs"]))
        print(f"  {CLASS_SHORT[cls_id]:>12} | {n:>11} | "
              f"{ok_/n*100:>10.2f}% | {m:>7.4f}")

    # ── HARDEST PAIRS ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  HARDEST RULE PAIRS (top 20)")
    print(f"{'='*70}")
    print(f"  {'RF':>4} | {'RG':>4} | {'Cls':>5} | {'BothExact':>12} | {'TauMAE':>7}")
    print(f"  {'-'*55}")
    sorted_pairs = sorted(pair_summary.items(),
                          key=lambda x: (x[1]["both_exact"] / x[1]["total"],
                                         np.mean(x[1]["tau_errs"])))
    for (rf, rg), s in sorted_pairs[:20]:
        tot  = s["total"]; ok_ = s["both_exact"]
        m    = float(np.mean(s["tau_errs"]))
        cf   = wolfram_class_full(rf); cg = wolfram_class_full(rg)
        print(f"  {rf:>4} | {rg:>4} | C{cf}xC{cg:>1} | "
              f"{ok_:>4}/{tot:<6} {ok_/tot*100:5.1f}% | {m:>7.4f}")

    # ── CEILING ───────────────────────────────────────────────────────────────
    var_   = fracs_arr[:, 8:16]
    total  = len(fracs_arr)
    no_sig = int((var_.max(axis=1) < 0.01).sum())
    weak   = int(((var_.max(axis=1) >= 0.01) & (var_.max(axis=1) < 0.05)).sum())
    strong = int((var_.max(axis=1) >= 0.05).sum())
    real   = (strong * 0.95 + weak * 0.65) / total * 100
    print(f"\n{'='*70}")
    print(f"  CEILING ANALYSIS")
    print(f"{'='*70}")
    print(f"  No signal    : {no_sig:>5,} ({no_sig/total*100:4.1f}%)  ← unidentifiable")
    print(f"  Weak signal  : {weak:>5,} ({weak/total*100:4.1f}%)  ← hard")
    print(f"  Strong signal: {strong:>5,} ({strong/total*100:4.1f}%)  ← solvable")
    print(f"  Realistic ceiling: ~{real:.0f}%")

    print(f"\n{'='*70}")
    print(f"  FINAL TEST COMPLETE  |  "
          f"Time: {time.time()-t_start:.1f}s  |  Log: {LOG_PATH}")
    print(f"{'='*70}")

    _log.close()

if __name__ == "__main__":
    main()
