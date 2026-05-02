"""
FINAL_TEST_TSCA.py  --  v9
============================
Directory: temporal_v9/

Comprehensive evaluation of ECANetTSCA v9.
Paths match TRAIN_TSCA.py and DATAGEN_TSCA.py exactly.

All tau metrics use SYMMETRIC MAE: min(|pred-tau|, |pred-(1-tau)|)
because tau_pred is always in [TAU_MIN, 0.5] by model design.

REPORTS:
  1. Overall accuracy (both rules exact, symmetric)
  2. Analytical tau baseline (before any training)
  3. Timestep assignment accuracy (how well prob_g tracks step_labels)
  4. Per-tau breakdown
  5. Tau estimation accuracy at various error thresholds
  6. Wolfram class combo breakdown
  7. Hardest rule pairs
  8. Ceiling analysis (information-theoretic limit)
  9. Verdict

Usage:
    python FINAL_TEST_TSCA.py
    (Run DATAGEN_TSCA.py and TRAIN_TSCA.py first)
"""

import torch
import numpy as np
import os, sys, time
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

# ══════════════════════════════════════════════════════════
BASE_DIR  = "/home/shovik.roy/Shrey/new_check_model/ECA_temporal_stocastic/temporal_v9"
CKPT_DIR  = os.path.join(BASE_DIR, "checkpoints_tsca_v9")
DATA_DIR  = os.path.join(BASE_DIR, "TSCA_Data")
TEST_PATH = os.path.join(DATA_DIR, "test")
BATCH     = 8
# ══════════════════════════════════════════════════════════

sys.path.insert(0, BASE_DIR)

from MODEL_TSCA import (
    ECANetTSCA, orbit_to_tokens,
    W, T, N_ORBITS, N_TOK, N_TRANS, FRAC_DIM, TAU_MIN, TAU_MAX
)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Device: CUDA -- {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps"); print("Device: Apple MPS")
else:
    device = torch.device("cpu"); print("Device: CPU")


# ── Wolfram classification ────────────────────────────────────────────────────
_CLASS_I   = {0, 8, 32, 40, 128, 136, 160, 168}
_CLASS_III = {18, 22, 30, 45, 60, 90, 105, 122, 126, 146, 150}
_CLASS_IV  = {41, 54, 106, 110}
_CLASS_LC  = {26, 73, 154}

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

def wolfram_class(r):
    m = _min_rep(r)
    if m in _CLASS_I:                           return 'A'
    if m in _CLASS_III | _CLASS_IV | _CLASS_LC: return 'C'
    return 'B'

def bits_to_rule(bits):
    b = (np.array(bits) > 0.5).astype(int)
    return int(sum(b[i] * (2 ** i) for i in range(8)))


# ── Dataset ───────────────────────────────────────────────────────────────────
class TSCATestDataset(Dataset):
    def __init__(self, path):
        self.orbits   = np.load(os.path.join(path, "orbits.npy"))
        self.fracs    = np.load(os.path.join(path, "frac_stats.npy"))
        self.rf_bits  = np.load(os.path.join(path, "rule_f_bits.npy"))
        self.rg_bits  = np.load(os.path.join(path, "rule_g_bits.npy"))
        self.taus     = np.load(os.path.join(path, "taus.npy"))
        self.rf_ids   = np.load(os.path.join(path, "rule_f_ids.npy"))
        self.rg_ids   = np.load(os.path.join(path, "rule_g_ids.npy"))
        self.slabels  = np.load(os.path.join(path, "step_labels.npy"))
        assert self.orbits.ndim == 4, \
            f"Expected [N,K,T,W], got {self.orbits.shape}"
        assert self.orbits.shape[1] == N_ORBITS, \
            f"Expected N_ORBITS={N_ORBITS}, got {self.orbits.shape[1]}"

    def __len__(self): return len(self.orbits)

    def __getitem__(self, idx):
        # orbit_to_tokens returns 4-feature raw tokens [L,B,R,A]
        toks = np.stack([orbit_to_tokens(self.orbits[idx, k])
                         for k in range(N_ORBITS)], axis=0)
        return (torch.tensor(toks,                dtype=torch.float32),
                torch.tensor(self.fracs[idx],     dtype=torch.float32),
                torch.tensor(self.rf_bits[idx],   dtype=torch.float32),
                torch.tensor(self.rg_bits[idx],   dtype=torch.float32),
                torch.tensor([self.taus[idx]],    dtype=torch.float32),
                int(self.rf_ids[idx]),
                int(self.rg_ids[idx]),
                torch.tensor(self.slabels[idx],   dtype=torch.float32))


# ── Model loader ──────────────────────────────────────────────────────────────
def load_model():
    # Try stage2 first, fall back to stage1
    s2 = os.path.join(CKPT_DIR, "stage2.pt")
    s1 = os.path.join(CKPT_DIR, "stage1.pt")
    ckpt_path = s2 if os.path.exists(s2) else s1
    if not os.path.exists(ckpt_path):
        print(f"ERROR: No checkpoint found in {CKPT_DIR}")
        print("Run TRAIN_TSCA.py first.")
        sys.exit(1)
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
    all_results   = []
    pair_summary  = defaultdict(lambda: {"both_exact":0,"total":0,"tau_errs":[]})
    tau_buckets   = defaultdict(lambda: {"both_exact":0,"total":0,"tau_errs":[]})
    class_summary = defaultdict(lambda: {"both_exact":0,"total":0,"tau_errs":[]})

    # Assignment accuracy: take best of forward/flipped orientation
    assign_fwd = 0; assign_flp = 0; assign_total = 0

    # Analytical tau baseline (before correction MLP)
    anal_preds = []; anal_trues = []

    for x, fracs, rf, rg, tau, idf, idg, slabels in loader:
        x_d = x.to(device); fr_d = fracs.to(device)

        # Full model forward
        prf, prg, pt, prob_g, _, _ = model(x_d, fr_d)

        # Analytical-only tau for baseline
        pt_anal = model.forward_tau_analytical(fr_d)
        anal_preds.append(pt_anal.cpu()); anal_trues.append(tau)

        # Assignment accuracy (best orientation)
        mean_labels = slabels.mean(dim=1).numpy()   # [B, N_TRANS]
        pg_np       = prob_g.cpu().numpy()
        pred_assign = (pg_np > 0.5).astype(np.float32)
        true_fwd    = (mean_labels > 0.5).astype(np.float32)
        assign_fwd    += (pred_assign == true_fwd).sum()
        assign_flp    += (pred_assign == (1.0 - true_fwd)).sum()
        assign_total  += pred_assign.size

        prf_np = torch.sigmoid(prf).cpu().numpy()
        prg_np = torch.sigmoid(prg).cpu().numpy()
        pt_np  = pt.cpu().numpy().reshape(-1)
        trf_np = rf.numpy(); trg_np = rg.numpy()
        tt_np  = tau.numpy().reshape(-1)

        for i in range(len(prf_np)):
            pf = prf_np[i]; pg_ = prg_np[i]; pl = float(pt_np[i])
            tf = trf_np[i]; tg  = trg_np[i]; tl = float(tt_np[i])

            both_fwd = (((pf > 0.5) == (tf > 0.5)).all() and
                        ((pg_ > 0.5) == (tg > 0.5)).all())
            both_flp = (((pf > 0.5) == (tg > 0.5)).all() and
                        ((pg_ > 0.5) == (tf > 0.5)).all())
            both_exact = both_fwd or both_flp

            # Symmetric tau MAE: min(|pred-tau|, |pred-(1-tau)|)
            tau_err = min(abs(pl - tl), abs(pl - (1.0 - tl)))

            true_f = int(idf[i]); true_g = int(idg[i])
            all_results.append({
                "true_f": true_f, "true_g": true_g,
                "pred_f": bits_to_rule(pf), "pred_g": bits_to_rule(pg_),
                "true_tau": tl, "pred_tau": pl,
                "both_exact": both_exact, "tau_err": tau_err,
            })

            pk = (min(true_f, true_g), max(true_f, true_g))
            pair_summary[pk]["both_exact"] += int(both_exact)
            pair_summary[pk]["total"]      += 1
            pair_summary[pk]["tau_errs"].append(tau_err)

            tb = round(round(tl * 10) / 10, 1)
            tau_buckets[tb]["both_exact"] += int(both_exact)
            tau_buckets[tb]["total"]      += 1
            tau_buckets[tb]["tau_errs"].append(tau_err)

            cf = wolfram_class(true_f); cg = wolfram_class(true_g)
            ck_ = tuple(sorted([cf, cg]))
            class_summary[ck_]["both_exact"] += int(both_exact)
            class_summary[ck_]["total"]      += 1
            class_summary[ck_]["tau_errs"].append(tau_err)

    # Best-orientation assignment accuracy
    assign_acc = max(assign_fwd, assign_flp) / max(assign_total, 1) * 100

    # Analytical tau symmetric MAE
    ap = torch.cat(anal_preds).squeeze()
    at = torch.cat(anal_trues).squeeze()
    anal_sym_mae = float(torch.minimum(
        (ap - at).abs(), (ap - (1.0 - at)).abs()).mean())

    return (all_results, dict(pair_summary), dict(tau_buckets),
            dict(class_summary), assign_acc, anal_sym_mae)


# ── Reports ───────────────────────────────────────────────────────────────────

def report_overall(results):
    n   = len(results)
    ok  = sum(r["both_exact"] for r in results)
    mae = np.mean([r["tau_err"] for r in results])
    print(f"\n{'='*65}\n  OVERALL RESULTS\n{'='*65}")
    print(f"  Test samples         : {n:,}")
    print(f"  Both exact           : {ok}/{n}  ({ok/n*100:.2f}%)  [symmetric]")
    print(f"  Tau MAE (symmetric)  : {mae:.4f}")
    print(f"    (symmetric = min(|pred-tau|, |pred-(1-tau)|))")
    print(f"    (tau_pred always ≤ 0.5 by model design)")


def report_analytical_tau(anal_sym_mae):
    print(f"\n{'='*65}\n  ANALYTICAL TAU BASELINE (before correction MLP)\n{'='*65}")
    print(f"  Analytical-only symmetric MAE : {anal_sym_mae:.4f}")
    if anal_sym_mae < 0.05:
        tag = "EXCELLENT -- correction MLP has minimal work to do"
    elif anal_sym_mae < 0.08:
        tag = "GOOD"
    else:
        tag = "HIGH -- many all-agreement pairs in test set"
    print(f"  {tag}")


def report_assignment(assign_acc):
    print(f"\n{'='*65}\n  TIMESTEP ASSIGNMENT ACCURACY\n{'='*65}")
    print(f"  Best-orientation accuracy : {assign_acc:.2f}%")
    print(f"  (how well prob_g[t] tracks which rule fired at timestep t)")
    print(f"  50% = random, 100% = perfect")
    if assign_acc > 80:
        print(f"  GOOD: assignment is learning correctly")
    elif assign_acc > 65:
        print(f"  MODERATE: assignment partially learned")
    else:
        print(f"  LOW: assignment not working well -- residual_weight may need tuning")


def report_tau_breakdown(tau_buckets):
    print(f"\n{'='*65}\n  PER-TAU BREAKDOWN\n{'='*65}")
    print(f"  {'Tau':>5} | {'BothExact':>12} | {'TauMAE':>7} | Visual")
    print(f"  {'-'*58}")
    for tau in sorted(tau_buckets.keys()):
        s   = tau_buckets[tau]
        ok  = s["both_exact"]; tot = s["total"]
        pct = ok / tot * 100
        mae = float(np.mean(s["tau_errs"]))
        bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        print(f"  {tau:>5.1f} | {ok:>4}/{tot:<6} {pct:5.1f}% | "
              f"{mae:>7.4f} | [{bar}]")
    print(f"  Note: tau near 0.5 is easiest (both rules fire ~equally).")
    print(f"        tau near 0.1/0.9 is hardest (one rule rarely fires).")


def report_tau_accuracy(results):
    errs = [r["tau_err"] for r in results]
    print(f"\n{'='*65}\n  TAU ESTIMATION ACCURACY (symmetric)\n{'='*65}")
    for t in [0.02, 0.05, 0.10, 0.15, 0.20]:
        frac = float(np.mean([e <= t for e in errs])) * 100
        bar  = "█" * int(frac / 5)
        print(f"  Error <= {t:.2f}  :  {frac:5.1f}%  {bar}")
    print(f"\n  Mean   : {float(np.mean(errs)):.4f}")
    print(f"  Median : {float(np.median(errs)):.4f}")
    print(f"  Max    : {float(np.max(errs)):.4f}")


def report_class_breakdown(class_summary):
    print(f"\n{'='*65}\n  WOLFRAM CLASS COMBO BREAKDOWN\n{'='*65}")
    print(f"  {'Combo':>6} | {'BothExact':>12} | {'TauMAE':>7} | {'N':>6}")
    print(f"  {'-'*45}")
    for key in sorted(class_summary.keys()):
        s   = class_summary[key]
        ok  = s["both_exact"]; tot = s["total"]
        mae = float(np.mean(s["tau_errs"]))
        print(f"  {key[0]}x{key[1]:>4} | {ok:>4}/{tot:<6} {ok/tot*100:5.1f}% | "
              f"{mae:>7.4f} | {tot:>6}")


def report_hardest_pairs(pair_summary, n=20):
    print(f"\n{'='*65}\n  HARDEST RULE PAIRS\n{'='*65}")
    print(f"  {'RF':>4} | {'RG':>4} | {'Cls':>5} | {'BothExact':>12} | "
          f"{'TauMAE':>7} | {'Disagree':>8}")
    print(f"  {'-'*62}")
    sorted_pairs = sorted(
        pair_summary.items(),
        key=lambda x: (x[1]["both_exact"] / x[1]["total"],
                       np.mean(x[1]["tau_errs"])))
    for (rf, rg), s in sorted_pairs[:n]:
        tot  = s["total"]; ok = s["both_exact"]
        pct  = ok / tot * 100
        mae  = float(np.mean(s["tau_errs"]))
        cls  = f"{wolfram_class(rf)}x{wolfram_class(rg)}"
        n_disagree = bin(rf ^ rg).count('1')
        print(f"  {rf:>4} | {rg:>4} | {cls:>5} | "
              f"{ok:>4}/{tot:<6} {pct:5.1f}% | {mae:>7.4f} | "
              f"{n_disagree:>8}")
    perfect = sum(1 for s in pair_summary.values()
                  if s["both_exact"] == s["total"])
    print(f"\n  Pairs with 100% both-exact : {perfect}/{len(pair_summary)}")
    print(f"  Note: 'Disagree' = number of bit positions where f[n]≠g[n].")
    print(f"        0 disagree = identical rules (unidentifiable).")
    print(f"        1-2 disagree = very hard (few discriminative patterns).")


def report_ceiling(fracs, taus):
    """Information-theoretic ceiling analysis."""
    var_   = fracs[:, 8:16]
    total  = len(fracs)
    no_sig = int((var_.max(axis=1) < 0.01).sum())
    weak   = int(((var_.max(axis=1) >= 0.01) & (var_.max(axis=1) < 0.05)).sum())
    strong = int((var_.max(axis=1) >= 0.05).sum())
    real   = (strong * 0.95 + weak * 0.65) / total * 100
    print(f"\n{'='*65}\n  CEILING ANALYSIS\n{'='*65}")
    print(f"  No signal   (max_var<0.01): {no_sig:>5,} ({no_sig/total*100:4.1f}%)"
          f"  ← unidentifiable")
    print(f"  Weak signal (0.01–0.05)  : {weak:>5,} ({weak/total*100:4.1f}%)"
          f"  ← hard")
    print(f"  Strong signal (≥0.05)    : {strong:>5,} ({strong/total*100:4.1f}%)"
          f"  ← solvable")
    print(f"  Realistic ceiling        : ~{real:.0f}%")


def report_verdict(results):
    n     = len(results)
    r_pct = sum(r["both_exact"] for r in results) / n * 100
    t_mae = np.mean([r["tau_err"] for r in results])
    print(f"\n{'='*65}\n  VERDICT\n{'='*65}")
    print(f"  Both exact (symmetric) : {r_pct:.2f}%")
    print(f"  Tau MAE (symmetric)    : {t_mae:.4f}")
    if   r_pct >= 95: v = "EXCELLENT (95%+)"
    elif r_pct >= 90: v = "EXCELLENT"
    elif r_pct >= 85: v = "VERY GOOD"
    elif r_pct >= 80: v = "GOOD"
    elif r_pct >= 70: v = "MODERATE"
    else:             v = "FAIR"
    print(f"\n  {v}")
    print("=" * 65)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()

    # Verify all required files
    required = ["orbits.npy", "frac_stats.npy", "rule_f_bits.npy",
                "rule_g_bits.npy", "taus.npy", "rule_f_ids.npy",
                "rule_g_ids.npy", "step_labels.npy"]
    for fname in required:
        p = os.path.join(TEST_PATH, fname)
        if not os.path.exists(p):
            print(f"ERROR: {p} not found.")
            print("Run DATAGEN_TSCA.py first.")
            sys.exit(1)

    print("=" * 65)
    print("  FINAL_TEST_TSCA v9  --  Full Evaluation")
    print("=" * 65)
    print(f"  Ckpt dir : {CKPT_DIR}")
    print(f"  Data     : {TEST_PATH}")
    print(f"  N_ORBITS : {N_ORBITS}")
    print(f"  tau_pred is always in [TAU_MIN, 0.5] by model design")
    print(f"  All tau metrics use symmetric MAE")
    print("=" * 65 + "\n")

    model  = load_model()
    ds     = TSCATestDataset(TEST_PATH)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False,
                        num_workers=0, pin_memory=(device.type == "cuda"))
    print(f"  Test samples : {len(ds):,}")
    print(f"  Running evaluation ...")

    (results, pair_summary, tau_buckets,
     class_summary, assign_acc, anal_sym_mae) = run_evaluation(model, loader)

    report_overall(results)
    report_analytical_tau(anal_sym_mae)
    report_assignment(assign_acc)
    report_tau_breakdown(tau_buckets)
    report_tau_accuracy(results)
    report_class_breakdown(class_summary)
    report_hardest_pairs(pair_summary)

    # Load raw fracs/taus for ceiling analysis
    fracs_all = np.load(os.path.join(TEST_PATH, "frac_stats.npy"))
    taus_all  = np.load(os.path.join(TEST_PATH, "taus.npy"))
    report_ceiling(fracs_all, taus_all)

    report_verdict(results)
    print(f"\n  Total evaluation time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
