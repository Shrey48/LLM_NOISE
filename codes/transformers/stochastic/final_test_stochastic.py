"""
FINAL_TEST_SCA.py  --  v1
===========================
Directory: stochastic_v1/

Comprehensive evaluation of ECANetSCA v1.
Mirrors FINAL_TEST_TSCA.py structure exactly.

All lambda metrics use SYMMETRIC MAE: min(|pred-lam|, |pred-(1-lam)|)
because lam_pred is always in [LAM_MIN, 0.5] by model design.
Identical reasoning to TSCA's symmetric tau MAE.

REPORTS:
  1. Overall accuracy (both rules exact, symmetric)
  2. Analytical lambda baseline
  3. Per-lambda breakdown (bucketed by true lambda)
  4. Lambda estimation accuracy at various thresholds
  5. Wolfram class combo breakdown
  6. Hardest rule pairs
  7. Ceiling analysis
  8. Verdict

Usage:
    python FINAL_TEST_SCA.py
    (Run DATAGEN_SCA.py and TRAIN_SCA.py first)
"""

import torch
import numpy as np
import os, sys, time
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

# ══════════════════════════════════════════════════════════════════
BASE_DIR = os.environ.get("BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR  = os.path.join(BASE_DIR, "checkpoints_sca_v1")
DATA_DIR  = os.path.join(BASE_DIR, "SCA_Data")
TEST_PATH = os.path.join(DATA_DIR, "test")
BATCH     = 8
# ══════════════════════════════════════════════════════════════════

sys.path.insert(0, BASE_DIR)

from MODEL_SCA import (
    ECANetSCA, orbit_to_tokens,
    W, T, N_ORBITS, N_TOK, N_TRANS, FRAC_DIM, WVAR_DIM, LAM_MIN, LAM_MAX
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
class SCATestDataset(Dataset):
    def __init__(self, path):
        self.orbits  = np.load(os.path.join(path, "orbits.npy"))
        self.fracs   = np.load(os.path.join(path, "frac_stats.npy"))
        self.wvars   = np.load(os.path.join(path, "within_var_stats.npy"))
        self.rf_bits = np.load(os.path.join(path, "rule_f_bits.npy"))
        self.rg_bits = np.load(os.path.join(path, "rule_g_bits.npy"))
        self.lams    = np.load(os.path.join(path, "lambdas.npy"))
        self.rf_ids  = np.load(os.path.join(path, "rule_f_ids.npy"))
        self.rg_ids  = np.load(os.path.join(path, "rule_g_ids.npy"))
        assert self.orbits.ndim == 4 and self.orbits.shape[1] == N_ORBITS

    def __len__(self): return len(self.orbits)

    def __getitem__(self, idx):
        toks = np.stack([orbit_to_tokens(self.orbits[idx, k])
                         for k in range(N_ORBITS)], axis=0)
        return (torch.tensor(toks,               dtype=torch.float32),
                torch.tensor(self.fracs[idx],    dtype=torch.float32),
                torch.tensor(self.wvars[idx],    dtype=torch.float32),
                torch.tensor(self.rf_bits[idx],  dtype=torch.float32),
                torch.tensor(self.rg_bits[idx],  dtype=torch.float32),
                torch.tensor([self.lams[idx]],   dtype=torch.float32),
                int(self.rf_ids[idx]),
                int(self.rg_ids[idx]))


# ── Model loader ──────────────────────────────────────────────────────────────
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

    anal_preds=[]; anal_trues=[]

    for x, fracs, wvars, rf, rg, lam, idf, idg in loader:
        x_d  = x.to(device); fr_d = fracs.to(device); wv_d = wvars.to(device)

        prf, prg, lp, _, _ = model(x_d, fr_d, wv_d)

        # Analytical-only lambda baseline
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

            # Symmetric lambda MAE: min(|pred-lam|, |pred-(1-lam)|)
            lam_err = min(abs(pl - tl), abs(pl - (1.0 - tl)))

            all_results.append({
                "true_f": true_f, "true_g": true_g,
                "pred_f": bits_to_rule(pf), "pred_g": bits_to_rule(pg_),
                "true_lam": tl, "pred_lam": pl,
                "both_exact": both_exact, "lam_err": lam_err,
            })

            pk = (min(true_f, true_g), max(true_f, true_g))
            pair_summary[pk]["both_exact"] += int(both_exact)
            pair_summary[pk]["total"]      += 1
            pair_summary[pk]["lam_errs"].append(lam_err)

            # Bucket by true lambda (round to nearest 0.1)
            tb = round(round(tl * 10) / 10, 1)
            lam_buckets[tb]["both_exact"] += int(both_exact)
            lam_buckets[tb]["total"]      += 1
            lam_buckets[tb]["lam_errs"].append(lam_err)

            cf = wolfram_class(true_f); cg = wolfram_class(true_g)
            ck_ = tuple(sorted([cf, cg]))
            class_summary[ck_]["both_exact"] += int(both_exact)
            class_summary[ck_]["total"]      += 1
            class_summary[ck_]["lam_errs"].append(lam_err)

    # Analytical lambda symmetric MAE
    ap = torch.cat(anal_preds).squeeze()
    at = torch.cat(anal_trues).squeeze()
    anal_sym_mae = float(torch.minimum(
        (ap - at).abs(), (ap - (1.0 - at)).abs()).mean())

    return (all_results, dict(pair_summary), dict(lam_buckets),
            dict(class_summary), anal_sym_mae)


# ── Reports (mirror FINAL_TEST_TSCA.py exactly) ───────────────────────────────

def report_overall(results):
    n   = len(results)
    ok  = sum(r["both_exact"] for r in results)
    mae = np.mean([r["lam_err"] for r in results])
    print(f"\n{'='*65}\n  OVERALL RESULTS\n{'='*65}")
    print(f"  Test samples         : {n:,}")
    print(f"  Both exact           : {ok}/{n}  ({ok/n*100:.2f}%)  [symmetric]")
    print(f"  Lambda MAE (sym)     : {mae:.4f}")
    print(f"    (symmetric = min(|pred-lam|, |pred-(1-lam)|))")
    print(f"    (lam_pred always ≤ 0.5 by model design)")


def report_analytical_lambda(anal_sym_mae):
    print(f"\n{'='*65}\n  ANALYTICAL LAMBDA BASELINE\n{'='*65}")
    print(f"  Symmetric MAE (analytical only) : {anal_sym_mae:.4f}")
    if anal_sym_mae < 0.04:
        tag = "EXCELLENT — within_var gives very clean signal"
    elif anal_sym_mae < 0.07:
        tag = "GOOD"
    else:
        tag = "HIGH — many all-agreement pairs"
    print(f"  {tag}")
    print(f"  Note: SCA analytical baseline is typically better than TSCA")
    print(f"  because W=50 cells give much more averaging per timestep")


def report_lambda_breakdown(lam_buckets):
    print(f"\n{'='*65}\n  PER-LAMBDA BREAKDOWN\n{'='*65}")
    print(f"  {'Lambda':>7} | {'BothExact':>12} | {'LamMAE':>7} | Visual")
    print(f"  {'-'*58}")
    for lam in sorted(lam_buckets.keys()):
        s   = lam_buckets[lam]
        ok  = s["both_exact"]; tot = s["total"]
        pct = ok / tot * 100 if tot > 0 else 0.0
        mae = float(np.mean(s["lam_errs"])) if s["lam_errs"] else 0.0
        bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        sym = " ←sym" if abs(lam - 0.5) < 0.01 else ""
        print(f"  {lam:>7.1f} | {ok:>4}/{tot:<6} {pct:5.1f}% | "
              f"{mae:>7.4f} | [{bar}]{sym}")
    print(f"\n  Note: lambda≈0.5 hardest to orient (symmetric).")
    print(f"        lambda≈0.1/0.9 hardest for rule identification (one rule rare).")


def report_lambda_accuracy(results):
    errs = [r["lam_err"] for r in results]
    print(f"\n{'='*65}\n  LAMBDA ESTIMATION ACCURACY (symmetric)\n{'='*65}")
    for t in [0.02, 0.05, 0.10, 0.15, 0.20]:
        frac = float(np.mean([e <= t for e in errs])) * 100
        bar  = "█" * int(frac / 5)
        print(f"  Error <= {t:.2f}  :  {frac:5.1f}%  {bar}")
    print(f"\n  Mean   : {float(np.mean(errs)):.4f}")
    print(f"  Median : {float(np.median(errs)):.4f}")
    print(f"  Max    : {float(np.max(errs)):.4f}")


def report_class_breakdown(class_summary):
    print(f"\n{'='*65}\n  WOLFRAM CLASS COMBO BREAKDOWN\n{'='*65}")
    print(f"  {'Combo':>6} | {'BothExact':>12} | {'LamMAE':>7} | {'N':>6}")
    print(f"  {'-'*45}")
    for key in sorted(class_summary.keys()):
        s   = class_summary[key]
        ok  = s["both_exact"]; tot = s["total"]
        mae = float(np.mean(s["lam_errs"])) if s["lam_errs"] else 0.0
        print(f"  {key[0]}x{key[1]:>4} | {ok:>4}/{tot:<6} {ok/tot*100:5.1f}% | "
              f"{mae:>7.4f} | {tot:>6}")


def report_hardest_pairs(pair_summary, n=20):
    print(f"\n{'='*65}\n  HARDEST RULE PAIRS\n{'='*65}")
    print(f"  {'RF':>4} | {'RG':>4} | {'Cls':>5} | {'BothExact':>12} | "
          f"{'LamMAE':>7} | {'Disagree':>8}")
    print(f"  {'-'*62}")
    sorted_pairs = sorted(
        pair_summary.items(),
        key=lambda x: (x[1]["both_exact"] / x[1]["total"],
                       np.mean(x[1]["lam_errs"])))
    for (rf, rg), s in sorted_pairs[:n]:
        tot  = s["total"]; ok = s["both_exact"]
        pct  = ok / tot * 100
        mae  = float(np.mean(s["lam_errs"]))
        cls  = f"{wolfram_class(rf)}x{wolfram_class(rg)}"
        n_dis = bin(rf ^ rg).count('1')
        print(f"  {rf:>4} | {rg:>4} | {cls:>5} | "
              f"{ok:>4}/{tot:<6} {pct:5.1f}% | {mae:>7.4f} | {n_dis:>8}")
    perfect = sum(1 for s in pair_summary.values()
                  if s["both_exact"] == s["total"])
    print(f"\n  Pairs with 100% both-exact : {perfect}/{len(pair_summary)}")
    print(f"  Note: 'Disagree' = bit positions where f[n]≠g[n].")
    print(f"        0 disagree → identical rules, unidentifiable.")
    print(f"        1-2 disagree → few discriminative patterns, hard.")


def report_ceiling(wvars, fracs, lams):
    var_  = fracs[:, 8:16]
    wvar_ = wvars
    total = len(fracs)
    lam_arr = np.array(lams)

    no_frac  = var_.max(axis=1) < 0.01
    no_wvar  = wvar_.max(axis=1) < 0.002
    no_sig   = int((no_frac & no_wvar).sum())
    weak     = int(((var_.max(axis=1) >= 0.01) & (var_.max(axis=1) < 0.05)).sum())
    strong   = int((var_.max(axis=1) >= 0.05).sum())
    near_half = int((np.abs(lam_arr - 0.5) < 0.05).sum())
    realistic = (strong * 0.95 + weak * 0.65) / total * 100

    print(f"\n{'='*65}\n  CEILING ANALYSIS\n{'='*65}")
    print(f"  Total samples        : {total:,}")
    print(f"  No signal            : {no_sig:>5,} ({no_sig/total*100:4.1f}%)"
          f"  ← rules agree on all patterns")
    print(f"  Weak signal          : {weak:>5,} ({weak/total*100:4.1f}%)"
          f"  ← hard")
    print(f"  Strong signal        : {strong:>5,} ({strong/total*100:4.1f}%)"
          f"  ← solvable")
    print(f"  lambda≈0.5           : {near_half:>5,} ({near_half/total*100:4.1f}%)"
          f"  ← hardest orientation")
    print(f"  Realistic ceiling    : ~{realistic:.0f}%")
    print(f"  SCA advantage vs TSCA: W=50 draws/timestep + within_var signal")


def report_verdict(results):
    n     = len(results)
    r_pct = sum(r["both_exact"] for r in results) / n * 100
    t_mae = np.mean([r["lam_err"] for r in results])
    print(f"\n{'='*65}\n  VERDICT\n{'='*65}")
    print(f"  Both exact (symmetric) : {r_pct:.2f}%")
    print(f"  Lambda MAE (symmetric) : {t_mae:.4f}")
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

    required = ["orbits.npy", "frac_stats.npy", "within_var_stats.npy",
                "rule_f_bits.npy", "rule_g_bits.npy",
                "lambdas.npy", "rule_f_ids.npy", "rule_g_ids.npy"]
    for fname in required:
        p = os.path.join(TEST_PATH, fname)
        if not os.path.exists(p):
            print(f"ERROR: {p} not found. Run DATAGEN_SCA.py first.")
            sys.exit(1)

    print("=" * 65)
    print("  FINAL_TEST_SCA v1  --  Full Evaluation")
    print("=" * 65)
    print(f"  Ckpt dir : {CKPT_DIR}")
    print(f"  Data     : {TEST_PATH}")
    print(f"  N_ORBITS : {N_ORBITS}")
    print(f"  Lambda   : continuous [{LAM_MIN}, {LAM_MAX}]")
    print(f"  lam_pred always in [LAM_MIN, 0.5] by model design")
    print(f"  All lambda metrics use symmetric MAE")
    print("=" * 65 + "\n")

    model  = load_model()
    ds     = SCATestDataset(TEST_PATH)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False,
                        num_workers=0, pin_memory=(device.type == "cuda"))
    print(f"  Test samples : {len(ds):,}")
    print(f"  Running evaluation ...")

    (results, pair_summary, lam_buckets,
     class_summary, anal_sym_mae) = run_evaluation(model, loader)

    report_overall(results)
    report_analytical_lambda(anal_sym_mae)
    report_lambda_breakdown(lam_buckets)
    report_lambda_accuracy(results)
    report_class_breakdown(class_summary)
    report_hardest_pairs(pair_summary)

    wvars_all = np.load(os.path.join(TEST_PATH, "within_var_stats.npy"))
    fracs_all = np.load(os.path.join(TEST_PATH, "frac_stats.npy"))
    lams_all  = np.load(os.path.join(TEST_PATH, "lambdas.npy"))
    report_ceiling(wvars_all, fracs_all, lams_all)

    report_verdict(results)
    print(f"\n  Total evaluation time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
