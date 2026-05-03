"""
DATAGEN_SCA.py  --  v1
========================
Directory: stochastic_v1/

Generates SCA_Data/ used by TRAIN_SCA.py and FINAL_TEST_SCA.py.

SCA = Stochastic CA, cell-level randomness, lambda ∈ [0.1, 0.9] continuous.

Output:
  SCA_Data/
    train_rules.npy      (179,)
    test_rules.npy       (77,)
    train_pairs.npy      (500, 2)
    test_pairs.npy       (500, 2)
    test/
      orbits.npy           (10000, 8, T, W)     float32
      frac_stats.npy       (10000, 24)           float32
      within_var_stats.npy (10000, 8)            float32
      rule_f_bits.npy      (10000, 8)            float32
      rule_g_bits.npy      (10000, 8)            float32
      lambdas.npy          (10000,)              float32
      rule_f_ids.npy       (10000,)              int32
      rule_g_ids.npy       (10000,)              int32

500 pairs × 20 samples = 10,000 test samples (mirrors TSCA structure).
Lambda sampled uniformly from [0.1, 0.9] for each test sample.

Training data is generated on-the-fly by OnTheFlyDataset in TRAIN_SCA.py.
"""

import numpy as np
import os, sys, time
from collections import Counter

# ══════════════════════════════════════════════════════════════════
BASE_DIR   = "/home/new_check_model/ECA_temporal_stocastic/stochastic_v1"
OUTPUT_DIR = os.path.join(BASE_DIR, "SCA_Data")
# ══════════════════════════════════════════════════════════════════

sys.path.insert(0, BASE_DIR)

from MODEL_SCA import (
    simulate_sca, compute_frac_stats, compute_within_var_stats,
    rule_to_bits, random_init,
    W, T, N_ORBITS, LAM_MIN, LAM_MAX
)

SEED          = 42
N_TRAIN       = 179
N_TEST        = 77
TRAIN_PAIRS   = 500
TEST_PAIRS    = 500
TEST_SAMPLES  = 20    # 500 × 20 = 10,000  (same as TSCA)

# ── Wolfram classification ─────────────────────────────────────────────────────
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


# ── Pair sampling (identical to TSCA) ─────────────────────────────────────────
def sample_pairs(rules, n_pairs, rng, label):
    all_pairs = [(f, g) for f in rules for g in rules if f != g]
    perm      = rng.permutation(len(all_pairs))
    all_pairs = [all_pairs[i] for i in perm]

    combo_pools = {}
    for f, g in all_pairs:
        key = tuple(sorted([wolfram_class(f), wolfram_class(g)]))
        combo_pools.setdefault(key, []).append((f, g))

    selected     = list(all_pairs[:min(n_pairs, len(all_pairs))])
    selected_set = set(selected)
    sel_combos   = set(tuple(sorted([wolfram_class(f), wolfram_class(g)]))
                       for f, g in selected)
    for combo in [('A','A'),('A','B'),('A','C'),('B','B'),('B','C'),('C','C')]:
        if combo not in sel_combos and combo in combo_pools:
            for p in combo_pools[combo][:10]:
                if p not in selected_set:
                    selected.append(p); selected_set.add(p)

    final = selected[:n_pairs]
    assert len(final) == n_pairs, f"{label}: got {len(final)}, wanted {n_pairs}"
    cov = Counter(tuple(sorted([wolfram_class(f), wolfram_class(g)]))
                  for f, g in final)
    print(f"  {label}: {len(final)} pairs  " +
          " ".join(f"{k[0]}x{k[1]}:{v}" for k, v in sorted(cov.items())))
    return final


# ── Dataset builder ───────────────────────────────────────────────────────────
def build_test_dataset(pairs, n_samples, rng):
    """
    Build test set: n_samples per pair, lambda uniform from [LAM_MIN, LAM_MAX].
    Total: len(pairs) × n_samples = 500 × 20 = 10,000 samples.
    """
    orbits_l=[]; frac_l=[]; wvar_l=[]
    rfb_l=[]; rgb_l=[]; lam_l=[]; rfi_l=[]; rgi_l=[]

    for pi, (f, g) in enumerate(pairs):
        bf = rule_to_bits(int(f)); bg = rule_to_bits(int(g))
        for _ in range(n_samples):
            lam    = float(rng.uniform(LAM_MIN, LAM_MAX))
            orbits = []
            for _ in range(N_ORBITS):
                init = random_init(rng)
                orb  = simulate_sca(int(f), int(g), init, lam, rng)
                orbits.append(orb)
            orbs_k = np.stack(orbits, axis=0)    # [K, T, W]
            fs     = compute_frac_stats(orbs_k)
            wv     = compute_within_var_stats(orbs_k)
            orbits_l.append(orbs_k); frac_l.append(fs); wvar_l.append(wv)
            rfb_l.append(bf); rgb_l.append(bg); lam_l.append(lam)
            rfi_l.append(int(f)); rgi_l.append(int(g))

        if (pi + 1) % 100 == 0 or (pi + 1) == len(pairs):
            print(f"    {pi+1}/{len(pairs)} pairs ...", end="\r")
    print()

    orbits = np.array(orbits_l,  dtype=np.float32)
    fracs  = np.array(frac_l,    dtype=np.float32)
    wvars  = np.array(wvar_l,    dtype=np.float32)
    rfb    = np.array(rfb_l,     dtype=np.float32)
    rgb    = np.array(rgb_l,     dtype=np.float32)
    lams   = np.array(lam_l,     dtype=np.float32)
    rfi    = np.array(rfi_l,     dtype=np.int32)
    rgi    = np.array(rgi_l,     dtype=np.int32)

    print(f"    {len(orbits):,} samples  orbits={orbits.shape}  "
          f"lam=[{lams.min():.2f},{lams.max():.2f}]")

    # Verify within_var signal quality
    print(f"\n    Within-var signal verification (5 samples):")
    for i in range(min(5, len(wvars))):
        lam_true = float(lams[i])
        max_wv   = float(wvars[i].max())
        expected = lam_true * (1.0 - lam_true)
        sym_lam  = min(lam_true, 1.0 - lam_true)
        lam_A_est = 0.5 - (max(0.0, 0.25 - max_wv) ** 0.5) if max_wv > 0.002 else 0.5
        sym_err   = min(abs(lam_A_est - lam_true),
                        abs((1.0 - lam_A_est) - lam_true))
        print(f"      [{i}] lam={lam_true:.3f}  expected_wvar={expected:.4f}  "
              f"max_wvar={max_wv:.4f}  sym_err={sym_err:.4f}")

    return orbits, fracs, wvars, rfb, rgb, lams, rfi, rgi


def save_test(path, orbits, fracs, wvars, rfb, rgb, lams, rfi, rgi):
    os.makedirs(path, exist_ok=True)
    for nm, arr in [("orbits",           orbits),
                    ("frac_stats",       fracs),
                    ("within_var_stats", wvars),
                    ("rule_f_bits",      rfb),
                    ("rule_g_bits",      rgb),
                    ("lambdas",          lams),
                    ("rule_f_ids",       rfi),
                    ("rule_g_ids",       rgi)]:
        np.save(os.path.join(path, nm + ".npy"), arr)
    mb = sum(a.nbytes for a in [orbits, fracs, wvars, rfb, rgb, lams, rfi, rgi]) / 1e6
    print(f"    Saved → {path}  ({mb:.1f} MB)")


# ── Ceiling analysis ──────────────────────────────────────────────────────────
def analyze_ceiling(wvars, fracs, lams):
    """
    Ceiling analysis for SCA.

    SCA is easier than TSCA because:
      - W=50 independent Bernoulli draws per pattern per timestep
        (vs 1 binary draw for the whole row in TSCA)
      - within_var gives a second independent estimator of lambda*(1-lambda)
      - More averaging → cleaner frac estimates

    No signal = rules agree on all patterns (same as TSCA).
    """
    var_   = fracs[:, 8:16]            # frac*(1-frac)
    wvar_  = wvars                      # within-timestep variance
    total  = len(fracs)

    # No signal: both frac_var and within_var are negligible
    no_frac_sig  = var_.max(axis=1) < 0.01
    no_wvar_sig  = wvar_.max(axis=1) < 0.002
    no_signal    = int((no_frac_sig & no_wvar_sig).sum())
    weak_frac    = int(((var_.max(axis=1) >= 0.01) &
                        (var_.max(axis=1) < 0.05)).sum())
    strong       = int((var_.max(axis=1) >= 0.05).sum())
    realistic    = (strong * 0.95 + weak_frac * 0.65) / total * 100

    lam_arr     = np.array(lams)
    near_half   = int((np.abs(lam_arr - 0.5) < 0.05).sum())

    print(f"\n  ═══ CEILING ANALYSIS ═══")
    print(f"  Total test samples        : {total:,}")
    print(f"  No signal (both <thresh)  : {no_signal:>5,} ({no_signal/total*100:4.1f}%)"
          f"  ← rules agree on all patterns")
    print(f"  Weak signal (0.01–0.05)   : {weak_frac:>5,} ({weak_frac/total*100:4.1f}%)"
          f"  ← hard")
    print(f"  Strong signal (≥0.05)     : {strong:>5,} ({strong/total*100:4.1f}%)"
          f"  ← solvable")
    print(f"  lambda≈0.5 samples        : {near_half:>5,} ({near_half/total*100:4.1f}%)"
          f"  ← orientation hardest")
    print(f"  Realistic ceiling         : ~{realistic:.0f}%")
    print(f"  NOTE: SCA easier than TSCA due to W=50 independent draws")
    print(f"        and within_var as additional lambda estimator")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0  = time.time()
    rng = np.random.default_rng(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print(f"  DATAGEN_SCA v1  |  N_ORBITS={N_ORBITS}  W={W}  T={T}")
    print(f"  Lambda: continuous uniform [{LAM_MIN}, {LAM_MAX}]")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 70)

    # Same SEED=42 rule split as TSCA for comparability
    rng_split   = np.random.default_rng(SEED)
    shuffled    = rng_split.permutation(256).tolist()
    train_rules = sorted(shuffled[:N_TRAIN])
    test_rules  = sorted(shuffled[N_TRAIN:])
    assert not set(train_rules) & set(test_rules)

    np.save(os.path.join(OUTPUT_DIR, "train_rules.npy"),
            np.array(train_rules, dtype=np.int32))
    np.save(os.path.join(OUTPUT_DIR, "test_rules.npy"),
            np.array(test_rules,  dtype=np.int32))

    for nm, rules in [("Train", train_rules), ("Test", test_rules)]:
        cc = Counter(wolfram_class(r) for r in rules)
        print(f"  {nm} rules: A={cc.get('A',0)} B={cc.get('B',0)} "
              f"C={cc.get('C',0)}  total={len(rules)}")

    print("\n  Sampling pairs ...")
    train_pairs = sample_pairs(train_rules, TRAIN_PAIRS, rng, "Train")
    test_pairs  = sample_pairs(test_rules,  TEST_PAIRS,  rng, "Test")
    np.save(os.path.join(OUTPUT_DIR, "train_pairs.npy"),
            np.array(train_pairs, dtype=np.int32))
    np.save(os.path.join(OUTPUT_DIR, "test_pairs.npy"),
            np.array(test_pairs,  dtype=np.int32))

    print(f"\n  Building test set "
          f"({TEST_PAIRS} pairs × {TEST_SAMPLES} samples = "
          f"{TEST_PAIRS*TEST_SAMPLES:,}) ...")
    test_data = build_test_dataset(test_pairs, TEST_SAMPLES, rng)
    save_test(os.path.join(OUTPUT_DIR, "test"), *test_data)

    analyze_ceiling(test_data[2], test_data[1], test_data[5])

    print(f"\n  Test samples : {len(test_data[0]):,}")
    print(f"  Time         : {time.time()-t0:.1f}s")
    print(f"\n  Training data generated on-the-fly — no train split needed.")
    print(f"  Next: python TRAIN_SCA.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
