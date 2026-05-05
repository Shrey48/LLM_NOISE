"""
DATAGEN_TSCA.py  --  v9
========================
Directory: temporal_v9/

Generates TSCA_Data/ used by TRAIN_TSCA.py and FINAL_TEST_TSCA.py.

This script regenerates it from scratch if needed (identical output).

Output:
  TSCA_Data/
    train_rules.npy      (179,)
    test_rules.npy       (77,)
    train_pairs.npy      (500,2)
    test_pairs.npy       (500,2)
    test/
      orbits.npy         (10000, 8, T, W)    float32
      frac_stats.npy     (10000, 24)          float32
      rule_f_bits.npy    (10000, 8)           float32
      rule_g_bits.npy    (10000, 8)           float32
      taus.npy           (10000,)             float32
      rule_f_ids.npy     (10000,)             int32
      rule_g_ids.npy     (10000,)             int32
      step_labels.npy    (10000, 8, T-1)      int8

Training data is generated on-the-fly by OnTheFlyDataset in TRAIN_TSCA.py.
Only the test split is saved to disk.
"""

import numpy as np
import os, sys, time
from collections import Counter

# ══════════════════════════════════════════════════════════
BASE_DIR = os.environ.get("BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "TSCA_Data")
# ══════════════════════════════════════════════════════════

sys.path.insert(0, BASE_DIR)

from MODEL_TSCA import (
    simulate_labeled, compute_frac_stats,
    rule_to_bits, random_init,
    W, T, N_ORBITS, TAU_MIN, TAU_MAX
)

SEED         = 42
N_TRAIN      = 179
N_TEST       = 77
TRAIN_PAIRS  = 500
TEST_PAIRS   = 500
TEST_SAMPLES = 20      # 500 × 20 = 10,000 test samples

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

# ── Pair sampling ─────────────────────────────────────────────────────────────
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
    orbits_l=[]; frac_l=[]; rfb_l=[]; rgb_l=[]
    taus_l=[]; rfi_l=[]; rgi_l=[]; slabels_l=[]

    for pi, (f, g) in enumerate(pairs):
        bf = rule_to_bits(int(f))
        bg = rule_to_bits(int(g))
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
            print(f"    {pi+1}/{len(pairs)} pairs ...", end="\r")
    print()

    orbits  = np.array(orbits_l,  dtype=np.float32)
    fracs   = np.array(frac_l,    dtype=np.float32)
    rfb     = np.array(rfb_l,     dtype=np.float32)
    rgb     = np.array(rgb_l,     dtype=np.float32)
    taus    = np.array(taus_l,    dtype=np.float32)
    rfi     = np.array(rfi_l,     dtype=np.int32)
    rgi     = np.array(rgi_l,     dtype=np.int32)
    slabels = np.array(slabels_l, dtype=np.int8)

    print(f"    {len(orbits):,} samples  orbits={orbits.shape}  "
          f"tau=[{taus.min():.2f},{taus.max():.2f}]")

    # Verify frac signal quality
    print(f"    Signal verification (5 samples):")
    for i in range(min(5, len(fracs))):
        sf       = np.sort(fracs[i, :8])
        interior = sf[(sf > 0.05) & (sf < 0.95)]
        tau_est  = float(interior.min()) if len(interior) >= 1 else 0.5
        tau_true = float(taus[i])
        sym_err  = min(abs(tau_est - tau_true), abs((1.0 - tau_est) - tau_true))
        print(f"      [{i}] tau={tau_true:.3f}  est={tau_est:.3f}  "
              f"sym_err={sym_err:.4f}  fracs={np.round(sf, 2)}")

    return orbits, fracs, rfb, rgb, taus, rfi, rgi, slabels

def save_test(path, orbits, fracs, rfb, rgb, taus, rfi, rgi, slabels):
    os.makedirs(path, exist_ok=True)
    for nm, arr in [("orbits",      orbits),
                    ("frac_stats",  fracs),
                    ("rule_f_bits", rfb),
                    ("rule_g_bits", rgb),
                    ("taus",        taus),
                    ("rule_f_ids",  rfi),
                    ("rule_g_ids",  rgi),
                    ("step_labels", slabels)]:
        np.save(os.path.join(path, nm + ".npy"), arr)
    mb = sum(a.nbytes for a in
             [orbits, fracs, rfb, rgb, taus, rfi, rgi, slabels]) / 1e6
    print(f"    Saved → {path}  ({mb:.1f} MB)")

# ── Ceiling analysis ──────────────────────────────────────────────────────────
def analyze_ceiling(fracs, taus):
    """
    Compute information-theoretic ceiling on both-exact accuracy.

    For a pair (f, g, tau), both rules and tau are identifiable only if
    there are disagreement patterns (f[n] ≠ g[n]) that appear in the orbits.
    max_var is a proxy for this: max_var = max(frac*(1-frac)) ≈ tau*(1-tau)
    at disagreement patterns, 0 at agreement patterns.

    No signal (max_var < 0.01): both rules indistinguishable from a single rule.
    Both rules are identical OR all patterns are agreement patterns.
    """
    var_ = fracs[:, 8:16]
    total       = len(fracs)
    no_signal   = int((var_.max(axis=1) < 0.01).sum())
    weak_signal = int(((var_.max(axis=1) >= 0.01) &
                       (var_.max(axis=1) < 0.05)).sum())
    strong      = int((var_.max(axis=1) >= 0.05).sum())

    realistic = (strong * 0.95 + weak_signal * 0.65) / total * 100

    print(f"\n  ═══ CEILING ANALYSIS ═══")
    print(f"  Total test samples   : {total:,}")
    print(f"  No signal (max_var<0.01): {no_signal:>5,} ({no_signal/total*100:4.1f}%)"
          f"  ← tau & rules unrecoverable (agree on all patterns)")
    print(f"  Weak signal (0.01–0.05): {weak_signal:>5,} ({weak_signal/total*100:4.1f}%)"
          f"  ← hard, model may struggle")
    print(f"  Strong signal (≥0.05): {strong:>5,} ({strong/total*100:4.1f}%)"
          f"  ← solvable with good model")
    print(f"  Theoretical ceiling  : {(total-no_signal)/total*100:.1f}%"
          f"  (100% on weak+strong)")
    print(f"  Realistic ceiling    : ~{realistic:.0f}%"
          f"  (95% strong, 65% weak)")
    print(f"  Current best (v8/v9) : ~86–87%  → headroom to ~{realistic:.0f}%")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0  = time.time()
    rng = np.random.default_rng(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print(f"  DATAGEN_TSCA v9  |  N_ORBITS={N_ORBITS}  W={W}  T={T}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  (Or symlink from temporal_v5/TSCA_Data -- same output)")
    print("=" * 70)

    # Rule split: same SEED=42 -- identical to all previous versions
    rng_split   = np.random.default_rng(SEED)
    shuffled    = rng_split.permutation(256).tolist()
    train_rules = sorted(shuffled[:N_TRAIN])
    test_rules  = sorted(shuffled[N_TRAIN:])
    assert not set(train_rules) & set(test_rules)
    assert len(train_rules) == N_TRAIN and len(test_rules) == N_TEST

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

    analyze_ceiling(test_data[1], test_data[4])

    print(f"\n  Test samples : {len(test_data[0]):,}")
    print(f"  Time         : {time.time()-t0:.1f}s")
    print(f"\n  Training data generated on-the-fly — no train split needed.")
    print(f"  Next: python TRAIN_TSCA.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
