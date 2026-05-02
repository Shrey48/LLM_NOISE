"""
DATAGEN_SKEW.py  --  s-Skewed ECA Dataset Generator (Enhanced, T=200)
======================================================================

Same structure as baseline but with T=200 time steps for more signal.
Tokens are now 5-dimensional (did_change feature added in orbit_to_tokens).

Phase 1: 179 rules x 200 samples = 35,800  (s=20, synchronous)
Phase 2: 179 rules x 20 s-values x 25 samples = 89,500  (stratified)
Test splits: 77 rules, same structure.

Usage: python DATAGEN_SKEW.py
"""

import numpy as np
import os
import sys
import time

# ── Config ────────────────────────────────────────────────────────────────────
W                  = 20
T                  = 200       # increased from 100 for more signal
SEED               = 42

N_TRAIN_RULES      = 179
N_TEST_RULES       = 77

P1_SAMPLES_TRAIN   = 200
P1_SAMPLES_TEST    = 100

P2_SAMPLES_PER_S   = 25
P2_SAMPLES_TEST    = 10

S_VALUES = list(range(1, W + 1))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "ECA_Data_Skew")


# ── ECA Simulation ────────────────────────────────────────────────────────────

def build_rule_table(rule_number):
    return np.array([(rule_number >> i) & 1 for i in range(8)], dtype=np.uint8)

def rule_to_bits(rule_number):
    return [(rule_number >> i) & 1 for i in range(8)]

def simulate_eca_skewed(rule_number, init_state, n_steps, s, rng):
    rule_table = build_rule_table(rule_number)
    orbit = np.zeros((n_steps, W), dtype=np.int8)
    state = init_state.copy().astype(np.uint8)
    orbit[0] = state
    offsets = np.arange(s, dtype=np.intp)
    for t in range(1, n_steps):
        left   = np.roll(state,  1)
        right  = np.roll(state, -1)
        idx    = (4 * left + 2 * state + right).astype(np.uint8)
        new_st = rule_table[idx]
        if s >= W:
            state = new_st
        else:
            start = int(rng.integers(0, W))
            cells = (start + offsets) % W
            state = state.copy()
            state[cells] = new_st[cells]
        orbit[t] = state
    return orbit

def random_init(rng):
    while True:
        init = rng.integers(0, 2, size=W, dtype=np.int8)
        if 0 < int(init.sum()) < W:
            return init


# ── Dataset Builders ──────────────────────────────────────────────────────────

def build_phase1(rules, samples_per_rule, rng, label):
    orbits, rule_bits, s_vals, rule_ids = [], [], [], []
    n_rules = len(rules)
    for ri, rule in enumerate(rules):
        bits = rule_to_bits(int(rule))
        for _ in range(samples_per_rule):
            init  = random_init(rng)
            orbit = simulate_eca_skewed(int(rule), init, T, W, rng)
            orbits.append(orbit)
            rule_bits.append(bits)
            s_vals.append(W)
            rule_ids.append(int(rule))
        if (ri + 1) % 30 == 0 or (ri + 1) == n_rules:
            print(f"      {label}: {ri+1}/{n_rules} rules done ...", end="\r")
    print()
    orbits    = np.array(orbits,    dtype=np.int8)
    rule_bits = np.array(rule_bits, dtype=np.int8)
    s_vals    = np.array(s_vals,    dtype=np.int32)
    rule_ids  = np.array(rule_ids,  dtype=np.int32)
    print(f"      {label}: {len(orbits):,} samples  (s={W}, sync)")
    return orbits, rule_bits, s_vals, rule_ids


def build_phase2(rules, samples_per_s, rng, label):
    orbits, rule_bits, s_vals, rule_ids = [], [], [], []
    n_rules = len(rules)
    for ri, rule in enumerate(rules):
        bits = rule_to_bits(int(rule))
        for s in S_VALUES:
            for _ in range(samples_per_s):
                init  = random_init(rng)
                orbit = simulate_eca_skewed(int(rule), init, T, s, rng)
                orbits.append(orbit)
                rule_bits.append(bits)
                s_vals.append(s)
                rule_ids.append(int(rule))
        if (ri + 1) % 20 == 0 or (ri + 1) == n_rules:
            print(f"      {label}: {ri+1}/{n_rules} rules done ...", end="\r")
    print()
    orbits    = np.array(orbits,    dtype=np.int8)
    rule_bits = np.array(rule_bits, dtype=np.int8)
    s_vals    = np.array(s_vals,    dtype=np.int32)
    rule_ids  = np.array(rule_ids,  dtype=np.int32)
    dist = {sv: int((s_vals == sv).sum()) for sv in S_VALUES}
    print(f"      {label}: {len(orbits):,} samples  (stratified)")
    print(f"      S distribution: { {k: v for k,v in dist.items()} }")
    return orbits, rule_bits, s_vals, rule_ids


def save_split(path, orbits, rule_bits, s_vals, rule_ids):
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "orbits.npy"),    orbits)
    np.save(os.path.join(path, "rule_bits.npy"), rule_bits)
    np.save(os.path.join(path, "s_values.npy"),  s_vals)
    np.save(os.path.join(path, "rule_ids.npy"),  rule_ids)
    size_mb = (orbits.nbytes + rule_bits.nbytes +
               s_vals.nbytes + rule_ids.nbytes) / 1e6
    print(f"      Saved to: {path}  ({size_mb:.1f} MB)")


def sanity_check(orbits, rule_bits, s_vals, rule_ids, n=3):
    print(f"    Sanity check ({n} samples):")
    for i in range(min(n, len(orbits))):
        rule = int(sum(int(rule_bits[i, b]) * (2**b) for b in range(8)))
        assert rule == rule_ids[i]
        assert orbits[i].shape == (T, W)
        assert set(np.unique(orbits[i])).issubset({0, 1})
        assert 1 <= s_vals[i] <= W
        print(f"      sample {i}: rule={rule:3d}  s={s_vals[i]:2d}  "
              f"orbit={orbits[i].shape}  OK")


def main():
    t_total = time.time()
    rng     = np.random.default_rng(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  DATAGEN_SKEW  --  s-Skewed ECA Dataset Generator (T=200)")
    print("=" * 65)
    print(f"  W={W}, T={T}  ->  {(T-1)*W} triplet tokens/sample (5 features)")
    print(f"  Rules : {N_TRAIN_RULES} train / {N_TEST_RULES} test  (SEED={SEED})")
    print(f"  S values: {S_VALUES}")
    print(f"  Output: {OUTPUT_DIR}")

    rng_split   = np.random.default_rng(SEED)
    shuffled    = rng_split.permutation(256).tolist()
    train_rules = sorted(shuffled[:N_TRAIN_RULES])
    test_rules  = sorted(shuffled[N_TRAIN_RULES:])
    assert len(set(train_rules) & set(test_rules)) == 0

    np.save(os.path.join(OUTPUT_DIR, "train_rules.npy"), np.array(train_rules, dtype=np.int32))
    np.save(os.path.join(OUTPUT_DIR, "test_rules.npy"),  np.array(test_rules,  dtype=np.int32))
    print(f"\n  Train rules : {train_rules[:8]}... ({len(train_rules)} total)")
    print(f"  Test  rules : {test_rules[:8]}... ({len(test_rules)}  total)")

    # Phase 1
    print(f"\n  PHASE 1  --  Synchronous (s={W})")
    print(f"  " + "-" * 55)
    t1 = time.time()
    print(f"    Building Phase 1 TRAIN...")
    p1_tr = build_phase1(train_rules, P1_SAMPLES_TRAIN, rng, "P1-train")
    save_split(os.path.join(OUTPUT_DIR, "phase1", "train"), *p1_tr)
    sanity_check(*p1_tr)
    print(f"    Building Phase 1 TEST...")
    p1_te = build_phase1(test_rules, P1_SAMPLES_TEST, rng, "P1-test ")
    save_split(os.path.join(OUTPUT_DIR, "phase1", "test"), *p1_te)
    sanity_check(*p1_te)
    print(f"  Phase 1 done  ({time.time()-t1:.1f}s)")

    # Phase 2
    print(f"\n  PHASE 2  --  s-Skewed (s=1..{W})")
    print(f"  " + "-" * 55)
    t1 = time.time()
    print(f"    Building Phase 2 TRAIN...")
    p2_tr = build_phase2(train_rules, P2_SAMPLES_PER_S, rng, "P2-train")
    save_split(os.path.join(OUTPUT_DIR, "phase2", "train"), *p2_tr)
    sanity_check(*p2_tr)
    print(f"    Building Phase 2 TEST...")
    p2_te = build_phase2(test_rules, P2_SAMPLES_TEST, rng, "P2-test ")
    save_split(os.path.join(OUTPUT_DIR, "phase2", "test"), *p2_te)
    sanity_check(*p2_te)
    print(f"  Phase 2 done  ({time.time()-t1:.1f}s)")

    total_size = sum(a.nbytes for split in [p1_tr,p1_te,p2_tr,p2_te] for a in split) / 1e6
    print(f"\n  {'='*55}")
    print(f"  SUMMARY: {len(p1_tr[0])+len(p1_te[0])+len(p2_tr[0])+len(p2_te[0]):,} "
          f"samples  ~{total_size:.0f} MB  {time.time()-t_total:.1f}s")
    print(f"  Next: python TRAIN_SKEW.py")
    print(f"  {'='*55}")


if __name__ == "__main__":
    main()
