"""
DATAGEN_NEW.py  --  ECA Dataset Generator for ECANet New
=========================================================

Generates 4 dataset splits with T=100, W=20.

PHASE 1 (synchronous, alpha=1.0 only):
  Train : 179 rules x 200 samples = 35,800 total
  Test  : 77  rules x 100 samples =  7,700 total
  Purpose: Clean rule inference baseline. No alpha noise.

PHASE 2 (asynchronous, stratified by alpha):
  Train : 179 rules x 10 alphas x 50 samples = 89,500 total
  Test  : 77  rules x 10 alphas x 20 samples = 15,400 total
  Every rule seen at EVERY alpha level exactly K times.
  No random imbalance -- guaranteed coverage.

STORAGE FORMAT (per split):
  orbits.npy    : int8   [N, 100, 20]  -- raw orbits
  rule_bits.npy : int8   [N, 8]        -- 8 rule bits
  alphas.npy    : float32 [N]          -- alpha value per sample
  rule_ids.npy  : int32  [N]           -- which rule (0-255) per sample

Samples are stored RULE-WISE (all samples for rule 0, then rule 1, etc.)
within each alpha group. DataLoader shuffle=True handles training order.

Rule split: SEED=42 (same as TRAIN3 -- results are directly comparable).

Usage:
  python DATAGEN_NEW.py

Output folder structure:
  ECA_Data_New/
    train_rules.npy
    test_rules.npy
    phase1/
      train/  orbits.npy  rule_bits.npy  alphas.npy  rule_ids.npy
      test/   ...
    phase2/
      train/  ...
      test/   ...
"""

import numpy as np
import os
import sys
import time

# ── Config ────────────────────────────────────────────────────────────────────
W                  = 20
T                  = 100
SEED               = 42    # same as TRAIN3 for comparability

N_TRAIN_RULES      = 179
N_TEST_RULES       = 77

# Phase 1
P1_SAMPLES_TRAIN   = 200   # per rule, alpha=1.0
P1_SAMPLES_TEST    = 100

# Phase 2
P2_SAMPLES_PER_ALPHA = 50  # per (rule, alpha) -- 179 x 10 x 50 = 89,500
P2_SAMPLES_TEST      = 20  # per (rule, alpha) -- 77  x 10 x 20 = 15,400

ALPHA_VALUES = [round(a * 0.1, 1) for a in range(1, 11)]  # 0.1 .. 1.0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "ECA_Data_New")


# ── ECA Simulation ────────────────────────────────────────────────────────────

def build_rule_table(rule_number):
    return np.array([(rule_number >> i) & 1 for i in range(8)], dtype=np.uint8)

def rule_to_bits(rule_number):
    return [(rule_number >> i) & 1 for i in range(8)]

def simulate_eca(rule_number, init_state, n_steps, alpha, rng):
    rule_table = build_rule_table(rule_number)
    orbit = np.zeros((n_steps, W), dtype=np.int8)
    state = init_state.copy().astype(np.uint8)
    orbit[0] = state
    for t in range(1, n_steps):
        left   = np.roll(state,  1)
        right  = np.roll(state, -1)
        idx    = (4 * left + 2 * state + right).astype(np.uint8)
        new_st = rule_table[idx]
        if alpha < 1.0:
            mask  = rng.random(W) < alpha
            state = np.where(mask, new_st, state).astype(np.uint8)
        else:
            state = new_st.astype(np.uint8)
        orbit[t] = state
    return orbit

def random_init(rng):
    """Non-trivial initial state (not all-zeros or all-ones)."""
    while True:
        init = rng.integers(0, 2, size=W, dtype=np.int8)
        if 0 < int(init.sum()) < W:
            return init


# ── Dataset Builders ──────────────────────────────────────────────────────────

def build_phase1(rules, samples_per_rule, rng, label):
    """
    Phase 1: synchronous only (alpha=1.0).
    Stored rule-wise: all samples for rule[0] first, then rule[1], etc.
    """
    orbits    = []
    rule_bits = []
    alphas    = []
    rule_ids  = []

    n_rules = len(rules)
    for ri, rule in enumerate(rules):
        bits = rule_to_bits(int(rule))
        for _ in range(samples_per_rule):
            init  = random_init(rng)
            orbit = simulate_eca(int(rule), init, T, 1.0, rng)
            orbits.append(orbit)
            rule_bits.append(bits)
            alphas.append(1.0)
            rule_ids.append(int(rule))

        if (ri + 1) % 30 == 0 or (ri + 1) == n_rules:
            print(f"      {label}: {ri+1}/{n_rules} rules done ...", end="\r")

    print()
    orbits    = np.array(orbits,    dtype=np.int8)
    rule_bits = np.array(rule_bits, dtype=np.int8)
    alphas    = np.array(alphas,    dtype=np.float32)
    rule_ids  = np.array(rule_ids,  dtype=np.int32)
    print(f"      {label}: {len(orbits):,} samples  (alpha=1.0 only)")
    return orbits, rule_bits, alphas, rule_ids


def build_phase2(rules, samples_per_alpha, rng, label):
    """
    Phase 2: stratified by (rule, alpha).
    For every rule, generates exactly `samples_per_alpha` samples
    at EACH of the 10 alpha levels. Perfect balance guaranteed.

    Storage order: rule-wise, then alpha-wise within each rule.
      rule_0/alpha_0.1: 50 samples
      rule_0/alpha_0.2: 50 samples
      ...
      rule_0/alpha_1.0: 50 samples
      rule_1/alpha_0.1: 50 samples
      ...
    DataLoader shuffle=True handles the training order.
    """
    orbits    = []
    rule_bits = []
    alphas    = []
    rule_ids  = []

    n_rules = len(rules)
    for ri, rule in enumerate(rules):
        bits = rule_to_bits(int(rule))
        for alpha in ALPHA_VALUES:
            for _ in range(samples_per_alpha):
                init  = random_init(rng)
                orbit = simulate_eca(int(rule), init, T, alpha, rng)
                orbits.append(orbit)
                rule_bits.append(bits)
                alphas.append(alpha)
                rule_ids.append(int(rule))

        if (ri + 1) % 20 == 0 or (ri + 1) == n_rules:
            print(f"      {label}: {ri+1}/{n_rules} rules done ...", end="\r")

    print()
    orbits    = np.array(orbits,    dtype=np.int8)
    rule_bits = np.array(rule_bits, dtype=np.int8)
    alphas    = np.array(alphas,    dtype=np.float32)
    rule_ids  = np.array(rule_ids,  dtype=np.int32)

    # Verify balance
    dist = {av: int((alphas == av).sum()) for av in ALPHA_VALUES}
    print(f"      {label}: {len(orbits):,} samples  (stratified)")
    print(f"      Alpha distribution: { {k: v for k,v in dist.items()} }")
    return orbits, rule_bits, alphas, rule_ids


# ── Save ─────────────────────────────────────────────────────────────────────

def save_split(path, orbits, rule_bits, alphas, rule_ids):
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "orbits.npy"),    orbits)
    np.save(os.path.join(path, "rule_bits.npy"), rule_bits)
    np.save(os.path.join(path, "alphas.npy"),    alphas)
    np.save(os.path.join(path, "rule_ids.npy"),  rule_ids)
    size_mb = (orbits.nbytes + rule_bits.nbytes +
               alphas.nbytes + rule_ids.nbytes) / 1e6
    print(f"      Saved to: {path}  ({size_mb:.1f} MB)")


# ── Sanity check ──────────────────────────────────────────────────────────────

def sanity_check(orbits, rule_bits, alphas, rule_ids, n=3):
    print(f"    Sanity check ({n} samples):")
    for i in range(min(n, len(orbits))):
        rule = int(sum(int(rule_bits[i, b]) * (2**b) for b in range(8)))
        assert rule == rule_ids[i], f"rule_ids mismatch at {i}"
        assert orbits[i].shape == (T, W), f"orbit shape wrong at {i}"
        assert set(np.unique(orbits[i])).issubset({0, 1}), f"non-binary orbit at {i}"
        alpha = alphas[i]
        print(f"      sample {i}: rule={rule:3d}  alpha={alpha:.1f}  "
              f"orbit={orbits[i].shape}  bits={list(rule_bits[i])}  OK")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t_total = time.time()
    rng     = np.random.default_rng(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  DATAGEN_NEW  --  ECA Dataset Generator")
    print("=" * 65)
    print(f"  W={W}, T={T}  ->  {(T-1)*W} triplet tokens/sample")
    print(f"  Rules : {N_TRAIN_RULES} train / {N_TEST_RULES} test  (SEED={SEED})")
    print(f"  Alphas: {ALPHA_VALUES}")
    print(f"  Output: {OUTPUT_DIR}")

    # ── Rule split (SEED=42 -- same as TRAIN3) ────────────────────────────────
    rng_split   = np.random.default_rng(SEED)
    shuffled    = rng_split.permutation(256).tolist()
    train_rules = sorted(shuffled[:N_TRAIN_RULES])
    test_rules  = sorted(shuffled[N_TRAIN_RULES:])
    assert len(set(train_rules) & set(test_rules)) == 0, "Rule overlap!"

    np.save(os.path.join(OUTPUT_DIR, "train_rules.npy"), np.array(train_rules, dtype=np.int32))
    np.save(os.path.join(OUTPUT_DIR, "test_rules.npy"),  np.array(test_rules,  dtype=np.int32))
    print(f"\n  Train rules : {train_rules[:8]}... ({len(train_rules)} total, saved)")
    print(f"  Test  rules : {test_rules[:8]}... ({len(test_rules)}  total, saved)")

    # ── Phase 1 train ─────────────────────────────────────────────────────────
    print(f"\n  PHASE 1  --  Synchronous (alpha=1.0)")
    print(f"  " + "-" * 55)
    t1 = time.time()

    print(f"    Building Phase 1 TRAIN ({N_TRAIN_RULES} rules x {P1_SAMPLES_TRAIN} samples)...")
    p1_tr = build_phase1(train_rules, P1_SAMPLES_TRAIN, rng, "P1-train")
    save_split(os.path.join(OUTPUT_DIR, "phase1", "train"), *p1_tr)
    sanity_check(*p1_tr)

    print(f"    Building Phase 1 TEST ({N_TEST_RULES} rules x {P1_SAMPLES_TEST} samples)...")
    p1_te = build_phase1(test_rules, P1_SAMPLES_TEST, rng, "P1-test ")
    save_split(os.path.join(OUTPUT_DIR, "phase1", "test"), *p1_te)
    sanity_check(*p1_te)

    print(f"  Phase 1 done  ({time.time()-t1:.1f}s)")

    # ── Phase 2 train ─────────────────────────────────────────────────────────
    print(f"\n  PHASE 2  --  Async (alpha in {ALPHA_VALUES[:3]}...)")
    print(f"  " + "-" * 55)
    t1 = time.time()

    print(f"    Building Phase 2 TRAIN ({N_TRAIN_RULES} rules x 10 alphas x {P2_SAMPLES_PER_ALPHA} samples)...")
    p2_tr = build_phase2(train_rules, P2_SAMPLES_PER_ALPHA, rng, "P2-train")
    save_split(os.path.join(OUTPUT_DIR, "phase2", "train"), *p2_tr)
    sanity_check(*p2_tr)

    print(f"    Building Phase 2 TEST ({N_TEST_RULES} rules x 10 alphas x {P2_SAMPLES_TEST} samples)...")
    p2_te = build_phase2(test_rules, P2_SAMPLES_TEST, rng, "P2-test ")
    save_split(os.path.join(OUTPUT_DIR, "phase2", "test"), *p2_te)
    sanity_check(*p2_te)

    print(f"  Phase 2 done  ({time.time()-t1:.1f}s)")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_size = sum(
        a.nbytes for split in [p1_tr, p1_te, p2_tr, p2_te] for a in split
    ) / 1e6

    print(f"\n  {'='*55}")
    print(f"  DATASET SUMMARY")
    print(f"  {'='*55}")
    print(f"  Phase 1 train : {len(p1_tr[0]):>7,} samples  (alpha=1.0)")
    print(f"  Phase 1 test  : {len(p1_te[0]):>7,} samples  (alpha=1.0)")
    print(f"  Phase 2 train : {len(p2_tr[0]):>7,} samples  (all alphas, stratified)")
    print(f"  Phase 2 test  : {len(p2_te[0]):>7,} samples  (all alphas, stratified)")
    print(f"  Total storage : ~{total_size:.0f} MB")
    print(f"  Total time    : {time.time()-t_total:.1f}s")
    print(f"\n  Next step: python TRAIN_NEW.py")
    print(f"  {'='*55}")


if __name__ == "__main__":
    main()
