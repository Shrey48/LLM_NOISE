"""
smoke_test_decoder_only.py -- CPU sanity check for decoder_only_control_alpha.py
==================================================================================
Verifies data loading, model instantiation, sequence construction on real
orbits, a forward pass, one training step (forward + backward + optimiser
step), and the evaluate() function, without requiring a GPU.

Run from the same directory as decoder_only_control_alpha.py.

Usage: python3 smoke_test_decoder_only.py --data_dir ECA_Data_New
"""

import argparse, os, sys
import numpy as np


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="ECA_Data_New")
    args = ap.parse_args()

    all_ok = True
    print("=" * 60)
    print("  Smoke test: decoder-only next-token control")
    print("=" * 60)

    print("\n-- Imports --")
    try:
        import torch
        import torch.nn as nn
        all_ok &= check("torch imports", True, f"version {torch.__version__}")
    except Exception as e:
        return check("torch imports", False, str(e))

    try:
        from decoder_only_control_alpha import (
            DecoderOnlyECA, make_sequence, alpha_to_class, ECASequenceDataset,
            evaluate, VOCAB_SIZE, N_BITS_RULE, N_BITS_ALPHA,
        )
        all_ok &= check("decoder_only_control_alpha imports", True)
    except Exception:
        print("  [FAIL] decoder_only_control_alpha imports")
        import traceback; traceback.print_exc()
        return

    print("\n-- Real data loading --")
    try:
        test_base = os.path.join(args.data_dir, "phase2", "test")
        orbits = np.load(os.path.join(test_base, "orbits.npy"))
        rule_ids = np.load(os.path.join(test_base, "rule_ids.npy"))
        alphas = np.load(os.path.join(test_base, "alphas.npy"))
        all_ok &= check("alphaECA test data loads", True, f"shape={orbits.shape}")
    except Exception:
        print("  [FAIL] Data loading")
        import traceback; traceback.print_exc()
        return

    print("\n-- Sequence construction on real data --")
    try:
        orbit_flat = orbits[0].flatten().astype(int).tolist()
        rule = int(rule_ids[0])
        alpha_class = alpha_to_class(float(alphas[0]))
        seq, mask = make_sequence(orbit_flat, rule, alpha_class)
        expected_len = orbits.shape[1] * orbits.shape[2] + 2 + N_BITS_RULE + N_BITS_ALPHA
        all_ok &= check("Sequence length correct", len(seq) == expected_len,
                        f"got {len(seq)}, expected {expected_len}")
        all_ok &= check("Loss mask length matches sequence", len(mask) == len(seq))
        all_ok &= check("Loss mask covers only completion",
                        sum(mask) == 1 + N_BITS_RULE + 1 + N_BITS_ALPHA,
                        f"got {sum(mask)} masked positions")
    except Exception:
        print("  [FAIL] Sequence construction")
        import traceback; traceback.print_exc()
        return

    print("\n-- Model instantiation --")
    try:
        seq_len = orbits.shape[1] * orbits.shape[2] + 2 + N_BITS_RULE + N_BITS_ALPHA
        model = DecoderOnlyECA(max_seq_len=seq_len + 10)
        n_params = sum(p.numel() for p in model.parameters())
        all_ok &= check("Model instantiates", True, f"{n_params:,} params")
    except Exception:
        print("  [FAIL] Model instantiation")
        import traceback; traceback.print_exc()
        return

    print("\n-- Forward pass on real data (batch of 2) --")
    try:
        ds = ECASequenceDataset(orbits[:2], rule_ids[:2], alphas[:2])
        x0, m0 = ds[0]
        x1, m1 = ds[1]
        x = torch.stack([x0, x1])
        mask = torch.stack([m0, m1])
        model.eval()
        with torch.no_grad():
            logits = model(x)
        all_ok &= check("Forward pass runs without error", True)
        all_ok &= check("Output shape correct",
                        logits.shape == (2, x.shape[1], VOCAB_SIZE),
                        f"got {tuple(logits.shape)}")
    except Exception:
        print("  [FAIL] Forward pass")
        import traceback; traceback.print_exc()
        all_ok = False

    print("\n-- One training step (forward + backward + optimiser step) --")
    try:
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        logits = model(x)
        pred = logits[:, :-1, :]
        target = x[:, 1:]
        m = mask[:, 1:].float()
        loss = nn.functional.cross_entropy(
            pred.reshape(-1, VOCAB_SIZE), target.reshape(-1), reduction="none")
        loss = (loss * m.reshape(-1)).sum() / m.sum()
        loss_before = loss.item()
        opt.zero_grad()
        loss.backward()

        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        all_ok &= check("Gradients computed for all parameters",
                        len(grad_norms) == sum(1 for _ in model.parameters()))
        all_ok &= check("Gradients are finite (no NaN/Inf)",
                        all(np.isfinite(g) for g in grad_norms))
        all_ok &= check("Loss is finite", np.isfinite(loss_before), f"loss={loss_before:.4f}")

        opt.step()
        with torch.no_grad():
            logits2 = model(x)
            pred2 = logits2[:, :-1, :]
            loss2 = nn.functional.cross_entropy(
                pred2.reshape(-1, VOCAB_SIZE), target.reshape(-1), reduction="none")
            loss_after = ((loss2 * m.reshape(-1)).sum() / m.sum()).item()
        all_ok &= check("Optimiser step changes the loss",
                        loss_after != loss_before,
                        f"before={loss_before:.4f} after={loss_after:.4f}")
    except Exception:
        print("  [FAIL] Training step")
        import traceback; traceback.print_exc()
        all_ok = False

    print("\n-- evaluate() on a tiny batch --")
    try:
        from torch.utils.data import DataLoader
        small_ds = ECASequenceDataset(orbits[:4], rule_ids[:4], alphas[:4])
        small_loader = DataLoader(small_ds, batch_size=2, shuffle=False)
        rule_acc, alpha_acc = evaluate(model, small_loader, "cpu")
        all_ok &= check("evaluate() runs without error", True,
                        f"rule_acc={rule_acc:.1f}% alpha_acc={alpha_acc:.1f}% (sanity check only)")
    except Exception:
        print("  [FAIL] evaluate()")
        import traceback; traceback.print_exc()
        all_ok = False

    print("\n" + "=" * 60)
    print("  ALL CHECKS PASSED" if all_ok else "  SOME CHECKS FAILED")
    print("=" * 60)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
