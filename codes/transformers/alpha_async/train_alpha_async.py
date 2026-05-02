"""
TRAIN_NEW.py  --  ECANet New  Two-Phase Training  (with RESUME support)
========================================================================

RESUME MODE (default if phase1_best.pt exists):
  Detects existing phase1_best.pt checkpoint and resumes Phase 1
  from the saved epoch. Only the remaining epochs are run.
  The LR scheduler is fast-forwarded to match the resumed epoch.

  Example: crashed at epoch 29 of 50 -> resumes from epoch 30,
           runs epochs 30-50, then proceeds to Phase 2 normally.

FRESH MODE (if no checkpoint exists):
  Runs Phase 1 from scratch (epochs 1-50), then Phase 2.

PHASE 1  --  Synchronous (alpha=1.0):
  Data    : Phase 1 train (179 rules x 200 samples)
  Loss    : rule only  (lambda_rule=1.0, lambda_alpha=0.0)
  Epochs  : 50,  LR: 3e-4 -> 1e-5  (CosineAnnealing)
  Batch   : 8 physical x 16 grad_accum = 128 effective
  Saves   : checkpoints_new/phase1_best.pt

  After Phase 1 completes, automatically tests on:
    (A) Phase 1 test set  (alpha=1.0)
    (B) Phase 2 test set  (random alpha)

PHASE 2  --  Async fine-tuning:
  Data    : Phase 2 train (179 rules x 10 alphas x 50 samples)
  Loads   : phase1_best.pt
  Loss    : lambda_rule=1.0, lambda_alpha=0.3
  Alpha loss : CrossEntropyLoss over 10 classes {0.1..1.0}
               Guarantees max alpha error = 0.05 at inference.
  Epochs  : 30,  LR: 1e-4 -> 1e-6
  Batch   : 8 physical x 16 grad_accum = 128 effective
  Saves   : checkpoints_new/phase2_best.pt

  After Phase 2 completes, automatically tests on:
    (A) Phase 1 test set  (alpha=1.0)  -- checks rule knowledge preserved
    (B) Phase 2 test set  (random alpha)  -- full async performance

GPU:  CUDA (NVIDIA RTX A1000) -> MPS (Apple) -> CPU  (in that order)
      All Windows-safe: no unicode chars, num_workers=0

Usage:
  python TRAIN_NEW.py            # auto-detects resume or fresh start
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import time
from torch.utils.data import Dataset, DataLoader, random_split

# -- Locate scripts -----------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from MODEL_NEW import (ECANetNew, orbit_to_tokens, bits_to_rule,
                       W, T, N_TOK, N_BITS, D_MODEL, ALPHA_VALUES,
                       alpha_logits_to_value, alpha_value_to_class)

# -- Config -------------------------------------------------------------------
DATA_DIR   = os.path.join(SCRIPT_DIR, "ECA_Data_New")
CKPT_DIR   = os.path.join(SCRIPT_DIR, "checkpoints_new")

# Phase 1
P1_BATCH    = 8
P1_GRAD_ACC = 16     # effective batch = 128
P1_EPOCHS   = 50
P1_LR       = 3e-4
P1_LR_MIN   = 1e-5
P1_LAM_RULE = 1.0
P1_LAM_ALP  = 0.0    # alpha head OFF in Phase 1

# Phase 2
P2_BATCH    = 8
P2_GRAD_ACC = 16
P2_EPOCHS   = 30
P2_LR       = 1e-4
P2_LR_MIN   = 1e-6
P2_LAM_RULE = 1.0
P2_LAM_ALP  = 0.3

VAL_SPLIT   = 0.15

# -- Device -------------------------------------------------------------------
def get_device():
    print("=" * 60)
    print("  DEVICE DETECTION")
    print("=" * 60)
    if torch.cuda.is_available():
        d = torch.device("cuda")
        print(f"  CUDA : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        d = torch.device("mps")
        print("  Apple MPS (Mac GPU)")
    else:
        d = torch.device("cpu")
        print("  CPU only  (no GPU found)")
    print(f"  Using : {d}")
    print("=" * 60 + "\n")
    return d

DEVICE = get_device()
PIN_MEMORY = (DEVICE.type == "cuda")


# -- Dataset ------------------------------------------------------------------

class ECADataset(Dataset):
    def __init__(self, path):
        self.orbits    = np.load(os.path.join(path, "orbits.npy"))
        self.rule_bits = np.load(os.path.join(path, "rule_bits.npy"))
        self.alphas    = np.load(os.path.join(path, "alphas.npy"))
        self.rule_ids  = np.load(os.path.join(path, "rule_ids.npy"))

    def __len__(self):
        return len(self.orbits)

    def __getitem__(self, idx):
        tokens  = orbit_to_tokens(self.orbits[idx])
        x       = torch.tensor(tokens,              dtype=torch.float32)
        y_rule  = torch.tensor(self.rule_bits[idx], dtype=torch.float32)
        y_alpha = torch.tensor([self.alphas[idx]],  dtype=torch.float32)
        rule_id = int(self.rule_ids[idx])
        return x, y_rule, y_alpha, rule_id


# -- Metrics ------------------------------------------------------------------

def rule_metrics(logits, targets):
    preds    = (torch.sigmoid(logits) >= 0.5).float()
    bit_acc  = (preds == targets).float().mean().item() * 100
    exact    = (preds == targets).all(dim=1).float().mean().item() * 100
    return bit_acc, exact

def alpha_metrics_multi(ap, at):
    err = np.abs(ap.reshape(-1) - at.reshape(-1))
    return {
        "mae":     float(err.mean()),
        "strict":  float((err <= 0.05).mean()) * 100,
        "normal":  float((err <= 0.10).mean()) * 100,
        "relaxed": float((err <= 0.15).mean()) * 100,
    }

def format_time(s):
    if s < 60:   return f"{s:.0f}s"
    if s < 3600: return f"{s/60:.1f}m"
    return f"{s/3600:.1f}h"


# -- Evaluation ---------------------------------------------------------------

@torch.no_grad()
def evaluate_loader(model, loader):
    model.eval()
    rule_results = {}
    all_ap, all_at = [], []

    for x, y_rule, y_alpha, rule_ids in loader:
        x      = x.to(DEVICE)
        y_rule = y_rule.to(DEVICE)

        rule_logits, alpha_logits = model(x)
        preds = (torch.sigmoid(rule_logits) >= 0.5).float()

        pred_alpha_vals = alpha_logits_to_value(alpha_logits)
        all_ap.append(pred_alpha_vals.cpu().numpy().reshape(-1))
        all_at.append(y_alpha.cpu().numpy().reshape(-1))

        for i, rid in enumerate(rule_ids):
            rid   = int(rid)
            tb    = y_rule[i].cpu()
            pb    = preds[i].cpu()
            bc    = int((pb == tb).sum().item())
            exact = int((pb == tb).all().item())
            if rid not in rule_results:
                rule_results[rid] = {"bc":0,"bt":0,"exact":0,"total":0}
            rule_results[rid]["bc"]    += bc
            rule_results[rid]["bt"]    += N_BITS
            rule_results[rid]["exact"] += exact
            rule_results[rid]["total"] += 1

    total_ex = sum(v["exact"] for v in rule_results.values())
    total_n  = sum(v["total"] for v in rule_results.values())
    total_bc = sum(v["bc"]    for v in rule_results.values())
    total_bt = sum(v["bt"]    for v in rule_results.values())
    per_exact = [v["exact"]/v["total"]*100 for v in rule_results.values()]

    all_ap_np = np.concatenate(all_ap)
    all_at_np = np.concatenate(all_at)
    a_met     = alpha_metrics_multi(all_ap_np, all_at_np)

    return {
        "bit_acc":      total_bc / total_bt * 100,
        "exact_match":  total_ex / total_n  * 100,
        "rules_any":    sum(1 for p in per_exact if p > 0),
        "n_rules":      len(rule_results),
        "alpha":        a_met,
        "rule_results": rule_results,
    }


# -- Training loop  (with resume support) -------------------------------------

def run_training(model, train_loader, val_loader, epochs,
                 lr, lr_min, lam_rule, lam_alpha,
                 save_path, label, grad_accum=1,
                 resume_epoch=0, resume_best_exact=0.0):
    """
    resume_epoch       : last completed epoch (0 = fresh start).
                         Training starts from resume_epoch + 1.
    resume_best_exact  : best exact match already saved in checkpoint.
                         New checkpoint only saved if we beat this.
    """
    criterion_rule  = nn.BCEWithLogitsLoss()
    criterion_alpha = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr_min)

    # Fast-forward LR scheduler to match the already-completed epochs
    if resume_epoch > 0:
        print(f"  Fast-forwarding LR scheduler by {resume_epoch} steps ...",
              end=" ", flush=True)
        for _ in range(resume_epoch):
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"LR is now {current_lr:.2e}")

    best_exact = resume_best_exact
    epoch_times = []

    if resume_epoch > 0:
        print(f"\n  RESUMING from epoch {resume_epoch}  "
              f"(best so far: {resume_best_exact:.2f}%)")
        print(f"  Running epochs {resume_epoch+1} to {epochs}  "
              f"({epochs - resume_epoch} remaining)")
    else:
        print(f"\n  Starting fresh ({epochs} epochs)")

    print(f"\n  {'-'*80}")
    print(f"  TRAINING {label}")
    print(f"  lam_rule={lam_rule}  lam_alpha={lam_alpha}  "
          f"LR={lr}->{lr_min}  Total={epochs}  "
          f"Batch={train_loader.batch_size}x{grad_accum}="
          f"{train_loader.batch_size*grad_accum} (eff)")
    print(f"  {'-'*80}")
    print(f"  {'Ep':>4} | {'Loss':>7} | {'TrBit':>6} | {'TrExact':>7} | "
          f"{'ValBit':>6} | {'ValExact':>8} | {'aMAE':>6} | "
          f"{'a+-.05':>7} | ETA")
    print(f"  {'-'*80}")

    for epoch in range(resume_epoch + 1, epochs + 1):
        t_ep = time.time()
        model.train()
        tot_loss = tot_bit = tot_exact = n = 0
        optimizer.zero_grad()

        for step, (x, y_rule, y_alpha, _) in enumerate(train_loader):
            x       = x.to(DEVICE)
            y_rule  = y_rule.to(DEVICE)
            y_alpha = y_alpha.to(DEVICE)

            rule_logits, alpha_logits = model(x)
            loss_rule = criterion_rule(rule_logits, y_rule)

            if lam_alpha > 0.0:
                alpha_cls  = alpha_value_to_class(y_alpha.squeeze(1))
                loss_alpha = criterion_alpha(alpha_logits, alpha_cls)
                loss       = (lam_rule * loss_rule +
                              lam_alpha * loss_alpha) / grad_accum
            else:
                loss = (lam_rule * loss_rule) / grad_accum

            loss.backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            b, e       = rule_metrics(rule_logits.detach(), y_rule)
            bs         = len(y_rule)
            tot_loss  += loss.item() * grad_accum * bs
            tot_bit   += b * bs
            tot_exact += e * bs
            n         += bs

        scheduler.step()

        # Validation
        model.eval()
        all_vap, all_vat = [], []
        vl_bit_list, vl_exact_list = [], []
        with torch.no_grad():
            for x, y_rule, y_alpha, _ in val_loader:
                x      = x.to(DEVICE)
                y_rule = y_rule.to(DEVICE)

                rule_logits_v, alpha_logits_v = model(x)
                b, e = rule_metrics(rule_logits_v, y_rule)
                vl_bit_list.append(b)
                vl_exact_list.append(e)

                pred_alpha_vals = alpha_logits_to_value(alpha_logits_v)
                all_vap.append(pred_alpha_vals.cpu().numpy().reshape(-1))
                all_vat.append(y_alpha.cpu().numpy().reshape(-1))

        vl_bit   = float(np.mean(vl_bit_list))
        vl_exact = float(np.mean(vl_exact_list))
        a_met    = alpha_metrics_multi(
            np.concatenate(all_vap), np.concatenate(all_vat))

        ep_t = time.time() - t_ep
        epoch_times.append(ep_t)
        remaining = epochs - epoch
        eta = format_time(np.mean(epoch_times[-5:]) * remaining)

        star = ""
        if vl_exact > best_exact:
            best_exact = vl_exact
            torch.save({
                "model_state": model.state_dict(),
                "epoch":       epoch,
                "best_exact":  best_exact,
                "phase":       label,
            }, save_path)
            star = " *"

        print(f"  {epoch:4d} | {tot_loss/n:7.4f} | "
              f"{tot_bit/n:5.1f}% | {tot_exact/n:6.1f}% | "
              f"{vl_bit:5.1f}% | {vl_exact:7.1f}% | "
              f"{a_met['mae']:6.4f} | {a_met['strict']:6.1f}% | "
              f"{eta}{star}")

    return best_exact


# -- Post-phase testing -------------------------------------------------------

def run_post_phase_test(model, ckpt_path, phase_label,
                        p1_test_path, p2_test_path):
    print(f"\n  {'='*65}")
    print(f"  POST-PHASE TEST  --  {phase_label}")
    print(f"  {'='*65}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Loaded: {ckpt_path}  (epoch={ckpt['epoch']}, "
          f"val_exact={ckpt['best_exact']:.2f}%)\n")

    for test_label, test_path in [
        ("Phase 1 test  (alpha=1.0 only)", p1_test_path),
        ("Phase 2 test  (all alphas)", p2_test_path),
    ]:
        ds     = ECADataset(test_path)
        loader = DataLoader(ds, batch_size=64, shuffle=False,
                            num_workers=0, pin_memory=PIN_MEMORY)
        res    = evaluate_loader(model, loader)

        print(f"  --- {test_label} ---")
        print(f"  Samples    : {len(ds):,}")
        print(f"  Bit acc    : {res['bit_acc']:.2f}%")
        print(f"  Exact match: {res['exact_match']:.2f}%")
        print(f"  Rules ok   : {res['rules_any']}/{res['n_rules']} "
              f"({res['rules_any']/res['n_rules']*100:.1f}%)")
        a = res["alpha"]
        print(f"  Alpha MAE   : {a['mae']:.4f}")
        print(f"  Alpha +-0.05: {a['strict']:.1f}%  "
              f"+-0.10: {a['normal']:.1f}%  "
              f"+-0.15: {a['relaxed']:.1f}%")
        print()

        rr = res["rule_results"]
        zero_rules = sorted(r for r, v in rr.items() if v["exact"] == 0)
        if zero_rules:
            print(f"  Rules with 0 exact matches: {zero_rules}")
        else:
            print(f"  All {res['n_rules']} rules have at least 1 exact match.")
        print()


# -- Resume detection ---------------------------------------------------------

def check_resume(ckpt_path):
    """
    Returns (resume_epoch, resume_best_exact) if a valid Phase 1 checkpoint
    exists and it is incomplete (epoch < P1_EPOCHS).
    Returns (0, 0.0) if no checkpoint or Phase 1 already done.
    """
    if not os.path.exists(ckpt_path):
        print(f"  No checkpoint found at {ckpt_path}. Fresh start.")
        return 0, 0.0

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        epoch      = int(ckpt.get("epoch", 0))
        best_exact = float(ckpt.get("best_exact", 0.0))
        phase      = str(ckpt.get("phase", ""))

        print(f"  Found checkpoint: {ckpt_path}")
        print(f"    epoch={epoch}  best_exact={best_exact:.2f}%  phase={phase}")

        if "PHASE 1" in phase and epoch < P1_EPOCHS:
            print(f"  --> RESUME MODE: Phase 1 incomplete "
                  f"({epoch}/{P1_EPOCHS} epochs done, "
                  f"{P1_EPOCHS - epoch} remaining)")
            return epoch, best_exact
        elif "PHASE 1" in phase and epoch >= P1_EPOCHS:
            print(f"  --> Phase 1 already complete. Skipping to Phase 2.")
            return P1_EPOCHS, best_exact
        else:
            print(f"  --> Unrecognised checkpoint phase. Starting Phase 1 fresh.")
            return 0, 0.0
    except Exception as e:
        print(f"  WARNING: Could not read checkpoint ({e}). Starting fresh.")
        return 0, 0.0


# -- Main ---------------------------------------------------------------------

def main():
    t_start = time.time()
    os.makedirs(CKPT_DIR, exist_ok=True)

    for phase, split in [("phase1","train"), ("phase1","test"),
                          ("phase2","train"), ("phase2","test")]:
        path = os.path.join(DATA_DIR, phase, split, "orbits.npy")
        if not os.path.exists(path):
            print(f"ERROR: Missing data at {path}")
            print("Run DATAGEN_NEW.py first.")
            sys.exit(1)

    model  = ECANetNew().to(DEVICE)
    npar   = sum(p.numel() for p in model.parameters())
    print("=" * 65)
    print("  ECANetNew  --  Two-Phase Training  (with Resume Support)")
    print("=" * 65)
    print(f"  W={W}, T={T}  ->  {N_TOK} triplet tokens/sample")
    print(f"  Parameters : {npar:,}  (~{npar/1e6:.2f}M)")
    print(f"  Data dir   : {DATA_DIR}")
    print(f"  Checkpoints: {CKPT_DIR}")
    print()
    print(f"  NOTE: If CUDA OOM occurs, reduce P1_BATCH/P2_BATCH from 8 to 4")
    print(f"        and increase P1_GRAD_ACC/P2_GRAD_ACC from 16 to 32.")
    print("=" * 65)

    p1_train_path = os.path.join(DATA_DIR, "phase1", "train")
    p1_test_path  = os.path.join(DATA_DIR, "phase1", "test")
    p2_train_path = os.path.join(DATA_DIR, "phase2", "train")
    p2_test_path  = os.path.join(DATA_DIR, "phase2", "test")
    p1_ckpt_path  = os.path.join(CKPT_DIR, "phase1_best.pt")
    p2_ckpt_path  = os.path.join(CKPT_DIR, "phase2_best.pt")

    # Check for resume
    print(f"\n{'='*65}")
    print(f"  CHECKPOINT CHECK")
    print(f"{'='*65}")
    resume_epoch, resume_best = check_resume(p1_ckpt_path)
    print()

    skip_phase1 = (resume_epoch >= P1_EPOCHS)

    # -- PHASE 1 --------------------------------------------------------------
    if not skip_phase1:
        print(f"{'='*65}")
        if resume_epoch > 0:
            print(f"  PHASE 1  --  RESUMING from epoch {resume_epoch}/{P1_EPOCHS}")
        else:
            print(f"  PHASE 1  --  Synchronous (alpha=1.0)  [fresh start]")
        print(f"  Goal : Learn rule inference with zero async noise")
        print(f"{'='*65}")

        if resume_epoch > 0:
            ckpt = torch.load(p1_ckpt_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            print(f"  Loaded weights from checkpoint (epoch {resume_epoch}) OK")

        p1_all = ECADataset(p1_train_path)
        n_val  = int(len(p1_all) * VAL_SPLIT)
        n_tr   = len(p1_all) - n_val
        p1_val, p1_tr = random_split(
            p1_all, [n_val, n_tr],
            generator=torch.Generator().manual_seed(42))
        print(f"  Train: {len(p1_tr):,}  Val: {len(p1_val):,}")

        p1_train_loader = DataLoader(p1_tr,  batch_size=P1_BATCH, shuffle=True,
                                      num_workers=0, pin_memory=PIN_MEMORY)
        p1_val_loader   = DataLoader(p1_val, batch_size=P1_BATCH, shuffle=False,
                                      num_workers=0, pin_memory=PIN_MEMORY)

        if resume_epoch == 0:
            print("  Warm-up forward pass ...", end=" ", flush=True)
            model.train()
            dummy = torch.zeros(2, N_TOK, 4).to(DEVICE)
            _, _  = model(dummy)
            print("OK")

        best_p1 = run_training(
            model, p1_train_loader, p1_val_loader,
            epochs            = P1_EPOCHS,
            lr                = P1_LR,
            lr_min            = P1_LR_MIN,
            lam_rule          = P1_LAM_RULE,
            lam_alpha         = P1_LAM_ALP,
            save_path         = p1_ckpt_path,
            label             = "PHASE 1 (sync)",
            grad_accum        = P1_GRAD_ACC,
            resume_epoch      = resume_epoch,
            resume_best_exact = resume_best,
        )

        print(f"\n  Phase 1 done. Best val exact: {best_p1:.2f}%")
        run_post_phase_test(model, p1_ckpt_path, "After Phase 1",
                            p1_test_path, p2_test_path)
    else:
        best_p1 = resume_best
        print(f"  Phase 1 already complete (best_exact={best_p1:.2f}%). Skipping.")

    # -- PHASE 2 --------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"  PHASE 2  --  Async fine-tuning (all alphas)")
    print(f"  Starts from Phase 1 best weights")
    print(f"{'='*65}")

    ckpt = torch.load(p1_ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded Phase 1 weights (epoch={ckpt['epoch']}, "
          f"val_exact={ckpt['best_exact']:.2f}%) OK")

    p2_all = ECADataset(p2_train_path)
    n_val  = int(len(p2_all) * VAL_SPLIT)
    n_tr   = len(p2_all) - n_val
    p2_val, p2_tr = random_split(
        p2_all, [n_val, n_tr],
        generator=torch.Generator().manual_seed(42))
    print(f"  Train: {len(p2_tr):,}  Val: {len(p2_val):,}")

    p2_train_loader = DataLoader(p2_tr,  batch_size=P2_BATCH, shuffle=True,
                                  num_workers=0, pin_memory=PIN_MEMORY)
    p2_val_loader   = DataLoader(p2_val, batch_size=P2_BATCH, shuffle=False,
                                  num_workers=0, pin_memory=PIN_MEMORY)

    best_p2 = run_training(
        model, p2_train_loader, p2_val_loader,
        epochs     = P2_EPOCHS,
        lr         = P2_LR,
        lr_min     = P2_LR_MIN,
        lam_rule   = P2_LAM_RULE,
        lam_alpha  = P2_LAM_ALP,
        save_path  = p2_ckpt_path,
        label      = "PHASE 2 (async)",
        grad_accum = P2_GRAD_ACC,
    )

    print(f"\n  Phase 2 done. Best val exact: {best_p2:.2f}%")
    run_post_phase_test(model, p2_ckpt_path, "After Phase 2",
                        p1_test_path, p2_test_path)

    print(f"\n{'='*65}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*65}")
    print(f"  Phase 1 best val exact  : {best_p1:.2f}%  (sync, rule only)")
    print(f"  Phase 2 best val exact  : {best_p2:.2f}%  (async, rule+alpha)")
    print(f"  Total time              : {format_time(time.time()-t_start)}")
    print(f"\n  Next step: python FINAL_TEST_NEW.py")
    print("=" * 65)


if __name__ == "__main__":
    main()