"""
TRAIN_SKEW.py  --  ECANet s-Skewed Two-Phase Training (Enhanced)
=================================================================

Phase 1: sync (s=20), rule only. Phase 2: s-skewed, rule + s.

S loss = CrossEntropy (20 classes) + beta * MSE (regression on s/W).
The regression auxiliary enforces ordinal structure.

Model outputs: rule_logits [B,8], s_logits [B,20], s_reg [B,1]

Usage: python TRAIN_SKEW.py
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import time
from torch.utils.data import Dataset, DataLoader, random_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from MODEL_SKEW import (ECANetSkew, orbit_to_tokens, bits_to_rule,
                        W, T, N_TOK, N_BITS, TOKEN_DIM, D_MODEL, S_VALUES,
                        s_logits_to_value, s_value_to_class, s_reg_to_value)

DATA_DIR   = os.path.join(SCRIPT_DIR, "ECA_Data_Skew")
CKPT_DIR   = os.path.join(SCRIPT_DIR, "checkpoints_skew")

# Phase 1
P1_BATCH    = 8
P1_GRAD_ACC = 16
P1_EPOCHS   = 50
P1_LR       = 3e-4
P1_LR_MIN   = 1e-5
P1_LAM_RULE = 1.0
P1_LAM_S    = 0.0

# Phase 2
P2_BATCH    = 8
P2_GRAD_ACC = 16
P2_EPOCHS   = 30
P2_LR       = 1e-4
P2_LR_MIN   = 1e-6
P2_LAM_RULE = 1.0
P2_LAM_S_CE = 0.3       # CrossEntropy weight
P2_LAM_S_REG = 0.1      # MSE regression weight

VAL_SPLIT   = 0.15


def get_device():
    if torch.cuda.is_available():
        d = torch.device("cuda")
        print(f"  CUDA : {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        d = torch.device("mps")
        print("  Apple MPS")
    else:
        d = torch.device("cpu")
        print("  CPU only")
    return d

DEVICE = get_device()
PIN_MEMORY = (DEVICE.type == "cuda")


class ECADataset(Dataset):
    def __init__(self, path):
        self.orbits    = np.load(os.path.join(path, "orbits.npy"))
        self.rule_bits = np.load(os.path.join(path, "rule_bits.npy"))
        self.s_values  = np.load(os.path.join(path, "s_values.npy"))
        self.rule_ids  = np.load(os.path.join(path, "rule_ids.npy"))

    def __len__(self):
        return len(self.orbits)

    def __getitem__(self, idx):
        tokens = orbit_to_tokens(self.orbits[idx])      # [N_TOK, 5]
        x      = torch.tensor(tokens,               dtype=torch.float32)
        y_rule = torch.tensor(self.rule_bits[idx],   dtype=torch.float32)
        y_s    = torch.tensor([self.s_values[idx]],  dtype=torch.int64)
        return x, y_rule, y_s, int(self.rule_ids[idx])


def rule_metrics(logits, targets):
    preds    = (torch.sigmoid(logits) >= 0.5).float()
    bit_acc  = (preds == targets).float().mean().item() * 100
    exact    = (preds == targets).all(dim=1).float().mean().item() * 100
    return bit_acc, exact

def s_metrics(s_pred, s_true):
    err = np.abs(s_pred.reshape(-1).astype(float) - s_true.reshape(-1).astype(float))
    return {
        "mae":     float(err.mean()),
        "exact":   float((err == 0).mean()) * 100,
        "off_by1": float((err <= 1).mean()) * 100,
        "off_by2": float((err <= 2).mean()) * 100,
    }

def format_time(s):
    if s < 60:   return f"{s:.0f}s"
    if s < 3600: return f"{s/60:.1f}m"
    return f"{s/3600:.1f}h"


@torch.no_grad()
def evaluate_loader(model, loader):
    model.eval()
    rule_results = {}
    all_sp, all_st = [], []

    for x, y_rule, y_s, rule_ids in loader:
        x      = x.to(DEVICE)
        y_rule = y_rule.to(DEVICE)

        rule_logits, s_logits, s_reg = model(x)
        preds = (torch.sigmoid(rule_logits) >= 0.5).float()

        pred_s = s_logits_to_value(s_logits)
        all_sp.append(pred_s.cpu().numpy().reshape(-1))
        all_st.append(y_s.cpu().numpy().reshape(-1))

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

    all_sp_np = np.concatenate(all_sp)
    all_st_np = np.concatenate(all_st)

    return {
        "bit_acc":      total_bc / total_bt * 100,
        "exact_match":  total_ex / total_n  * 100,
        "rules_any":    sum(1 for v in rule_results.values() if v["exact"]>0),
        "n_rules":      len(rule_results),
        "s":            s_metrics(all_sp_np, all_st_np),
        "rule_results": rule_results,
    }


def run_training(model, train_loader, val_loader, epochs,
                 lr, lr_min, lam_rule, lam_s_ce, lam_s_reg,
                 save_path, label, grad_accum=1,
                 resume_epoch=0, resume_best_exact=0.0):

    criterion_rule  = nn.BCEWithLogitsLoss()
    criterion_s_ce  = nn.CrossEntropyLoss()
    criterion_s_reg = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr_min)

    if resume_epoch > 0:
        for _ in range(resume_epoch):
            scheduler.step()
        print(f"  LR fast-forwarded to {optimizer.param_groups[0]['lr']:.2e}")

    best_exact = resume_best_exact
    epoch_times = []

    print(f"\n  {'-'*90}")
    print(f"  TRAINING {label}")
    print(f"  lam_rule={lam_rule}  lam_s_ce={lam_s_ce}  lam_s_reg={lam_s_reg}  "
          f"LR={lr}->{lr_min}  Epochs={epochs}")
    print(f"  {'-'*90}")
    print(f"  {'Ep':>4} | {'Loss':>7} | {'TrBit':>6} | {'TrEx':>5} | "
          f"{'VBit':>5} | {'VEx':>6} | {'sMAE':>5} | "
          f"{'sEx%':>5} | {'s+-1':>5} | ETA")
    print(f"  {'-'*90}")

    for epoch in range(resume_epoch + 1, epochs + 1):
        t_ep = time.time()
        model.train()
        tot_loss = tot_bit = tot_exact = n = 0
        optimizer.zero_grad()

        for step, (x, y_rule, y_s, _) in enumerate(train_loader):
            x      = x.to(DEVICE)
            y_rule = y_rule.to(DEVICE)
            y_s    = y_s.to(DEVICE)

            rule_logits, s_logits, s_reg = model(x)
            loss_rule = criterion_rule(rule_logits, y_rule)

            loss = lam_rule * loss_rule

            if lam_s_ce > 0.0 or lam_s_reg > 0.0:
                s_cls = s_value_to_class(y_s.squeeze(1))

                if lam_s_ce > 0.0:
                    loss = loss + lam_s_ce * criterion_s_ce(s_logits, s_cls)

                if lam_s_reg > 0.0:
                    # Regression target: s/W in [0, 1]
                    s_target = y_s.float().squeeze(1) / W
                    loss = loss + lam_s_reg * criterion_s_reg(
                        s_reg.squeeze(-1), s_target)

            loss = loss / grad_accum
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
        all_vsp, all_vst = [], []
        vl_bit_list, vl_exact_list = [], []
        with torch.no_grad():
            for x, y_rule, y_s, _ in val_loader:
                x      = x.to(DEVICE)
                y_rule = y_rule.to(DEVICE)
                rule_logits_v, s_logits_v, _ = model(x)
                b, e = rule_metrics(rule_logits_v, y_rule)
                vl_bit_list.append(b)
                vl_exact_list.append(e)
                all_vsp.append(s_logits_to_value(s_logits_v).cpu().numpy().reshape(-1))
                all_vst.append(y_s.cpu().numpy().reshape(-1))

        vl_bit   = float(np.mean(vl_bit_list))
        vl_exact = float(np.mean(vl_exact_list))
        s_met    = s_metrics(np.concatenate(all_vsp), np.concatenate(all_vst))

        ep_t = time.time() - t_ep
        epoch_times.append(ep_t)
        eta = format_time(np.mean(epoch_times[-5:]) * (epochs - epoch))

        star = ""
        if vl_exact > best_exact:
            best_exact = vl_exact
            torch.save({"model_state": model.state_dict(),
                        "epoch": epoch, "best_exact": best_exact,
                        "phase": label}, save_path)
            star = " *"

        print(f"  {epoch:4d} | {tot_loss/n:7.4f} | "
              f"{tot_bit/n:5.1f}% | {tot_exact/n:4.1f}% | "
              f"{vl_bit:4.1f}% | {vl_exact:5.1f}% | "
              f"{s_met['mae']:5.2f} | {s_met['exact']:4.1f}% | "
              f"{s_met['off_by1']:4.1f}% | {eta}{star}")

    return best_exact


def run_post_phase_test(model, ckpt_path, phase_label, p1_test_path, p2_test_path):
    print(f"\n  {'='*65}")
    print(f"  POST-PHASE TEST  --  {phase_label}")
    print(f"  {'='*65}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Loaded: epoch={ckpt['epoch']}  val_exact={ckpt['best_exact']:.2f}%\n")

    for test_label, test_path in [
        (f"Phase 1 test (s={W}, sync)", p1_test_path),
        ("Phase 2 test (all s)", p2_test_path),
    ]:
        ds     = ECADataset(test_path)
        loader = DataLoader(ds, batch_size=64, shuffle=False,
                            num_workers=0, pin_memory=PIN_MEMORY)
        res    = evaluate_loader(model, loader)
        sm     = res["s"]
        print(f"  --- {test_label} ---")
        print(f"  Samples={len(ds):,}  BitAcc={res['bit_acc']:.2f}%  "
              f"Exact={res['exact_match']:.2f}%  "
              f"Rules={res['rules_any']}/{res['n_rules']}")
        print(f"  S: MAE={sm['mae']:.3f}  Exact={sm['exact']:.1f}%  "
              f"+-1={sm['off_by1']:.1f}%  +-2={sm['off_by2']:.1f}%\n")


def check_resume(ckpt_path):
    if not os.path.exists(ckpt_path):
        return 0, 0.0
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        epoch      = int(ckpt.get("epoch", 0))
        best_exact = float(ckpt.get("best_exact", 0.0))
        phase      = str(ckpt.get("phase", ""))
        print(f"  Found checkpoint: epoch={epoch}  best={best_exact:.2f}%  phase={phase}")
        if "PHASE 1" in phase and epoch < P1_EPOCHS:
            return epoch, best_exact
        elif "PHASE 1" in phase:
            return P1_EPOCHS, best_exact
        return 0, 0.0
    except:
        return 0, 0.0


def main():
    t_start = time.time()
    os.makedirs(CKPT_DIR, exist_ok=True)

    for phase, split in [("phase1","train"),("phase1","test"),
                          ("phase2","train"),("phase2","test")]:
        path = os.path.join(DATA_DIR, phase, split, "orbits.npy")
        if not os.path.exists(path):
            print(f"ERROR: Missing {path}. Run DATAGEN_SKEW.py first.")
            sys.exit(1)

    model = ECANetSkew().to(DEVICE)
    npar  = sum(p.numel() for p in model.parameters())

    print("=" * 65)
    print("  ECANetSkew  --  Enhanced Two-Phase Training")
    print("=" * 65)
    print(f"  W={W}, T={T}, N_TOK={N_TOK}, TOKEN_DIM={TOKEN_DIM}")
    print(f"  Parameters: {npar:,}  (~{npar/1e6:.2f}M)")
    print(f"  S loss: CE(20-class) + MSE(regression on s/W)")
    print("=" * 65)

    p1_train_path = os.path.join(DATA_DIR, "phase1", "train")
    p1_test_path  = os.path.join(DATA_DIR, "phase1", "test")
    p2_train_path = os.path.join(DATA_DIR, "phase2", "train")
    p2_test_path  = os.path.join(DATA_DIR, "phase2", "test")
    p1_ckpt = os.path.join(CKPT_DIR, "phase1_best.pt")
    p2_ckpt = os.path.join(CKPT_DIR, "phase2_best.pt")

    resume_epoch, resume_best = check_resume(p1_ckpt)
    skip_phase1 = (resume_epoch >= P1_EPOCHS)

    # Phase 1
    if not skip_phase1:
        print(f"\n  PHASE 1  --  Synchronous (s={W})")
        if resume_epoch > 0:
            ckpt = torch.load(p1_ckpt, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model_state"])

        p1_all = ECADataset(p1_train_path)
        n_val  = int(len(p1_all) * VAL_SPLIT)
        p1_val, p1_tr = random_split(
            p1_all, [n_val, len(p1_all)-n_val],
            generator=torch.Generator().manual_seed(42))
        print(f"  Train: {len(p1_tr):,}  Val: {len(p1_val):,}")

        best_p1 = run_training(
            model,
            DataLoader(p1_tr,  batch_size=P1_BATCH, shuffle=True,
                       num_workers=0, pin_memory=PIN_MEMORY),
            DataLoader(p1_val, batch_size=P1_BATCH, shuffle=False,
                       num_workers=0, pin_memory=PIN_MEMORY),
            P1_EPOCHS, P1_LR, P1_LR_MIN, P1_LAM_RULE, 0.0, 0.0,
            p1_ckpt, "PHASE 1 (sync)", P1_GRAD_ACC,
            resume_epoch, resume_best)

        print(f"\n  Phase 1 best: {best_p1:.2f}%")
        run_post_phase_test(model, p1_ckpt, "After Phase 1",
                            p1_test_path, p2_test_path)
    else:
        best_p1 = resume_best
        print(f"  Phase 1 complete ({best_p1:.2f}%). Skipping.")

    # Phase 2
    print(f"\n{'='*65}")
    print(f"  PHASE 2  --  s-Skewed fine-tuning")
    print(f"{'='*65}")

    ckpt = torch.load(p1_ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded Phase 1 weights (epoch={ckpt['epoch']})")

    p2_all = ECADataset(p2_train_path)
    n_val  = int(len(p2_all) * VAL_SPLIT)
    p2_val, p2_tr = random_split(
        p2_all, [n_val, len(p2_all)-n_val],
        generator=torch.Generator().manual_seed(42))
    print(f"  Train: {len(p2_tr):,}  Val: {len(p2_val):,}")

    best_p2 = run_training(
        model,
        DataLoader(p2_tr,  batch_size=P2_BATCH, shuffle=True,
                   num_workers=0, pin_memory=PIN_MEMORY),
        DataLoader(p2_val, batch_size=P2_BATCH, shuffle=False,
                   num_workers=0, pin_memory=PIN_MEMORY),
        P2_EPOCHS, P2_LR, P2_LR_MIN, P2_LAM_RULE, P2_LAM_S_CE, P2_LAM_S_REG,
        p2_ckpt, "PHASE 2 (s-skewed)", P2_GRAD_ACC)

    print(f"\n  Phase 2 best: {best_p2:.2f}%")
    run_post_phase_test(model, p2_ckpt, "After Phase 2",
                        p1_test_path, p2_test_path)

    print(f"\n{'='*65}")
    print(f"  COMPLETE  P1={best_p1:.2f}%  P2={best_p2:.2f}%  "
          f"Time={format_time(time.time()-t_start)}")
    print(f"  Next: python FINAL_TEST_SKEW.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
