"""
TRAIN_SCA.py  --  v1
======================
Directory: stochastic_v1/

Training for ECANetSCA v1. Mirrors TSCA v9 TRAIN_TSCA.py structure.

════════════════════════════════════════════════════════════════
KEY DIFFERENCES FROM TSCA TRAINING
════════════════════════════════════════════════════════════════

SAME as TSCA v9:
  - Two-stage training (80ep + 80ep)
  - symmetric_loss_rules_only: orientation from rule BCE only
  - lam_mse_loss: MSE(lam_pred, min(true_lam, 1-true_lam))
    Same as TSCA tau_mse_loss — lam_pred always ≤ 0.5, target = min(lam,1-lam)
  - Adaptive COR_MAX in correction MLP (lam_mlp)
  - lam removed from symmetric_loss (same as TSCA CHANGE 4)
  - Curriculum: start narrow, reach full range by epoch 11
  - WarmupCosineScheduler, GRAD_ACC=32, BATCH=4
  - Separate LR for lam_mlp in Stage 2 (same as tau_mlp in TSCA)

DIFFERENT from TSCA v9:
  - No assignment_loss (no step_labels — SCA has no per-timestep rule label)
  - No ts_residual / residual_weight (TSCA FIX 3 removed — no use in SCA)
  - within_var_stats fed to model as extra input (new SCA signal)
  - OnTheFlyDataset generates SCA orbits (simulate_sca not simulate_labeled)
  - StaticDataset loads within_var_stats in addition to frac_stats

LOSS STRUCTURE:
  L = symmetric_rule_loss          (rule BCE, orientation from rules only)
    + LAM_LAM_MSE * lam_mse_loss   (MSE toward min(lam, 1-lam))
    + LAM_DIRECT  * direct_rule_loss_symmetric

  No lam in symmetric_loss (same reasoning as TSCA CHANGE 4).
  lam_mse_loss provides clean unambiguous gradient for lambda.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys, time
from torch.utils.data import Dataset, DataLoader

# ══════════════════════════════════════════════════════════════════
BASE_DIR = "/home/shovik.roy/Shrey/new_check_model/ECA_temporal_stocastic/stochastic_v1"
DATA_DIR = os.path.join(BASE_DIR, "SCA_Data")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints_sca_v1")
# ══════════════════════════════════════════════════════════════════

sys.path.insert(0, BASE_DIR)

from MODEL_SCA import (
    ECANetSCA, orbit_to_tokens, simulate_sca,
    random_init, rule_to_bits, compute_frac_stats, compute_within_var_stats,
    W, T, N_ORBITS, N_TOK, N_TRANS, N_BITS, FRAC_DIM, WVAR_DIM,
    LAM_MIN, LAM_MAX
)
N_RAW = 4

# ── Stage 1 hyperparameters ───────────────────────────────────────────────────
S1_EPOCHS      = 80
S1_LR          = 3e-4
S1_LR_MIN      = 1e-5
WARMUP_STEPS   = 100
S1_BATCH       = 4
S1_GRAD_ACC    = 32

S1_LAM_F       = 1.0    # rule BCE weight (in symmetric_rule_loss)
S1_LAM_G       = 1.0
S1_LAM_MSE     = 3.0    # lambda MSE weight (same as TSCA tau_mse weight)
S1_LAM_DIRECT  = 0.5    # direct decoder BCE

# Curriculum: same shape as TSCA (narrow → full by epoch 11)
S1_CURRICULUM  = [
    (10, 0.20),    # ep 1-10:  lam ∈ [0.30, 0.70]
    (11, 0.40),    # ep 11:    lam ∈ [0.10, 0.90]  full range
    (80, 0.40),    # ep 12-80: stay full range
]

# ── Stage 2 hyperparameters ───────────────────────────────────────────────────
S2_EPOCHS      = 80
S2_LR_MAIN     = 3e-5
S2_LR_LAM      = 1e-5
S2_LR_MIN      = 1e-7
S2_WARMUP      = 30
S2_BATCH       = 4
S2_GRAD_ACC    = 32

S2_LAM_MSE     = 2.0
S2_LAM_DIRECT  = 0.3

S2_CURRICULUM  = [(80, 0.40)]    # always full range in Stage 2

PAIRS_PER_EPOCH  = 1000
SAMPLES_PER_PAIR = 8

_CKPT_KEY = "lam_mlp.correction_net.0.weight"


# ── Warmup Cosine Scheduler (identical to TSCA) ───────────────────────────────
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, cosine_scheduler,
                 lr_start=1e-6, lr_peak=None):
        self.opt        = optimizer
        self.warmup     = warmup_steps
        self.cos_sch    = cosine_scheduler
        self.lr_start   = lr_start
        self.lr_peak    = lr_peak or optimizer.param_groups[0]["lr"]
        self._step      = 0
        self._in_warmup = True
        self._set_lr(lr_start)

    def _set_lr(self, lr):
        for pg in self.opt.param_groups:
            pg["lr"] = lr * pg.get("_lr_ratio", 1.0)

    def step_after_update(self):
        self._step += 1
        if self._step <= self.warmup:
            frac = self._step / self.warmup
            self._set_lr(self.lr_start + (self.lr_peak - self.lr_start) * frac)
            if self._step == self.warmup:
                self._in_warmup = False

    def epoch_step(self):
        if not self._in_warmup:
            self.cos_sch.step()


# ── Device ────────────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        d  = torch.device("cuda")
        nm = torch.cuda.get_device_name(0)
        gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  CUDA: {nm}  ({gb:.1f} GB)")
    elif torch.backends.mps.is_available():
        d = torch.device("mps"); print("  Apple MPS")
    else:
        d = torch.device("cpu"); print("  CPU only")
    return d

DEVICE = get_device()
PIN    = (DEVICE.type == "cuda")


# ══════════════════════════════════════════════════════════════════
# DATASETS
# ══════════════════════════════════════════════════════════════════

class StaticDataset(Dataset):
    """Static test dataset (SCA version — includes within_var_stats)."""
    def __init__(self, path):
        self.orbits = np.load(os.path.join(path, "orbits.npy"))
        self.fracs  = np.load(os.path.join(path, "frac_stats.npy"))
        self.wvars  = np.load(os.path.join(path, "within_var_stats.npy"))
        self.rfb    = np.load(os.path.join(path, "rule_f_bits.npy"))
        self.rgb    = np.load(os.path.join(path, "rule_g_bits.npy"))
        self.lams   = np.load(os.path.join(path, "lambdas.npy"))
        assert self.orbits.ndim == 4 and self.orbits.shape[1] == N_ORBITS

    def __len__(self): return len(self.orbits)

    def __getitem__(self, i):
        toks = np.stack([orbit_to_tokens(self.orbits[i, k])
                         for k in range(N_ORBITS)], axis=0)
        return (torch.tensor(toks,          dtype=torch.float32),
                torch.tensor(self.fracs[i], dtype=torch.float32),
                torch.tensor(self.wvars[i], dtype=torch.float32),
                torch.tensor(self.rfb[i],   dtype=torch.float32),
                torch.tensor(self.rgb[i],   dtype=torch.float32),
                torch.tensor([self.lams[i]], dtype=torch.float32))


class OnTheFlyDataset(Dataset):
    """
    On-the-fly SCA training dataset.
    Lambda sampled uniformly from [LAM_MIN, LAM_MAX] (or narrower via curriculum).
    """
    def __init__(self, rules, lam_half, n_pairs, spp, seed):
        rng  = np.random.default_rng(seed)
        lo   = max(LAM_MIN, 0.5 - lam_half)
        hi   = min(LAM_MAX, 0.5 + lam_half)
        n    = len(rules)

        seen = set(); pairs = []; att = 0
        while len(pairs) < n_pairs and att < n_pairs * 20:
            att += 1
            f = int(rules[rng.integers(0, n)])
            g = int(rules[rng.integers(0, n)])
            if f == g: continue
            p = (min(f, g), max(f, g))
            if p not in seen: pairs.append((f, g)); seen.add(p)
        while len(pairs) < n_pairs:
            f = int(rules[rng.integers(0, n)])
            g = int(rules[rng.integers(0, n)])
            if f != g: pairs.append((f, g))

        self.toks_l=[]; self.fracs_l=[]; self.wvars_l=[]
        self.rfb_l=[]; self.rgb_l=[]; self.lam_l=[]

        for f, g in pairs:
            bf = rule_to_bits(f); bg = rule_to_bits(g)
            for _ in range(spp):
                lam    = float(rng.uniform(lo, hi))
                orbits = []
                for _ in range(N_ORBITS):
                    init = random_init(rng)
                    orb  = simulate_sca(f, g, init, lam, rng)
                    orbits.append(orb)
                orbs_k = np.stack(orbits)
                fs     = compute_frac_stats(orbs_k)
                wv     = compute_within_var_stats(orbs_k)
                toks   = np.stack([orbit_to_tokens(orbs_k[k])
                                   for k in range(N_ORBITS)], axis=0)
                self.toks_l.append(toks);  self.fracs_l.append(fs)
                self.wvars_l.append(wv);   self.rfb_l.append(bf)
                self.rgb_l.append(bg);     self.lam_l.append(lam)

    def __len__(self): return len(self.toks_l)

    def __getitem__(self, i):
        return (torch.tensor(self.toks_l[i],  dtype=torch.float32),
                torch.tensor(self.fracs_l[i], dtype=torch.float32),
                torch.tensor(self.wvars_l[i], dtype=torch.float32),
                torch.tensor(self.rfb_l[i],   dtype=torch.float32),
                torch.tensor(self.rgb_l[i],   dtype=torch.float32),
                torch.tensor([self.lam_l[i]], dtype=torch.float32))


def get_lam_half(ep, curriculum):
    prev_end = 0; prev_h = curriculum[0][1]
    for end, h in curriculum:
        if ep <= end:
            prog = (ep - prev_end - 1) / max(end - prev_end - 1, 1)
            return round(prev_h + (h - prev_h) * prog, 4)
        prev_end = end; prev_h = h
    return curriculum[-1][1]


# ══════════════════════════════════════════════════════════════════
# LOSSES (mirrors TSCA v9 exactly for the parts that apply)
# ══════════════════════════════════════════════════════════════════

def symmetric_rule_loss(prf, prg, trf, trg, bce):
    """
    Orientation from rule BCE only — identical to TSCA CHANGE 4.
    min over (F→f,G→g) vs (F→g,G→f) of BCE losses.
    """
    B  = prf.shape[0]
    lf = torch.zeros(B, device=prf.device)
    lb = torch.zeros(B, device=prf.device)
    for i in range(B):
        lf[i] = bce(prf[i], trf[i]) + bce(prg[i], trg[i])
        lb[i] = bce(prf[i], trg[i]) + bce(prg[i], trf[i])
    return torch.minimum(lf, lb).mean()


def lam_mse_loss(lam_pred, true_lam):
    """
    Clean lambda MSE — identical in structure to TSCA tau_mse_loss.

    lam_pred is in [LAM_MIN, 0.5] (by model design).
    Target is min(true_lam, 1-true_lam) also in [LAM_MIN, 0.5].
    Simple MSE. No orientation ambiguity.

    lam_pred  : [B, 1]
    true_lam  : [B, 1]  true lambda ∈ [LAM_MIN, LAM_MAX]
    """
    target = torch.minimum(true_lam, 1.0 - true_lam)   # [B,1] ∈ [LAM_MIN, 0.5]
    return F.mse_loss(lam_pred, target)


def direct_rule_loss_symmetric(rf_dir, rg_dir, trf, trg, bce):
    """Symmetric BCE for direct decoder path. Identical to TSCA."""
    B  = rf_dir.shape[0]
    lf = torch.zeros(B, device=rf_dir.device)
    lb = torch.zeros(B, device=rf_dir.device)
    for i in range(B):
        lf[i] = bce(rf_dir[i], trf[i]) + bce(rg_dir[i], trg[i])
        lb[i] = bce(rf_dir[i], trg[i]) + bce(rg_dir[i], trf[i])
    return torch.minimum(lf, lb).mean()


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(prf, prg, plam, trf, trg, tlam):
    pf = torch.sigmoid(prf).numpy()
    pg = torch.sigmoid(prg).numpy()
    pt = plam.numpy().reshape(-1)
    tf = trf.numpy(); tg = trg.numpy()
    tt = tlam.numpy().reshape(-1)

    bf = (pf > 0.5).astype(np.float32)
    bg = (pg > 0.5).astype(np.float32)

    fwd = (bf == tf).all(1) & (bg == tg).all(1)
    flp = (bf == tg).all(1) & (bg == tf).all(1)

    # Symmetric lambda MAE: min(|pred-lam|, |pred-(1-lam)|)
    lam_err = np.minimum(np.abs(pt - tt), np.abs(pt - (1.0 - tt)))

    return ((bf == tf).mean() * 100,
            (bg == tg).mean() * 100,
            (fwd | flp).mean() * 100,
            float(lam_err.mean()))


def fmt(s):
    if s < 60:   return f"{s:.0f}s"
    if s < 3600: return f"{s/60:.1f}m"
    return f"{s/3600:.1f}h"


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_prf=[]; all_prg=[]; all_pt=[]
    all_trf=[]; all_trg=[]; all_tt=[]
    for x, fracs, wvars, yf, yg, yt in loader:
        rf, rg, lp, _, _ = model(x.to(DEVICE), fracs.to(DEVICE), wvars.to(DEVICE))
        all_prf.append(rf.cpu()); all_trf.append(yf)
        all_prg.append(rg.cpu()); all_trg.append(yg)
        all_pt.append(lp.cpu());  all_tt.append(yt)
    return compute_metrics(
        torch.cat(all_prf), torch.cat(all_prg), torch.cat(all_pt),
        torch.cat(all_trf), torch.cat(all_trg), torch.cat(all_tt))


def check_resume(path):
    if not os.path.exists(path): return 0, 0.0
    try:
        ck   = torch.load(path, map_location="cpu", weights_only=False)
        ep   = int(ck.get("epoch", 0))
        best = float(ck.get("best_both", 0.0))
        if _CKPT_KEY not in ck.get("model_state", {}):
            print("  Stale checkpoint. Discarding."); os.remove(path); return 0, 0.0
        print(f"  Resuming from epoch {ep}, best={best:.2f}%")
        return ep, best
    except Exception as e:
        print(f"  Checkpoint unreadable ({e}). Starting fresh."); return 0, 0.0


# ══════════════════════════════════════════════════════════════════
# STAGE 1
# ══════════════════════════════════════════════════════════════════

def train_stage1(model, train_rules, test_ld, save_path):
    print(f"\n{'='*72}")
    print(f"  STAGE 1  |  {S1_EPOCHS} epochs  |  warmup {WARMUP_STEPS} steps")
    print(f"  LR {S1_LR}→{S1_LR_MIN}")
    print(f"  Losses: rules_sym + lam_mse({S1_LAM_MSE}) + direct({S1_LAM_DIRECT})")
    print(f"  lam_pred always ≤ 0.5 (symmetric magnitude, mirrors TSCA tau)")
    print(f"  No assignment loss (SCA has no per-timestep rule labels)")
    print(f"{'='*72}")

    resume_ep, best_both = check_resume(save_path)
    if resume_ep > 0:
        ck = torch.load(save_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model_state"])

    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=S1_LR, weight_decay=1e-2)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=S1_EPOCHS, eta_min=S1_LR_MIN)
    sch = WarmupCosineScheduler(opt, WARMUP_STEPS, cos, lr_start=1e-6, lr_peak=S1_LR)
    if resume_ep > 0:
        sch._in_warmup = False
        for _ in range(resume_ep): cos.step()

    # Verify analytical lambda baseline before training
    if resume_ep == 0:
        model.eval()
        preds_v=[]; trues_v=[]
        with torch.no_grad():
            for x, fracs, wvars, yf, yg, yt in test_ld:
                pt = model.forward_lambda_analytical(fracs.to(DEVICE), wvars.to(DEVICE))
                preds_v.append(pt.cpu()); trues_v.append(yt)
        pred_t = torch.cat(preds_v).squeeze()
        true_t = torch.cat(trues_v).squeeze()
        sym_mae = float(torch.minimum(
            (pred_t - true_t).abs(),
            (pred_t - (1.0 - true_t)).abs()).mean())
        raw_mae = float((pred_t - true_t).abs().mean())
        print(f"  Analytical lambda (epoch 0):")
        print(f"    Symmetric MAE = {sym_mae:.4f}  (target < 0.05)")
        print(f"    Raw MAE       = {raw_mae:.4f}  (expected ~0.20 — model returns min(lam,1-lam))")
        print(f"    STATUS: {'GOOD' if sym_mae < 0.06 else 'HIGH -- MLP will compensate'}")
        print()

    ep_times = []
    print(f"  {'Ep':>4} | {'RuleLoss':>9} | {'LamMSE':>7} | {'DirLoss':>7} | "
          f"{'RFB':>6} | {'RGB':>6} | {'Both':>6} | {'LamMAE':>7} | "
          f"{'Range':>11} | ETA")
    print(f"  {'-'*105}")

    for ep in range(resume_ep + 1, S1_EPOCHS + 1):
        t1 = time.time()
        lh = get_lam_half(ep, S1_CURRICULUM)
        lo = round(max(LAM_MIN, 0.5 - lh), 2)
        hi = round(min(LAM_MAX, 0.5 + lh), 2)

        tr_ds = OnTheFlyDataset(train_rules, lh, PAIRS_PER_EPOCH,
                                SAMPLES_PER_PAIR, seed=10000 + ep * 7)
        tr_ld = DataLoader(tr_ds, batch_size=S1_BATCH, shuffle=True,
                           num_workers=0, pin_memory=PIN)

        model.train()
        tot_rule=tot_lam=tot_dir=n=0
        opt.zero_grad()

        for step, batch in enumerate(tr_ld):
            x, fracs, wvars, yf, yg, yt = batch
            x     = x.to(DEVICE);     fracs = fracs.to(DEVICE)
            wvars = wvars.to(DEVICE);  yf    = yf.to(DEVICE)
            yg    = yg.to(DEVICE);     yt    = yt.to(DEVICE)

            rf, rg, lp, rf_dir, rg_dir = model(x, fracs, wvars)

            rule_l = symmetric_rule_loss(rf, rg, yf, yg, bce)
            lam_l  = lam_mse_loss(lp, yt)
            dir_l  = direct_rule_loss_symmetric(rf_dir, rg_dir, yf, yg, bce)

            loss = (rule_l + S1_LAM_MSE * lam_l + S1_LAM_DIRECT * dir_l) / S1_GRAD_ACC
            loss.backward()

            if (step + 1) % S1_GRAD_ACC == 0 or (step + 1) == len(tr_ld):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
                sch.step_after_update()

            tot_rule += rule_l.item() * len(x)
            tot_lam  += lam_l.item()  * len(x)
            tot_dir  += dir_l.item()  * len(x)
            n        += len(x)

        sch.epoch_step()
        rfb, rgb, both, lmae = evaluate(model, test_ld)
        ep_times.append(time.time() - t1)
        eta  = fmt(np.mean(ep_times[-5:]) * (S1_EPOCHS - ep))
        star = ""

        if both > best_both:
            best_both = both
            torch.save({"model_state": model.state_dict(),
                        "stage": "stage1", "epoch": ep,
                        "best_both": best_both}, save_path)
            star = " *"

        print(f"  {ep:4d} | {tot_rule/n:9.5f} | {tot_lam/n:7.5f} | "
              f"{tot_dir/n:7.3f} | "
              f"{rfb:5.1f}% | {rgb:5.1f}% | {both:5.1f}% | "
              f"{lmae:7.4f} | [{lo:.2f},{hi:.2f}] | {eta}{star}",
              flush=True)

    print(f"\n  Stage 1 complete. Best both-exact: {best_both:.2f}%")
    return best_both


# ══════════════════════════════════════════════════════════════════
# STAGE 2
# ══════════════════════════════════════════════════════════════════

def train_stage2(model, train_rules, test_ld, save_path):
    print(f"\n{'='*72}")
    print(f"  STAGE 2  |  {S2_EPOCHS} epochs  |  fine-tune all params")
    print(f"  LambdaMLP LR={S2_LR_LAM}  Main LR={S2_LR_MAIN}")
    print(f"  lam_mse lam={S2_LAM_MSE}")
    print(f"{'='*72}")

    resume_ep, best_both = check_resume(save_path)
    if resume_ep > 0:
        ck = torch.load(save_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model_state"])

    bce = nn.BCEWithLogitsLoss()

    lam_params  = list(model.lam_mlp.parameters())
    lam_ids     = {id(p) for p in lam_params}
    main_params = [p for p in model.parameters() if id(p) not in lam_ids]

    opt = torch.optim.AdamW([
        {"params": lam_params,  "lr": S2_LR_LAM,
         "_lr_ratio": S2_LR_LAM / S2_LR_MAIN},
        {"params": main_params, "lr": S2_LR_MAIN,
         "_lr_ratio": 1.0},
    ], weight_decay=1e-2)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=S2_EPOCHS, eta_min=S2_LR_MIN)
    sch = WarmupCosineScheduler(opt, S2_WARMUP, cos,
                                lr_start=1e-7, lr_peak=S2_LR_MAIN)
    if resume_ep > 0:
        sch._in_warmup = False
        for _ in range(resume_ep): cos.step()

    ep_times = []
    print(f"  {'Ep':>4} | {'RuleLoss':>9} | {'LamMSE':>7} | {'DirLoss':>7} | "
          f"{'RFB':>6} | {'RGB':>6} | {'Both':>6} | {'LamMAE':>7} | "
          f"{'Range':>11} | ETA")
    print(f"  {'-'*105}")

    for ep in range(resume_ep + 1, S2_EPOCHS + 1):
        t1 = time.time()
        lh = get_lam_half(ep, S2_CURRICULUM)
        lo = round(max(LAM_MIN, 0.5 - lh), 2)
        hi = round(min(LAM_MAX, 0.5 + lh), 2)

        tr_ds = OnTheFlyDataset(train_rules, lh, PAIRS_PER_EPOCH,
                                SAMPLES_PER_PAIR, seed=20000 + ep * 13)
        tr_ld = DataLoader(tr_ds, batch_size=S2_BATCH, shuffle=True,
                           num_workers=0, pin_memory=PIN)

        model.train()
        tot_rule=tot_lam=tot_dir=n=0
        opt.zero_grad()

        for step, batch in enumerate(tr_ld):
            x, fracs, wvars, yf, yg, yt = batch
            x     = x.to(DEVICE);     fracs = fracs.to(DEVICE)
            wvars = wvars.to(DEVICE);  yf    = yf.to(DEVICE)
            yg    = yg.to(DEVICE);     yt    = yt.to(DEVICE)

            rf, rg, lp, rf_dir, rg_dir = model(x, fracs, wvars)

            rule_l = symmetric_rule_loss(rf, rg, yf, yg, bce)
            lam_l  = lam_mse_loss(lp, yt)
            dir_l  = direct_rule_loss_symmetric(rf_dir, rg_dir, yf, yg, bce)

            loss = (rule_l + S2_LAM_MSE * lam_l + S2_LAM_DIRECT * dir_l) / S2_GRAD_ACC
            loss.backward()

            if (step + 1) % S2_GRAD_ACC == 0 or (step + 1) == len(tr_ld):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
                sch.step_after_update()

            tot_rule += rule_l.item() * len(x)
            tot_lam  += lam_l.item()  * len(x)
            tot_dir  += dir_l.item()  * len(x)
            n        += len(x)

        sch.epoch_step()
        rfb, rgb, both, lmae = evaluate(model, test_ld)
        ep_times.append(time.time() - t1)
        eta  = fmt(np.mean(ep_times[-5:]) * (S2_EPOCHS - ep))
        star = ""

        if both > best_both:
            best_both = both
            torch.save({"model_state": model.state_dict(),
                        "stage": "stage2", "epoch": ep,
                        "best_both": best_both}, save_path)
            star = " *"

        print(f"  {ep:4d} | {tot_rule/n:9.5f} | {tot_lam/n:7.5f} | "
              f"{tot_dir/n:7.3f} | "
              f"{rfb:5.1f}% | {rgb:5.1f}% | {both:5.1f}% | "
              f"{lmae:7.4f} | [{lo:.2f},{hi:.2f}] | {eta}{star}",
              flush=True)

    print(f"\n  Stage 2 complete. Best both-exact: {best_both:.2f}%")
    return best_both


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    os.makedirs(CKPT_DIR, exist_ok=True)

    for fname in ["orbits.npy", "frac_stats.npy", "within_var_stats.npy"]:
        p = os.path.join(DATA_DIR, "test", fname)
        if not os.path.exists(p):
            print(f"ERROR: {p} not found. Run DATAGEN_SCA.py first.")
            sys.exit(1)

    train_rules = np.load(os.path.join(DATA_DIR, "train_rules.npy")).tolist()

    model = ECANetSCA().to(DEVICE)
    npar  = sum(p.numel() for p in model.parameters())
    print(f"\nECANetSCA v1  |  {npar:,} params  ({npar/1e6:.2f}M)")
    print(f"KEY DESIGN DECISIONS:")
    print(f"  Lambda continuous [0.1, 0.9] — same as tau in TSCA")
    print(f"  lam_pred always ≤ 0.5 (symmetric magnitude)")
    print(f"  lam_mse_loss = MSE(lam_pred, min(lam, 1-lam)) — identical to TSCA")
    print(f"  New signal: within_var_stats estimates lambda*(1-lambda) directly")
    print(f"  No assignment loss (no per-timestep rule labels in SCA)")
    print(f"  No ts_residual (per-timestep residual is zero-mean in SCA)")
    print(f"Checkpoints → {CKPT_DIR}\n")

    s1_ckpt = os.path.join(CKPT_DIR, "stage1.pt")
    s2_ckpt = os.path.join(CKPT_DIR, "stage2.pt")

    test_ds = StaticDataset(os.path.join(DATA_DIR, "test"))
    test_ld = DataLoader(test_ds, batch_size=8, shuffle=False,
                         num_workers=0, pin_memory=PIN)
    print(f"Test set: {len(test_ds):,} samples")

    best_s1 = train_stage1(model, train_rules, test_ld, s1_ckpt)

    ck = torch.load(s1_ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model_state"])
    print(f"\n  Loaded Stage 1 best (both={best_s1:.2f}%)")

    best_s2 = train_stage2(model, train_rules, test_ld, s2_ckpt)

    print(f"\n{'='*72}\n  TRAINING COMPLETE\n{'='*72}")
    print(f"  Stage 1 best : {best_s1:.2f}%")
    print(f"  Stage 2 best : {best_s2:.2f}%")

    ck = torch.load(s2_ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model_state"])
    rfb, rgb, both, lmae = evaluate(model, test_ld)
    print(f"\n  FINAL TEST  (epoch {ck['epoch']})")
    print(f"  Rule F bit acc : {rfb:.2f}%")
    print(f"  Rule G bit acc : {rgb:.2f}%")
    print(f"  Both exact     : {both:.2f}%  (symmetric)")
    print(f"  Lambda MAE     : {lmae:.4f}  (symmetric min(|pred-lam|,|pred-(1-lam)|))")
    print(f"  Total time     : {fmt(time.time()-t_start)}")
    print(f"\n  Next: python FINAL_TEST_SCA.py")


if __name__ == "__main__":
    main()
