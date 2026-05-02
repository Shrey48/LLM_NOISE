"""
TRAIN_TSCA.py  --  v9
======================
Directory: temporal_v9/

DEFINITIVE TRAINING. Built from complete cross-version analysis.

════════════════════════════════════════════════════════════════
KEY TRAINING CHANGES FROM ALL PREVIOUS VERSIONS
════════════════════════════════════════════════════════════════

CHANGE 1 — No tau_anchor_loss
  The anchor loss was present in v8/v8-fix/FINAL. Analysis showed it
  caused TauMAE to degrade from 0.057 to 0.125 as curriculum widened,
  by fighting the symmetric_loss over orientation. Eliminated entirely.

CHANGE 2 — symmetric_loss uses min(tau, 1-tau) correctly
  Since tau_pred is always ≤ 0.5, the symmetric_loss works as follows:
    Forward orientation: tau_target = true_tau, tau_pred ≈ min(tau,1-tau)
    Flipped orientation: tau_target = 1-true_tau, tau_pred ≈ min(tau,1-tau)
  Both orientations produce the same MSE loss for the tau component!
  So the symmetric_loss minimum is taken only based on the RULE bits.
  The tau MSE is identical regardless of orientation flip.

  Consequence: lam_tau in symmetric_loss is now less critical.
  The tau gradient flows purely from the symmetric tau MSE loss.
  We use a SEPARATE tau_mse_loss that targets min(true_tau, 1-true_tau)
  directly. This gives the correction MLP clean, unambiguous gradient.

CHANGE 3 — tau_mse_loss replaces tau_anchor_loss
  target = min(true_tau, 1-true_tau)  in [TAU_MIN, 0.5]
  pred   = tau_pred                    in [TAU_MIN, 0.5]
  loss   = MSE(pred, target)
  This is NOT the anchor loss from v8 (which computed min of BOTH pred
  and target). Here only the TARGET is min(...), not the prediction.
  The gradient correctly pushes pred toward the correct value.
  Used with lam=3.0 throughout training.

CHANGE 4 — lam_tau=0.0 in symmetric_loss
  Since tau_mse_loss handles tau supervision cleanly, we remove tau from
  symmetric_loss entirely. This prevents the symmetric_loss from picking
  orientation based on tau (which was causing instability in v8).
  symmetric_loss now only decides orientation from rule BCE losses.
  tau is supervised separately by tau_mse_loss.

CHANGE 5 — Curriculum stays at full range from epoch 11 onward
  Previous versions showed rapid improvement at epochs 13-14 (60%+ accuracy)
  when the curriculum reached full range. We now stay at full range from
  epoch 11. The analytical tau handles full range from epoch 0.

CHANGE 6 — assignment_loss weight increased to 3.0 in Stage 1
  With the residual-first score (FIX 3 in MODEL), the assignment signal is
  now much stronger from epoch 1 (residual_weight=5.0 provides direct
  correct signal). Higher lam_assign can now drive faster convergence.
  Previous concern was that 3.0 starved rule learning — but FIX 3 means
  assignment converges faster, freeing rule learning earlier.

UNCHANGED from previous versions:
  - assignment_loss_symmetric (orientation-aware)
  - direct_rule_loss_symmetric
  - WarmupCosineScheduler
  - Stage structure (80ep + 80ep)
  - Gradient clipping, GRAD_ACC=32, BATCH=4
  - Separate LRs for tau_mlp in Stage 2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys, time
from torch.utils.data import Dataset, DataLoader

# ══════════════════════════════════════════════════════════
BASE_DIR = "/home/shovik.roy/Shrey/new_check_model/ECA_temporal_stocastic/temporal_v9"
DATA_DIR = os.path.join(BASE_DIR, "TSCA_Data")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints_tsca_v9")
# ══════════════════════════════════════════════════════════

sys.path.insert(0, BASE_DIR)

from MODEL_TSCA import (
    ECANetTSCA, orbit_to_tokens, simulate_labeled,
    random_init, rule_to_bits, compute_frac_stats,
    W, T, N_ORBITS, N_TOK, N_TRANS, N_BITS, FRAC_DIM,
    TAU_MIN, TAU_MAX
)
N_RAW = 4

# ── Stage 1 hyperparameters ───────────────────────────────────────────────────
S1_EPOCHS     = 80
S1_LR         = 3e-4
S1_LR_MIN     = 1e-5
WARMUP_STEPS  = 100
S1_BATCH      = 4
S1_GRAD_ACC   = 32

# CHANGE 4: tau removed from symmetric_loss
S1_LAM_F      = 1.0
S1_LAM_G      = 1.0
S1_LAM_TAU    = 0.0     # tau supervised separately by tau_mse_loss

# CHANGE 3: clean tau MSE loss
S1_LAM_TAU_MSE = 3.0   # strong tau supervision throughout

# CHANGE 6: strong assignment with residual-first signal
S1_LAM_ASSIGN  = 3.0
S1_LAM_DIRECT  = 0.5

# CHANGE 5: reach full range by epoch 11
S1_CURRICULUM = [
    (10, 0.20),    # ep 1-10:  U(0.30, 0.70)
    (11, 0.40),    # ep 11:    U(0.10, 0.90) -- full range
    (80, 0.40),    # ep 12-80: stay full range
]

# ── Stage 2 hyperparameters ───────────────────────────────────────────────────
S2_EPOCHS      = 80
S2_LR_MAIN     = 3e-5
S2_LR_TAU      = 1e-5
S2_LR_MIN      = 1e-7
S2_WARMUP      = 30
S2_BATCH       = 4
S2_GRAD_ACC    = 32

S2_LAM_F       = 1.0
S2_LAM_G       = 1.0
S2_LAM_TAU     = 0.0    # still supervised by tau_mse_loss
S2_LAM_TAU_MSE = 2.0    # gentler in Stage 2
S2_LAM_ASSIGN  = 1.5
S2_LAM_DIRECT  = 0.3

S2_CURRICULUM = [
    (80, 0.40),    # always full range in Stage 2
]

PAIRS_PER_EPOCH  = 1000
SAMPLES_PER_PAIR = 8
_CKPT_KEY        = "tau_mlp.correction_net.0.weight"


# ── Warmup Cosine Scheduler ───────────────────────────────────────────────────
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
    def __init__(self, path):
        self.orbits = np.load(os.path.join(path, "orbits.npy"))
        self.fracs  = np.load(os.path.join(path, "frac_stats.npy"))
        self.rfb    = np.load(os.path.join(path, "rule_f_bits.npy"))
        self.rgb    = np.load(os.path.join(path, "rule_g_bits.npy"))
        self.taus   = np.load(os.path.join(path, "taus.npy"))
        assert self.orbits.ndim == 4 and self.orbits.shape[1] == N_ORBITS

    def __len__(self): return len(self.orbits)

    def __getitem__(self, i):
        toks = np.stack([orbit_to_tokens(self.orbits[i, k])
                         for k in range(N_ORBITS)], axis=0)
        return (torch.tensor(toks,           dtype=torch.float32),
                torch.tensor(self.fracs[i],  dtype=torch.float32),
                torch.tensor(self.rfb[i],    dtype=torch.float32),
                torch.tensor(self.rgb[i],    dtype=torch.float32),
                torch.tensor([self.taus[i]], dtype=torch.float32))


class OnTheFlyDataset(Dataset):
    def __init__(self, rules, lam_half, n_pairs, spp, seed):
        rng = np.random.default_rng(seed)
        lo  = max(TAU_MIN, 0.5 - lam_half)
        hi  = min(TAU_MAX, 0.5 + lam_half)
        n   = len(rules)
        seen = set(); pairs = []; att = 0
        while len(pairs) < n_pairs and att < n_pairs * 20:
            att += 1
            f = int(rules[rng.integers(0, n)])
            g = int(rules[rng.integers(0, n)])
            if f == g: continue
            p = (min(f, g), max(f, g))
            if p not in seen: pairs.append((f, g)); seen.add(p)
        while len(pairs) < n_pairs:
            f = int(rules[rng.integers(0, n)]); g = int(rules[rng.integers(0, n)])
            if f != g: pairs.append((f, g))

        self.toks_l=[]; self.fracs_l=[]; self.rfb_l=[]
        self.rgb_l=[]; self.taus_l=[]; self.slabels_l=[]

        for f, g in pairs:
            bf = rule_to_bits(f); bg = rule_to_bits(g)
            for _ in range(spp):
                tau = float(rng.uniform(lo, hi))
                orbits=[]; slbls=[]
                for _ in range(N_ORBITS):
                    init      = random_init(rng)
                    orb, slbl = simulate_labeled(f, g, init, tau, rng)
                    orbits.append(orb); slbls.append(slbl)
                orbs_k = np.stack(orbits); slbs_k = np.stack(slbls)
                fs     = compute_frac_stats(orbs_k)
                toks   = np.stack([orbit_to_tokens(orbs_k[k])
                                   for k in range(N_ORBITS)], axis=0)
                self.toks_l.append(toks); self.fracs_l.append(fs)
                self.slabels_l.append(slbs_k); self.rfb_l.append(bf)
                self.rgb_l.append(bg); self.taus_l.append(tau)

    def __len__(self): return len(self.toks_l)

    def __getitem__(self, i):
        return (torch.tensor(self.toks_l[i],    dtype=torch.float32),
                torch.tensor(self.fracs_l[i],   dtype=torch.float32),
                torch.tensor(self.rfb_l[i],     dtype=torch.float32),
                torch.tensor(self.rgb_l[i],     dtype=torch.float32),
                torch.tensor([self.taus_l[i]],  dtype=torch.float32),
                torch.tensor(self.slabels_l[i], dtype=torch.float32))


def get_lam_half(ep, curriculum):
    prev_end = 0; prev_h = curriculum[0][1]
    for end, h in curriculum:
        if ep <= end:
            prog = (ep - prev_end - 1) / max(end - prev_end - 1, 1)
            return round(prev_h + (h - prev_h) * prog, 4)
        prev_end = end; prev_h = h
    return curriculum[-1][1]


# ══════════════════════════════════════════════════════════════════
# LOSSES
# ══════════════════════════════════════════════════════════════════

def symmetric_loss_rules_only(prf, prg, trf, trg, bce):
    """
    CHANGE 4: tau removed from symmetric_loss entirely.
    Orientation is determined by rule BCE losses only.
    This removes the instability caused by tau competing with rules
    for orientation selection.
    """
    B  = prf.shape[0]
    lf = torch.zeros(B, device=prf.device)
    lb = torch.zeros(B, device=prf.device)
    for i in range(B):
        lf[i] = bce(prf[i], trf[i]) + bce(prg[i], trg[i])
        lb[i] = bce(prf[i], trg[i]) + bce(prg[i], trf[i])
    return torch.minimum(lf, lb).mean()


def tau_mse_loss(tau_pred, ttau):
    """
    CHANGE 3: Clean tau supervision targeting min(tau, 1-tau).

    tau_pred is in [TAU_MIN, 0.5] (by model design).
    Target is min(true_tau, 1-true_tau) also in [TAU_MIN, 0.5].
    Simple MSE. No orientation ambiguity. No anchor vs symmetric fight.

    tau_pred : [B, 1]
    ttau     : [B, 1]  -- true tau in [TAU_MIN, TAU_MAX]
    """
    target = torch.minimum(ttau, 1.0 - ttau)   # [B,1] in [TAU_MIN, 0.5]
    return F.mse_loss(tau_pred, target)


def assignment_loss_symmetric(prob_g, step_labels):
    """Orientation-aware assignment loss. Unchanged from v5b (correct)."""
    mean_labels = step_labels.mean(dim=1)
    loss_fwd = F.binary_cross_entropy(
        prob_g, mean_labels, reduction='none').mean(dim=1)
    loss_flp = F.binary_cross_entropy(
        prob_g, 1.0 - mean_labels, reduction='none').mean(dim=1)
    return torch.minimum(loss_fwd, loss_flp).mean()


def direct_rule_loss_symmetric(rf_dir, rg_dir, trf, trg, bce):
    B  = rf_dir.shape[0]
    lf = torch.zeros(B, device=rf_dir.device)
    lb = torch.zeros(B, device=rf_dir.device)
    for i in range(B):
        lf[i] = bce(rf_dir[i], trf[i]) + bce(rg_dir[i], trg[i])
        lb[i] = bce(rf_dir[i], trg[i]) + bce(rg_dir[i], trf[i])
    return torch.minimum(lf, lb).mean()


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(prf, prg, ptau, trf, trg, ttau):
    pf = prf.cpu().numpy(); pg = prg.cpu().numpy()
    pt = ptau.cpu().numpy().reshape(-1)
    tf = trf.cpu().numpy(); tg = trg.cpu().numpy()
    tt = ttau.cpu().numpy().reshape(-1)
    bf = (1/(1+np.exp(-pf)) > 0.5).astype(np.float32)
    bg = (1/(1+np.exp(-pg)) > 0.5).astype(np.float32)
    fwd = (bf==tf).all(1) & (bg==tg).all(1)
    flp = (bf==tg).all(1) & (bg==tf).all(1)
    # Symmetric tau MAE: min(|pred-tau|, |pred-(1-tau)|)
    tau_err = np.minimum(np.abs(pt - tt), np.abs(pt - (1.0 - tt)))
    return ((bf==tf).mean()*100, (bg==tg).mean()*100,
            (fwd|flp).mean()*100, float(tau_err.mean()))

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
    for x, fracs, yf, yg, yt in loader:
        rf, rg, tau, _, _, _ = model(x.to(DEVICE), fracs.to(DEVICE))
        all_prf.append(rf.cpu()); all_trf.append(yf)
        all_prg.append(rg.cpu()); all_trg.append(yg)
        all_pt.append(tau.cpu()); all_tt.append(yt)
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
    print(f"  Losses: rules_sym + tau_mse({S1_LAM_TAU_MSE}) + "
          f"assign({S1_LAM_ASSIGN}) + direct({S1_LAM_DIRECT})")
    print(f"  tau_pred always ≤ 0.5 (symmetric magnitude)")
    print(f"  No tau_anchor_loss -- tau_mse_loss is clean and unambiguous")
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

    # Verify analytical tau before training
    if resume_ep == 0:
        model.eval()
        preds_v=[]; trues_v=[]
        with torch.no_grad():
            for x, fracs, yf, yg, yt in test_ld:
                pt = model.forward_tau_analytical(fracs.to(DEVICE))
                preds_v.append(pt.cpu()); trues_v.append(yt)
        pred_t = torch.cat(preds_v).squeeze()
        true_t = torch.cat(trues_v).squeeze()
        # Symmetric MAE: min(|pred-tau|, |pred-(1-tau)|)
        sym_mae = float(torch.minimum(
            (pred_t - true_t).abs(),
            (pred_t - (1.0 - true_t)).abs()).mean())
        raw_mae = float((pred_t - true_t).abs().mean())
        print(f"  Analytical tau (epoch 0):")
        print(f"    Symmetric MAE = {sym_mae:.4f}  (target < 0.06)")
        print(f"    Raw MAE       = {raw_mae:.4f}  (expected ~0.20, NOT a warning)")
        print(f"    Raw MAE is always ~0.20 because analytical returns min(tau,1-tau)")
        print(f"    STATUS: {'GOOD' if sym_mae < 0.06 else 'HIGH -- transformer will compensate'}")
        print()
        # Verify residual_weight
        rw = model.rule_head.residual_weight.item()
        print(f"  residual_weight init = {rw:.4f}  (expected 5.0)")
        print()

    ep_times = []
    print(f"  {'Ep':>4} | {'RuleLoss':>9} | {'TauMSE':>7} | {'Asgn':>6} | "
          f"{'Dir':>6} | {'RFB':>6} | {'RGB':>6} | {'Both':>6} | "
          f"{'TauMAE':>7} | {'Range':>11} | ETA")
    print(f"  {'-'*112}")

    for ep in range(resume_ep + 1, S1_EPOCHS + 1):
        t1  = time.time()
        lh  = get_lam_half(ep, S1_CURRICULUM)
        lo  = round(max(TAU_MIN, 0.5 - lh), 2)
        hi  = round(min(TAU_MAX, 0.5 + lh), 2)

        tr_ds = OnTheFlyDataset(train_rules, lh, PAIRS_PER_EPOCH,
                                SAMPLES_PER_PAIR, seed=10000 + ep * 7)
        tr_ld = DataLoader(tr_ds, batch_size=S1_BATCH, shuffle=True,
                           num_workers=0, pin_memory=PIN)

        model.train()
        tot_rule=tot_tau=tot_asgn=tot_dir=n=0
        opt.zero_grad()

        for step, batch in enumerate(tr_ld):
            x, fracs, yf, yg, yt, slabels = batch
            x      = x.to(DEVICE);  fracs   = fracs.to(DEVICE)
            yf     = yf.to(DEVICE); yg      = yg.to(DEVICE)
            yt     = yt.to(DEVICE); slabels = slabels.to(DEVICE)

            rf, rg, tau, prob_g, rf_dir, rg_dir = model(x, fracs)

            # CHANGE 4: tau not in symmetric_loss
            rule_l = symmetric_loss_rules_only(rf, rg, yf, yg, bce)
            # CHANGE 3: clean tau MSE targeting min(tau, 1-tau)
            tmse_l = tau_mse_loss(tau, yt)
            asgn_l = assignment_loss_symmetric(prob_g, slabels)
            dir_l  = direct_rule_loss_symmetric(rf_dir, rg_dir, yf, yg, bce)

            loss = (rule_l +
                    S1_LAM_TAU_MSE * tmse_l +
                    S1_LAM_ASSIGN  * asgn_l +
                    S1_LAM_DIRECT  * dir_l) / S1_GRAD_ACC
            loss.backward()

            if (step + 1) % S1_GRAD_ACC == 0 or (step + 1) == len(tr_ld):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
                sch.step_after_update()

            tot_rule += rule_l.item() * len(x)
            tot_tau  += tmse_l.item() * len(x)
            tot_asgn += asgn_l.item() * len(x)
            tot_dir  += dir_l.item()  * len(x)
            n        += len(x)

        sch.epoch_step()
        rfb, rgb, both, tmae = evaluate(model, test_ld)
        ep_times.append(time.time() - t1)
        eta  = fmt(np.mean(ep_times[-5:]) * (S1_EPOCHS - ep))
        star = ""

        if both > best_both:
            best_both = both
            torch.save({"model_state": model.state_dict(),
                        "stage": "stage1", "epoch": ep,
                        "best_both": best_both}, save_path)
            star = " *"

        print(f"  {ep:4d} | {tot_rule/n:9.5f} | {tot_tau/n:7.5f} | "
              f"{tot_asgn/n:6.3f} | {tot_dir/n:6.3f} | "
              f"{rfb:5.1f}% | {rgb:5.1f}% | {both:5.1f}% | "
              f"{tmae:7.4f} | [{lo:.2f},{hi:.2f}] | {eta}{star}",
              flush=True)

    print(f"\n  Stage 1 complete. Best both-exact: {best_both:.2f}%")
    return best_both


# ══════════════════════════════════════════════════════════════════
# STAGE 2
# ══════════════════════════════════════════════════════════════════

def train_stage2(model, train_rules, test_ld, save_path):
    print(f"\n{'='*72}")
    print(f"  STAGE 2  |  {S2_EPOCHS} epochs  |  fine-tune all params")
    print(f"  TauMLP LR={S2_LR_TAU}  Main LR={S2_LR_MAIN}")
    print(f"  tau_mse lam={S2_LAM_TAU_MSE}  assign lam={S2_LAM_ASSIGN}")
    print(f"{'='*72}")

    resume_ep, best_both = check_resume(save_path)
    if resume_ep > 0:
        ck = torch.load(save_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model_state"])

    bce = nn.BCEWithLogitsLoss()

    tau_params  = list(model.tau_mlp.parameters())
    tau_ids     = {id(p) for p in tau_params}
    main_params = [p for p in model.parameters() if id(p) not in tau_ids]

    opt = torch.optim.AdamW([
        {"params": tau_params,  "lr": S2_LR_TAU,
         "_lr_ratio": S2_LR_TAU / S2_LR_MAIN},
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
    print(f"  {'Ep':>4} | {'RuleLoss':>9} | {'TauMSE':>7} | {'Asgn':>6} | "
          f"{'Dir':>6} | {'RFB':>6} | {'RGB':>6} | {'Both':>6} | "
          f"{'TauMAE':>7} | {'Range':>11} | ETA")
    print(f"  {'-'*112}")

    for ep in range(resume_ep + 1, S2_EPOCHS + 1):
        t1  = time.time()
        lh  = get_lam_half(ep, S2_CURRICULUM)
        lo  = round(max(TAU_MIN, 0.5 - lh), 2)
        hi  = round(min(TAU_MAX, 0.5 + lh), 2)

        tr_ds = OnTheFlyDataset(train_rules, lh, PAIRS_PER_EPOCH,
                                SAMPLES_PER_PAIR, seed=20000 + ep * 13)
        tr_ld = DataLoader(tr_ds, batch_size=S2_BATCH, shuffle=True,
                           num_workers=0, pin_memory=PIN)

        model.train()
        tot_rule=tot_tau=tot_asgn=tot_dir=n=0
        opt.zero_grad()

        for step, batch in enumerate(tr_ld):
            x, fracs, yf, yg, yt, slabels = batch
            x      = x.to(DEVICE);  fracs   = fracs.to(DEVICE)
            yf     = yf.to(DEVICE); yg      = yg.to(DEVICE)
            yt     = yt.to(DEVICE); slabels = slabels.to(DEVICE)

            rf, rg, tau, prob_g, rf_dir, rg_dir = model(x, fracs)

            rule_l = symmetric_loss_rules_only(rf, rg, yf, yg, bce)
            tmse_l = tau_mse_loss(tau, yt)
            asgn_l = assignment_loss_symmetric(prob_g, slabels)
            dir_l  = direct_rule_loss_symmetric(rf_dir, rg_dir, yf, yg, bce)

            loss = (rule_l +
                    S2_LAM_TAU_MSE * tmse_l +
                    S2_LAM_ASSIGN  * asgn_l +
                    S2_LAM_DIRECT  * dir_l) / S2_GRAD_ACC
            loss.backward()

            if (step + 1) % S2_GRAD_ACC == 0 or (step + 1) == len(tr_ld):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
                sch.step_after_update()

            tot_rule += rule_l.item() * len(x)
            tot_tau  += tmse_l.item() * len(x)
            tot_asgn += asgn_l.item() * len(x)
            tot_dir  += dir_l.item()  * len(x)
            n        += len(x)

        sch.epoch_step()
        rfb, rgb, both, tmae = evaluate(model, test_ld)
        ep_times.append(time.time() - t1)
        eta  = fmt(np.mean(ep_times[-5:]) * (S2_EPOCHS - ep))
        star = ""

        if both > best_both:
            best_both = both
            torch.save({"model_state": model.state_dict(),
                        "stage": "stage2", "epoch": ep,
                        "best_both": best_both}, save_path)
            star = " *"

        print(f"  {ep:4d} | {tot_rule/n:9.5f} | {tot_tau/n:7.5f} | "
              f"{tot_asgn/n:6.3f} | {tot_dir/n:6.3f} | "
              f"{rfb:5.1f}% | {rgb:5.1f}% | {both:5.1f}% | "
              f"{tmae:7.4f} | [{lo:.2f},{hi:.2f}] | {eta}{star}",
              flush=True)

    print(f"\n  Stage 2 complete. Best both-exact: {best_both:.2f}%")
    return best_both


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    os.makedirs(CKPT_DIR, exist_ok=True)

    for fname in ["orbits.npy", "frac_stats.npy"]:
        p = os.path.join(DATA_DIR, "test", fname)
        if not os.path.exists(p):
            print(f"ERROR: {p} not found.")
            print("Symlink TSCA_Data from temporal_v5 (same data, no regeneration):")
            print(f"  ln -s <path_to_temporal_v5>/TSCA_Data {DATA_DIR}")
            sys.exit(1)

    train_rules = np.load(os.path.join(DATA_DIR, "train_rules.npy")).tolist()

    model = ECANetTSCA().to(DEVICE)
    npar  = sum(p.numel() for p in model.parameters())
    print(f"\nECANetTSCA v9  |  {npar:,} params  ({npar/1e6:.2f}M)")
    print(f"KEY CHANGES vs all previous versions:")
    print(f"  FIX 1: No tau_anchor_loss (it fought symmetric_loss)")
    print(f"  FIX 2: tau_pred always ≤ 0.5 (symmetric magnitude)")
    print(f"  FIX 3: Residual-first score (direct path, not buried in 256 dims)")
    print(f"  FIX 4: tau removed from symmetric_loss (rules only)")
    print(f"  CHANGE 3: tau_mse_loss = MSE(tau_pred, min(tau,1-tau)) -- clean")
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
    rfb, rgb, both, tmae = evaluate(model, test_ld)
    print(f"\n  FINAL TEST  (epoch {ck['epoch']})")
    print(f"  Rule F bit acc : {rfb:.2f}%")
    print(f"  Rule G bit acc : {rgb:.2f}%")
    print(f"  Both exact     : {both:.2f}%  (symmetric)")
    print(f"  Tau MAE        : {tmae:.4f}  (symmetric min(|pred-tau|,|pred-(1-tau)|))")
    print(f"  Total time     : {fmt(time.time()-t_start)}")
    print(f"\n  Next: python FINAL_TEST_TSCA.py")


if __name__ == "__main__":
    main()
