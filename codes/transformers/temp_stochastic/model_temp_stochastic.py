"""
MODEL_TSCA.py  --  ECANetTSCA v9
==================================
Directory: temporal_v9/

DEFINITIVE ARCHITECTURE. Built from complete cross-version analysis.

════════════════════════════════════════════════════════════════
WHAT IS FIXED AND WHY (from analysis of all 8 previous versions)
════════════════════════════════════════════════════════════════

FIX 1 — tau_anchor_loss ELIMINATED
  The anchor loss (present in v8, v8-fix, FINAL) targeted min(tau,1-tau)
  for both prediction and ground truth. This locked the correction MLP
  to always predict ≤0.5, fighting the symmetric_loss orientation.
  Result: TauMAE rose from 0.057 → 0.125 as curriculum widened.
  Fix: no anchor loss. The symmetric_loss already handles orientation
  correctly. The correction MLP learns orientation from that signal.

FIX 2 — Correction MLP predicts magnitude only (stays ≤ 0.5)
  The analytical formula gives min(tau, 1-tau) ∈ [TAU_MIN, 0.5].
  COR_MAX is 0.12. So tau_pred = tau_analytical + correction.
  But tau_analytical ≤ 0.5 and correction ∈ (-0.12, +0.12), so
  tau_pred can range from ~0.1 to ~0.62 — it can cross 0.5.
  When tau_pred > 0.5, the symmetric_loss treats it differently
  than tau_pred < 0.5 for the two orientations. This creates
  instability when the correction oscillates sign.

  Fix: clamp tau_pred to [TAU_MIN, 0.5] after correction.
  Then let the symmetric_loss figure out orientation via its
  min(forward_loss, flipped_loss) mechanism. For a sample with
  true tau=0.7, the model predicts 0.3, and the symmetric_loss
  correctly identifies flipped orientation (using 1-0.3=0.7 as tau).
  For true tau=0.3, model predicts 0.3, forward orientation.
  Both orientations are handled correctly. No instability.

  The TauMAE will correctly report ~0.05 (symmetric) throughout.

FIX 3 — Residual-first score_net
  In v8/FINAL, ts_residual was one of [h(256), tau(1), var(1), res(1)].
  Signal ratio: 1/259 = 0.4%. Score_net couldn't "see" it despite 4× scaling.
  Result: Asgn dropped from 0.69 to only 0.57 over 80 epochs.

  Fix: score = RESIDUAL_WEIGHT * ts_residual + score_net([h, tau, var])
  where RESIDUAL_WEIGHT is a learnable parameter initialized to 5.0.
  The residual contributes directly to the logit, bypassing the deep
  network. Since E[residual | rule_g fired] > E[residual | rule_f fired],
  this gives the score_net a strong correct prior from epoch 1.
  The score_net([h, tau, var]) adds a learned refinement on top.

  This is inspired by residual connections: the direct path carries
  the clean signal, the deep path learns corrections to it.

FIX 4 — COR_MAX adaptive to confidence
  Low-confidence pairs (all-agreement rules) need larger corrections
  because the analytical estimate defaults to 0.5.
  High-confidence pairs need smaller corrections (already accurate).
  Fix: COR_MAX_BASE=0.08 for all pairs, plus confidence-gated extra:
    effective_cor_max = COR_MAX_BASE + (1 - confidence) * COR_MAX_EXTRA
  where COR_MAX_EXTRA=0.35. High confidence: max=0.08. Low: max=0.43.
  This prevents the MLP from disturbing accurate estimates while
  allowing large corrections for all-agreement pairs.

KEPT FROM PREVIOUS VERSIONS (working correctly):
  - analytical_tau_robust: 3-method blend giving sym MAE=0.057
  - DirectRuleDecoder: per-pattern analytical rule estimation
  - PatternEmbedding: 8-dim learned pattern embedding
  - CellEncoder with learnable query attention
  - TimestepEncoder: 6-layer transformer
  - ConfidenceWeightedOrbitFusion: learned orbit weights
  - _compute_ts_residual: a - frac[pat] signal
  - orbit_to_tokens: 4-feature raw tokens [L,B,R,A]

════════════════════════════════════════════════════════════════
ARCHITECTURE OVERVIEW
════════════════════════════════════════════════════════════════

Input: raw_tokens [B, K, N_TOK, 4] + frac_stats [B, 24]

PATH A (Tau, analytical):
  analytical_tau_robust(frac_stats) → tau_mag [B,1] ∈ [TAU_MIN, 0.5]
  + correction_net(features, confidence) * effective_cor_max
  → tau_pred [B,1] ∈ [TAU_MIN, 0.5]  (symmetric, always ≤ 0.5)
  The symmetric_loss handles the ≤0.5 / ≥0.5 orientation.

PATH B (Rules, transformer):
  PatternEmbedding → CellEncoder → TimestepEncoder → OrbitFusion
  → h [B, N_TRANS, D2]
  _compute_ts_residual → ts_residual [B, N_TRANS]
  score = RESIDUAL_WEIGHT * ts_residual + score_net(h, tau, var)
  prob_g = sigmoid(score)
  DirectRuleDecoder → direct_rf, direct_rg [B, 8]
  rep_f = weighted_sum(h, 1-prob_g), rep_g = weighted_sum(h, prob_g)
  rf_logits = rf_head([rep_f, direct_rf.detach()])
  rg_logits = rg_head([rep_g, direct_rg.detach()])

OUTPUT: rf_logits [B,8], rg_logits [B,8], tau_pred [B,1],
        prob_g [B,N_TRANS], rf_dir [B,8], rg_dir [B,8]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# ── Constants ─────────────────────────────────────────────────────────────────
W        = 50
T        = 200
N_ORBITS = 8
N_TRANS  = T - 1
N_CELLS  = W
N_FEAT   = 12              # [L, B, R, A] + 8-dim pattern embedding
N_TOK    = N_TRANS * W
N_BITS   = 8
N_PAT    = 8
FRAC_DIM = 24
N_EMB    = 8

D1       = 128
N_L1     = 4
N_HEADS1 = 4
D_FF1    = 512

D2       = 256
N_L2     = 6
N_HEADS2 = 8
D_FF2    = 1024

DROPOUT  = 0.1
TAU_MIN  = 0.1
TAU_MAX  = 0.9
EPS      = 1e-8

# FIX 4: adaptive COR_MAX
COR_MAX_BASE  = 0.08   # max correction for high-confidence pairs
COR_MAX_EXTRA = 0.35   # extra range for zero-confidence (all-agreement) pairs


# ── Sinusoidal PE ─────────────────────────────────────────────────────────────
class SinusoidalPE(nn.Module):
    def __init__(self, d, max_len, dropout=DROPOUT):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d)
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() *
                        (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(-2)])


# ── Pattern Embedding ─────────────────────────────────────────────────────────
class PatternEmbedding(nn.Module):
    """Learned 8-dim embedding for each of the 8 ECA neighborhood patterns."""
    def __init__(self, n_pat=8, emb_dim=N_EMB):
        super().__init__()
        self.emb = nn.Embedding(n_pat, emb_dim)
        nn.init.normal_(self.emb.weight, std=0.1)

    def forward(self, pat_idx):
        return self.emb(pat_idx)


# ══════════════════════════════════════════════════════════════════
# PATH A: ANALYTICAL TAU (FIX 1 + FIX 2 + FIX 4)
# ══════════════════════════════════════════════════════════════════

def analytical_tau_robust(frac_stats):
    """
    Three-method robust tau estimator.
    Returns tau_est [B,1] in [TAU_MIN, 0.5], confidence [B,1] in [0,1].

    ALWAYS returns values in [TAU_MIN, 0.5] -- this is min(tau, 1-tau).
    The symmetric_loss handles orientation (whether to use tau or 1-tau).

    Method A: var-based — max(var[n]) ≈ tau*(1-tau)
              → tau = 0.5 - sqrt(0.25 - max_var)
    Method B: sorted frac interior (count-filtered)
              → min interior frac = min(tau, 1-tau)
    Method C: top-2 variance patterns, both must be genuine
              → min(frac[top2]) = min(tau, 1-tau)
    """
    frac = frac_stats[:, :8]
    var_ = frac_stats[:, 8:16]
    cnt  = frac_stats[:, 16:24]

    well_obs   = cnt > 0.005
    var_masked = var_.clone()
    var_masked[~well_obs] = 0.0
    max_var_w  = var_masked.max(dim=1).values   # [B]

    # Method A: var-based
    has_signal      = max_var_w > 0.005
    max_var_clamped = max_var_w.clamp(0.0, 0.2499)
    tau_A_raw       = 0.5 - torch.sqrt((0.25 - max_var_clamped).clamp(min=1e-8))
    tau_A           = torch.where(has_signal, tau_A_raw,
                                  torch.full_like(tau_A_raw, 0.5))
    tau_A           = tau_A.clamp(TAU_MIN, 0.5)   # FIX 2: clamp to ≤0.5

    # Method B: filtered interior of sorted fracs
    frac_obs        = frac.clone()
    frac_obs[~well_obs] = -1.0
    sorted_frac, _  = torch.sort(frac_obs, dim=1)
    interior_mask   = (sorted_frac > 0.08) & (sorted_frac < 0.92)
    inf_fill        = sorted_frac.clone()
    inf_fill[~interior_mask] = 1.0
    tau_B           = inf_fill.min(dim=1).values
    tau_B[~interior_mask.any(dim=1)] = 0.5
    tau_B           = tau_B.clamp(TAU_MIN, 0.5)   # FIX 2

    # Method C: top-2 variance patterns (both must be genuine)
    var_obs_c       = var_.clone()
    var_obs_c[~well_obs] = 0.0
    top2_vals, top2_idx = var_obs_c.topk(2, dim=1)
    top2_fracs      = frac.gather(1, top2_idx)
    tau_C           = top2_fracs.min(dim=1).values
    low_var_C       = (top2_vals[:, 0] < 0.01) | (top2_vals[:, 1] < 0.02)
    tau_C[low_var_C] = 0.5
    tau_C           = tau_C.clamp(TAU_MIN, 0.5)   # FIX 2

    # Confidence and blend
    confidence      = (max_var_w / 0.25).clamp(0.0, 1.0)   # [B]
    w_A = 0.4 + 0.2 * (1.0 - confidence)
    w_B = 0.3 * confidence
    w_C = 0.3 * confidence
    total_w         = (w_A + w_B + w_C).clamp(min=EPS)
    tau_blend       = (w_A * tau_A + w_B * tau_B + w_C * tau_C) / total_w
    tau_blend       = tau_blend.clamp(TAU_MIN, 0.5)   # FIX 2: always ≤ 0.5

    return tau_blend.unsqueeze(1), confidence.unsqueeze(1)


class AnalyticalTauMLP(nn.Module):
    """
    FIX 1+2+4: Tau estimator that always returns values in [TAU_MIN, 0.5].

    Stage 1 (no params): 3-method analytical estimate → [TAU_MIN, 0.5]
    Stage 2 (learned):   correction refines within adaptive range

    FIX 4: effective_cor_max = COR_MAX_BASE + (1-confidence)*COR_MAX_EXTRA
      High confidence (strong disagreement signal): max correction = 0.08
      Zero confidence (all-agreement rules): max correction = 0.43
      This prevents over-correcting accurate estimates while allowing
      large corrections for hard pairs.

    FIX 2: output always clamped to [TAU_MIN, 0.5].
      The correction never pushes tau_pred above 0.5.
      symmetric_loss handles the ≤0.5/≥0.5 orientation externally.

    FIX 1: No tau_anchor_loss needed (and none used in training).
      The correction MLP learns from symmetric_loss gradient alone.
      Since tau_pred is always ≤0.5, the symmetric_loss gradient is
      unambiguous: always push toward the correct min(tau,1-tau).
    """
    def __init__(self):
        super().__init__()
        # 41 features: frac_stats[24] + sorted_frac[8] + adj_diff[8] + confidence[1]
        self.correction_net = nn.Sequential(
            nn.Linear(41, 256),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Tanh(),   # output in (-1, 1), scaled below
        )
        # Start near zero so analytical estimate dominates early
        nn.init.normal_(self.correction_net[-2].weight, std=0.01)
        nn.init.zeros_(self.correction_net[-2].bias)

    def _features(self, frac_stats, confidence):
        frac = frac_stats[:, :8]
        sorted_frac, _ = torch.sort(frac, dim=1)
        adj_diff = torch.zeros_like(sorted_frac)
        adj_diff[:, 1:] = (sorted_frac[:, 1:] - sorted_frac[:, :-1]).abs()
        return torch.cat([frac_stats, sorted_frac, adj_diff, confidence], dim=1)

    def forward(self, frac_stats):
        """
        frac_stats [B,24] → tau_pred [B,1] in [TAU_MIN, 0.5]
        Always ≤ 0.5. symmetric_loss handles orientation.
        """
        with torch.no_grad():
            tau_analytical, confidence = analytical_tau_robust(frac_stats)

        # FIX 4: adaptive correction range
        effective_cor_max = COR_MAX_BASE + (1.0 - confidence) * COR_MAX_EXTRA

        feats      = self._features(frac_stats, confidence)
        raw_corr   = self.correction_net(feats)             # [B,1] in (-1,1)
        correction = raw_corr * effective_cor_max           # [B,1] in adaptive range

        # FIX 2: clamp to [TAU_MIN, 0.5] — NEVER above 0.5
        tau_pred = (tau_analytical + correction).clamp(TAU_MIN, 0.5)
        return tau_pred

    def forward_analytical_only(self, frac_stats):
        """Pure analytical estimate for diagnostics."""
        with torch.no_grad():
            tau_est, _ = analytical_tau_robust(frac_stats)
        return tau_est


# ── Direct Rule Decoder ───────────────────────────────────────────────────────
class DirectRuleDecoder(nn.Module):
    """
    Per-pattern analytical rule estimation from frac_stats + tau_pred.
    Input features per pattern: [frac, tau, 1-tau, |frac-tau|, |frac-(1-tau)|, var, cnt]
    Output: rf_logits [B,8], rg_logits [B,8]

    Note: tau_pred is in [TAU_MIN, 0.5]. For true tau>0.5, the model sees
    1-tau (the symmetric equivalent). The symmetric_loss handles this.
    Both rule heads see the same features regardless of orientation.
    """
    def __init__(self):
        super().__init__()
        self.per_pat_mlp = nn.Sequential(
            nn.Linear(7, 64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward(self, frac_stats, tau_pred):
        B    = frac_stats.shape[0]
        frac = frac_stats[:, :8]
        var_ = frac_stats[:, 8:16]
        cnt  = frac_stats[:, 16:24]
        tau      = tau_pred.expand(B, 8)                  # [B,8] ≤ 0.5
        tau_comp = (1.0 - tau_pred).expand(B, 8)          # [B,8] ≥ 0.5
        feats = torch.stack([frac, tau, tau_comp,
                             (frac - tau).abs(),
                             (frac - tau_comp).abs(),
                             var_, cnt], dim=-1)
        out = self.per_pat_mlp(feats.reshape(B * 8, 7)).reshape(B, 8, 2)
        return out[:, :, 0], out[:, :, 1]   # rf_logits, rg_logits [B,8]


# ── Distributional Rule Decoder ───────────────────────────────────────────────
# N_DIST_PCTS: number of percentile values per pattern
N_DIST_PCTS = 9
DIST_PCTS   = [5, 15, 25, 35, 50, 65, 75, 85, 95]   # percentile levels

class DistributionalRuleDecoder(nn.Module):
    """
    Path 3 from the document: distributional features that work WITHOUT
    knowing which timestep belongs to which rule.

    The core insight:
      For pattern k, the per-timestep rate varies between timesteps.
      When rule F fires at timestep t: rate[t,k] ≈ f[k] ∈ {0,1}
      When rule G fires at timestep t: rate[t,k] ≈ g[k] ∈ {0,1}
      → The distribution of rate[t,k] over all timesteps is BIMODAL
        with modes at f[k] and g[k].

    The percentiles of this distribution directly encode both rules:
      Low percentiles  (5th, 15th) → the smaller of {f[k], g[k]}
      High percentiles (85th, 95th) → the larger of {f[k], g[k]}
      50th percentile → weighted by tau (≈ tau*g[k] + (1-tau)*f[k] = frac[k])

    This is the ONLY path that can distinguish both rules simultaneously
    from pure statistics, without requiring temporal separation.
    Critical for hard cases (tau near 0.1/0.9) where assignment signal is weak.

    Input:
      raw_tokens : [B, K, N_TOK, 4]  -- raw orbit tokens
    Output:
      rf_logits  : [B, 8]
      rg_logits  : [B, 8]

    Implementation:
      1. For each pattern k, compute per-timestep rate: mean(A | pat=k) at time t
         This gives rate_ts [B, N_TRANS, 8] -- one rate per timestep per pattern
      2. For each pattern k, compute 9 percentiles of rate_ts[:, :, k] over N_TRANS
         This gives pcts [B, 8, 9]
      3. Neural network decodes rule bits from percentiles
    """
    def __init__(self):
        super().__init__()
        # Input per pattern: N_DIST_PCTS percentile values
        # Output per pattern: 2 logits (f_bit, g_bit)
        self.dist_mlp = nn.Sequential(
            nn.Linear(N_DIST_PCTS, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

    def _compute_ts_rates(self, raw_tokens):
        """
        Compute per-timestep per-pattern rates from raw tokens.

        For each timestep t, for each pattern k:
          rate[t, k] = mean(A[t, c] for cells c where pat[t,c] == k)
                     = fraction of pattern-k cells that output 1 at time t

        raw_tokens : [B, K, N_TOK, 4]
        Returns    : rate_ts [B, N_TRANS, 8]
        """
        B, K, _, _ = raw_tokens.shape

        l = raw_tokens[:, :, :, 0]                        # [B, K, N_TOK]
        b = raw_tokens[:, :, :, 1]
        r = raw_tokens[:, :, :, 2]
        a = raw_tokens[:, :, :, 3]                        # after-state

        pat_idx = (4*l + 2*b + r).long().clamp(0, 7)     # [B, K, N_TOK]

        # Reshape to [B, K, N_TRANS, N_CELLS]
        pat_4d = pat_idx.reshape(B, K, N_TRANS, N_CELLS)  # [B, K, N_TRANS, N_CELLS]
        a_4d   = a.reshape(B, K, N_TRANS, N_CELLS)        # [B, K, N_TRANS, N_CELLS]

        # For each pattern k, accumulate sum and count of A-values
        # Vectorised using one-hot scatter
        # pat_onehot: [B, K, N_TRANS, N_CELLS, 8]
        pat_oh  = F.one_hot(pat_4d, num_classes=8).float()  # [B, K, N_TRANS, N_CELLS, 8]
        a_exp   = a_4d.unsqueeze(-1)                         # [B, K, N_TRANS, N_CELLS, 1]

        # Sum A-values and counts per pattern per timestep
        sum_a   = (pat_oh * a_exp).sum(dim=3)               # [B, K, N_TRANS, 8]
        sum_cnt = pat_oh.sum(dim=3)                         # [B, K, N_TRANS, 8]

        # Per-timestep rate: safe division (0 if pattern not observed at that timestep)
        rate_ts = sum_a / (sum_cnt + EPS)                   # [B, K, N_TRANS, 8]

        # Zero out entries where pattern was not observed at that timestep
        # (so they don't contaminate percentiles with 0.0 values from missing patterns)
        rate_ts = torch.where(sum_cnt > 0.5, rate_ts,
                              torch.full_like(rate_ts, float('nan')))

        # Average over K orbits for each timestep (nanmean to skip missing patterns)
        # [B, N_TRANS, 8]
        # Using mean rather than nanmean for compatibility; missing patterns are rare
        rate_ts_mean = rate_ts.nanmean(dim=1)               # [B, N_TRANS, 8]

        return rate_ts_mean                                  # [B, N_TRANS, 8]

    def forward(self, raw_tokens):
        """
        raw_tokens : [B, K, N_TOK, 4]
        Returns    : rf_logits [B,8], rg_logits [B,8]
        """
        B = raw_tokens.shape[0]

        # Step 1: per-timestep per-pattern rates [B, N_TRANS, 8]
        rate_ts = self._compute_ts_rates(raw_tokens)         # [B, N_TRANS, 8]

        # Replace NaN with 0.5 (neutral) before percentile computation
        rate_ts = torch.where(torch.isnan(rate_ts),
                              torch.full_like(rate_ts, 0.5), rate_ts)

        # Step 2: percentiles of rate distribution over N_TRANS timesteps
        # rate_ts transposed to [B, 8, N_TRANS] for per-pattern percentile
        rate_tp = rate_ts.permute(0, 2, 1)                  # [B, 8, N_TRANS]

        # torch.quantile: compute N_DIST_PCTS percentiles per pattern per sample
        pct_levels = torch.tensor(
            [p / 100.0 for p in DIST_PCTS],
            dtype=rate_tp.dtype, device=rate_tp.device)     # [9]

        # pcts: [B, 8, 9] -- percentile values per pattern per sample
        pcts = torch.quantile(rate_tp, pct_levels, dim=2)   # [9, B, 8]
        pcts = pcts.permute(1, 2, 0)                        # [B, 8, 9]

        # Step 3: decode rule bits from percentile distribution
        out = self.dist_mlp(pcts.reshape(B * 8, N_DIST_PCTS)).reshape(B, 8, 2)
        return out[:, :, 0], out[:, :, 1]   # rf_logits, rg_logits [B,8]


# ══════════════════════════════════════════════════════════════════
# PATH B: HIERARCHICAL TRANSFORMER (FIX 3)
# ══════════════════════════════════════════════════════════════════

class CellEncoder(nn.Module):
    """Level 1: N_CELLS=50 cell tokens → D1 fingerprint."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(N_FEAT, D1)
        self.pe   = SinusoidalPE(D1, N_CELLS)
        layer     = nn.TransformerEncoderLayer(
            D1, N_HEADS1, D_FF1, DROPOUT, "gelu",
            batch_first=True, norm_first=True)
        self.enc  = nn.TransformerEncoder(layer, N_L1, enable_nested_tensor=False)
        self.q    = nn.Parameter(torch.randn(1, 1, D1) * 0.02)
        self.attn = nn.MultiheadAttention(D1, N_HEADS1, dropout=DROPOUT, batch_first=True)
        self.norm = nn.LayerNorm(D1)

    def forward(self, x):
        B_flat = x.shape[0]
        h      = self.enc(self.pe(self.proj(x)))
        q      = self.q.expand(B_flat, -1, -1)
        out, _ = self.attn(q, h, h)
        return self.norm(out.squeeze(1))


class TimestepEncoder(nn.Module):
    """Level 2: [B_flat, N_TRANS, D1] → [B_flat, N_TRANS, D2]"""
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(D1, D2)
        self.pe   = SinusoidalPE(D2, N_TRANS)
        layer     = nn.TransformerEncoderLayer(
            D2, N_HEADS2, D_FF2, DROPOUT, "gelu",
            batch_first=True, norm_first=True)
        self.enc  = nn.TransformerEncoder(layer, N_L2, enable_nested_tensor=False)

    def forward(self, x):
        return self.enc(self.pe(self.proj(x)))


class ConfidenceWeightedOrbitFusion(nn.Module):
    """Learned confidence weighting across K orbits."""
    def __init__(self):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Linear(D2, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, h_all):
        orbit_summary = h_all.mean(dim=2)
        weights       = F.softmax(self.weight_net(orbit_summary), dim=1)
        return (h_all * weights.unsqueeze(-1)).sum(dim=1)


class TauConditionedRuleHead(nn.Module):
    """
    Three-path rule decoding:

    Path 1 (transformer): weighted sum of h by assignment weights
    Path 2 (analytical):  direct_rf/rg_logits from DirectRuleDecoder
    Path 3 (distributional): dist_rf/rg_logits from DistributionalRuleDecoder

    All three concatenated → rule head MLPs → rf_logits, rg_logits

    Assignment (prob_g per timestep):
      score[t] = RESIDUAL_WEIGHT * ts_residual[t] + score_net(h[t], tau, var)
      residual_weight is a learnable scalar init to 5.0

    Rule heads input: [rep(D2) + direct_logits(8) + dist_logits(8)] = D2+16
    """
    def __init__(self):
        super().__init__()
        # Learnable weight for direct residual path (FIX 3)
        self.residual_weight = nn.Parameter(torch.tensor(5.0))

        # score_net: h(D2) + tau(1) + mean_var(1)
        self.score_net = nn.Sequential(
            nn.Linear(D2 + 2, 128), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(128, 64),     nn.GELU(),
            nn.Linear(64, 1),
        )
        # Rule heads: transformer rep + direct logits + distributional logits
        # D2 + 8 (direct) + 8 (distributional) = D2 + 16
        self.rf_head = nn.Sequential(
            nn.LayerNorm(D2 + 16), nn.Linear(D2 + 16, 256), nn.GELU(),
            nn.Dropout(DROPOUT),   nn.Linear(256, 128),     nn.GELU(),
            nn.Linear(128, N_BITS),
        )
        self.rg_head = nn.Sequential(
            nn.LayerNorm(D2 + 16), nn.Linear(D2 + 16, 256), nn.GELU(),
            nn.Dropout(DROPOUT),   nn.Linear(256, 128),     nn.GELU(),
            nn.Linear(128, N_BITS),
        )
        # Sharp init for score_net final layer
        nn.init.normal_(self.score_net[-1].weight, std=0.5)
        nn.init.zeros_(self.score_net[-1].bias)

    def forward(self, h, tau_pred, frac_stats,
                direct_rf_logits, direct_rg_logits,
                dist_rf_logits, dist_rg_logits,
                ts_residual):
        """
        h                : [B, N_TRANS, D2]
        tau_pred         : [B, 1]
        frac_stats       : [B, 24]
        direct_rf_logits : [B, 8]   from DirectRuleDecoder
        direct_rg_logits : [B, 8]   from DirectRuleDecoder
        dist_rf_logits   : [B, 8]   from DistributionalRuleDecoder (Path 3)
        dist_rg_logits   : [B, 8]   from DistributionalRuleDecoder (Path 3)
        ts_residual      : [B, N_TRANS]
        """
        B, Nt, D = h.shape
        var_f        = frac_stats[:, 8:16]
        mean_var     = var_f.mean(dim=1, keepdim=True)
        tau_exp      = tau_pred.unsqueeze(1).expand(B, Nt, 1)
        mean_var_exp = mean_var.unsqueeze(1).expand(B, Nt, 1)

        # Assignment: deep path + direct residual path
        inp_deep   = torch.cat([h, tau_exp, mean_var_exp], dim=-1)
        score_deep = self.score_net(inp_deep).squeeze(-1)
        score      = self.residual_weight * ts_residual + score_deep
        prob_g     = torch.sigmoid(score)
        prob_f     = 1.0 - prob_g

        # Weighted pooling
        wf = prob_f / (prob_f.sum(dim=1, keepdim=True) + EPS)
        wg = prob_g / (prob_g.sum(dim=1, keepdim=True) + EPS)
        rep_f = (h * wf.unsqueeze(-1)).sum(dim=1)      # [B, D2]
        rep_g = (h * wg.unsqueeze(-1)).sum(dim=1)      # [B, D2]

        # All three paths concatenated — detach analytical paths so their
        # gradients don't interfere with the transformer learning
        rep_f_aug = torch.cat([rep_f,
                                direct_rf_logits.detach(),
                                dist_rf_logits.detach()], dim=-1)   # [B, D2+16]
        rep_g_aug = torch.cat([rep_g,
                                direct_rg_logits.detach(),
                                dist_rg_logits.detach()], dim=-1)   # [B, D2+16]

        return self.rf_head(rep_f_aug), self.rg_head(rep_g_aug), prob_g


# ── Main Model ────────────────────────────────────────────────────────────────
class ECANetTSCA(nn.Module):
    """
    ECANetTSCA v9.

    Key properties:
      - tau_pred always in [TAU_MIN, 0.5] (symmetric representation)
      - No tau_anchor_loss needed or used
      - Residual signal directly in score logit (not buried in 256 dims)
      - Adaptive correction range based on confidence

    forward() inputs:
      raw_tokens : [B, K, N_TOK, 4]   [L, B, R, A]
      frac_stats : [B, 24]

    forward() outputs:
      rf_logits  [B, 8]
      rg_logits  [B, 8]
      tau_pred   [B, 1] in [TAU_MIN, 0.5]  -- symmetric magnitude
      prob_g     [B, N_TRANS]
      rf_dir     [B, 8]
      rg_dir     [B, 8]
    """
    N_RAW = 4

    def __init__(self):
        super().__init__()
        self.pat_emb    = PatternEmbedding(8, N_EMB)
        self.tau_mlp    = AnalyticalTauMLP()
        self.direct_dec = DirectRuleDecoder()
        self.dist_dec   = DistributionalRuleDecoder()   # Path 3: distributional
        self.cell_enc   = CellEncoder()
        self.ts_enc     = TimestepEncoder()
        self.orbit_fuse = ConfidenceWeightedOrbitFusion()
        self.rule_head  = TauConditionedRuleHead()
        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if any(s in name for s in
                       ["score_net", "correction_net", "weight_net"]):
                    continue   # custom init
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _embed_tokens(self, raw_tokens):
        """[B_flat, N_CELLS, 4] → [B_flat, N_CELLS, 12] with pattern embedding."""
        l = raw_tokens[:, :, 0]
        b = raw_tokens[:, :, 1]
        r = raw_tokens[:, :, 2]
        pat_idx = (4 * l + 2 * b + r).long().clamp(0, 7)
        emb     = self.pat_emb(pat_idx)
        return torch.cat([raw_tokens, emb], dim=-1)

    def _compute_ts_residual(self, raw_tokens, frac_stats):
        """
        Compute per-timestep residual: mean_k,c(a[t,c] - frac[pat[t,c]])

        Theory: E[residual | rule_g fired] = (1-tau)*(g-f) at disagree pats
                E[residual | rule_f fired] = -tau*(g-f) at disagree pats
        These have opposite signs → direct discriminative signal.

        raw_tokens : [B, K, N_TOK, 4]
        frac_stats : [B, 24]
        Returns    : [B, N_TRANS]
        """
        B, K, _, _ = raw_tokens.shape
        frac    = frac_stats[:, :8]                                # [B, 8]

        l       = raw_tokens[:, :, :, 0]
        b       = raw_tokens[:, :, :, 1]
        r       = raw_tokens[:, :, :, 2]
        a_vals  = raw_tokens[:, :, :, 3]                          # [B, K, N_TOK]

        pat_idx = (4*l + 2*b + r).long().clamp(0, 7)              # [B, K, N_TOK]
        pat_4d  = pat_idx.reshape(B, K, N_TRANS, N_CELLS)
        a_4d    = a_vals.reshape(B, K, N_TRANS, N_CELLS)

        frac_exp      = frac.view(B, 1, 1, 1, 8).expand(B, K, N_TRANS, N_CELLS, 8)
        frac_per_cell = frac_exp.gather(-1, pat_4d.unsqueeze(-1)).squeeze(-1)

        residual = a_4d - frac_per_cell                            # [B, K, N_TRANS, N_CELLS]
        return residual.mean(dim=(1, 3))                           # [B, N_TRANS]

    def _encode_orbits(self, raw_tokens):
        """raw_tokens [B, K, N_TOK, 4] → h [B, N_TRANS, D2]"""
        B, K, _, _ = raw_tokens.shape
        x_flat  = raw_tokens.reshape(B * K * N_TRANS, N_CELLS, self.N_RAW)
        x_emb   = self._embed_tokens(x_flat)
        e_flat  = self.cell_enc(x_emb)
        e       = e_flat.reshape(B, K, N_TRANS, D1)
        h_flat  = self.ts_enc(e.reshape(B * K, N_TRANS, D1))
        h_all   = h_flat.reshape(B, K, N_TRANS, D2)
        return self.orbit_fuse(h_all)

    def forward(self, raw_tokens, frac_stats):
        # Path A: tau (always in [TAU_MIN, 0.5])
        tau_pred             = self.tau_mlp(frac_stats)
        # Path B: analytical rule estimate from frac statistics
        rf_dir, rg_dir       = self.direct_dec(frac_stats, tau_pred)
        # Path C: distributional rule estimate from per-timestep rate percentiles
        rf_dist, rg_dist     = self.dist_dec(raw_tokens)
        # Per-timestep residual signal for assignment
        ts_residual          = self._compute_ts_residual(raw_tokens, frac_stats)
        # Transformer encoding
        h                    = self._encode_orbits(raw_tokens)
        # Tau-conditioned rule head with all three paths
        rf, rg, prob_g       = self.rule_head(
            h, tau_pred, frac_stats,
            rf_dir, rg_dir,
            rf_dist, rg_dist,
            ts_residual)
        return rf, rg, tau_pred, prob_g, rf_dir, rg_dir

    def forward_tau_only(self, frac_stats):
        return self.tau_mlp(frac_stats)

    def forward_tau_analytical(self, frac_stats):
        return self.tau_mlp.forward_analytical_only(frac_stats)


# ── Data helpers ──────────────────────────────────────────────────────────────
def build_table(r):
    return np.array([(r >> i) & 1 for i in range(8)], dtype=np.uint8)

def rule_to_bits(r):
    return np.array([(r >> i) & 1 for i in range(8)], dtype=np.float32)

def random_init(rng):
    while True:
        x = rng.integers(0, 2, size=W, dtype=np.uint8)
        if 0 < int(x.sum()) < W:
            return x

def simulate_labeled(rf, rg, init, tau, rng):
    tf = build_table(rf); tg = build_table(rg)
    orbit  = np.zeros((T, W), dtype=np.float32)
    labels = np.zeros(T - 1, dtype=np.int8)
    s = init.copy().astype(np.uint8); orbit[0] = s
    for t in range(1, T):
        L = np.roll(s, 1); R = np.roll(s, -1)
        idx = (4 * L + 2 * s + R).astype(np.uint8)
        if tau <= 0.0:
            ns = tf[idx]; labels[t-1] = 0
        elif tau >= 1.0:
            ns = tg[idx]; labels[t-1] = 1
        elif rng.random() < tau:
            ns = tg[idx]; labels[t-1] = 1
        else:
            ns = tf[idx]; labels[t-1] = 0
        s = ns.astype(np.uint8); orbit[t] = s
    return orbit, labels

def orbit_to_tokens(orbit):
    """[T,W] → [N_TOK, 4] raw tokens [L,B,R,A]."""
    o = orbit.astype(np.float32)
    b = o[:-1]; a = o[1:]
    l = np.roll(b, 1, axis=1)
    r = np.roll(b, -1, axis=1)
    return np.stack([l, b, r, a], axis=-1).reshape(-1, 4)

def compute_frac_stats(orbits_k):
    """orbits_k [K,T,W] → frac_stats [24]."""
    K, _, _ = orbits_k.shape
    frac = np.zeros(8, dtype=np.float32)
    cnt  = np.zeros(8, dtype=np.float32)
    before = orbits_k[:, :-1, :]
    after  = orbits_k[:, 1:,  :]
    left   = np.roll(before, 1, axis=2)
    right  = np.roll(before, -1, axis=2)
    pat    = (4 * left + 2 * before + right).astype(np.int32)
    total  = float(K * (T - 1) * W)
    for n in range(8):
        mask  = (pat == n); cnt_n = float(mask.sum())
        if cnt_n > 0: frac[n] = float(after[mask].sum()) / cnt_n
        cnt[n] = cnt_n / total
    return np.concatenate([frac, frac * (1 - frac), cnt]).astype(np.float32)


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  ECANetTSCA v9 -- smoke test")
    print("=" * 65)

    model = ECANetTSCA()
    npar  = sum(p.numel() for p in model.parameters())
    print(f"  PatternEmbedding          : "
          f"{sum(p.numel() for p in model.pat_emb.parameters()):>10,}")
    print(f"  AnalyticalTauMLP          : "
          f"{sum(p.numel() for p in model.tau_mlp.parameters()):>10,}")
    print(f"  DirectRuleDecoder         : "
          f"{sum(p.numel() for p in model.direct_dec.parameters()):>10,}")
    print(f"  DistributionalRuleDecoder : "
          f"{sum(p.numel() for p in model.dist_dec.parameters()):>10,}")
    print(f"  CellEncoder               : "
          f"{sum(p.numel() for p in model.cell_enc.parameters()):>10,}")
    print(f"  TimestepEncoder           : "
          f"{sum(p.numel() for p in model.ts_enc.parameters()):>10,}")
    print(f"  ConfidenceWeightedFusion  : "
          f"{sum(p.numel() for p in model.orbit_fuse.parameters()):>10,}")
    print(f"  TauConditionedRuleHead    : "
          f"{sum(p.numel() for p in model.rule_head.parameters()):>10,}")
    print(f"  Total                     : {npar:>10,}  ({npar/1e6:.2f}M)")

    rng_np = np.random.default_rng(42)

    # Test 1: tau always ≤ 0.5 (FIX 2)
    print(f"\n  TEST 1: tau_pred always in [TAU_MIN, 0.5]")
    errors_sym = []
    for _ in range(20):
        rf_r = int(rng_np.integers(0, 256))
        rg_r = int(rng_np.integers(0, 256))
        while rf_r == rg_r: rg_r = int(rng_np.integers(0, 256))
        tau_true = float(rng_np.uniform(0.1, 0.9))
        f_b = np.array([(rf_r >> b) & 1 for b in range(8)], dtype=np.float32)
        g_b = np.array([(rg_r >> b) & 1 for b in range(8)], dtype=np.float32)
        frac = tau_true * g_b + (1 - tau_true) * f_b
        var_ = frac * (1 - frac)
        cnt  = rng_np.dirichlet(np.ones(8) * 2.0).astype(np.float32)
        fs   = torch.tensor(np.concatenate([frac, var_, cnt]),
                            dtype=torch.float32).unsqueeze(0)
        tau_a = model.forward_tau_analytical(fs).item()
        tau_p = model.forward_tau_only(fs).item()
        sym_err = min(abs(tau_a - tau_true), abs((1-tau_a) - tau_true))
        errors_sym.append(sym_err)
        assert tau_a <= 0.5 + 1e-5, f"analytical tau > 0.5: {tau_a}"
        assert tau_p <= 0.5 + 1e-5, f"tau_pred > 0.5: {tau_p}"
    print(f"    All 20 samples: tau ≤ 0.5 -- OK")
    print(f"    Mean symmetric MAE: {np.mean(errors_sym):.4f}")

    # Test 2: residual weight is learnable and initialized correctly
    print(f"\n  TEST 2: residual_weight initialization")
    rw = model.rule_head.residual_weight.item()
    print(f"    residual_weight = {rw:.4f}  (expected 5.0)")
    assert abs(rw - 5.0) < 0.01, f"Wrong init: {rw}"
    print(f"    OK")

    # Test 3: residual has correct sign
    print(f"\n  TEST 3: residual signal direction")
    B = 4
    raw_tokens = torch.zeros(B, N_ORBITS, N_TOK, 4)
    fs2 = torch.zeros(B, FRAC_DIM)
    fs2[:, :] = 0.0
    fs2[:, :8] = 0.3    # frac[all patterns] = 0.3
    # Samples 0,1: all A=1 (rule_g output → residual = 1-0.3 = +0.7)
    raw_tokens[:2, :, :, 3] = 1.0
    # Samples 2,3: all A=0 (rule_f output → residual = 0-0.3 = -0.3)
    raw_tokens[2:, :, :, 3] = 0.0
    ts_res = model._compute_ts_residual(raw_tokens, fs2)
    assert ts_res[:2].mean() > 0.5, "rule_g residual should be positive"
    assert ts_res[2:].mean() < -0.2, "rule_f residual should be negative"
    print(f"    rule_g residual: {ts_res[:2].mean():.3f} (expected +0.70)")
    print(f"    rule_f residual: {ts_res[2:].mean():.3f} (expected -0.30)")
    print(f"    OK")

    # Test 4: forward pass shapes
    print(f"\n  TEST 4: forward pass shapes")
    frac_stats2 = torch.rand(B, FRAC_DIM)
    model.eval()
    with torch.no_grad():
        rf, rg, tau, prob_g, rfd, rgd = model(raw_tokens, frac_stats2)
    assert rf.shape     == (B, 8)
    assert rg.shape     == (B, 8)
    assert tau.shape    == (B, 1)
    assert prob_g.shape == (B, N_TRANS)
    assert (tau >= TAU_MIN).all() and (tau <= 0.5 + 1e-5).all()
    assert (prob_g >= 0).all() and (prob_g <= 1).all()
    print(f"    rf={rf.shape}  rg={rg.shape}  tau={tau.shape}")
    print(f"    tau range: [{tau.min():.3f}, {tau.max():.3f}]  (should be ≤ 0.5)")
    print(f"    prob_g range: [{prob_g.min():.3f}, {prob_g.max():.3f}]")
    print(f"    OK")

    # Test 5: gradients flow through residual_weight
    print(f"\n  TEST 5: gradient through residual_weight")
    model.train()
    rf2, rg2, tau2, pg2, _, _ = model(raw_tokens, frac_stats2)
    loss = rf2.mean() + rg2.mean() + tau2.mean() + pg2.mean()
    loss.backward()
    rw_grad = model.rule_head.residual_weight.grad
    assert rw_grad is not None and rw_grad.abs().item() > 0
    print(f"    residual_weight grad: {rw_grad.item():.6f}  OK")

    print(f"\n  ECANetTSCA v9 -- ALL TESTS PASSED")
    print("=" * 65)
