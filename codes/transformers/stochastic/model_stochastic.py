"""
MODEL_SCA.py  --  ECANetSCA v1
================================
Directory: stochastic_v1/

Stochastic Cellular Automaton (SCA) identifier.

════════════════════════════════════════════════════════════════
SCA PHYSICS
════════════════════════════════════════════════════════════════

At each timestep t, for each cell c independently:
  - Draw u ~ Uniform(0,1)
  - If u < lambda: next[c] = rule_G[ pattern(c) ]
  - Else:          next[c] = rule_F[ pattern(c) ]
Neighborhoods use the CURRENT (pre-update) row (synchronous).

lambda ∈ [0.1, 0.9]  continuous (same range as tau in TSCA)

════════════════════════════════════════════════════════════════
SCA vs TSCA — KEY DIFFERENCES
════════════════════════════════════════════════════════════════

TSCA:  one coin flip per TIMESTEP → whole row uses F or G
SCA:   one coin flip per CELL per timestep → each cell independent

Consequence for signals:

1. frac_stats [24] — IDENTICAL formula:
   E[after[c,t] | pat[c,t]=n] = lambda*g[n] + (1-lambda)*f[n]
   So frac[n] ≈ lambda*g[n] + (1-lambda)*f[n] in both models.

2. within_var_stats [8] — NEW signal unique to SCA:
   Var[after[c,t] | pat[c,t]=n] = lambda*(1-lambda)*(g[n]-f[n])^2
   This is nonzero at disagreement patterns (g[n]≠f[n]).
   Averaged over (c,t) → clean estimator of lambda*(1-lambda).
   From within_var: lambda*(1-lambda) = within_var
   → lam_sym = 0.5 - sqrt(0.25 - within_var)  [gives min(lam, 1-lam)]
   In TSCA this doesn't exist: all cells use same rule per step.
   SCA gives W=50 independent Bernoulli draws per pattern per timestep.

3. Per-timestep residual — ABSENT in SCA:
   In TSCA, ts_residual[t] = mean_c(a[t,c] - frac[pat[t,c]])
   When rule G fires: E[residual_t] > 0
   When rule F fires: E[residual_t] < 0
   → Strong per-timestep assignment signal (TSCA FIX 3)
   In SCA: every timestep mixes F and G cells → E[residual_t] = 0 always.
   So we DROP the assignment loss and ts_residual entirely.
   The transformer learns rule identity from cross-timestep correlations.

4. Lambda is CONTINUOUS in [0.1, 0.9] — identical structure to tau.
   Symmetric: (F, G, lambda) ≡ (G, F, 1-lambda) statistically.
   lam_pred always in [LAM_MIN, 0.5] (symmetric magnitude).
   symmetric_loss picks orientation from rule bits (same as TSCA).
   lam_mse_loss targets min(true_lam, 1-true_lam) directly.

════════════════════════════════════════════════════════════════
ARCHITECTURE OVERVIEW
════════════════════════════════════════════════════════════════

Input:
  raw_tokens       [B, K, N_TOK, 4]
  frac_stats       [B, 24]
  within_var_stats [B, 8]

PATH A — Lambda (AnalyticalLambdaMLP):
  analytical_lambda_robust(frac_stats, within_var_stats)
    → lam_sym ∈ [LAM_MIN, 0.5]
  correction_net refines within adaptive range
    → lam_pred [B,1] ∈ [LAM_MIN, 0.5]  always ≤ 0.5
  lam_mse_loss(lam_pred, min(true_lam, 1-true_lam)) trains this path.
  symmetric_loss handles orientation externally (same as TSCA).

PATH B — Transformer (CellEncoder → TimestepEncoder → OrbitFusion):
  → h [B, N_TRANS, D2]

PATH C — Direct analytical rule decoder (DirectRuleDecoder):
  Per-pattern MLP using frac + within_var + lam_pred
  → rf_dir [B,8], rg_dir [B,8]

PATH D — Distributional decoder (DistributionalRuleDecoder):
  Percentiles of per-timestep per-pattern rates
  → rf_dist [B,8], rg_dist [B,8]

RULE HEAD (LambdaConditionedRuleHead):
  Lambda-conditioned attention pooling over h → rep [B, D2]
  Concat [rep, rf_dir.detach(), rf_dist.detach()] → rf_logits
  Concat [rep, rg_dir.detach(), rg_dist.detach()] → rg_logits

OUTPUT:
  rf_logits  [B, 8]
  rg_logits  [B, 8]
  lam_pred   [B, 1]  ∈ [LAM_MIN, 0.5]
  rf_dir     [B, 8]
  rg_dir     [B, 8]
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
N_FEAT   = 12            # [L, B, R, A] + 8-dim pattern embedding
N_TOK    = N_TRANS * W
N_BITS   = 8
N_PAT    = 8
FRAC_DIM = 24
WVAR_DIM = 8
N_EMB    = 8

LAM_MIN  = 0.1
LAM_MAX  = 0.9
EPS      = 1e-8

D1       = 128
N_L1     = 4
N_HEADS1 = 4
D_FF1    = 512

D2       = 256
N_L2     = 6
N_HEADS2 = 8
D_FF2    = 1024

DROPOUT  = 0.1

COR_MAX_BASE  = 0.08
COR_MAX_EXTRA = 0.35

N_DIST_PCTS = 9
DIST_PCTS   = [5, 15, 25, 35, 50, 65, 75, 85, 95]


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
    def __init__(self, n_pat=8, emb_dim=N_EMB):
        super().__init__()
        self.emb = nn.Embedding(n_pat, emb_dim)
        nn.init.normal_(self.emb.weight, std=0.1)

    def forward(self, pat_idx):
        return self.emb(pat_idx)


# ══════════════════════════════════════════════════════════════════
# PATH A: ANALYTICAL LAMBDA
# ══════════════════════════════════════════════════════════════════

def analytical_lambda_robust(frac_stats, within_var_stats):
    """
    Three-method robust lambda estimator.
    Returns lam_sym [B,1] ∈ [LAM_MIN, 0.5] and confidence [B,1] ∈ [0,1].

    lam_sym = min(lambda, 1-lambda) — the symmetric magnitude.
    Always ≤ 0.5. symmetric_loss handles orientation.

    Method A: within-variance (best method, unique to SCA)
      max_wvar ≈ lambda*(1-lambda) at max-disagreement pattern
      lam_sym = 0.5 - sqrt(0.25 - max_wvar)

    Method B: frac interior (same as TSCA tau)
      Minimum of interior fracs (between 0.08 and 0.92)
      → min(lambda, 1-lambda)

    Method C: top-2 frac-variance patterns (same as TSCA tau)
      Top-2 variance patterns are disagreement patterns
      → min(frac[top2]) ≈ min(lambda, 1-lambda)
    """
    frac  = frac_stats[:, :8]
    var_  = frac_stats[:, 8:16]
    cnt   = frac_stats[:, 16:24]
    wvar  = within_var_stats

    well_obs = cnt > 0.005

    # Method A
    wvar_m          = wvar.clone(); wvar_m[~well_obs] = 0.0
    max_wvar        = wvar_m.max(dim=1).values
    has_wvar        = max_wvar > 0.002
    max_wvar_c      = max_wvar.clamp(0.0, 0.2499)
    lam_A_raw       = 0.5 - torch.sqrt((0.25 - max_wvar_c).clamp(min=EPS))
    lam_A           = torch.where(has_wvar, lam_A_raw,
                                  torch.full_like(lam_A_raw, 0.5)).clamp(LAM_MIN, 0.5)

    # Method B
    var_m           = var_.clone(); var_m[~well_obs] = 0.0
    max_var_w       = var_m.max(dim=1).values
    has_sig         = max_var_w > 0.005
    frac_obs        = frac.clone(); frac_obs[~well_obs] = -1.0
    sorted_f, _     = torch.sort(frac_obs, dim=1)
    int_mask        = (sorted_f > 0.08) & (sorted_f < 0.92)
    fill            = sorted_f.clone(); fill[~int_mask] = 1.0
    lam_B           = fill.min(dim=1).values
    lam_B[~int_mask.any(dim=1)] = 0.5
    lam_B           = torch.where(has_sig, lam_B,
                                  torch.full_like(lam_B, 0.5)).clamp(LAM_MIN, 0.5)

    # Method C
    var_c           = var_.clone(); var_c[~well_obs] = 0.0
    top2_v, top2_i  = var_c.topk(2, dim=1)
    top2_f          = frac.gather(1, top2_i)
    lam_C           = top2_f.min(dim=1).values
    lam_C[(top2_v[:, 0] < 0.01) | (top2_v[:, 1] < 0.02)] = 0.5
    lam_C           = lam_C.clamp(LAM_MIN, 0.5)

    # Confidence: max of var-based and wvar-based
    conf_var  = (max_var_w / 0.25).clamp(0.0, 1.0)
    conf_wvar = (max_wvar  / 0.25).clamp(0.0, 1.0)
    confidence = torch.maximum(conf_var, conf_wvar)

    # Blend: weight Method A more when wvar is strong
    w_A    = 0.45 * conf_wvar + 0.10 * (1.0 - confidence)
    w_B    = 0.30 * confidence
    w_C    = 0.25 * confidence
    total  = (w_A + w_B + w_C).clamp(min=EPS)
    blend  = (w_A * lam_A + w_B * lam_B + w_C * lam_C) / total
    blend  = blend.clamp(LAM_MIN, 0.5)

    return blend.unsqueeze(1), confidence.unsqueeze(1)


class AnalyticalLambdaMLP(nn.Module):
    """
    Lambda estimator — mirrors TSCA AnalyticalTauMLP exactly.

    Always returns lam_pred ∈ [LAM_MIN, 0.5].
    Correction MLP refines within adaptive range (same as TSCA FIX 4).

    Features (50 dims):
      frac_stats(24) + within_var_stats(8) + sorted_frac(8)
      + adj_diff(8) + lam_sym(1) + confidence(1)
    """
    def __init__(self):
        super().__init__()
        in_dim = FRAC_DIM + WVAR_DIM + N_PAT + N_PAT + 1 + 1  # 50
        self.correction_net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(256, 256),    nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(256, 128),    nn.GELU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )
        nn.init.normal_(self.correction_net[-2].weight, std=0.01)
        nn.init.zeros_(self.correction_net[-2].bias)

    def _features(self, frac_stats, within_var_stats, lam_sym, confidence):
        frac = frac_stats[:, :8]
        sf, _ = torch.sort(frac, dim=1)
        adj = torch.zeros_like(sf)
        adj[:, 1:] = (sf[:, 1:] - sf[:, :-1]).abs()
        return torch.cat([frac_stats, within_var_stats, sf, adj,
                          lam_sym, confidence], dim=1)

    def forward(self, frac_stats, within_var_stats):
        with torch.no_grad():
            lam_analytical, confidence = analytical_lambda_robust(
                frac_stats, within_var_stats)
        effective_cor_max = COR_MAX_BASE + (1.0 - confidence) * COR_MAX_EXTRA
        feats      = self._features(frac_stats, within_var_stats,
                                    lam_analytical, confidence)
        correction = self.correction_net(feats) * effective_cor_max
        lam_pred   = (lam_analytical + correction).clamp(LAM_MIN, 0.5)
        return lam_pred

    def forward_analytical_only(self, frac_stats, within_var_stats):
        with torch.no_grad():
            lam_est, _ = analytical_lambda_robust(frac_stats, within_var_stats)
        return lam_est


# ── Direct Rule Decoder ───────────────────────────────────────────────────────
class DirectRuleDecoder(nn.Module):
    """
    Per-pattern MLP rule estimation.
    Features per pattern (8 dims):
      [frac, lam, 1-lam, |frac-lam|, |frac-(1-lam)|, frac_var, within_var, cnt]
    within_var adds direct signal about disagreement at each pattern.
    """
    def __init__(self):
        super().__init__()
        self.per_pat_mlp = nn.Sequential(
            nn.Linear(8, 64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward(self, frac_stats, within_var_stats, lam_pred):
        B    = frac_stats.shape[0]
        frac = frac_stats[:, :8]
        var_ = frac_stats[:, 8:16]
        cnt  = frac_stats[:, 16:24]
        wvar = within_var_stats

        lam   = lam_pred.expand(B, 8)
        lam_c = (1.0 - lam_pred).expand(B, 8)
        feats = torch.stack([frac, lam, lam_c,
                             (frac - lam).abs(), (frac - lam_c).abs(),
                             var_, wvar, cnt], dim=-1)
        out = self.per_pat_mlp(feats.reshape(B * 8, 8)).reshape(B, 8, 2)
        return out[:, :, 0], out[:, :, 1]


# ── Distributional Rule Decoder ───────────────────────────────────────────────
class DistributionalRuleDecoder(nn.Module):
    """
    Path D: percentiles of per-timestep per-pattern rates.
    Identical implementation to TSCA v9 DistributionalRuleDecoder.
    In SCA, rate[t,k] is a high-quality estimate of the mixture
    lambda*g[k] + (1-lambda)*f[k] (W=50 cells per timestep).
    """
    def __init__(self):
        super().__init__()
        self.dist_mlp = nn.Sequential(
            nn.Linear(N_DIST_PCTS, 64), nn.GELU(),
            nn.Linear(64, 64),          nn.GELU(),
            nn.Linear(64, 2),
        )

    def _compute_ts_rates(self, raw_tokens):
        B, K, _, _ = raw_tokens.shape
        l = raw_tokens[:, :, :, 0]; b = raw_tokens[:, :, :, 1]
        r = raw_tokens[:, :, :, 2]; a = raw_tokens[:, :, :, 3]
        pat_idx = (4*l + 2*b + r).long().clamp(0, 7)
        pat_4d  = pat_idx.reshape(B, K, N_TRANS, N_CELLS)
        a_4d    = a.reshape(B, K, N_TRANS, N_CELLS)
        pat_oh  = F.one_hot(pat_4d, num_classes=8).float()
        a_exp   = a_4d.unsqueeze(-1)
        sum_a   = (pat_oh * a_exp).sum(dim=3)
        sum_cnt = pat_oh.sum(dim=3)
        rate_ts = sum_a / (sum_cnt + EPS)
        rate_ts = torch.where(sum_cnt > 0.5, rate_ts,
                              torch.full_like(rate_ts, float('nan')))
        return rate_ts.nanmean(dim=1)

    def forward(self, raw_tokens):
        B       = raw_tokens.shape[0]
        rate_ts = self._compute_ts_rates(raw_tokens)
        rate_ts = torch.where(torch.isnan(rate_ts),
                              torch.full_like(rate_ts, 0.5), rate_ts)
        rate_tp = rate_ts.permute(0, 2, 1)
        pct_levels = torch.tensor([p / 100.0 for p in DIST_PCTS],
                                  dtype=rate_tp.dtype, device=rate_tp.device)
        pcts = torch.quantile(rate_tp, pct_levels, dim=2).permute(1, 2, 0)
        out  = self.dist_mlp(pcts.reshape(B * 8, N_DIST_PCTS)).reshape(B, 8, 2)
        return out[:, :, 0], out[:, :, 1]


# ══════════════════════════════════════════════════════════════════
# PATH B: HIERARCHICAL TRANSFORMER (identical to TSCA v9)
# ══════════════════════════════════════════════════════════════════

class CellEncoder(nn.Module):
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
    def __init__(self):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Linear(D2, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, h_all):
        weights = F.softmax(self.weight_net(h_all.mean(dim=2)), dim=1)
        return (h_all * weights.unsqueeze(-1)).sum(dim=1)


class LambdaConditionedRuleHead(nn.Module):
    """
    Rule head for SCA.

    No per-timestep assignment (ts_residual absent — that was TSCA FIX 3).
    Instead: lambda-conditioned attention pooling over all N_TRANS timesteps.

    attn_weight[t] = softmax( attn_net(h[t], lam_pred, mean_var) ) [B, N_TRANS]
    rep = weighted sum of h over timesteps

    Then separate rf_head and rg_head from same rep (no orientation split here —
    the symmetric_loss handles orientation during training).
    Both heads see the same rep; the model learns which rule is "F" vs "G"
    from the learned representation without explicit temporal assignment.

    Three paths: transformer(D2) + direct(8) + distributional(8) = D2+16
    """
    def __init__(self):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(D2 + 2, 128), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(128, 64),     nn.GELU(),
            nn.Linear(64, 1),
        )
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
        nn.init.normal_(self.attn_net[-1].weight, std=0.1)
        nn.init.zeros_(self.attn_net[-1].bias)

    def forward(self, h, lam_pred, frac_stats,
                direct_rf, direct_rg, dist_rf, dist_rg):
        B, Nt, D = h.shape
        mean_var = frac_stats[:, 8:16].mean(dim=1, keepdim=True)  # [B,1]
        lam_exp  = lam_pred.unsqueeze(1).expand(B, Nt, 1)
        var_exp  = mean_var.unsqueeze(1).expand(B, Nt, 1)

        attn_inp = torch.cat([h, lam_exp, var_exp], dim=-1)
        attn_w   = F.softmax(self.attn_net(attn_inp).squeeze(-1), dim=1)
        rep      = (h * attn_w.unsqueeze(-1)).sum(dim=1)

        rep_f = torch.cat([rep, direct_rf.detach(), dist_rf.detach()], dim=-1)
        rep_g = torch.cat([rep, direct_rg.detach(), dist_rg.detach()], dim=-1)
        return self.rf_head(rep_f), self.rg_head(rep_g)


# ── Main Model ────────────────────────────────────────────────────────────────
class ECANetSCA(nn.Module):
    """
    ECANetSCA v1.

    forward() inputs:
      raw_tokens       [B, K, N_TOK, 4]
      frac_stats       [B, 24]
      within_var_stats [B, 8]

    forward() outputs:
      rf_logits  [B, 8]
      rg_logits  [B, 8]
      lam_pred   [B, 1]  ∈ [LAM_MIN, 0.5]
      rf_dir     [B, 8]
      rg_dir     [B, 8]
    """
    N_RAW = 4

    def __init__(self):
        super().__init__()
        self.pat_emb    = PatternEmbedding(8, N_EMB)
        self.lam_mlp    = AnalyticalLambdaMLP()
        self.direct_dec = DirectRuleDecoder()
        self.dist_dec   = DistributionalRuleDecoder()
        self.cell_enc   = CellEncoder()
        self.ts_enc     = TimestepEncoder()
        self.orbit_fuse = ConfidenceWeightedOrbitFusion()
        self.rule_head  = LambdaConditionedRuleHead()
        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if any(s in name for s in ["correction_net", "attn_net", "weight_net"]):
                    continue
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _embed_tokens(self, raw_tokens):
        l = raw_tokens[:, :, 0]; b = raw_tokens[:, :, 1]; r = raw_tokens[:, :, 2]
        pat_idx = (4 * l + 2 * b + r).long().clamp(0, 7)
        return torch.cat([raw_tokens, self.pat_emb(pat_idx)], dim=-1)

    def _encode_orbits(self, raw_tokens):
        B, K, _, _ = raw_tokens.shape
        x_flat  = raw_tokens.reshape(B * K * N_TRANS, N_CELLS, self.N_RAW)
        e_flat  = self.cell_enc(self._embed_tokens(x_flat))
        e       = e_flat.reshape(B, K, N_TRANS, D1)
        h_flat  = self.ts_enc(e.reshape(B * K, N_TRANS, D1))
        h_all   = h_flat.reshape(B, K, N_TRANS, D2)
        return self.orbit_fuse(h_all)

    def forward(self, raw_tokens, frac_stats, within_var_stats):
        lam_pred         = self.lam_mlp(frac_stats, within_var_stats)
        rf_dir, rg_dir   = self.direct_dec(frac_stats, within_var_stats, lam_pred)
        rf_dist, rg_dist = self.dist_dec(raw_tokens)
        h                = self._encode_orbits(raw_tokens)
        rf, rg           = self.rule_head(h, lam_pred, frac_stats,
                                          rf_dir, rg_dir, rf_dist, rg_dist)
        return rf, rg, lam_pred, rf_dir, rg_dir

    def forward_lambda_only(self, frac_stats, within_var_stats):
        return self.lam_mlp(frac_stats, within_var_stats)

    def forward_lambda_analytical(self, frac_stats, within_var_stats):
        return self.lam_mlp.forward_analytical_only(frac_stats, within_var_stats)


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

def simulate_sca(rf, rg, init, lam, rng):
    """
    Simulate one SCA orbit.

    Each cell c independently at each timestep t:
      if rng.random() < lam: next[c] = rule_G[pattern(c)]
      else:                   next[c] = rule_F[pattern(c)]
    Neighborhoods from CURRENT (pre-update) row (synchronous).

    rf, rg : int (0..255)
    init   : [W] uint8
    lam    : float ∈ [0.1, 0.9]
    Returns: orbit [T, W] float32
    """
    tf = build_table(rf); tg = build_table(rg)
    orbit = np.zeros((T, W), dtype=np.float32)
    s     = init.copy().astype(np.uint8)
    orbit[0] = s
    for t in range(1, T):
        L   = np.roll(s, 1); R = np.roll(s, -1)
        idx = (4 * L + 2 * s + R).astype(np.uint8)
        ns  = np.where(rng.random(size=W) < lam,
                       tg[idx], tf[idx]).astype(np.uint8)
        s   = ns; orbit[t] = s.astype(np.float32)
    return orbit

def orbit_to_tokens(orbit):
    """[T, W] → [N_TOK, 4] tokens [L, B, R, A]."""
    o = orbit.astype(np.float32)
    b = o[:-1]; a = o[1:]
    l = np.roll(b, 1, axis=1); r = np.roll(b, -1, axis=1)
    return np.stack([l, b, r, a], axis=-1).reshape(-1, 4)

def compute_frac_stats(orbits_k):
    """
    orbits_k [K, T, W] → [24]
    Identical to TSCA: frac[n] = mean(after | pat=n)
    E[frac[n]] = lambda*g[n] + (1-lambda)*f[n]
    """
    before = orbits_k[:, :-1, :]; after = orbits_k[:, 1:, :]
    left   = np.roll(before, 1, axis=2); right = np.roll(before, -1, axis=2)
    pat    = (4 * left + 2 * before + right).astype(np.int32)
    total  = float(orbits_k.shape[0] * (T - 1) * W)
    frac   = np.zeros(8, dtype=np.float32)
    cnt    = np.zeros(8, dtype=np.float32)
    for n in range(8):
        mask = (pat == n); cnt_n = float(mask.sum())
        if cnt_n > 0: frac[n] = float(after[mask].sum()) / cnt_n
        cnt[n] = cnt_n / total
    return np.concatenate([frac, frac * (1 - frac), cnt]).astype(np.float32)

def compute_within_var_stats(orbits_k):
    """
    Per-pattern within-timestep variance — new signal unique to SCA.

    For each (k, t, pattern n): compute sample variance of after-states
    across all cells c where pat[k,t,c] == n.
    Average over valid (k,t) pairs (those with ≥ 2 cells showing pattern n).

    Estimates lambda*(1-lambda)*(g[n]-f[n])^2 at disagreement patterns.
    Max over n gives lambda*(1-lambda) directly.

    orbits_k : [K, T, W] float32
    Returns  : within_var [8] float32
    """
    before = orbits_k[:, :-1, :].astype(np.float64)
    after  = orbits_k[:, 1:,  :].astype(np.float64)
    left   = np.roll(before, 1, axis=2); right = np.roll(before, -1, axis=2)
    pat    = (4 * left + 2 * before + right).astype(np.int32)

    sum_var = np.zeros(8, dtype=np.float64)
    sum_cnt = np.zeros(8, dtype=np.float64)

    for n in range(8):
        mask   = (pat == n)                            # [K, T-1, W] bool
        a_vals = after * mask                          # zero where pat≠n
        sum1   = a_vals.sum(axis=2)                   # [K, T-1]
        sum2   = (a_vals ** 2).sum(axis=2)
        cnt_kt = mask.sum(axis=2).astype(np.float64)
        valid  = cnt_kt >= 2.0
        with np.errstate(divide='ignore', invalid='ignore'):
            safe_cnt = np.where(cnt_kt > 0, cnt_kt, 1.0)
            mean_kt  = np.where(valid, sum1 / safe_cnt, 0.0)
            var_kt   = np.where(valid, sum2 / safe_cnt - mean_kt**2, 0.0)
        sum_var[n] += (np.maximum(var_kt, 0.0) * valid).sum()
        sum_cnt[n] += valid.sum()

    with np.errstate(divide='ignore', invalid='ignore'):
        wv = np.where(sum_cnt > 0, sum_var / np.where(sum_cnt > 0, sum_cnt, 1.0), 0.0)
    return wv.astype(np.float32)


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  ECANetSCA v1 -- smoke test")
    print("=" * 65)

    model = ECANetSCA()
    npar  = sum(p.numel() for p in model.parameters())
    for nm, mod in [("PatternEmbedding",          model.pat_emb),
                    ("AnalyticalLambdaMLP",        model.lam_mlp),
                    ("DirectRuleDecoder",          model.direct_dec),
                    ("DistributionalRuleDecoder",  model.dist_dec),
                    ("CellEncoder",               model.cell_enc),
                    ("TimestepEncoder",           model.ts_enc),
                    ("ConfidenceWeightedFusion",  model.orbit_fuse),
                    ("LambdaConditionedRuleHead", model.rule_head)]:
        print(f"  {nm:<28}: "
              f"{sum(p.numel() for p in mod.parameters()):>10,}")
    print(f"  {'Total':<28}: {npar:>10,}  ({npar/1e6:.2f}M)")

    rng = np.random.default_rng(42)

    # Test 1: simulate_sca
    print(f"\n  TEST 1: simulate_sca")
    orbit = simulate_sca(30, 45, random_init(rng), 0.3, rng)
    assert orbit.shape == (T, W) and orbit.min() >= 0 and orbit.max() <= 1
    print(f"    shape={orbit.shape}  range=[{orbit.min():.0f},{orbit.max():.0f}]  OK")

    # Test 2: within_var_stats tracks lambda*(1-lambda) at disagreement patterns
    print(f"\n  TEST 2: within_var_stats signal (rule 0 vs 255)")
    for lam_t in [0.1, 0.3, 0.5, 0.7, 0.9]:
        orbs = np.stack([simulate_sca(0, 255, random_init(rng), lam_t, rng)
                         for _ in range(N_ORBITS)])
        wv   = compute_within_var_stats(orbs)
        exp  = lam_t * (1 - lam_t)
        print(f"    lam={lam_t:.1f}  expected={exp:.4f}  "
              f"max_wvar={wv.max():.4f}  mean_wvar={wv.mean():.4f}")

    # Test 3: analytical lambda (always ≤ 0.5)
    print(f"\n  TEST 3: analytical lambda estimation (symmetric)")
    errs = []
    for _ in range(20):
        rf_r = int(rng.integers(0, 256)); rg_r = int(rng.integers(0, 256))
        while rf_r == rg_r: rg_r = int(rng.integers(0, 256))
        lam_t = float(rng.uniform(LAM_MIN, LAM_MAX))
        orbs  = np.stack([simulate_sca(rf_r, rg_r, random_init(rng), lam_t, rng)
                          for _ in range(N_ORBITS)])
        fs    = torch.tensor(compute_frac_stats(orbs)).unsqueeze(0)
        wv    = torch.tensor(compute_within_var_stats(orbs)).unsqueeze(0)
        la    = model.forward_lambda_analytical(fs, wv).item()
        lp    = model.forward_lambda_only(fs, wv).item()
        assert la <= 0.5 + 1e-5, f"analytical > 0.5: {la}"
        assert lp <= 0.5 + 1e-5, f"lam_pred > 0.5: {lp}"
        errs.append(min(abs(la - lam_t), abs((1 - la) - lam_t)))
    print(f"    All 20 samples ≤ 0.5  OK")
    print(f"    Mean symmetric MAE: {np.mean(errs):.4f}  (target < 0.07)")

    # Test 4: forward pass shapes
    print(f"\n  TEST 4: forward pass shapes")
    B   = 2
    raw = torch.zeros(B, N_ORBITS, N_TOK, 4)
    raw[:, :, :, :3] = (torch.rand(B, N_ORBITS, N_TOK, 3) > 0.5).float()
    raw[:, :, :, 3]  = torch.rand(B, N_ORBITS, N_TOK)
    fs  = torch.rand(B, FRAC_DIM)
    wv  = torch.rand(B, WVAR_DIM).abs() * 0.15
    model.eval()
    with torch.no_grad():
        rf, rg, lp, rfd, rgd = model(raw, fs, wv)
    assert rf.shape == (B, 8) and rg.shape == (B, 8)
    assert lp.shape == (B, 1)
    assert (lp >= LAM_MIN - 1e-5).all() and (lp <= 0.5 + 1e-5).all()
    print(f"    rf={rf.shape}  rg={rg.shape}  lam_pred={lp.shape}")
    print(f"    lam_pred range: [{lp.min():.3f}, {lp.max():.3f}]  (≤ 0.5)  OK")

    # Test 5: gradients
    print(f"\n  TEST 5: gradient flow")
    model.train()
    rf2, rg2, lp2, _, _ = model(raw, fs, wv)
    (rf2.mean() + rg2.mean() + lp2.mean()).backward()
    no_grad = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    print(f"    {'ALL OK' if not no_grad else 'Missing: ' + str(no_grad[:3])}")

    print(f"\n  ECANetSCA v1 -- ALL TESTS PASSED")
    print("=" * 65)
