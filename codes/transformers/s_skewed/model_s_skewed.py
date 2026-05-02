"""
MODEL_SKEW.py  --  ECANet s-Skewed  (Enhanced Architecture)
=============================================================

Enhanced from baseline for maximum s-prediction accuracy.

FOUR ENHANCEMENTS OVER BASELINE:

  1. 5-FEATURE TOKENS (was 4):
     Token = (left, center, right, center_next, did_change)
     The 5th feature `did_change = float(center != center_next)` directly
     encodes whether a cell was updated. Under s-skewed update, changed
     cells form contiguous blocks of size s -- this is THE key signal.

  2. SPATIAL-AWARE POOLING:
     1D convolution over adjacent cell tokens within each timestep.
     Captures contiguous-block structure of s-skewed updates.

  3. MULTI-SCALE AGGREGATION:
     Three parallel branches: global stats, temporal stats, spatial stats.
     Each sees a different view of the s parameter.

  4. DUAL S-HEAD (classification + regression):
     s_logits [B, 20] : CrossEntropyLoss (discrete precision)
     s_reg    [B, 1]  : MSE loss on s/W (ordinal regulariser)

CONSTANTS:
  W=20, T=200, N_TRANS=199, N_TOK=3980, TOKEN_DIM=5
  D_MODEL=128, N_HEADS=4, N_LAYERS=4, D_FF=512

Input  : [B, 3980, 5]
Output : rule_logits [B, 8], s_logits [B, 20], s_reg [B, 1]
"""

import torch
import torch.nn as nn
import math
import numpy as np

# ── Exported constants ────────────────────────────────────────────────────────
W         = 20
T         = 200
N_TRANS   = T - 1           # 199
N_TOK     = N_TRANS * W     # 3980
N_BITS    = 8
TOKEN_DIM = 5               # (left, center, right, center_next, did_change)
D_MODEL   = 128
N_HEADS   = 4
N_LAYERS  = 4
D_FF      = 512
DROPOUT   = 0.1

S_VALUES    = list(range(1, W + 1))
N_S_CLASSES = len(S_VALUES)           # 20


# ── 2D Positional Encoding ────────────────────────────────────────────────────

class TripletPE2D(nn.Module):
    def __init__(self, d_model, n_trans, n_cells, dropout=DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        assert d_model % 2 == 0
        half = d_model // 2

        def sinpe(n_pos, n_dim):
            pe  = torch.zeros(n_pos, n_dim)
            pos = torch.arange(n_pos, dtype=torch.float).unsqueeze(1)
            div = torch.exp(torch.arange(0, n_dim, 2, dtype=torch.float)
                            * (-math.log(10000.0) / n_dim))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            return pe

        time_pe = sinpe(n_trans, half)
        cell_pe = sinpe(n_cells, half)
        t_idx   = torch.arange(n_trans).repeat_interleave(n_cells)
        c_idx   = torch.arange(n_cells).repeat(n_trans)
        full_pe = torch.cat([time_pe[t_idx], cell_pe[c_idx]], dim=-1)
        self.register_buffer("pe", full_pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


# ── Spatial-Aware Multi-Scale Pooling ─────────────────────────────────────────

class SpatialMultiScalePool(nn.Module):
    """
    Branch A -- Global: mean/std/max/min over all tokens -> [B, 4D]
    Branch B -- Temporal: pool cells per step, then stats over steps -> [B, 2D]
    Branch C -- Spatial: pool steps per cell, conv for local correlation,
                then stats over cells -> [B, 2D]
    Combined: [B, 8D] -> projected to [B, D]
    """
    def __init__(self, d_model=D_MODEL, n_trans=N_TRANS, n_cells=W, dropout=DROPOUT):
        super().__init__()
        self.n_trans = n_trans
        self.n_cells = n_cells

        self.spatial_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.proj = nn.Sequential(
            nn.Linear(8 * d_model, 4 * d_model),
            nn.GELU(),
            nn.LayerNorm(4 * d_model),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
        )

    def forward(self, h):
        B, NT, D = h.shape

        # Branch A: Global
        g_mean = h.mean(dim=1)
        g_std  = h.std(dim=1)
        g_max  = h.max(dim=1).values
        g_min  = h.min(dim=1).values
        branch_a = torch.cat([g_mean, g_std, g_max, g_min], dim=-1)

        # Reshape to 2D grid
        h2d = h.view(B, self.n_trans, self.n_cells, D)

        # Branch B: Temporal
        t_pool = h2d.mean(dim=2)                         # [B, N_TRANS, D]
        t_mean = t_pool.mean(dim=1)
        t_std  = t_pool.std(dim=1)
        branch_b = torch.cat([t_mean, t_std], dim=-1)

        # Branch C: Spatial with conv
        c_pool = h2d.mean(dim=1)                         # [B, W, D]
        c_conv = self.spatial_conv(
            c_pool.transpose(1, 2)
        ).transpose(1, 2)                                # [B, W, D]
        c_mean = c_conv.mean(dim=1)
        c_std  = c_conv.std(dim=1)
        branch_c = torch.cat([c_mean, c_std], dim=-1)

        combined = torch.cat([branch_a, branch_b, branch_c], dim=-1)
        return self.proj(combined)


# ── Main Model ────────────────────────────────────────────────────────────────

class ECANetSkew(nn.Module):
    def __init__(self):
        super().__init__()

        self.triplet_proj = nn.Linear(TOKEN_DIM, D_MODEL)
        self.pos_enc      = TripletPE2D(D_MODEL, N_TRANS, W, DROPOUT)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=D_FF,
            dropout=DROPOUT, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=N_LAYERS, enable_nested_tensor=False)

        # Rule stream
        self.rule_head = nn.Sequential(
            nn.Linear(D_MODEL, 64), nn.GELU(),
            nn.Dropout(DROPOUT), nn.Linear(64, N_BITS),
        )

        # S stream (enhanced)
        self.s_pool = SpatialMultiScalePool(D_MODEL, N_TRANS, W, DROPOUT)

        self.s_cls_head = nn.Sequential(
            nn.Linear(D_MODEL, 64), nn.GELU(),
            nn.Dropout(DROPOUT), nn.Linear(64, N_S_CLASSES),
        )

        self.s_reg_head = nn.Sequential(
            nn.Linear(D_MODEL, 64), nn.GELU(),
            nn.Dropout(DROPOUT), nn.Linear(64, 1), nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.triplet_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)

        rule_out = self.rule_head(h.mean(dim=1))
        s_feat   = self.s_pool(h)
        s_logits = self.s_cls_head(s_feat)
        s_reg    = self.s_reg_head(s_feat)

        return rule_out, s_logits, s_reg


# ── S helpers ─────────────────────────────────────────────────────────────────

_S_LUT = None

def s_logits_to_value(logits):
    global _S_LUT
    if _S_LUT is None or _S_LUT.device != logits.device:
        _S_LUT = torch.tensor(S_VALUES, dtype=torch.float32, device=logits.device)
    return _S_LUT[logits.argmax(dim=-1)]

def s_value_to_class(s_val):
    if isinstance(s_val, torch.Tensor):
        return (s_val.long() - 1).clamp(0, N_S_CLASSES - 1)
    elif isinstance(s_val, np.ndarray):
        return np.clip(s_val.astype(np.int64) - 1, 0, N_S_CLASSES - 1)
    return max(0, min(N_S_CLASSES - 1, int(round(float(s_val))) - 1))

def s_reg_to_value(s_reg):
    return (s_reg.squeeze(-1) * W).round().clamp(1, W)


# ── ECA helpers ───────────────────────────────────────────────────────────────

def build_rule_table(rule_number):
    return np.array([(rule_number >> i) & 1 for i in range(8)], dtype=np.uint8)

def rule_to_bits(rule_number):
    return np.array([(rule_number >> i) & 1 for i in range(8)], dtype=np.float32)

def bits_to_rule(bits):
    if isinstance(bits, torch.Tensor):
        bits = bits.cpu().numpy()
    return int(sum(int(round(float(b))) * (2 ** i) for i, b in enumerate(bits)))

def simulate_eca_skewed(rule_number, init_state, n_steps, s, rng, w=W):
    rule_table = build_rule_table(rule_number)
    orbit = np.zeros((n_steps, w), dtype=np.float32)
    state = init_state.copy().astype(np.uint8)
    orbit[0] = state
    offsets = np.arange(s, dtype=np.intp)
    for t in range(1, n_steps):
        left   = np.roll(state,  1)
        right  = np.roll(state, -1)
        idx    = (4 * left + 2 * state + right).astype(np.uint8)
        new_st = rule_table[idx]
        if s >= w:
            state = new_st
        else:
            start = int(rng.integers(0, w))
            cells = (start + offsets) % w
            state = state.copy()
            state[cells] = new_st[cells]
        orbit[t] = state
    return orbit

def orbit_to_tokens(orbit):
    """
    Convert [T, W] orbit -> [N_TOK, 5] tokens.
    Features: (left, center, right, center_next, did_change)
    """
    orbit  = orbit.astype(np.float32)
    before = orbit[:-1]
    after  = orbit[1:]
    left   = np.roll(before,  1, axis=1)
    right  = np.roll(before, -1, axis=1)
    did_change = (before != after).astype(np.float32)
    tokens = np.stack([left, before, right, after, did_change], axis=-1)
    return tokens.reshape(-1, 5)


if __name__ == "__main__":
    model = ECANetSkew()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ECANetSkew (Enhanced) ready")
    print(f"  W={W}, T={T}, N_TOK={N_TOK}, TOKEN_DIM={TOKEN_DIM}")
    print(f"  D_MODEL={D_MODEL}, N_HEADS={N_HEADS}, N_LAYERS={N_LAYERS}")
    print(f"  S classes : {N_S_CLASSES}  |  Parameters: {n_params:,}")

    dummy = torch.zeros(2, N_TOK, TOKEN_DIM)
    rule_logits, s_logits, s_reg = model(dummy)
    print(f"  Forward: rule={rule_logits.shape}  "
          f"s_cls={s_logits.shape}  s_reg={s_reg.shape}")

    for sv in S_VALUES:
        assert S_VALUES[s_value_to_class(sv)] == sv
    print(f"  Round-trip OK.  MODEL_SKEW.py ready.")
