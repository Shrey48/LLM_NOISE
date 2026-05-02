"""
MODEL_NEW.py  --  ECANet New  (Hybrid Architecture)
=====================================================

Rule stream  : Triplet tokenization + 2D PE + Transformer + mean pool
               Taken directly from TRAIN3 -- proven best rule accuracy.

Alpha stream : Same transformer backbone, then StatisticalPool
               (mean + std + max + min across all tokens) -> MLP
               Alpha predicted as 10-class classification over
               {0.1, 0.2, ..., 1.0} -- guarantees max error = 0.05.

CONSTANTS exported (used by DATAGEN, TRAIN, FINAL_TEST):
  W=20, T=100, N_TRANS=99, N_TOK=1980, N_BITS=8
  D_MODEL=128, N_HEADS=4, N_LAYERS=4, D_FF=512

Input to model  : [B, N_TOK=1980, 4]  -- pre-built triplet tokens
Output          : rule_logits [B, 8],  alpha_logits [B, 10]

Alpha decoding  : argmax(alpha_logits) -> class index -> ALPHA_VALUES[idx]
                  Max possible error = 0.05 (half of 0.1 step size)
"""

import torch
import torch.nn as nn
import math

# ── Exported constants ────────────────────────────────────────────────────────
W        = 20
T        = 100
N_TRANS  = T - 1          # 99 transitions
N_TOK    = N_TRANS * W    # 1980 triplet tokens per sample
N_BITS   = 8
D_MODEL  = 128
N_HEADS  = 4
N_LAYERS = 4
D_FF     = 512
DROPOUT  = 0.1

ALPHA_VALUES   = [round(a * 0.1, 1) for a in range(1, 11)]  # 0.1 .. 1.0
N_ALPHA_CLASSES = len(ALPHA_VALUES)   # 10


# ── 2D Positional Encoding ────────────────────────────────────────────────────

class TripletPE2D(nn.Module):
    """
    2D positional encoding for triplet tokens.

    Each of the 1980 tokens sits at (transition_t, cell_i) in a 2D grid.
    Standard 1D PE over 1980 positions conflates time and space.

    This PE encodes them separately:
      First D//2 dims  : sinusoidal over transition index t  (0..98)
      Last  D//2 dims  : sinusoidal over cell index i        (0..19)

    Lets the transformer group tokens by:
      - same cell_i at different t  => temporal pattern of one cell
      - same t across all i         => one complete timestep snapshot

    Taken unchanged from TRAIN3 -- key contributor to rule accuracy.
    """
    def __init__(self, d_model, n_trans, n_cells, dropout=DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        assert d_model % 2 == 0, "d_model must be even"
        half = d_model // 2

        def sinpe(n_pos, n_dim):
            pe  = torch.zeros(n_pos, n_dim)
            pos = torch.arange(n_pos, dtype=torch.float).unsqueeze(1)
            div = torch.exp(torch.arange(0, n_dim, 2, dtype=torch.float)
                            * (-math.log(10000.0) / n_dim))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            return pe

        time_pe = sinpe(n_trans, half)    # (N_TRANS, D/2)
        cell_pe = sinpe(n_cells, half)    # (W,       D/2)

        t_idx   = torch.arange(n_trans).repeat_interleave(n_cells)  # (N_TOK,)
        c_idx   = torch.arange(n_cells).repeat(n_trans)             # (N_TOK,)
        full_pe = torch.cat([time_pe[t_idx], cell_pe[c_idx]], dim=-1)  # (N_TOK, D)
        self.register_buffer("pe", full_pe.unsqueeze(0))               # (1, N_TOK, D)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


# ── Statistical Pool (alpha stream) ──────────────────────────────────────────

class StatisticalPool(nn.Module):
    """
    Statistical pooling for alpha prediction.

    WHY NOT MEAN POOL:
      Mean pooling loses distributional shape. Two orbits with very
      different alphas (0.2 vs 0.8) can have similar mean activations
      if the rule itself is very active or very quiescent.

    WHAT WE DO INSTEAD:
      Compute 4 statistics across the N_TOK=1980 token dimension:
        mean  -- average activation level
        std   -- spread / variability (key: low alpha -> high variance
                 because frozen cells create very different activations
                 from updated cells)
        max   -- peak activation
        min   -- floor activation

      Concatenate: [B, 4*D_MODEL=512]
      MLP: 512 -> 256 -> D_MODEL=128

      This gives a rich summary of the orbit's "texture" from which
      alpha can be classified. The 10-class output guarantees
      max error = 0.05 (half a step of the 0.1 grid).

    LayerNorm after first projection stabilises training.
    """
    def __init__(self, d_model=D_MODEL, dropout=DROPOUT):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(4 * d_model, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, d_model),
            nn.GELU(),
        )

    def forward(self, x):
        # x: [B, N_TOK, D_MODEL]
        mean_ = x.mean(dim=1)           # [B, D_MODEL]
        std_  = x.std(dim=1)            # [B, D_MODEL]
        max_  = x.max(dim=1).values     # [B, D_MODEL]
        min_  = x.min(dim=1).values     # [B, D_MODEL]
        stats = torch.cat([mean_, std_, max_, min_], dim=-1)  # [B, 512]
        return self.proj(stats)          # [B, D_MODEL]


# ── Main Model ────────────────────────────────────────────────────────────────

class ECANetNew(nn.Module):
    """
    ECANet New  --  Combined Rule + Alpha Architecture

    SHARED BACKBONE:
      triplet_proj : Linear(4 -> 128)
        Each token = (left, center, right, center_next)
        This is ONE complete rule observation. Zero noise, zero mixing.
        The model just needs to aggregate these observations.

      pos_enc : TripletPE2D
        2D sinusoidal: time dimension + cell dimension separately.

      encoder : 4x TransformerEncoderLayer
        d=128, h=4, ff=512, GELU, PreNorm (norm_first=True)
        Bidirectional attention across all 1980 tokens.

    RULE STREAM (from TRAIN3 -- best rule accuracy):
      h.mean(dim=1)  ->  [B, 128]
      rule_head: Linear(128->64) -> GELU -> Dropout -> Linear(64->8)
      Loss: BCEWithLogitsLoss (8 independent bits)
      Active: Phase 1 + Phase 2

    ALPHA STREAM (10-class classification):
      StatisticalPool(h) -> [B, 128]
      alpha_head: Linear(128->64)->GELU->Drop->Linear(64->32)->GELU->Linear(32->10)
      Output: [B, 10] raw logits over ALPHA_VALUES = {0.1, 0.2, ..., 1.0}
      Loss: CrossEntropyLoss
      Decode: argmax -> class index -> ALPHA_VALUES[idx]
      Max possible error: 0.05 (guaranteed by discrete class structure)
      Active: Phase 2 only (lambda_alpha=0 in Phase 1)

    Forward input  : x [B, N_TOK=1980, 4]  (pre-built triplet tokens)
    Forward output : rule_logits [B, 8],  alpha_logits [B, 10]
    """
    def __init__(self):
        super().__init__()

        # ── Shared backbone ───────────────────────────────────────────────────
        self.triplet_proj = nn.Linear(4, D_MODEL)
        self.pos_enc      = TripletPE2D(D_MODEL, N_TRANS, W, DROPOUT)

        enc_layer = nn.TransformerEncoderLayer(
            d_model         = D_MODEL,
            nhead           = N_HEADS,
            dim_feedforward = D_FF,
            dropout         = DROPOUT,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,      # pre-norm: more stable training
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=N_LAYERS,
            enable_nested_tensor=False)  # avoids UserWarning with norm_first

        # ── Rule stream ───────────────────────────────────────────────────────
        self.rule_head = nn.Sequential(
            nn.Linear(D_MODEL, 64),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, N_BITS),
        )

        # ── Alpha stream (10-class classification) ────────────────────────────
        self.alpha_pool = StatisticalPool(D_MODEL, DROPOUT)
        self.alpha_head = nn.Sequential(
            nn.Linear(D_MODEL, 64),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, N_ALPHA_CLASSES),   # 10 logits, NO Sigmoid
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, N_TOK=1980, 4]
        h = self.triplet_proj(x)   # [B, N_TOK, D_MODEL]
        h = self.pos_enc(h)        # [B, N_TOK, D_MODEL]
        h = self.encoder(h)        # [B, N_TOK, D_MODEL]

        # Rule stream
        rule_out = self.rule_head(h.mean(dim=1))   # [B, 8]

        # Alpha stream -- 10-class logits
        alpha_out = self.alpha_head(self.alpha_pool(h))  # [B, 10]

        return rule_out, alpha_out


# ── Alpha classification helpers ──────────────────────────────────────────────

import numpy as np

_ALPHA_LUT = None   # lazy-init on first call to avoid CUDA issues

def alpha_logits_to_value(logits):
    """
    Convert [B, 10] alpha logits -> [B] float alpha values via argmax.

    Max possible error = 0.05 (half of 0.1 grid step).
    Works on any device; output tensor stays on same device as logits.

    Usage:
        rule_logits, alpha_logits = model(x)
        alpha_vals = alpha_logits_to_value(alpha_logits)  # [B] float
    """
    global _ALPHA_LUT
    if _ALPHA_LUT is None or _ALPHA_LUT.device != logits.device:
        _ALPHA_LUT = torch.tensor(ALPHA_VALUES, dtype=torch.float32,
                                  device=logits.device)
    idx = logits.argmax(dim=-1)   # [B]
    return _ALPHA_LUT[idx]        # [B]


def alpha_value_to_class(alpha_val):
    """
    Convert float alpha value(s) -> integer class index (0..9).

    Maps: 0.1->0, 0.2->1, ..., 1.0->9

    Accepts:
      - Python float / int  -> int
      - numpy scalar / array -> int64 array
      - torch Tensor        -> LongTensor (same device)
    """
    if isinstance(alpha_val, torch.Tensor):
        return torch.round((alpha_val - 0.1) / 0.1).long()
    elif isinstance(alpha_val, np.ndarray):
        return np.round((alpha_val - 0.1) / 0.1).astype(np.int64)
    return int(round((float(alpha_val) - 0.1) / 0.1))


# ── ECA helpers (shared across all scripts) ───────────────────────────────────

def build_rule_table(rule_number):
    """LUT: index = 4*left + 2*center + right -> output cell value."""
    return np.array([(rule_number >> i) & 1 for i in range(8)], dtype=np.uint8)

def rule_to_bits(rule_number):
    return np.array([(rule_number >> i) & 1 for i in range(8)], dtype=np.float32)

def bits_to_rule(bits):
    if isinstance(bits, torch.Tensor):
        bits = bits.cpu().numpy()
    return int(sum(int(round(float(b))) * (2 ** i) for i, b in enumerate(bits)))

def simulate_eca(rule_number, init_state, n_steps, alpha, rng, w=W):
    """Generate a full ECA orbit. Returns float32 array [n_steps, W]."""
    rule_table = build_rule_table(rule_number)
    orbit = np.zeros((n_steps, w), dtype=np.float32)
    state = init_state.copy().astype(np.uint8)
    orbit[0] = state
    for t in range(1, n_steps):
        left   = np.roll(state,  1)
        right  = np.roll(state, -1)
        idx    = (4 * left + 2 * state + right).astype(np.uint8)
        new_st = rule_table[idx]
        if alpha < 1.0:
            mask  = rng.random(w) < alpha
            state = np.where(mask, new_st, state).astype(np.uint8)
        else:
            state = new_st.astype(np.uint8)
        orbit[t] = state
    return orbit

def orbit_to_tokens(orbit):
    """
    Convert a [T, W] orbit to [N_TOK, 4] triplet tokens.

    For each transition t in 0..T-2, for each cell i in 0..W-1:
      token = (left[t,i], center[t,i], right[t,i], center_next[t+1,i])

    This is one complete rule observation -- exactly 4 inputs, no mixing.
    The backbone only needs to aggregate these observations.
    """
    orbit  = orbit.astype(np.float32)
    before = orbit[:-1]                         # [T-1, W]
    after  = orbit[1:]                          # [T-1, W]
    left   = np.roll(before,  1, axis=1)        # [T-1, W]
    right  = np.roll(before, -1, axis=1)        # [T-1, W]
    tokens = np.stack([left, before, right, after], axis=-1)  # [T-1, W, 4]
    return tokens.reshape(-1, 4)                # [N_TOK, 4]


if __name__ == "__main__":
    # Quick sanity check
    import sys
    model = ECANetNew()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ECANetNew ready")
    print(f"  W={W}, T={T}, N_TOK={N_TOK}")
    print(f"  D_MODEL={D_MODEL}, N_HEADS={N_HEADS}, N_LAYERS={N_LAYERS}, D_FF={D_FF}")
    print(f"  Alpha classes : {N_ALPHA_CLASSES}  (guaranteed max error = 0.05)")
    print(f"  Parameters: {n_params:,}  (~{n_params/1e6:.2f}M)")

    dummy = torch.zeros(2, N_TOK, 4)
    rule_logits, alpha_logits = model(dummy)
    alpha_vals = alpha_logits_to_value(alpha_logits)
    print(f"  Forward check: rule_logits {rule_logits.shape}, "
          f"alpha_logits {alpha_logits.shape}")
    print(f"  Decoded alpha values: {alpha_vals.tolist()}")

    # Test helpers
    for av in ALPHA_VALUES:
        cls = alpha_value_to_class(av)
        assert ALPHA_VALUES[cls] == av, f"Round-trip failed for alpha={av}"
    print(f"  alpha_value_to_class round-trip: OK")
    print("MODEL_NEW.py  --  OK")
