"""
decoder_only_control_alpha.py -- decoder-only next-token control on alphaECA
==============================================================================
A causally masked transformer trained from random initialisation with standard
next-token cross-entropy (Appendix, Baseline 5).

  - Sequence: [orbit digits (0/1)] [SEP] [8 rule bits] [SEP] [4 alpha-class bits]
  - Vocabulary: {0, 1, SEP}
  - Loss on the completion tokens only (rule bits + alpha bits), matching the
    LLM fine-tuning protocol.
  - No analytical estimator, no task-specific pooling, no pretrained weights.
  - 4 layers, d_model=128, 4 heads, FF 512 (~1.05M parameters).
  - Evaluation is teacher-forced; the final-epoch accuracy is reported.

Usage:
  python3 decoder_only_control_alpha.py --data_dir ECA_Data_New --epochs 30
"""

import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

VOCAB_SIZE = 3
SEP = 2
N_BITS_RULE = 8
N_BITS_ALPHA = 4
ALPHA_VALUES = [round(a * 0.1, 1) for a in range(1, 11)]


class DecoderOnlyECA(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=128, n_heads=4, n_layers=4, max_seq_len=2100):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.tok_emb(x) + self.pos_emb(pos)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        h = self.decoder(h, mask=causal_mask, is_causal=True)
        return self.head(self.ln_f(h))


def make_sequence(orbit_flat, rule_number, alpha_class):
    rule_bits = [(rule_number >> i) & 1 for i in range(N_BITS_RULE)]
    alpha_bits = [(alpha_class >> i) & 1 for i in range(N_BITS_ALPHA)]
    seq = list(orbit_flat) + [SEP] + rule_bits + [SEP] + alpha_bits
    prompt_len = len(orbit_flat)
    loss_mask = [0] * prompt_len + [1] * (1 + N_BITS_RULE + 1 + N_BITS_ALPHA)
    return seq, loss_mask


def alpha_to_class(alpha_val):
    return min(range(10), key=lambda i: abs(ALPHA_VALUES[i] - alpha_val))


class ECASequenceDataset(Dataset):
    def __init__(self, orbits, rule_ids, alphas):
        self.orbits = orbits
        self.rule_ids = rule_ids
        self.alphas = alphas

    def __len__(self):
        return len(self.orbits)

    def __getitem__(self, idx):
        orbit_flat = self.orbits[idx].flatten().astype(int).tolist()
        rule = int(self.rule_ids[idx])
        alpha_class = alpha_to_class(float(self.alphas[idx]))
        seq, mask = make_sequence(orbit_flat, rule, alpha_class)
        return torch.tensor(seq, dtype=torch.long), torch.tensor(mask, dtype=torch.long)


def evaluate(model, loader, device):
    model.eval()
    n = 0
    rule_correct = 0
    alpha_correct = 0
    with torch.no_grad():
        for x, mask in loader:
            x, mask = x.to(device), mask.to(device)
            logits = model(x)
            pred = logits[:, :-1, :].argmax(-1)
            target = x[:, 1:]
            m = mask[:, 1:]

            for b in range(x.size(0)):
                completion_positions = (m[b] == 1).nonzero(as_tuple=True)[0]
                if len(completion_positions) == 0:
                    continue
                pred_tokens = pred[b, completion_positions]
                true_tokens = target[b, completion_positions]
                rule_pred = pred_tokens[1:1 + N_BITS_RULE]
                rule_true = true_tokens[1:1 + N_BITS_RULE]
                alpha_pred = pred_tokens[2 + N_BITS_RULE:2 + N_BITS_RULE + N_BITS_ALPHA]
                alpha_true = true_tokens[2 + N_BITS_RULE:2 + N_BITS_RULE + N_BITS_ALPHA]
                if torch.equal(rule_pred, rule_true):
                    rule_correct += 1
                if torch.equal(alpha_pred, alpha_true):
                    alpha_correct += 1
                n += 1
    model.train()
    return rule_correct / n * 100 if n else 0.0, alpha_correct / n * 100 if n else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="ECA_Data_New")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--output_dir", default="decoder_only_results")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    train_orbits = np.load(os.path.join(args.data_dir, "phase2", "train", "orbits.npy"))
    train_rules = np.load(os.path.join(args.data_dir, "phase2", "train", "rule_ids.npy"))
    train_alphas = np.load(os.path.join(args.data_dir, "phase2", "train", "alphas.npy"))
    test_orbits = np.load(os.path.join(args.data_dir, "phase2", "test", "orbits.npy"))
    test_rules = np.load(os.path.join(args.data_dir, "phase2", "test", "rule_ids.npy"))
    test_alphas = np.load(os.path.join(args.data_dir, "phase2", "test", "alphas.npy"))

    print(f"Train: {len(train_orbits):,}  Test: {len(test_orbits):,}")
    T, W = train_orbits.shape[1], train_orbits.shape[2]
    seq_len = T * W + 2 + N_BITS_RULE + N_BITS_ALPHA
    print(f"Orbit shape: T={T} W={W}  Full sequence length: {seq_len}")

    train_ds = ECASequenceDataset(train_orbits, train_rules, train_alphas)
    test_ds = ECASequenceDataset(test_orbits, test_rules, test_alphas)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = DecoderOnlyECA(d_model=args.d_model, n_layers=args.n_layers,
                           max_seq_len=seq_len + 10).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        total_loss, n_batches = 0.0, 0
        for x, mask in train_loader:
            x, mask = x.to(device), mask.to(device)
            logits = model(x)
            pred = logits[:, :-1, :]
            target = x[:, 1:]
            m = mask[:, 1:].float()
            loss = nn.functional.cross_entropy(
                pred.reshape(-1, VOCAB_SIZE), target.reshape(-1), reduction="none")
            loss = (loss * m.reshape(-1)).sum() / m.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        sched.step()

        if epoch % 2 == 0 or epoch == args.epochs:
            rule_acc, alpha_acc = evaluate(model, test_loader, device)
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}  loss={total_loss/n_batches:.4f}  "
                  f"test_rule={rule_acc:.2f}%  test_alpha={alpha_acc:.2f}%  "
                  f"elapsed={elapsed:.0f}s", flush=True)

    rule_acc, alpha_acc = evaluate(model, test_loader, device)
    print(f"\nFINAL  rule={rule_acc:.2f}%  alpha={alpha_acc:.2f}%")

    with open(os.path.join(args.output_dir, "decoder_only_alpha_results.json"), "w") as f:
        json.dump({
            "n_params": n_params, "epochs": args.epochs,
            "final_rule_accuracy": rule_acc, "final_alpha_accuracy": alpha_acc,
        }, f, indent=2)
    print(f"Saved: {os.path.join(args.output_dir, 'decoder_only_alpha_results.json')}")


if __name__ == "__main__":
    main()
