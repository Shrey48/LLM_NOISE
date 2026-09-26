"""
vit_from_scratch_alpha.py -- Vision Transformer trained from scratch on rendered orbits
=========================================================================================
Appendix, Baseline 6.

  - The orbit is rendered as in the image-input evaluation: each cell is an
    8x8 pixel block (0 = black, 1 = white), giving an 800x160 single-channel image.
  - Standard ViT: 8x8 patches (one patch per orbit cell, 2,000 patches),
    learnable position embeddings, CLS token, bidirectional transformer encoder
    (4 layers, d_model=128, 4 heads, FF 512, PreNorm, GELU, dropout 0.1), and
    heads for the rule (8-bit BCE) and alpha (10-class cross-entropy).
  - Random initialisation: no pretrained weights, no analytical estimator,
    no task-specific pooling. ~1,075,602 parameters.
  - AdamW (lr 3e-4, weight decay 0.01), cosine schedule, 30 epochs, batch 32,
    gradient clipping 1.0; the final-epoch accuracy is reported.

Usage:
  python3 vit_from_scratch_alpha.py --data_dir ECA_Data_New --epochs 30
"""

import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ALPHA_VALUES = [round(a * 0.1, 1) for a in range(1, 11)]
CELL_SIZE = 8
IMG_H, IMG_W = 800, 160  # 100 rows x 20 cols, each cell scaled to 8x8


class ViTECA(nn.Module):
    def __init__(self, img_h=IMG_H, img_w=IMG_W, patch_size=CELL_SIZE, d_model=128,
                 n_heads=4, n_layers=4, n_bits_rule=8, n_alpha_classes=10):
        super().__init__()
        self.patch_size = patch_size
        n_patches_h = img_h // patch_size
        n_patches_w = img_w // patch_size
        n_patches = n_patches_h * n_patches_w
        patch_dim = patch_size * patch_size

        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)

        self.rule_head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, n_bits_rule))
        self.alpha_head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, n_alpha_classes))

        self.n_patches_h, self.n_patches_w = n_patches_h, n_patches_w

    def forward(self, img):
        B = img.shape[0]
        p = self.patch_size
        patches = img.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(B, self.n_patches_h * self.n_patches_w, p * p)
        x = self.patch_embed(patches)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.encoder(x)
        x = self.ln_f(x[:, 0])
        return self.rule_head(x), self.alpha_head(x)


def orbit_to_image(orbit: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(orbit.astype(np.float32), CELL_SIZE, axis=0), CELL_SIZE, axis=1)


def alpha_to_class(alpha_val):
    return min(range(10), key=lambda i: abs(ALPHA_VALUES[i] - alpha_val))


def rule_to_bits(rule_number):
    return [(rule_number >> i) & 1 for i in range(8)]


class ECAImageDataset(Dataset):
    def __init__(self, orbits, rule_ids, alphas):
        self.orbits = orbits
        self.rule_ids = rule_ids
        self.alphas = alphas

    def __len__(self):
        return len(self.orbits)

    def __getitem__(self, idx):
        img = orbit_to_image(self.orbits[idx])
        rule_bits = torch.tensor(rule_to_bits(int(self.rule_ids[idx])), dtype=torch.float32)
        alpha_class = torch.tensor(alpha_to_class(float(self.alphas[idx])), dtype=torch.long)
        return torch.tensor(img).unsqueeze(0), rule_bits, alpha_class


def evaluate(model, loader, device):
    model.eval()
    n = 0
    rule_correct = 0
    alpha_correct = 0
    with torch.no_grad():
        for img, rule_bits, alpha_class in loader:
            img, rule_bits, alpha_class = img.to(device), rule_bits.to(device), alpha_class.to(device)
            rule_out, alpha_out = model(img)
            rule_pred = (torch.sigmoid(rule_out) > 0.5).float()
            alpha_pred = alpha_out.argmax(-1)
            for b in range(img.size(0)):
                if torch.equal(rule_pred[b], rule_bits[b]):
                    rule_correct += 1
                if alpha_pred[b].item() == alpha_class[b].item():
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
    ap.add_argument("--output_dir", default="vit_results")
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

    train_ds = ECAImageDataset(train_orbits, train_rules, train_alphas)
    test_ds = ECAImageDataset(test_orbits, test_rules, test_alphas)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ViTECA(d_model=args.d_model, n_layers=args.n_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        total_loss, n_batches = 0.0, 0
        for img, rule_bits, alpha_class in train_loader:
            img, rule_bits, alpha_class = img.to(device), rule_bits.to(device), alpha_class.to(device)
            rule_out, alpha_out = model(img)
            loss = (nn.functional.binary_cross_entropy_with_logits(rule_out, rule_bits) +
                    nn.functional.cross_entropy(alpha_out, alpha_class))
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

    with open(os.path.join(args.output_dir, "vit_alpha_results.json"), "w") as f:
        json.dump({
            "n_params": n_params, "epochs": args.epochs,
            "final_rule_accuracy": rule_acc, "final_alpha_accuracy": alpha_acc,
        }, f, indent=2)
    print(f"Saved: {os.path.join(args.output_dir, 'vit_alpha_results.json')}")


if __name__ == "__main__":
    main()
