"""Train WallNet on self-play data and export weights for the Rust candle loader.

Usage:
    uv run train.py --data selfplay.jsonl --out wallnet.safetensors [--epochs 20]

Loss = soft-target policy cross-entropy + value MSE. The policy target is a
visit-count distribution (already a probability), so we use the full soft
cross-entropy rather than a single hard label.
"""

import argparse

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import DataLoader, random_split

from dataset import SelfPlayDataset
from model import WallNet


def policy_loss(logits, target):
    # soft cross-entropy: -sum_a target(a) * log_softmax(logits)(a), mean over batch
    logp = F.log_softmax(logits, dim=-1)
    return -(target * logp).sum(dim=-1).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="selfplay.jsonl")
    ap.add_argument("--out", default="wallnet.safetensors")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--value-weight", type=float, default=1.0)
    args = ap.parse_args()

    ds = SelfPlayDataset(args.data)
    n_val = max(1, int(len(ds) * args.val_frac))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch)

    model = WallNet()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for f, pi, z in train_dl:
            opt.zero_grad()
            logits, v = model(f)
            loss = policy_loss(logits, pi) + args.value_weight * F.mse_loss(v, z)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vp = vv = nb = 0.0
            for f, pi, z in val_dl:
                logits, v = model(f)
                vp += policy_loss(logits, pi).item()
                vv += F.mse_loss(v, z).item()
                nb += 1
            print(
                f"epoch {epoch:>3}/{args.epochs}  "
                f"val_policy {vp / nb:.4f}  val_value {vv / nb:.4f}"
            )

    # Contiguous f32 tensors keyed by the names the candle loader expects.
    state = {k: v.detach().contiguous() for k, v in model.state_dict().items()}
    save_file(state, args.out)
    print(f"saved {args.out}  ({sum(p.numel() for p in model.parameters())} params)")


if __name__ == "__main__":
    main()
