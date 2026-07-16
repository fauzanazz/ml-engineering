"""Train WallNet on self-play data and export weights for the Rust candle loader.

Usage:
    uv run train.py --data selfplay.jsonl --out wallnet.safetensors [--epochs 20]

Loss = soft-target policy cross-entropy + value MSE. The policy target is a
visit-count distribution (already a probability), so we use the full soft
cross-entropy rather than a single hard label.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, random_split

from dataset import SelfPlayDataset
from encoding import WALLCHESS, get_game_spec
from model import WallNet, WallNetCNN, WallNetResT


def policy_loss(logits, target, pw=None):
    # soft cross-entropy: -sum_a target(a) * log_softmax(logits)(a)
    # pw: optional per-sample policy weight (0 = value-only sample). Normalized by
    # sum(pw) so the loss scale is independent of the value-only fraction.
    logp = F.log_softmax(logits, dim=-1)
    ce = -(target * logp).sum(dim=-1)  # per-sample
    if pw is None:
        return ce.mean()
    denom = pw.sum().clamp(min=1.0)
    return (ce * pw).sum() / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["selfplay.jsonl"],
                    help="full data: policy + value targets (the d3 search anchor)")
    ap.add_argument("--value-data", nargs="*", default=[],
                    help="value-only data (self-play z): trained on value, policy ignored")
    ap.add_argument("--out", default="wallnet.safetensors")
    ap.add_argument("--game", choices=["wallchess", "checkers"], default="wallchess",
                    help="training contract; checkers currently supports only --arch mlp")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0,
                    help="torch seed for reproducible init/split/shuffle")
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--policy-loss-weight", type=float, default=1.0,
                    help="0 trains value-only while leaving the policy head unused")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--arch", choices=["mlp", "cnn", "restnet"], default="mlp",
                    help="mlp: 2-layer MLP (default); cnn: CNN encoder; "
                         "restnet: interleaved Residual+Transformer (POC, not WASM-deployable)")
    ap.add_argument("--cnn-channels", type=int, default=32,
                    help="CNN: base channel count (halved before flatten)")
    ap.add_argument("--rest-channels", type=int, default=64,
                    help="ResTNet: tower channel width")
    ap.add_argument("--rest-blocks", default="RRTRRT",
                    help="ResTNet: block string, R=residual T=transformer (must start with R)")
    ap.add_argument("--device", default="auto",
                    help="Training device: auto (mps > cuda > cpu), cpu, mps, cuda")
    ap.add_argument("--init", default=None,
                    help="optional safetensors warm-start before optimizer/checkpoint resume")
    ap.add_argument("--checkpoint", default=None,
                    help="torch checkpoint for resumable mid-training restarts")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore an existing --checkpoint and start fresh")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"device: {device}")

    spec = get_game_spec(args.game)
    if spec is not WALLCHESS and args.arch != "mlp":
        raise SystemExit(f"{args.game} training supports only --arch mlp; {args.arch} is Wall Chess-specific")
    print(f"game: {spec.id}  feature_len={spec.feature_len}  action_count={spec.action_count}")

    ds = SelfPlayDataset(args.data, policy_weight=1.0, spec=spec)
    if args.value_data:
        from torch.utils.data import ConcatDataset
        vds = SelfPlayDataset(args.value_data, policy_weight=0.0, spec=spec)
        print(f"data: {len(ds)} full + {len(vds)} value-only = {len(ds) + len(vds)}")
        ds = ConcatDataset([ds, vds])
    n_val = max(1, int(len(ds) * args.val_frac))
    n_train = len(ds) - n_val
    data_gen = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=data_gen)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_dl = DataLoader(val_ds, batch_size=args.batch)

    if args.arch == "cnn":
        model = WallNetCNN(channels=args.cnn_channels).to(device)
    elif args.arch == "restnet":
        model = WallNetResT(channels=args.rest_channels, blocks=args.rest_blocks).to(device)
    else:
        model = WallNet(hidden=args.hidden, spec=spec).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    best_val = float("inf")
    best_state = None
    start_epoch = 1
    ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    init_path = Path(args.init) if args.init else None
    if init_path and not init_path.exists():
        raise FileNotFoundError(init_path)
    if init_path and (not ckpt_path or not ckpt_path.exists() or args.no_resume):
        model.load_state_dict(load_file(str(init_path), device="cpu"))
        print(f"warm-started from {init_path}")
    if ckpt_path and ckpt_path.exists() and not args.no_resume:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_val = float(ckpt["best_val"])
        best_state = {k: v.to(device).detach().clone().contiguous() for k, v in ckpt["best_state"].items()}
        start_epoch = int(ckpt["epoch"]) + 1
        print(f"resumed {ckpt_path} at epoch {start_epoch}/{args.epochs}")
    if best_state is None:
        best_state = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        tp = tv = tb = 0.0
        for f, pi, z, pw in train_dl:
            f, pi, z, pw = f.to(device), pi.to(device), z.to(device), pw.to(device)
            opt.zero_grad()
            logits, v = model(f)
            pl = policy_loss(logits, pi, pw)
            vl = F.mse_loss(v, z)
            loss = args.policy_loss_weight * pl + args.value_weight * vl
            tp += pl.item()
            tv += vl.item()
            tb += 1
            loss.backward()
            opt.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            vp = vv = nb = 0.0
            for f, pi, z, pw in val_dl:
                f, pi, z, pw = f.to(device), pi.to(device), z.to(device), pw.to(device)
                logits, v = model(f)
                vp += policy_loss(logits, pi, pw).item()
                vv += F.mse_loss(v, z).item()
                nb += 1
            val_combined = args.policy_loss_weight * vp / nb + args.value_weight * vv / nb
            marker = ""
            if val_combined < best_val:
                best_val = val_combined
                best_state = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
                marker = " *"
            print(
                f"epoch {epoch:>3}/{args.epochs}  train_policy {tp / tb:.4f}  train_value {tv / tb:.4f}  "
                f"val_policy {vp / nb:.4f}  val_value {vv / nb:.4f}{marker}"
            )

        if ckpt_path:
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optim": opt.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val": best_val,
                    "best_state": {k: v.detach().cpu().contiguous() for k, v in best_state.items()},
                },
                ckpt_path,
            )

    # Save best checkpoint (lowest combined val loss), not last epoch.
    # Move to CPU before saving — safetensors requires CPU tensors.
    cpu_state = {k: v.cpu().detach().contiguous() for k, v in best_state.items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_file(cpu_state, args.out)
    print(f"saved {args.out}  ({sum(p.numel() for p in model.parameters())} params)")


if __name__ == "__main__":
    main()
