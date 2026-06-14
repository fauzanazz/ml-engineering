"""Distill a ResTNet teacher into a deployable MLP student.

The teacher (WallNetResT) has spatial/global structure the flat MLP lacks, but
only the MLP is loadable by the Rust/WASM net runtime (core/src/net.rs). So we
train the MLP to match the teacher's SOFT outputs (dark knowledge: the full
relative move distribution + calibrated value), not just the sparse hard targets.

Loss = T^2 * KL(student || teacher_policy@T)  +  value_weight * MSE(values)
       (+ optional alpha blend with the original hard pi/z targets)

Usage:
    uv run distill.py --teacher rest.safetensors --data a.jsonl b.jsonl \
        --out wallnet-student.safetensors --epochs 40

Output is a standard MLP safetensors (l1/l2/policy/value) — drop-in for arena
and WASM.
"""

import argparse

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, random_split

from dataset import SelfPlayDataset
from model import WallNet, WallNetResT


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def soft_ce(student_logits, target_probs):
    """-sum target * log_softmax(student), mean over batch."""
    return -(target_probs * F.log_softmax(student_logits, dim=-1)).sum(-1).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="ResTNet teacher safetensors")
    ap.add_argument("--data", nargs="+", required=True, help="positions to distill over")
    ap.add_argument("--out", default="wallnet-student.safetensors")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--value-weight", type=float, default=0.5)
    ap.add_argument("--hidden", type=int, default=512, help="student MLP width")
    ap.add_argument("--temp", type=float, default=2.0, help="distillation temperature")
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="blend weight on the original hard pi/z targets (0=pure distill)")
    # teacher arch (must match how the teacher was trained)
    ap.add_argument("--rest-channels", type=int, default=64)
    ap.add_argument("--rest-blocks", default="RRTRRT")
    args = ap.parse_args()

    device = pick_device("auto")
    print(f"device: {device}  temp: {args.temp}  alpha: {args.alpha}")

    teacher = WallNetResT(channels=args.rest_channels, blocks=args.rest_blocks).to(device)
    teacher.load_state_dict(load_file(args.teacher))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    ds = SelfPlayDataset(args.data)
    n_val = max(1, int(len(ds) * args.val_frac))
    train_ds, val_ds = random_split(
        ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(0)
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch)

    student = WallNet(hidden=args.hidden).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    T = args.temp

    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        student.train()
        for f, pi, z, _pw in train_dl:
            f, pi, z = f.to(device), pi.to(device), z.to(device)
            with torch.no_grad():
                t_logits, t_val = teacher(f)
                t_soft = F.softmax(t_logits / T, dim=-1)
            s_logits, s_val = student(f)
            # KL via soft-CE at temperature T (scaled by T^2 to keep grad magnitude)
            loss = (T * T) * soft_ce(s_logits / T, t_soft)
            loss = loss + args.value_weight * F.mse_loss(s_val, t_val)
            if args.alpha > 0:
                loss = loss + args.alpha * (
                    soft_ce(s_logits, pi) + args.value_weight * F.mse_loss(s_val, z)
                )
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        # Validate against the TEACHER (distillation fidelity) — the student's job
        # is to reproduce the teacher, so that's what we early-stop on.
        student.eval()
        with torch.no_grad():
            vp = vv = nb = 0.0
            for f, pi, z, _pw in val_dl:
                f = f.to(device)
                t_logits, t_val = teacher(f)
                t_soft = F.softmax(t_logits, dim=-1)
                s_logits, s_val = student(f)
                vp += soft_ce(s_logits, t_soft).item()
                vv += F.mse_loss(s_val, t_val).item()
                nb += 1
            combined = vp / nb + args.value_weight * vv / nb
            marker = ""
            if combined < best_val:
                best_val = combined
                best_state = {k: v.detach().clone().contiguous() for k, v in student.state_dict().items()}
                marker = " *"
            print(f"epoch {epoch:>3}/{args.epochs}  fid_policy {vp/nb:.4f}  fid_value {vv/nb:.4f}{marker}")

    cpu_state = {k: v.cpu().detach().contiguous() for k, v in best_state.items()}
    save_file(cpu_state, args.out)
    print(f"saved {args.out}  ({sum(p.numel() for p in student.parameters())} params)")


if __name__ == "__main__":
    main()
