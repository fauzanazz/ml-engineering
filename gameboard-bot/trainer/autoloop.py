"""Autonomous Wall Chess CNN self-play loop.

Default mode loops forever:
  heuristic/search data -> CNN policy+value training -> MCTS arena gate -> promote -> repeat.

State lives under --run-dir and is written atomically so a killed process resumes
from the last completed stage. Per-epoch training checkpoints resume mid-train.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
TRAINER = ROOT / "trainer"
RESULT_RE = re.compile(r"^RESULT (\{.*\})$", re.MULTILINE)


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(default)
    return json.loads(path.read_text())


def run(cmd: list[str], cwd: Path, log: Path, env: dict[str, str] | None = None) -> str:
    log.parent.mkdir(parents=True, exist_ok=True)
    merged = os.environ.copy()
    if env:
        merged.update(env)
    with log.open("a") as fh:
        fh.write("\n$ " + " ".join(cmd) + "\n")
        fh.flush()
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=merged,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        fh.write(proc.stdout)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout)
    return proc.stdout


def split_count(total: int, shards: int) -> list[int]:
    shards = max(1, min(max(1, shards), max(1, total)))
    base, extra = divmod(total, shards)
    return [base + (1 if i < extra else 0) for i in range(shards) if base + (1 if i < extra else 0) > 0]


def run_parallel(jobs: list[Any], workers: int) -> None:
    if not jobs:
        return
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as pool:
        for _ in pool.map(lambda job: job(), jobs):
            pass


def promote(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dst)


def stage_done(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def run_to_file(
    cmd: list[str],
    out_arg: int,
    cwd: Path,
    out: Path,
    log: Path,
    env: dict[str, str] | None = None,
) -> None:
    if stage_done(out):
        return
    tmp = out.with_suffix(out.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    patched = list(cmd)
    patched[out_arg] = str(tmp)
    run(patched, cwd=cwd, log=log, env=env)
    tmp.replace(out)


def build_bins(args: argparse.Namespace, state: dict[str, Any]) -> None:
    net_arena = CORE / "target" / "release" / "net_arena"
    if state.get("built") and net_arena.exists():
        return
    run(
        [
            "cargo",
            "build",
            "--release",
            "--features",
            "net",
            "--bin",
            "search_data",
            "--bin",
            "expert_data",
            "--bin",
            "selfplay_data",
            "--bin",
            "net_search_arena",
            "--bin",
            "net_arena",
        ],
        cwd=CORE,
        log=args.run_dir / "logs" / "build.log",
    )
    state["built"] = True
    atomic_json(args.state_file, state)


def generate_data(args: argparse.Namespace, state: dict[str, Any], iter_dir: Path) -> list[Path]:
    data_dir = iter_dir / "data"
    log_dir = iter_dir / "logs"
    data_dir.mkdir(parents=True, exist_ok=True)
    search_bin = CORE / "target" / "release" / "search_data"
    expert_bin = CORE / "target" / "release" / "expert_data"
    selfplay_bin = CORE / "target" / "release" / "selfplay_data"
    seed_base = 0xC11CE000 + int(state["iteration"]) * 100
    variants = [
        {},
        {"WC_EVAL_W_PATH": "45", "WC_EVAL_W_WALL": "90"},
        {"WC_EVAL_W_PATH": "55", "WC_EVAL_W_WALL": "115", "WC_EVAL_W_LEAD_QUANT": "15"},
    ]
    outputs: list[Path] = []
    jobs: list[Any] = []

    searches = [
        ("opening", max(1, args.search_samples // 3), args.search_depth, 8, variants[0]),
        ("midgame", max(1, args.search_samples // 3), args.search_depth, 36, variants[1]),
        ("late", max(1, args.search_samples - 2 * (args.search_samples // 3)), max(1, args.search_depth - 1), 96, variants[2]),
    ]
    for i, (name, samples, depth, random_plies, env) in enumerate(searches):
        out = data_dir / f"search-{name}.jsonl"
        outputs.append(out)
        jobs.append(
            lambda out=out, name=name, samples=samples, depth=depth, random_plies=random_plies, env=env, i=i: run_to_file(
                [str(search_bin), str(samples), str(depth), str(out), str(random_plies), "350", "70", "16"],
                out_arg=3,
                cwd=CORE,
                out=out,
                log=log_dir / f"data-search-{name}.log",
                env={**env, "SEARCH_DATA_SEED": str(seed_base + i)},
            )
        )

    expert_out = data_dir / "expert-trajectories.jsonl"
    outputs.append(expert_out)
    jobs.append(
        lambda: run_to_file(
            [str(expert_bin), str(args.expert_games), str(args.search_depth), str(expert_out), str(args.max_plies)],
            cwd=CORE,
            out_arg=3,
            out=expert_out,
            log=log_dir / "data-expert.log",
            env={"EXPERT_DATA_SEED": str(seed_base + 50), **variants[int(state["iteration"]) % len(variants)]},
        )
    )

    best = previous_candidate(args, state)
    selfplay_counts = split_count(args.selfplay_games, args.selfplay_shards)
    for shard, games in enumerate(selfplay_counts):
        out = data_dir / ("selfplay.jsonl" if len(selfplay_counts) == 1 else f"selfplay-{shard:02d}.jsonl")
        outputs.append(out)

        def selfplay_job(out=out, games=games, shard=shard) -> None:
            cmd = [str(selfplay_bin), str(games), str(args.selfplay_sims), str(out)]
            if best:
                cmd.append(str(best))
            run_to_file(
                cmd,
                out_arg=3,
                cwd=CORE,
                out=out,
                log=log_dir / f"data-selfplay-{shard:02d}.log",
                env={"SELFPLAY_SEED": str(seed_base + 75 + shard)},
            )

        jobs.append(selfplay_job)

    run_parallel(jobs, args.data_workers)
    return outputs


def previous_candidate(args: argparse.Namespace, state: dict[str, Any]) -> Path | None:
    history = state.get("history", [])
    if history:
        path = Path(history[-1]["candidate"])
        if stage_done(path):
            return path
    return current_best(args)


def replay_data(args: argparse.Namespace, state: dict[str, Any], current: list[Path]) -> list[Path]:
    out = list(current)
    patterns = ["search-*.jsonl", "expert-*.jsonl", "selfplay*.jsonl"]
    cur = int(state["iteration"])
    start = max(0, cur - max(0, args.replay_iters))
    for i in range(start, cur):
        data_dir = args.run_dir / f"iter-{i:06d}" / "data"
        for pattern in patterns:
            for path in sorted(data_dir.glob(pattern)):
                if stage_done(path):
                    out.append(path)
    return out


def train_candidate(args: argparse.Namespace, state: dict[str, Any], iter_dir: Path, data: list[Path]) -> Path:
    out = iter_dir / "candidate.safetensors"
    ckpt = iter_dir / "train.ckpt.pt"
    if stage_done(out):
        return out
    cmd = [
        "uv",
        "run",
        "python",
        "train.py",
        "--game",
        "wallchess",
        "--arch",
        "cnn",
        "--cnn-channels",
        str(args.cnn_channels),
        "--data",
        *map(str, data),
        "--out",
        str(out),
        "--epochs",
        str(args.epochs),
        "--batch",
        str(args.batch),
        "--lr",
        str(args.lr),
        "--policy-loss-weight",
        str(args.policy_loss_weight),
        "--checkpoint",
        str(ckpt),
    ]
    init = previous_candidate(args, state)
    if init:
        cmd.extend(["--init", str(init)])
    run(cmd, cwd=TRAINER, log=iter_dir / "logs" / "train.log")
    return out


def gate_candidate(args: argparse.Namespace, candidate: Path, iter_dir: Path) -> dict[str, Any]:
    result_path = iter_dir / "gate.json"
    if result_path.exists():
        return json.loads(result_path.read_text())

    def run_shard(item: tuple[int, int]) -> dict[str, Any]:
        shard, games = item
        stdout = run(
            [
                str(CORE / "target" / "release" / "net_arena"),
                str(candidate),
                str(games),
                str(args.selfplay_sims),
                str(args.arena_depth),
                str(args.max_plies),
                str(args.opening_plies),
                "0",
                str(args.arena_threshold),
            ],
            cwd=CORE,
            log=iter_dir / "logs" / f"gate-{shard:02d}.log",
            env={"NET_ARENA_SEED": str(0xA6E10000 + int(iter_dir.name.split("-")[-1]) * 100 + shard)},
        )
        m = RESULT_RE.search(stdout)
        if not m:
            raise RuntimeError("net_arena did not print RESULT JSON")
        return json.loads(m.group(1))

    counts = split_count(args.arena_games, args.arena_shards)
    with ThreadPoolExecutor(max_workers=max(1, min(args.arena_shards, len(counts)))) as pool:
        shard_results = list(pool.map(run_shard, list(enumerate(counts))))

    wins = sum(int(r["candidate_wins"]) for r in shard_results)
    draws = sum(int(r["draws"]) for r in shard_results)
    games = sum(int(r["games"]) for r in shard_results)
    opponent_wins = sum(int(r["opponent_wins"]) for r in shard_results)
    candidate_score = (wins + 0.5 * draws) / max(1, games)
    result = {
        "candidate_score": candidate_score,
        "candidate_wins": wins,
        "draws": draws,
        "games": games,
        "opponent": "heuristic",
        "opponent_wins": opponent_wins,
        "promote": candidate_score >= args.arena_threshold,
        "threshold": args.arena_threshold,
        "shards": len(shard_results),
    }
    atomic_json(result_path, result)
    return result


def current_best(args: argparse.Namespace) -> Path | None:
    path = args.run_dir / "best" / "current.safetensors"
    return path if stage_done(path) else None


def complete_iteration(args: argparse.Namespace, state: dict[str, Any], iter_dir: Path, candidate: Path, gate: dict[str, Any]) -> None:
    promoted = bool(gate["promote"])
    if promoted:
        best = args.run_dir / "best" / "current.safetensors"
        promote(candidate, best)
        atomic_json(
            args.run_dir / "best" / "manifest.json",
            {
                "iteration": state["iteration"],
                "model": str(best),
                "source": str(candidate),
                "gate": gate,
                "arch": "cnn-value",
                "feature_len": 462,
            },
        )
    state.setdefault("history", []).append(
        {
            "iteration": state["iteration"],
            "candidate": str(candidate),
            "gate": gate,
            "promoted": promoted,
        }
    )
    state["iteration"] = int(state["iteration"]) + 1
    state["stage"] = "build"
    atomic_json(args.state_file, state)


def one_iteration(args: argparse.Namespace, state: dict[str, Any]) -> None:
    build_bins(args, state)
    iter_dir = args.run_dir / f"iter-{int(state['iteration']):06d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    state["stage"] = "data"
    atomic_json(args.state_file, state)
    data = generate_data(args, state, iter_dir)
    state["stage"] = "train"
    atomic_json(args.state_file, state)
    candidate = train_candidate(args, state, iter_dir, replay_data(args, state, data))
    state["stage"] = "gate"
    atomic_json(args.state_file, state)
    gate = gate_candidate(args, candidate, iter_dir)
    state["stage"] = "promote"
    atomic_json(args.state_file, state)
    complete_iteration(args, state, iter_dir, candidate, gate)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "wallchess-cnn-loop")
    ap.add_argument("--search-samples", type=int, default=3000)
    ap.add_argument("--search-depth", type=int, default=3)
    ap.add_argument("--expert-games", type=int, default=40)
    ap.add_argument("--selfplay-games", type=int, default=30)
    ap.add_argument("--selfplay-sims", type=int, default=120)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--data-workers", type=int, default=8)
    ap.add_argument("--selfplay-shards", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--policy-loss-weight", type=float, default=0.25,
                    help="policy loss weight; 0 is value-only")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cnn-channels", type=int, default=32)
    ap.add_argument("--arena-games", type=int, default=80)
    ap.add_argument("--arena-shards", type=int, default=4)
    ap.add_argument("--arena-depth", type=int, default=2)
    ap.add_argument("--arena-threshold", type=float, default=0.60)
    ap.add_argument("--max-plies", type=int, default=140)
    ap.add_argument("--opening-plies", type=int, default=4)
    ap.add_argument("--sleep-seconds", type=float, default=0.0)
    ap.add_argument("--once", action="store_true", help="smoke mode: run one iteration, then exit")
    ap.add_argument("--iterations", type=int, default=None,
                    help="run this many iterations, then exit")
    ap.add_argument("--replay-iters", type=int, default=8,
                    help="include this many previous iterations of data when training")
    args = ap.parse_args()
    args.run_dir = args.run_dir.resolve()
    args.state_file = args.run_dir / "state.json"
    state = load_json(args.state_file, {"iteration": 0, "stage": "build", "built": False, "history": []})

    done = 0
    while True:
        one_iteration(args, state)
        done += 1
        if args.once or (args.iterations is not None and done >= args.iterations):
            break
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
