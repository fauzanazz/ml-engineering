"""Load self-play JSONL (emitted by `selfplay_data`) into dense tensors."""

import json

import torch
from torch.utils.data import Dataset

from encoding import ACTION_COUNT, FEATURE_LEN


class SelfPlayDataset(Dataset):
    """Each record: {"z": float, "f": [FEATURE_LEN floats], "pi": [[idx, prob], ...]}.

    `f` is the me-frame feature vector, `pi` the visit-count policy (sparse), `z`
    the game outcome from that state's side-to-move POV.
    """

    def __init__(self, path: str | list[str], policy_weight: float = 1.0):
        """policy_weight scales this source's policy loss per sample. Set 0.0 for
        value-only data (self-play z targets without trusting its weak policy)."""
        paths = [path] if isinstance(path, str) else path
        feats, pols, vals = [], [], []
        for one_path in paths:
            with open(one_path) as fh:
                for line_no, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    f = rec["f"]
                    if len(f) != FEATURE_LEN:
                        raise ValueError(
                            f"{one_path}:{line_no}: feature len {len(f)} != {FEATURE_LEN}"
                        )
                    pol = torch.zeros(ACTION_COUNT, dtype=torch.float32)
                    for idx, prob in rec["pi"]:
                        pol[int(idx)] = float(prob)
                    feats.append(torch.tensor(f, dtype=torch.float32))
                    pols.append(pol)
                    vals.append(float(rec["z"]))

        if not feats:
            raise ValueError(f"{paths}: no samples")
        self.feats = torch.stack(feats)
        self.pols = torch.stack(pols)
        self.vals = torch.tensor(vals, dtype=torch.float32)
        self.pw = torch.full((len(vals),), float(policy_weight), dtype=torch.float32)

    def __len__(self):
        return len(self.vals)

    def __getitem__(self, i):
        return self.feats[i], self.pols[i], self.vals[i], self.pw[i]
