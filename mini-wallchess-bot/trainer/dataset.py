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

    def __init__(self, path: str):
        feats, pols, vals = [], [], []
        with open(path) as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                f = rec["f"]
                if len(f) != FEATURE_LEN:
                    raise ValueError(
                        f"{path}:{line_no}: feature len {len(f)} != {FEATURE_LEN}"
                    )
                pol = torch.zeros(ACTION_COUNT, dtype=torch.float32)
                for idx, prob in rec["pi"]:
                    pol[int(idx)] = float(prob)
                feats.append(torch.tensor(f, dtype=torch.float32))
                pols.append(pol)
                vals.append(float(rec["z"]))

        if not feats:
            raise ValueError(f"{path}: no samples")
        self.feats = torch.stack(feats)
        self.pols = torch.stack(pols)
        self.vals = torch.tensor(vals, dtype=torch.float32)

    def __len__(self):
        return len(self.vals)

    def __getitem__(self, i):
        return self.feats[i], self.pols[i], self.vals[i]
