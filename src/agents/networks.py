"""Permutation-equivariant set network over a variable number of active patient rows.

Per-row input: [SOFA/24, expected ICU duration, waiting time, hazard] + learned embedding of categorical status.
Shared encoder -> masked mean/max pooling -> hospital context (+ free-bed fraction) -> per-row score and STOP score.
Patient IDs are never inputs. Padding rows are masked out of pooling and of the action distribution.
"""
import numpy as np
import torch
import torch.nn as nn


def collate(obs_list, actions=None):
    B = len(obs_list); N = max(o["n"] for o in obs_list)
    x = np.zeros((B, N, 4), np.float32); st = np.zeros((B, N), np.int64)
    rowm = np.zeros((B, N), bool); am = np.zeros((B, N + 1), bool); free = np.zeros(B, np.float32)
    for b, o in enumerate(obs_list):
        n = o["n"]; x[b, :n] = o["x"]; st[b, :n] = o["status"]; rowm[b, :n] = True
        am[b, :n] = o["mask"][:n]; am[b, N] = o["mask"][n]; free[b] = o["free"]
    out = [torch.from_numpy(a) for a in (x, st, rowm, am, free)]
    if actions is not None:
        a = np.array([N if act == o["n"] else act for act, o in zip(actions, obs_list)])
        out.append(torch.from_numpy(a))
    return out


class SetNet(nn.Module):
    def __init__(self, h=64, critic=False):
        super().__init__()
        self.emb = nn.Embedding(2, 8)
        self.enc = nn.Sequential(nn.Linear(12, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.ctx = nn.Sequential(nn.Linear(2 * h + 1, h), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 1))
        self.stop = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        self.v = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1)) if critic else None

    def forward(self, x, st, rowm, free):
        N = x.shape[1]
        e = self.enc(torch.cat([x, self.emb(st)], -1))
        m = rowm.unsqueeze(-1).float()
        mean = (e * m).sum(1) / m.sum(1).clamp(min=1.0)
        mx = e.masked_fill(~rowm.unsqueeze(-1), -1e9).max(1).values
        c = self.ctx(torch.cat([mean, mx, free[:, None]], -1))
        rs = self.head(torch.cat([e, c[:, None].expand(-1, N, -1)], -1)).squeeze(-1)
        q = torch.cat([rs, self.stop(c)], 1)
        v = self.v(c).squeeze(-1) if self.v is not None else None
        return q, v
