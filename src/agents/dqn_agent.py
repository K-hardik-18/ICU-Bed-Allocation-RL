"""DQN and Double-DQN with action masking and elapsed-time-aware discounting gamma^(dt/tau0)."""
import random
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F

from .networks import SetNet, collate

NEG = -1e9


class DQNAgent:
    def __init__(self, cfg, double=False, seed=0):
        d = cfg["train"]["dqn"]; self.double = double; self.name = "ddqn" if double else "dqn"
        self.gamma = cfg["env"]["gamma"]; self.tau0 = cfg["env"]["tau0_h"]
        torch.manual_seed(seed); self.rng = random.Random(seed); self.nrng = np.random.default_rng(seed)
        self.q = SetNet(d["hidden"]); self.tq = SetNet(d["hidden"]); self.tq.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=d["lr"])
        self.buf = deque(maxlen=d["buffer"]); self.d = d; self.updates = 0

    @torch.no_grad()
    def act(self, obs, eps=0.0):
        mask = obs["mask"]
        if self.nrng.random() < eps:
            n = obs["n"]   # exploratory action: STOP only with small probability, otherwise a uniformly random eligible patient
            if self.nrng.random() < self.d.get("explore_stop_prob", 0.05): return n
            return int(self.nrng.choice(np.where(mask[:-1])[0]))
        x, st, rm, am, fr = collate([obs])
        q, _ = self.q(x, st, rm, fr)
        q = q.masked_fill(~am, NEG)
        a = int(q.argmax(1)); n = obs["n"]
        return n if a == x.shape[1] else a

    def store(self, o, a, r, o2, done, dt):
        self.buf.append((o, a, r, o2, done, dt))

    def update(self):
        d = self.d
        if len(self.buf) < max(d["batch"], d["warmup"]): return None
        batch = self.rng.sample(self.buf, d["batch"])
        o, a, r, o2, dn, dt = zip(*batch)
        x, st, rm, am, fr, at = collate(list(o), a)
        x2, st2, rm2, am2, fr2 = collate(list(o2))
        r = torch.tensor(r, dtype=torch.float32); dn = torch.tensor(dn, dtype=torch.float32)
        disc = self.gamma ** (torch.tensor(dt, dtype=torch.float32) / self.tau0)
        q, _ = self.q(x, st, rm, fr)
        qsa = q.gather(1, at[:, None]).squeeze(1)
        with torch.no_grad():
            qt, _ = self.tq(x2, st2, rm2, fr2)
            if self.double:   # online net selects, target net evaluates
                qo, _ = self.q(x2, st2, rm2, fr2)
                a2 = qo.masked_fill(~am2, NEG).argmax(1, keepdim=True)
                nxt = qt.gather(1, a2).squeeze(1)
            else:
                nxt = qt.masked_fill(~am2, NEG).max(1).values
            nxt = torch.where(dn > 0, torch.zeros_like(nxt), nxt)
            y = r + disc * nxt
        loss = F.smooth_l1_loss(qsa, y)
        self.opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.q.parameters(), 10.0); self.opt.step()
        self.updates += 1
        if self.updates % d["target_update"] == 0: self.tq.load_state_dict(self.q.state_dict())
        return loss.item()

    def save(self, path): torch.save(self.q.state_dict(), path)
    def load(self, path): self.q.load_state_dict(torch.load(path))
