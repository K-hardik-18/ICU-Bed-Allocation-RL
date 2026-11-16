"""PPO (clipped surrogate) with masked categorical policy over {ADMIT(i)} U {STOP}; duration-aware GAE."""
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .networks import SetNet, collate

NEG = -1e9


class PPOAgent:
    name = "ppo"

    def __init__(self, cfg, seed=0):
        d = cfg["train"]["ppo"]; self.d = d; self.gamma = cfg["env"]["gamma"]; self.tau0 = cfg["env"]["tau0_h"]
        torch.manual_seed(seed); self.nrng = np.random.default_rng(seed)
        self.net = SetNet(d["hidden"], critic=True)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=d["lr"])

    @torch.no_grad()
    def act(self, obs, explore=False):
        x, st, rm, am, fr = collate([obs]); n = obs["n"]
        q, v = self.net(x, st, rm, fr)
        logits = q.masked_fill(~am, NEG)
        if explore:
            dist = Categorical(logits=logits); a = int(dist.sample()); lp = float(dist.log_prob(torch.tensor([a])))
            self._last = (lp, float(v))
        else:
            a = int(logits.argmax(1))
        return n if a == x.shape[1] else a

    def last(self): return self._last

    def update(self, traj):
        """traj: list of episodes; each a list of dicts(o, a, lp, v, r, dt, done)."""
        d = self.d; O, A, LP, ADV, RET = [], [], [], [], []
        for ep in traj:
            adv = 0.0
            for i in range(len(ep) - 1, -1, -1):
                s = ep[i]; g = self.gamma ** (s["dt"] / self.tau0)
                nv = 0.0 if s["done"] else ep[i + 1]["v"]
                delta = s["r"] + g * nv - s["v"]
                adv = delta + (0.0 if s["done"] else g * d["gae_lambda"] * adv)
                s["adv"] = adv; s["ret"] = adv + s["v"]
            for s in ep:
                O.append(s["o"]); A.append(s["a"]); LP.append(s["lp"]); ADV.append(s["adv"]); RET.append(s["ret"])
        ADV = torch.tensor(ADV, dtype=torch.float32); RET = torch.tensor(RET, dtype=torch.float32); LP = torch.tensor(LP, dtype=torch.float32)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        n = len(O); stats = []
        for _ in range(d["epochs"]):
            perm = self.nrng.permutation(n)
            for i in range(0, n, d["minibatch"]):
                idx = perm[i:i + d["minibatch"]]
                x, st, rm, am, fr, at = collate([O[j] for j in idx], [A[j] for j in idx])
                q, v = self.net(x, st, rm, fr)
                dist = Categorical(logits=q.masked_fill(~am, NEG))
                lp = dist.log_prob(at); ratio = torch.exp(lp - LP[idx])
                adv = ADV[idx]
                pl = -torch.min(ratio * adv, torch.clamp(ratio, 1 - d["clip"], 1 + d["clip"]) * adv).mean()
                vl = F.mse_loss(v, RET[idx]); ent = dist.entropy().mean()
                loss = pl + d["vf"] * vl - d["ent"] * ent
                self.opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5); self.opt.step()
                stats.append((pl.item(), vl.item(), ent.item()))
        return np.mean(stats, axis=0)

    def save(self, path): torch.save(self.net.state_dict(), path)
    def load(self, path): self.net.load_state_dict(torch.load(path))
