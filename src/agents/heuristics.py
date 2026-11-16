"""Non-learning reference policies (random baseline + two transparent heuristics). All use the same action interface."""
import numpy as np


class RandomAgent:
    name = "random"
    def __init__(self, seed=0): self.rng = np.random.default_rng(seed)
    def act(self, obs):
        cand = np.where(obs["mask"][:-1])[0]
        return int(self.rng.choice(cand))      # never STOP, never an ineligible row


class ShortestPredictedFirst:
    name = "spf_heuristic"
    def act(self, obs):
        x = obs["x"]; m = obs["mask"][:-1]
        return int(np.argmin(np.where(m, x[:, 1], np.inf)))


class HighestSofaFirst:
    name = "sofa_heuristic"
    def act(self, obs):
        x = obs["x"]; m = obs["mask"][:-1]
        return int(np.argmax(np.where(m, x[:, 0], -np.inf)))
