"""Shared helpers: config loading and project paths."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "synthetic"
CKPT = ROOT / "checkpoints"
RES = ROOT / "results"
FIG = ROOT / "figures"


def load_config(path=None):
    with open(path or ROOT / "config.yaml") as f:
        return yaml.safe_load(f)
