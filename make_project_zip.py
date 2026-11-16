"""Build project_zip/: self-contained report bundle (main.tex, ref.bib, rlreport.cls, every referenced figure physically copied).
Command: python make_project_zip.py"""
import re, shutil
from pathlib import Path

root = Path(__file__).resolve().parent; z = root / "project_zip"; (z / "figures").mkdir(parents=True, exist_ok=True)
tex = (root / "main.tex").read_text()
figs = sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)))
for old in (z / "figures").glob("*.png"):      # bundle holds exactly the figures main.tex references
    if old.name not in figs: old.unlink()
for f in figs:
    shutil.copy2(root / "figures" / f, z / "figures" / f)
for f in ("main.tex", "ref.bib", "rlreport.cls"):
    shutil.copy2(root / f, z / f)
missing = [f for f in figs if not (z / "figures" / f).exists()]
print(f"copied {len(figs)} figures; missing: {missing}")
