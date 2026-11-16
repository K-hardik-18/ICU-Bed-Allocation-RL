import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DPI = 400
COL = 3.45      # single-column width (in) for rlreport
WIDE = 7.1      # full-width figure (in)
PAL = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00", purple="#CC79A7", sky="#56B4E9", grey="#666666", black="#222222")
ALGO_COL = dict(random=PAL["grey"], dqn=PAL["blue"], ddqn=PAL["orange"], ppo=PAL["green"], spf_heuristic=PAL["purple"], sofa_heuristic=PAL["red"])
ALGO_LBL = dict(random="Random", dqn="DQN", ddqn="DDQN", ppo="PPO", spf_heuristic="Shortest-pred.-first", sofa_heuristic="Highest-SOFA-first")

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "figure.dpi": 100, "savefig.dpi": DPI, "font.family": "DejaVu Sans"})


def save(fig, name, outdir=None):
    from ..common import FIG
    p = (outdir or FIG) / name
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return p


def compose(rows, name, width=WIDE, labels=True, gap=0.06, label_h=0.16):
    """Tile existing PNG figures into one multi-panel figure. rows: list of lists of figure file names.
    Each row is scaled to the full width; panels are labelled (a), (b), ... ."""
    import matplotlib.image as mpimg
    from ..common import FIG
    imgs = [[mpimg.imread(FIG / f) for f in r] for r in rows]
    heights = []
    for r in imgs:
        aspect = sum(im.shape[1] / im.shape[0] for im in r)
        heights.append((width - gap * (len(r) - 1)) / aspect)
    H = sum(heights) + len(rows) * (label_h if labels else 0) + gap * (len(rows) - 1)
    fig = plt.figure(figsize=(width, H)); y = H; k = 0
    for r, h in zip(imgs, heights):
        y -= (label_h if labels else 0) + h; x = 0.0
        for im in r:
            w = h * im.shape[1] / im.shape[0]
            ax = fig.add_axes([x / width, y / H, w / width, h / H]); ax.imshow(im); ax.axis("off")
            if labels:
                fig.text((x + 0.02) / width, (y + h + 0.02) / H, f"({chr(97 + k)})", fontsize=9, fontweight="bold", va="bottom")
            x += w + gap; k += 1
        y -= gap
    return save(fig, name)
