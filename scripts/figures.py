"""Figures from out/audit_report.json: reliability curves, ECE binning sweep,
slice ECEs, plus out/band_table.md.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
R = json.loads((ROOT / "out" / "audit_report.json").read_text())

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # RM, length null, judge

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 10,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _curve(rows, min_n=25):
    # drop sparse bins
    pts = [(r["mean_pred"], r["emp_freq"], r["n"]) for r in rows if r["n"] and r["n"] >= min_n]
    return [p for p, _, _ in pts], [e for _, e, _ in pts], [n for _, _, n in pts]


def fig1():
    fig, (ax, axn) = plt.subplots(
        2, 1, figsize=(6.4, 6.2), height_ratios=[4, 1], sharex=True)
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=INK2, zorder=1)
    ax.annotate("honest line", (0.80, 0.84), rotation=38, color=INK2, fontsize=9)
    x, y, n = _curve(R["reliability"])
    ax.plot(x, y, "-o", lw=2, ms=6, color=BLUE, zorder=3,
            mec=SURFACE, mew=1)                      # 2px surface ring
    ax.annotate("trained RM", (x[-1], y[-1]), xytext=(6, -2),
                textcoords="offset points", color=INK, fontweight="bold")
    if "reliability_length_null" in R:
        xl, yl, _ = _curve(R["reliability_length_null"])
        ax.plot(xl, yl, "-o", lw=2, ms=6, color=ORANGE, zorder=2,
                mec=SURFACE, mew=1)
        ax.annotate("length null", (xl[0], yl[0]), xytext=(-8, 10), ha="right",
                    textcoords="offset points", color=INK)
    if "reliability_judge" in R:
        xj, yj, _ = _curve(R["reliability_judge"], min_n=40)
        ax.plot(xj, yj, "-o", lw=2, ms=6, color=AQUA, zorder=2, mec=SURFACE, mew=1)
        if xj:
            ax.annotate("zero-shot judge", (xj[-1], yj[-1]), xytext=(6, 6),
                        textcoords="offset points", color=INK)
    ax.set_ylabel("what actually happened (empirical frequency)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("When the reward model says p, how often is A preferred?",
                 loc="left", fontweight="bold", color=INK)
    axn.bar(x, n, width=0.075, color=BLUE, alpha=0.5)
    axn.set_ylabel("n", rotation=0, labelpad=12)
    axn.set_xlabel("what the model claimed (predicted probability)")
    fig.tight_layout()
    fig.savefig(ROOT / "out" / "fig1_reliability.png", dpi=300)


def fig2():
    keys = list(R["ece_sensitivity"])
    x = range(len(keys))
    rm = [R["ece_sensitivity"][k] for k in keys]
    nv = [R["ece_sensitivity_length_null"][k] for k in keys]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(x, nv, "o", ms=8, color=ORANGE, mec=SURFACE, mew=1)
    ax.plot(x, rm, "o", ms=8, color=BLUE, mec=SURFACE, mew=1)
    for i, (a, b) in enumerate(zip(rm, nv)):
        # whichever dot is lower gets its label below, the higher one above
        up, dn = (7, -12) if b >= a else (-12, 7)
        ax.annotate(f"{a:.3f}", (i, a), xytext=(0, dn), textcoords="offset points",
                    ha="center", fontsize=8, color=BLUE)
        ax.annotate(f"{b:.3f}", (i, b), xytext=(0, up), textcoords="offset points",
                    ha="center", fontsize=8, color=ORANGE)
    ax.text(len(keys) - 0.6, rm[-1] - 0.011, "trained RM", color=BLUE, fontsize=9,
            ha="right", va="top", fontweight="bold")
    ax.text(len(keys) - 0.6, nv[-1] + 0.011, "length null", color=ORANGE, fontsize=9,
            ha="right", va="bottom", fontweight="bold")
    ax.set_xticks(list(x),
                  [k.replace("width_", "eq-width\n").replace("mass_", "eq-mass\n")
                   for k in keys])
    ax.set_ylabel("ECE")
    ax.set_ylim(-0.005, max(nv) * 1.35)
    ax.set_title("ECE under different binnings",
                 loc="left", fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(ROOT / "out" / "fig2_ece_binning.png", dpi=300)


def fig3():
    panels = [p for p in ("slice_length_gap", "slice_domain", "slice_strength", "slice_agreement") if p in R]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.6), sharey=True)
    axes = [axes] if len(panels) == 1 else list(axes)
    for ax, key in zip(axes, panels):
        rows = R[key]["rows"]
        names = [r["slice"] for r in rows]
        vals = [r["ece_10w"] for r in rows]
        bars = ax.bar(range(len(rows)), vals, width=0.62, color=BLUE)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8, color=INK2)
        ax.set_xticks(range(len(rows)), names, rotation=20, ha="right", fontsize=8)
        ax.set_title(R[key]["slice_by"], loc="left", fontsize=10, color=INK)
    axes[0].set_ylabel("ECE (10-bin)")
    fig.suptitle("Where the overconfidence lives", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(ROOT / "out" / "fig3_slices.png", dpi=300)


def table4():
    lines = ["| run | T | n | accuracy | brier | log loss | ECE(10) |",
             "|---|---|---|---|---|---|---|"]
    for b in R["band"]:
        lines.append(f"| {b['run']} | {b['T']} | {b['n']} | {b['accuracy']:.3f} "
                     f"| {b['brier']:.4f} | {b['log_loss']:.4f} | {b['ece_10w']:.4f} |")
    h = R["headline"]
    lines += ["", f"Worst of {h['n_runs']} runs at T=1: Brier {h['worst_brier']:.4f}, ECE {h['worst_ece_10w']:.4f} "
                  f"(best Brier {h['best_brier']:.4f}). Temperature rows are a sensitivity sweep."]
    (ROOT / "out" / "band_table.md").write_text("\n".join(lines))


if __name__ == "__main__":
    fig1(); fig2(); fig3(); table4()
    print("wrote fig1-3 PNGs + band_table.md -> out/")
