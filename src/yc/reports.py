"""README findings, one function per topic: `yc report <topic>` writes charts to out/."""

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from yc import plots

CORE = list(range(1, 11))
LONG = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]
NAMES = ["level", "slope", "curvature"]


def changes(w: pd.DataFrame, cols: list[int]) -> pd.DataFrame:
    """Month-on-month changes in basis points over the longest run of consecutive months with
    every maturity in `cols`."""
    x = w[cols]
    ok = x.notna().all(axis=1)
    run = (~ok).cumsum()
    best = run[ok].value_counts().idxmax()
    x = x[ok & (run == best)]
    return (x.diff().dropna() * 100)


def report_pca(out_dir: Path) -> dict:
    """PCA of monthly changes in RBI's month-end G-sec yields: 1-10 years over the whole sample,
    by half, and 1-30 years over the months all ten long-set maturities exist; how much each
    component explains, its shape, and how well three components rebuild each month's move."""
    from yc import data, pca

    w = data.yields()
    d = changes(w, CORE)
    p = pca.pca(d.to_numpy())
    res = {"1-10 years": f"{len(d)} monthly changes, {d.index[0]:%b %Y}-{d.index[-1]:%b %Y}"}
    res["  volatility, bp a month"] = ", ".join(f"{m}y {v:.0f}" for m, v in d.std().items())
    res["  share of variance"] = ", ".join(f"PC{i + 1} {s:.1%}" for i, s in
                                           enumerate(p["share"][:5])) + (
        f"; first three {p['share'][:3].sum():.1%}")
    for k, nm in enumerate(NAMES):
        res[f"  {nm} loadings"] = " ".join(f"{v:+.2f}" for v in p["loadings"][:, k])
    rec = p["scores"][:, :3] @ p["loadings"][:, :3].T + p["mean"]
    err = d.to_numpy() - rec
    res["  three components rebuild each month's move"] = (
        f"error {np.sqrt((err ** 2).mean()):.1f} bp root-mean-square, largest "
        f"{np.abs(err).max():.0f} bp, against "
        f"{np.sqrt(((d - d.mean()) ** 2).to_numpy().mean()):.1f} bp of raw moves")
    halves = {}
    for lab, a, b in (("1996-2009", None, "2009"), ("2010-2026", "2010", None)):
        q = pca.pca(d.loc[a:b].to_numpy())
        halves[lab] = q["share"]
        res[f"  {lab} ({len(d.loc[a:b])} months)"] = ", ".join(
            f"{nm} {s:.1%}" for nm, s in zip(NAMES, q["share"][:3], strict=True))
    dl = changes(w, LONG)
    q = pca.pca(dl.to_numpy())
    res["1-30 years"] = (f"{len(dl)} monthly changes, {dl.index[0]:%b %Y}-{dl.index[-1]:%b %Y}; "
                         + ", ".join(f"PC{i + 1} {s:.1%}" for i, s in enumerate(q["share"][:4])))
    for k in range(3):
        res[f"  PC{k + 1} loadings ({', '.join(f'{m}y' for m in LONG)})"] = " ".join(
            f"{v:+.2f}" for v in q["loadings"][:, k])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    cols = (plots.SERIES[0], plots.SERIES[1], plots.SERIES[2])
    for k, (nm, col) in enumerate(zip(NAMES, cols, strict=True)):
        ax.plot(CORE, p["loadings"][:, k], color=col, lw=2.2, marker="o", ms=5,
                label=f"{nm} ({p['share'][k]:.0%} of movement)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.axhline(0, color=plots.TEXT_2, lw=0.8)
    ax.set_xticks(CORE, [f"{m}y" for m in CORE])
    ax.set_xlabel("years to maturity")
    ax.set_ylabel("loading (how much each yield moves\nper unit of the component)")
    ax.set_title(f"The three shapes, 1-10 years, {d.index[0]:%Y}-{d.index[-1]:%Y}", fontsize=10.5,
                 loc="left")
    ax = axes[1]
    sets = (("1-10y, all", p["share"]), ("1-10y, 1996-2009", halves["1996-2009"]),
            ("1-10y, 2010-2026", halves["2010-2026"]),
            (f"1-30y, {dl.index[0]:%Y}-{dl.index[-1]:%Y}", q["share"]))
    x = np.arange(5)
    for i, (lab, sh) in enumerate(sets):
        ax.bar(x + (i - 1.5) * 0.2, 100 * sh[:5], 0.19, label=lab,
               color=(plots.SERIES[0], plots.MUTED, plots.BLUE_RAMP[2], plots.SERIES[3])[i])
    ax.set_xticks(x, ["1st", "2nd", "3rd", "4th", "5th"])
    ax.set_xlabel("component")
    ax.set_ylabel("share of all yield movement, %")
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("How much each component explains", fontsize=10.5, loc="left")
    fig.suptitle("Three shapes explained 95% of Indian yield moves; level alone 83%",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    plots.save(fig, out_dir, "pca")
    plt.close(fig)
    return res


REPORTS: dict[str, Callable[[Path], dict]] = {
    "pca": report_pca,
}
