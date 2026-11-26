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


def report_stability(out_dir: Path) -> dict:
    """Rolling PCA of 1-10 year monthly changes, 60-month windows: each window's shares; how
    closely each window's components match the full-sample ones (absolute cosine) and the overlap
    of the three-shape set (mean squared cosine, which ignores order and sign); out of sample, how
    much of the next 12 months' movement the window's three shapes capture, against the best any
    three could (those months' own first three) and a parallel shift alone; and the 10-year
    yield's kink against its neighbours (10y minus the average of 9y and 11y)."""
    from yc import data, pca

    w = data.yields()
    d = changes(w, CORE)
    X = d.to_numpy()
    L = pca.pca(X)["loadings"][:, :3]
    r = pca.rolling(X, list(d.index))
    df = pd.DataFrame({
        "end": [x["end"] for x in r],
        **{NAMES[k]: [x["share"][k] for x in r] for k in range(3)},
        **{f"match {NAMES[k]}": [abs(x["loadings"][:, k] @ L[:, k]) for x in r]
           for k in range(3)},
        "overlap": [np.linalg.norm(x["loadings"].T @ L) ** 2 / 3 for x in r],
        "captured": [x.get("next_captured", np.nan) for x in r],
        "best": [x.get("next_best", np.nan) for x in r]}).set_index("end")
    par = np.ones((len(CORE), 1)) / np.sqrt(len(CORE))
    df["parallel"] = [pca.captured(X[e:e + 12], par) if e + 12 <= len(X) else np.nan
                      for e in range(60, len(X) + 1)]
    res = {"windows": f"{len(df)} windows of 60 months ending "
                      f"{df.index[0]:%b %Y}-{df.index[-1]:%b %Y}"}
    for nm in NAMES:
        res[f"  {nm} share"] = (f"median {df[nm].median():.1%}, range {df[nm].min():.1%}-"
                                f"{df[nm].max():.1%}")
    for nm in NAMES:
        m = df[f"match {nm}"]
        res[f"  {nm} matches the full-sample shape (|cosine|)"] = (
            f"median {m.median():.2f}, lowest {m.min():.2f} ({m.idxmin():%b %Y}); below 0.9 in "
            f"{(m < 0.9).mean():.0%} of windows")
    ov = df["overlap"]
    res["  three-shape set overlap"] = (f"median {ov.median():.2f}, lowest {ov.min():.2f} "
                                         f"({ov.idxmin():%b %Y})")
    ok = df["captured"].notna()
    res["next 12 months' movement captured"] = (
        f"{ok.sum()} windows: previous window's three shapes median "
        f"{df.loc[ok, 'captured'].median():.1%} (lowest {df.loc[ok, 'captured'].min():.1%}); "
        f"best possible three {df.loc[ok, 'best'].median():.1%}; parallel shift alone "
        f"{df.loc[ok, 'parallel'].median():.1%}")
    lw = [x for x in r if x["end"] == pd.Timestamp("2018-12-31")][0]
    res["  window to Dec 2018, second component loadings 1-10y"] = " ".join(
        f"{v:+.2f}" for v in lw["loadings"][:, 1])
    kink = (w[10] - (w[9] + w[11]) / 2) * 100
    dk = kink.diff().dropna()
    mid = (dk.index.year >= 2014) & (dk.index.year <= 2020)
    res["10-year kink against 9y and 11y, monthly change"] = (
        f"standard deviation {dk[mid].std():.1f} bp in 2014-2020 against {dk[~mid].std():.1f} bp in"
        " other years; largest " + ", ".join(
            f"{t:%b %Y} {v:+.0f} bp" for t, v in dk.loc[dk.abs().nlargest(3).index].items()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    for nm, col in zip(NAMES, (plots.SERIES[0], plots.SERIES[1], plots.SERIES[2]), strict=True):
        ax.plot(df.index, 100 * df[nm], color=col, lw=2, label=nm)
    ax.set_ylabel("share of movement in the 5 years\nto that month, %")
    ax.set_yscale("log")
    ax.set_yticks([1, 3, 10, 30, 100])
    ax.set_yticklabels(["1%", "3%", "10%", "30%", "100%"])
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title("Each shape's share, rolling 5 years", fontsize=10.5, loc="left")
    ax.set_xlabel("end of the 5-year window")
    ax = axes[1]
    for c, lab, col in (("match level", "level matches full-sample shape", plots.SERIES[0]),
                        ("match slope", "slope matches", plots.SERIES[1]),
                        ("captured", "next year captured by the 3 shapes", plots.TEXT_2)):
        ax.plot(df.index, df[c], color=col, lw=2 if c != "captured" else 1.4, label=lab,
                ls="-" if c != "captured" else "--")
    ax.axvspan(pd.Timestamp("2014-01-01"), pd.Timestamp("2020-12-31"), color=plots.GRID, lw=0)
    ax.text(pd.Timestamp("2014-03-01"), 0.03, "10-year yield\njumping on its own", fontsize=8.5,
            color=plots.TEXT_2)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("end of the 5-year window")
    ax.set_ylabel("similarity / share (1 = identical / all)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 0.12))
    ax.set_title("Level never changed; slope gave way for a while", fontsize=10.5, loc="left")
    fig.suptitle("The level shape held for 25 years; old shapes still captured most of the next "
                 "year",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    plots.save(fig, out_dir, "stability")
    plt.close(fig)
    return res


REPORTS: dict[str, Callable[[Path], dict]] = {
    "pca": report_pca,
    "stability": report_stability,
}
