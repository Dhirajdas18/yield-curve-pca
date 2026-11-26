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


def _shares(share) -> str:
    return ", ".join(f"{nm} {s_:.1%}" for nm, s_ in zip(NAMES, share, strict=False))


def report_levels(out_dir: Path) -> dict:
    """PCA on yield levels against monthly changes over the same months (1-10 years): shares,
    how alike the shapes are, how persistent each series is (lag-one autocorrelation), and what
    PCA on levels does to unrelated series (500 sets of 10 independent random walks of the same
    length, seed 0). Then choices: covariance against correlation PCA, and maturity sets
    (without the 1-year, four maturities, without the 10-year)."""
    from yc import data, pca

    w = data.yields()
    d = changes(w, CORE)
    lv = w.loc[d.index[0] - pd.offsets.MonthEnd(1):d.index[-1], CORE]
    pl, pc = pca.pca(lv.to_numpy() * 100), pca.pca(d.to_numpy())
    res = {"months": f"{len(lv)} levels and {len(d)} changes, "
                     f"{lv.index[0]:%b %Y}-{lv.index[-1]:%b %Y}"}
    res["levels, share"] = _shares(pl["share"])
    res["changes, share"] = _shares(pc["share"])
    res["  shape match, levels against changes (|cosine|)"] = ", ".join(
        f"{nm} {abs(pl['loadings'][:, k] @ pc['loadings'][:, k]):.2f}"
        for k, nm in enumerate(NAMES))
    ar = [np.corrcoef(lv[m].to_numpy()[1:], lv[m].to_numpy()[:-1])[0, 1] for m in CORE]
    arc = [np.corrcoef(d[m].to_numpy()[1:], d[m].to_numpy()[:-1])[0, 1] for m in CORE]
    res["  lag-one autocorrelation"] = (f"levels {min(ar):.3f}-{max(ar):.3f}; changes "
                                        f"{min(arc):+.2f} to {max(arc):+.2f}")
    rng = np.random.default_rng(0)
    rw_l, rw_c = [], []
    for _ in range(500):
        e = rng.normal(0, 1, (len(lv), len(CORE)))
        rw_l.append(pca.pca(e.cumsum(axis=0))["share"][0])
        rw_c.append(pca.pca(e)["share"][0])
    rw_l, rw_c = np.array(rw_l), np.array(rw_c)
    res["independent random walks, first component's share"] = (
        f"on levels median {np.median(rw_l):.1%} (5-95%: {np.percentile(rw_l, 5):.1%}-"
        f"{np.percentile(rw_l, 95):.1%}); on their changes median {np.median(rw_c):.1%}")
    z = (d - d.mean()) / d.std()
    pz = pca.pca(z.to_numpy())
    res["correlation PCA on changes"] = (
        _shares(pz["share"]) + "; level loadings "
        + " ".join(f"{v:.2f}" for v in pz["loadings"][:, 0]))
    res["  covariance PCA level loadings"] = " ".join(f"{v:.2f}" for v in pc["loadings"][:, 0])
    variants = {"1-10 years": CORE, "2-10 years (no 1-year)": CORE[1:],
                "1, 2, 5, 10 years": [1, 2, 5, 10], "1-9 years (no 10-year)": CORE[:-1]}
    vs = {}
    for nm, cols in variants.items():
        dd = changes(w, cols)
        q = pca.pca(dd.to_numpy())
        vs[nm] = q["share"][:3]
        res[f"changes, {nm} ({len(dd)} months)"] = _shares(q["share"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    bins = np.linspace(0, 1, 41)
    ax.hist(rw_c, bins=bins, color=plots.MUTED, label="unrelated series, changes")
    ax.hist(rw_l, bins=bins, color=plots.SERIES[1], alpha=0.85, label="unrelated series, levels")
    for v, lab, col in ((pc["share"][0], "Indian yields, changes", plots.SERIES[0]),
                        (pl["share"][0], "Indian yields, levels", plots.SERIES[0])):
        ax.axvline(v, color=col, lw=2, ls="-" if "changes" in lab else "--")
        ax.text(v, ax.get_ylim()[1] * (0.9 if "levels" in lab else 0.75), f" {lab}\n {v:.0%}",
                fontsize=8.5, color=plots.TEXT, ha="right" if "levels" in lab else "left")
    ax.xaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
    ax.set_xlabel("share of movement in the first component")
    ax.set_ylabel("number of simulated sets (of 500)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title("Levels make even unrelated series look like one", fontsize=10.5, loc="left")
    ax = axes[1]
    x = np.arange(len(CORE))
    ax.plot(CORE, pc["loadings"][:, 0], color=plots.SERIES[0], lw=2.2, marker="o", ms=5,
            label=f"covariance PCA (level {pc['share'][0]:.0%})")
    ax.plot(CORE, pz["loadings"][:, 0], color=plots.SERIES[3], lw=2.2, marker="o", ms=5,
            label=f"correlation PCA (level {pz['share'][0]:.0%})")
    ax.axhline(1 / np.sqrt(len(CORE)), color=plots.TEXT_2, lw=0.8, ls=":")
    ax.text(10, 1 / np.sqrt(len(CORE)) + 0.005, "a parallel shift", fontsize=8.5,
            color=plots.TEXT_2, ha="right", va="bottom")
    ax.set_xticks(CORE, [f"{m}y" for m in CORE])
    ax.set_ylabel("level loading")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title("Scaling each maturity flattens the level shape", fontsize=10.5, loc="left")
    del x, vs
    fig.suptitle("Run PCA on changes: levels inflated the first share from 83% to 98%",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    plots.save(fig, out_dir, "levels")
    plt.close(fig)
    return res


BOOKS = {"1-10 year ladder": np.ones(10), "7-year bullet": np.eye(10)[6] * 10,
         "1-3 year book": np.r_[np.full(3, 10 / 3), np.zeros(7)]}
HEDGES = {"duration (10y)": ([10], "duration"), "PCA level+slope (2y, 10y)": ([2, 10], "pca"),
          "PCA three shapes (2y, 5y, 10y)": ([2, 5, 10], "pca"),
          "minimum variance (2y, 5y, 10y)": ([2, 5, 10], "minvar")}


def report_hedge(out_dir: Path) -> dict:
    """Hedging three bond books (rupees per bp at 1-10 years, total 10) out of sample: each
    month's hedge sized on the previous 60 months of 1-10 year changes, scored on that month.
    Risk = standard deviation of the hedged monthly P&L as a share of the unhedged; worst month
    the same way; by period; the PCA three-shape hedge's largest positions; and the long hedge
    bond moved from 10 to 9 years."""
    from yc import data, hedge

    w = data.yields()
    d = changes(w, CORE)
    X, dates = d.to_numpy(), d.index[60:]
    res = {"scored months": f"{len(dates)}, {dates[0]:%b %Y}-{dates[-1]:%b %Y}"}
    table = {}
    for bn, b in BOOKS.items():
        base = pd.Series([-b @ x for x in X[60:]], index=dates)
        cells = []
        for hn, (bonds, m) in HEDGES.items():
            r = pd.Series(hedge.backtest(X, CORE, b, bonds, m), index=dates)
            ratio = r.std() / base.std()
            table[(bn, hn)] = ratio
            by = ", ".join(f"{a}-{c[-2:]} {r.loc[a:c].std() / base.loc[a:c].std():.0%}"
                           for a, c in (("2001", "2013"), ("2014", "2020"), ("2021", "2026")))
            cells.append(f"{hn} {ratio:.0%} (worst month {r.abs().max() / base.abs().max():.0%}; "
                         f"{by})")
        res[f"{bn}: unhedged risk Rs {base.std():.0f} a month"] = "; ".join(cells)
    b = BOOKS["1-10 year ladder"]
    gross = pd.Series([np.abs(hedge.hedge(b, X[t - 60:t], CORE, [2, 5, 10], "pca")).sum() / 10
                       for t in range(60, len(X))], index=dates)
    r3 = pd.Series(hedge.backtest(X, CORE, b, [2, 5, 10], "pca"), index=dates)
    res["  ladder, PCA three-shape hedge size (gross hedge / book)"] = (
        f"median {gross.median():.1f}; largest " + ", ".join(
            f"{t:%b %Y} {v:.0f}x" for t, v in gross.nlargest(3).items())
        + f"; its five worst months carry {(r3 ** 2).nlargest(5).sum() / (r3 ** 2).sum():.0%} of "
        "its variance")
    base = np.array([-b @ x for x in X[60:]])
    nine = []
    for hn, (bonds, m) in HEDGES.items():
        r9 = hedge.backtest(X, CORE, b, [9 if x == 10 else x for x in bonds], m)
        nine.append(f"{hn.split(' (')[0]} {r9.std() / base.std():.0%}")
    res["  ladder with the 9-year bond in place of the 10-year"] = ", ".join(nine)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    x = np.arange(len(BOOKS))
    cols = (plots.MUTED, plots.SERIES[1], plots.SERIES[2], plots.SERIES[0])
    for i, (hn, col) in enumerate(zip(HEDGES, cols, strict=True)):
        v = [100 * table[(bn, hn)] for bn in BOOKS]
        bars = ax.bar(x + (i - 1.5) * 0.2, np.minimum(v, 150), 0.19, color=col, label=hn)
        for bar, vv in zip(bars, v, strict=True):
            ax.annotate(f"{vv:.0f}", (bar.get_x() + bar.get_width() / 2, min(vv, 150)),
                        xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7.5,
                        color=plots.TEXT)
    ax.axhline(100, color=plots.TEXT_2, lw=0.8, ls="--")
    ax.text(x[-1] + 0.45, 101, "no hedge", fontsize=8, color=plots.TEXT_2, ha="right",
            va="bottom")
    ax.set_xticks(x, list(BOOKS))
    ax.set_ylim(0, 165)
    ax.set_ylabel("monthly risk left after hedging,\n% of unhedged (capped at 150)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Out of sample, 2001-2026", fontsize=10.5, loc="left")
    ax = axes[1]
    base_s = pd.Series(base, index=dates)
    for hn, col in (("duration (10y)", plots.MUTED), ("PCA level+slope (2y, 10y)", plots.SERIES[1]),
                    ("minimum variance (2y, 5y, 10y)", plots.SERIES[0])):
        bonds, m = HEDGES[hn]
        r = pd.Series(hedge.backtest(X, CORE, b, bonds, m), index=dates)
        roll = r.rolling(24).std() / base_s.rolling(24).std()
        ax.plot(roll.index, 100 * roll, color=col, lw=2, label=hn)
    ax.axhline(100, color=plots.TEXT_2, lw=0.8, ls="--")
    ax.set_ylabel("risk left, % of unhedged\n(rolling 2 years, ladder)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title("Duration hedging failed in 2014-2020", fontsize=10.5, loc="left")
    fig.suptitle("A regression hedge on three bonds beat duration; the exact three-shape hedge "
                 "blew up", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    plots.save(fig, out_dir, "hedge")
    plt.close(fig)
    return res


def report_predict(out_dir: Path) -> dict:
    """Does today's curve predict the next 12 months? Outcomes: the change in the 1-year and
    10-year yields, and the 10-year bond's return above the 1-year yield (yc.predict). Predictors:
    slope (10y - 1y) and curvature (2 x 5y - 1y - 10y). In-sample regressions with Newey-West t
    (12 lags, the outcomes overlap); out of sample from January 2006 with an expanding window that
    only uses outcomes already known, scored against the historical mean (R² below zero = worse than
    the mean). Also the 1-year yield's next-year change after flat or inverted curves."""
    from yc import data
    from yc import predict as P

    w = data.yields()
    sg = P.signals(w)
    outs = {"1-year yield change": w[1].shift(-12) - w[1],
            "10-year yield change": w[10].shift(-12) - w[10],
            "10-year bond return above the 1-year yield": P.excess_return(w)}
    res, oos_tab = {}, {}
    for k, y in outs.items():
        for cols in (["slope"], ["slope", "curvature"]):
            df = pd.concat([y, sg[cols]], axis=1).dropna()
            b, t, r2 = P.nw_t(df.iloc[:, 0].to_numpy(), df[cols].to_numpy(), 12)
            f = P.oos(y, sg[cols])
            oos_tab[(k, "+".join(cols))] = P.oos_r2(f)
            hit = float((np.sign(f["model"] - f["mean"]) == np.sign(f["y"] - f["mean"])).mean())
            res[f"{k}, on {' and '.join(cols)}"] = (
                f"{len(df)} months: " + ", ".join(f"{c} {bb:+.2f} (t {tt:.2f})" for c, bb, tt in
                                                  zip(cols, b[1:], t[1:], strict=True))
                + f", R² {r2:.1%}; out of sample {len(f)} forecasts ({f.index[0]:%b %Y}-"
                f"{f.index[-1]:%b %Y}): R² {oos_tab[(k, '+'.join(cols))]:+.1%}, right side of the "
                f"mean {hit:.0%}")
    ex = outs["10-year bond return above the 1-year yield"].dropna()
    res["10-year bond return above the 1-year yield"] = (
        f"mean {ex.mean():+.2f}% a year, standard deviation {ex.std():.2f}")
    for short in (2, 3):
        y = w[short].shift(-12) - w[short]
        x = (w[10] - w[short]).rename("slope").to_frame()
        df = pd.concat([y, x], axis=1).dropna()
        b, t, r2 = P.nw_t(df.iloc[:, 0].to_numpy(), df[["slope"]].to_numpy(), 12)
        res[f"  robustness: {short}-year yield change on 10y - {short}y slope"] = (
            f"slope {b[1]:+.2f} (t {t[1]:.2f}), R² {r2:.1%}; out of sample R² "
            f"{P.oos_r2(P.oos(y, x)):+.1%}")
    ch1 = outs["1-year yield change"]
    for lab, mask in (("slope below 0.25 pp", sg["slope"] < 0.25),
                      ("slope 0.25-1.5 pp", sg["slope"].between(0.25, 1.5)),
                      ("slope above 1.5 pp", sg["slope"] > 1.5)):
        x = ch1[mask].dropna()
        res[f"  1-year yield's next-year change, {lab}"] = (
            f"{len(x)} months: mean {100 * x.mean():+.0f} bp, rose in {(x > 0).mean():.0%}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    df = pd.concat([ch1.rename("y"), sg["slope"]], axis=1).dropna()
    ax.scatter(df["slope"], 100 * df["y"], s=12, color=plots.SERIES[0], alpha=0.6,
               edgecolors="none")
    b = np.polyfit(df["slope"], 100 * df["y"], 1)
    xs = np.linspace(df["slope"].min(), df["slope"].max(), 50)
    ax.plot(xs, np.polyval(b, xs), color=plots.TEXT_2, lw=1.5)
    ax.axhline(0, color=plots.TEXT_2, lw=0.8)
    ax.set_xlabel("today's slope: 10-year minus 1-year yield, percentage points")
    ax.set_ylabel("1-year yield's change over\nthe next 12 months, bp")
    ax.set_title("A steep curve came before a rising 1-year yield", fontsize=10.5, loc="left")
    ax = axes[1]
    labs = list(outs)
    x = np.arange(len(labs))
    for i, (cols, col) in enumerate((("slope", plots.SERIES[0]),
                                     ("slope+curvature", plots.SERIES[2]))):
        v = [100 * oos_tab[(k, cols)] for k in labs]
        bars = ax.bar(x + (i - 0.5) * 0.36, v, 0.34, color=col,
                      label=cols.replace("+", " and "))
        for bar, vv in zip(bars, v, strict=True):
            ax.annotate(f"{vv:+.0f}%", (bar.get_x() + bar.get_width() / 2, vv),
                        xytext=(0, 3 if vv >= 0 else -3), textcoords="offset points",
                        ha="center", va="bottom" if vv >= 0 else "top", fontsize=8,
                        color=plots.TEXT)
    ax.axhline(0, color=plots.TEXT_2, lw=0.8)
    ax.set_xticks(x, ["1-year yield\nchange", "10-year yield\nchange",
                      "10-year bond's\nextra return"])
    ax.set_ylabel("out-of-sample R², % (below 0 =\nworse than the average)")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.margins(y=0.25)
    ax.set_title("Out of sample, 2006-2025", fontsize=10.5, loc="left")
    fig.suptitle("The slope predicted the 1-year yield, much less beyond it; nothing predicted "
                 "long rates or bond returns", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    plots.save(fig, out_dir, "predict")
    plt.close(fig)
    return res


def report_ns(out_dir: Path) -> dict:
    """Nelson-Siegel with a fixed decay against PCA, 1-10 years: the decay chosen on 1996-2009
    curves, and PCA shapes estimated on 1996-2009 (on levels for fitting curves, on changes for
    capturing moves), both scored on 2010-2026: root-mean-square error of each month's fitted
    curve, overall and by maturity; the share of each month's change captured; and how the
    Nelson-Siegel factors' monthly changes line up with the PCA components'."""
    from yc import data, ns, pca

    w = data.yields()
    d = changes(w, CORE)
    lv = w.loc[d.index[0] - pd.offsets.MonthEnd(1):, CORE].dropna()
    tr, te = lv.loc[:"2009"], lv.loc["2010":]
    lam = ns.choose_lam(tr.to_numpy(), CORE)
    _, fitted = ns.fit(te.to_numpy(), CORE, lam)
    ns_err = (te.to_numpy() - fitted) * 100
    p = pca.pca(tr.to_numpy())
    L, mu = p["loadings"][:, :3], p["mean"]
    pc_err = (te.to_numpy() - ((te.to_numpy() - mu) @ L @ L.T + mu)) * 100
    q = pca.pca(te.to_numpy())
    own = (te.to_numpy() - ((te.to_numpy() - q["mean"]) @ q["loadings"][:, :3]
                            @ q["loadings"][:, :3].T + q["mean"])) * 100
    rms = lambda e: float(np.sqrt((e ** 2).mean()))  # noqa: E731
    res = {"training": f"{len(tr)} curves {tr.index[0]:%b %Y}-{tr.index[-1]:%b %Y}; test {len(te)} "
                       f"curves {te.index[0]:%b %Y}-{te.index[-1]:%b %Y}",
           "Nelson-Siegel decay chosen on training curves": f"{lam:.2f} years (curvature peaks "
                                                            f"near {1.79 * lam:.1f} years)"}
    res["test curves fitted, root-mean-square error"] = (
        f"Nelson-Siegel {rms(ns_err):.1f} bp; PCA shapes from training {rms(pc_err):.1f} bp; PCA "
        f"fitted on the test curves themselves {rms(own):.1f} bp")
    res["  by maturity, Nelson-Siegel"] = " ".join(f"{m}y {v:.1f}" for m, v in
                                                     zip(CORE, np.sqrt((ns_err ** 2).mean(0)),
                                                         strict=True))
    res["  by maturity, PCA from training"] = " ".join(f"{m}y {v:.1f}" for m, v in
                                                        zip(CORE, np.sqrt((pc_err ** 2).mean(0)),
                                                            strict=True))
    dtr, dte = d.loc[:"2009"].to_numpy(), d.loc["2010":].to_numpy()
    Q, _ = np.linalg.qr(ns.basis(CORE, lam))
    res["test monthly changes captured"] = (
        f"Nelson-Siegel shapes {pca.captured(dte, Q):.1%}; PCA shapes from training "
        f"{pca.captured(dte, pca.pca(dtr)['loadings'][:, :3]):.1%}; best possible three "
        f"{pca.pca(dte)['share'][:3].sum():.1%}")
    bt, _ = ns.fit(lv.to_numpy(), CORE, lam)
    db = np.diff(bt, axis=0)[-len(d):]
    sc = pca.pca(d.to_numpy())["scores"][:, :3]
    C = np.corrcoef(np.c_[db, sc].T)[:3, 3:]
    for i, nm in enumerate(("level b0", "slope b1", "curvature b2")):
        res[f"  Nelson-Siegel {nm} change, correlation with PCA level / slope / curvature"] = (
            " / ".join(f"{v:+.2f}" for v in C[i]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    tau = np.linspace(0.25, 30, 200)
    Bf = ns.basis(tau, lam)
    for k, (nm, col) in enumerate(zip(NAMES, (plots.SERIES[0], plots.SERIES[1], plots.SERIES[2]),
                                      strict=True)):
        ax.plot(tau, Bf[:, k], color=col, lw=2.2, label=f"{nm} (b{k})")
    ax.axvspan(1, 10, color=plots.GRID, lw=0)
    ax.text(5.5, 1.05, "maturities used", fontsize=8.5, color=plots.TEXT_2, ha="center")
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("years to maturity")
    ax.set_ylabel("how much a yield moves per unit\nof each Nelson-Siegel factor")
    ax.legend(frameon=False, fontsize=9, loc="center right")
    ax.set_title(f"Nelson-Siegel's three fixed shapes (decay {lam:.2f} years)", fontsize=10.5,
                 loc="left")
    ax = axes[1]
    x = np.arange(len(CORE))
    ax.bar(x - 0.2, np.sqrt((ns_err ** 2).mean(0)), 0.38, color=plots.SERIES[0],
           label=f"Nelson-Siegel ({rms(ns_err):.1f} bp overall)")
    ax.bar(x + 0.2, np.sqrt((pc_err ** 2).mean(0)), 0.38, color=plots.MUTED,
           label=f"PCA shapes from 1996-2009 ({rms(pc_err):.1f} bp)")
    ax.set_xticks(x, [f"{m}y" for m in CORE])
    ax.set_ylabel("2010-2026 fitting error, bp\n(root mean square)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title("Both fitted 2010-2026 curves to about 8 bp", fontsize=10.5, loc="left")
    fig.suptitle("A three-number formula fitted the curve as well as PCA, out of sample",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    plots.save(fig, out_dir, "ns")
    plt.close(fig)
    return res


def report_verdict(out_dir: Path) -> dict:
    """What the reference model should take from the curve. (1) NIFTY's monthly return on the month's level
    move (average change of 1-10 year yields) and slope move (change in 10y - 1y), Newey-West t
    (3 lags), whole period and halves. (2) The curve as a regime input: NIFTY's next-month and
    next-12-month return on today's slope and on the past 3 months' level move, out of sample from
    January 2018 (expanding window, scored against the historical mean). (3) Single stocks in
    the reference model's point-in-time top 200: each stock's rate sensitivity (its return on NIFTY and the
    level move) in 2013-2019 against 2020-2026. NIFTY and stocks from the NSE bhavcopy archive."""
    from yc import data
    from yc import predict as P
    from yc import verdict as V

    w = data.yields()
    cm = V.curve_moves(w)
    n = V.nifty_monthly()
    j = pd.concat([n, cm], axis=1, sort=True).dropna()
    res = {"NIFTY months": f"{len(j)}, {j.index[0]:%b %Y}-{j.index[-1]:%b %Y}"}
    periods = (("all", None, None), ("2013-2019", None, "2019"), ("2020-2026", "2020", None))
    for lab, a, b in periods:
        x = j.loc[a:b]
        bb, t, r2 = P.nw_t(x["nifty"].to_numpy() * 100, x[["level", "slope"]].to_numpy(), 3)
        res[f"NIFTY on curve moves, {lab} ({len(x)} months)"] = (
            f"{100 * bb[1]:+.1f}% per 100 bp level rise (t {t[1]:.2f}); {100 * bb[2]:+.1f}% per "
            f"100 bp steepening (t {t[2]:.2f}); R² {r2:.1%}")
    sg = P.signals(w)
    lev3 = cm["level"].rolling(3).sum().rename("past 3 months' level move")
    nxt = {"next month": (n.shift(-1), 1),
           "next 12 months": ((1 + n).rolling(12).apply(np.prod, raw=True).shift(-12) - 1, 12)}
    for lab, (y, h) in nxt.items():
        for x in (sg[["slope"]], lev3.to_frame()):
            f = P.oos(y, x, h=h, start="2018-01-31", min_obs=48)
            df = pd.concat([y, x], axis=1, sort=True).dropna()
            bb, t, r2 = P.nw_t(df.iloc[:, 0].to_numpy(), df.iloc[:, 1:].to_numpy(), max(h, 3))
            res[f"NIFTY {lab} on {x.columns[0]}"] = (
                f"in-sample t {t[1]:.2f}, R² {r2:.1%}; out of sample {len(f)} forecasts, R² "
                f"{P.oos_r2(f):+.1%}")
    px, uni = V.stock_panel()
    r = px.pct_change(fill_method=None)
    r = r.where(uni.shift(1).reindex_like(r).fillna(False).astype(bool))
    X = pd.concat([n, cm["level"]], axis=1, sort=True).dropna()
    b1, b2 = V.stock_rate_betas(r, X, None, "2019"), V.stock_rate_betas(r, X, "2020", None)
    bj = pd.concat([b1.rename("a"), b2.rename("b")], axis=1).dropna()
    res["stock rate sensitivity, % per 100 bp level rise beyond NIFTY's"] = (
        f"{len(b1)} stocks in 2013-2019 (median {b1.median():+.1f}), {len(b2)} in 2020-2026 "
        f"(median {b2.median():+.1f}); {len(bj)} in both: rank correlation "
        f"{bj['a'].rank().corr(bj['b'].rank()):+.2f}")
    top, bot = bj["a"] >= bj["a"].quantile(0.8), bj["a"] <= bj["a"].quantile(0.2)
    res["  most and least rate-sensitive fifth in 2013-2019, in 2020-2026"] = (
        f"least sensitive (most positive) {bj.loc[top, 'b'].mean():+.1f}; most negative "
        f"{bj.loc[bot, 'b'].mean():+.1f}; most negative five then: " + ", ".join(
            f"{s_} {v:+.0f} then {bj.at[s_, 'b']:+.0f}" for s_, v in bj["a"].nsmallest(5).items()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    ax.scatter(j["level"], 100 * j["nifty"], s=14, color=plots.SERIES[0], alpha=0.65,
               edgecolors="none")
    bfit = np.polyfit(j["level"], 100 * j["nifty"], 1)
    xs = np.linspace(j["level"].min(), j["level"].max(), 50)
    ax.plot(xs, np.polyval(bfit, xs), color=plots.TEXT_2, lw=1.5)
    ax.axhline(0, color=plots.TEXT_2, lw=0.8)
    ax.axvline(0, color=plots.TEXT_2, lw=0.8)
    ax.set_xlabel("the month's level move: average change of 1-10 year yields, bp")
    ax.set_ylabel("NIFTY's return that month, %")
    ax.set_title("Rates explained little of NIFTY's month", fontsize=10.5, loc="left")
    ax = axes[1]
    ax.scatter(bj["a"], bj["b"], s=16, color=plots.SERIES[1], alpha=0.7, edgecolors="none")
    lim = float(np.nanpercentile(np.abs(bj.to_numpy()), 98)) * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.plot([-lim, lim], [-lim, lim], color=plots.TEXT_2, lw=0.8, ls="--")
    ax.axhline(0, color=plots.TEXT_2, lw=0.6)
    ax.axvline(0, color=plots.TEXT_2, lw=0.6)
    ax.set_xlabel("stock's rate sensitivity 2013-2019,\n% per 100 bp rise beyond NIFTY's")
    ax.set_ylabel("same stock, 2020-2026")
    ax.set_title("A stock's rate sensitivity didn't carry over", fontsize=10.5, loc="left")
    fig.suptitle("For an equity model the curve is a risk to measure, not a signal to use",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    plots.save(fig, out_dir, "verdict")
    plt.close(fig)
    return res


REPORTS: dict[str, Callable[[Path], dict]] = {
    "pca": report_pca,
    "stability": report_stability,
    "levels": report_levels,
    "hedge": report_hedge,
    "predict": report_predict,
    "ns": report_ns,
    "verdict": report_verdict,
}
