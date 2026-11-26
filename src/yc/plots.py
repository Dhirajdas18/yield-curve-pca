"""Shared chart style for README charts. Colours follow the entity (each series keeps its colour in
every chart); the page uses the same palette through site.py."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e1"
# Categorical slots in fixed order (validated: CVD + normal-vision separation pass).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
COLORS = {"MKT": SERIES[0], "SMB": SERIES[1], "HML": SERIES[2], "WML": SERIES[3]}
MUTED = "#a3a29c"
# Sequential blue ramp (light to dark) for magnitudes such as correlation.
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


# Dark mode: the same hues stepped for a dark surface (reference palette's dark column).
DARK = {
    SURFACE: "#1a1a19",
    TEXT: "#f0efec",
    TEXT_2: "#c3c2b7",
    GRID: "#2e2d2b",
    "#f1f0ec": "#2e2d2b",  # diverging midpoint
    "white": "#242422",
    SERIES[0]: "#3987e5",
    SERIES[1]: "#d95926",
    SERIES[2]: "#199e70",
    SERIES[3]: "#c98500",
    SERIES[4]: "#d55181",
    SERIES[5]: "#0ca30c",
    SERIES[6]: "#9085e9",
    SERIES[7]: "#e66767",
}
# Line colours only: on dark, longer rolling windows get lighter (still ordered by length).
# Kept out of DARK so heatmap colour scales built from the same ramp stay in order.
DARK_LINES = {
    BLUE_RAMP[2]: "#256abf",
    BLUE_RAMP[4]: "#6da7ec",
    BLUE_RAMP[6]: "#cde2fb",
}


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_2,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "lines.linewidth": 2,
            "legend.frameon": False,
            "legend.labelcolor": TEXT_2,
            "font.size": 10,
            "figure.dpi": 110,
        }
    )


def label_line_end(ax, x, y, text: str, color: str) -> None:
    """Direct label at a line's right end (identity never by colour alone)."""
    ax.annotate(
        text,
        (x, y),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color=TEXT,
        bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": color, "lw": 1},
    )


# Boomerang growth for README GIFs: every data element grows the way the data does (lines draw
# along their path, bars rise from their baseline, points appear in order, ranges extend from their
# start), holds long enough to read, shrinks back faster, short pause, loop. Strong ease-out both
# ways (never ease-in). Axes, grids and reference lines stay still; labels fade in at the end.
FPS = 25
IN_FRAMES = 35  # 1.4 s
OUT_FRAMES = 20  # 0.8 s
HOLD_MS = 8000
PAUSE_MS = 300
FRAME_MS = 1000 // FPS
LOOP_MS = (IN_FRAMES + OUT_FRAMES) * FRAME_MS + HOLD_MS + PAUSE_MS
GIF_DPI = 90
_EASE = (0.23, 1.0, 0.32, 1.0)  # cubic-bezier(0.23, 1, 0.32, 1)


def ease_out(p: float) -> float:
    """Strong ease-out: cubic-bezier(0.23, 1, 0.32, 1) evaluated at progress p in [0, 1]."""
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    x1, y1, x2, y2 = _EASE

    def bez(t, a, b):
        return 3 * (1 - t) ** 2 * t * a + 3 * (1 - t) * t**2 * b + t**3

    lo, hi = 0.0, 1.0
    for _ in range(40):  # solve x(t) = p
        mid = (lo + hi) / 2
        if bez(mid, x1, x2) < p:
            lo = mid
        else:
            hi = mid
    return bez((lo + hi) / 2, y1, y2)


def _frame(fig, bbox):
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=GIF_DPI, bbox_inches=bbox)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _grow_line(line):
    """Draw a data line along its own path, with an interpolated tip for smooth motion."""
    x = np.asarray(line.get_xdata(orig=False), dtype=float)
    y = np.asarray(line.get_ydata(orig=False), dtype=float)
    n = len(x)
    if line.get_linestyle() in ("None", "", " "):  # markers only: points appear in order
        return lambda p: line.set_data(x[: int(round(p * n))], y[: int(round(p * n))])

    def at(p):
        f = p * (n - 1)
        k = int(np.floor(f))
        if k >= n - 1:
            line.set_data(x, y)
            return
        t = f - k
        line.set_data(
            np.append(x[: k + 1], x[k] + t * (x[k + 1] - x[k])),
            np.append(y[: k + 1], y[k] + t * (y[k + 1] - y[k])),
        )

    return at


def _grow_points(coll):
    offsets = coll.get_offsets().copy()
    face, sizes = coll.get_facecolors().copy(), coll.get_sizes().copy()
    n = len(offsets)

    def at(p):
        k = int(round(p * n))
        coll.set_offsets(offsets[:k] if k else np.empty((0, 2)))
        if len(face) > 1:
            coll.set_facecolors(face[:k])
        if len(sizes) > 1:
            coll.set_sizes(sizes[:k])

    return at


def _grow_segments(coll):
    segs = [np.asarray(sg, dtype=float).copy() for sg in coll.get_segments()]

    def at(p):
        coll.set_segments(
            [
                np.vstack([sg[0], sg[0] + p * (sg[-1] - sg[0])])
                if len(sg) == 2
                else sg[: max(1, int(round(p * len(sg))))]
                for sg in segs
            ]
        )

    return at


def _fade(art):
    alpha = art.get_alpha() if art.get_alpha() is not None else 1.0
    return lambda p: art.set_alpha(alpha * p)


def _animator(fig):
    """A function at(p) that sets every data element of `fig` to growth progress p in [0, 1]."""
    from matplotlib.collections import LineCollection, PathCollection, PolyCollection

    steps, texts = [], []
    for ax in fig.axes:
        ax.set_xlim(ax.get_xlim())  # freeze limits: growing data must not rescale the axes
        ax.set_ylim(ax.get_ylim())
        in_bars = set()
        for cont in ax.containers:
            if not hasattr(cont, "orientation"):
                continue
            for bar in cont.patches:
                in_bars.add(bar)
                # Stacked or floating bars scale their start too, so pieces stay attached.
                if cont.orientation == "horizontal":
                    w, x0 = bar.get_width(), bar.get_x()
                    steps.append(lambda p, b=bar, w=w, x0=x0: (b.set_x(x0 * p), b.set_width(w * p)))
                else:
                    h, y0 = bar.get_height(), bar.get_y()
                    steps.append(
                        lambda p, b=bar, h=h, y0=y0: (b.set_y(y0 * p), b.set_height(h * p))
                    )
        for patch in ax.patches:
            if patch not in in_bars:
                steps.append(_fade(patch))
        for line in ax.lines:
            if len(line.get_xdata()) > 2 or line.get_linestyle() in ("None", "", " "):
                steps.append(_grow_line(line))  # two-point lines are reference lines: still
        for coll in ax.collections:
            if isinstance(coll, PathCollection):
                steps.append(_grow_points(coll))
            elif isinstance(coll, LineCollection):
                steps.append(_grow_segments(coll))
            elif isinstance(coll, PolyCollection):
                steps.append(_fade(coll))
        for img in ax.images:
            steps.append(_fade(img))
        texts += [(t, t.get_alpha() if t.get_alpha() is not None else 1.0) for t in ax.texts]

    def at(p: float) -> None:
        for step in steps:
            step(p)
        fade = min(max((p - 0.8) / 0.2, 0.0), 1.0)  # labels fade in over the last 20%
        for t, alpha in texts:
            t.set_alpha(alpha * fade)
            if getattr(t, "arrow_patch", None) is not None:
                t.arrow_patch.set_alpha(alpha * fade)
            if t.get_bbox_patch() is not None:
                t.get_bbox_patch().set_alpha(fade)

    return at


def _save_gif(fig, path: Path) -> None:
    from PIL import Image

    # Crop every frame to the finished chart's tight box, same as the PNG (stable frame size).
    bbox = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.1)
    grow = _animator(fig)

    def at(progress: float):
        grow(progress)
        return _frame(fig, bbox)

    frames = [at(ease_out(i / IN_FRAMES)) for i in range(1, IN_FRAMES + 1)]
    durations = [FRAME_MS] * IN_FRAMES
    frames.append(at(1.0))
    durations.append(HOLD_MS)
    frames += [at(1 - ease_out(i / OUT_FRAMES)) for i in range(1, OUT_FRAMES + 1)]
    durations += [FRAME_MS] * OUT_FRAMES
    frames.append(at(0.0))
    durations.append(PAUSE_MS)
    grow(1.0)  # leave the figure finished
    palette = frames[IN_FRAMES].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=1,
    )


def save(fig, out_dir: Path, name: str, animate: bool = True) -> Path:
    """Save a PNG, plus an animated GIF version for the README unless animate=False."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    if animate:
        _save_gif(fig, out_dir / f"{name}.gif")
    plt.close(fig)
    return path
