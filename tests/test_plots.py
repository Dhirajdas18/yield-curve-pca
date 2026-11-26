import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from yc import plots


def test_ease_out_endpoints_and_monotonic():
    p = np.linspace(0, 1, 101)
    v = [plots.ease_out(x) for x in p]
    assert v[0] == 0 and abs(v[-1] - 1) < 1e-9
    assert all(b >= a for a, b in zip(v, v[1:], strict=False))
    assert plots.ease_out(0.2) > 0.5  # strong ease-out: most of the motion happens early


def test_save_writes_png_and_looping_boomerang_gif(tmp_path):
    plots.style()
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [1, 3, 2])
    ax.annotate("label", (2, 2))
    png = plots.save(fig, tmp_path, "demo")
    gif = tmp_path / "demo.gif"
    assert png.exists() and gif.exists()
    with Image.open(gif) as im:
        assert im.info.get("loop") == 0
        n = im.n_frames
        durations = []
        for i in range(n):
            im.seek(i)
            durations.append(im.info["duration"])
    # Identical consecutive frames (the tail of the ease-out) are merged, durations summed.
    assert 10 < n <= plots.IN_FRAMES + 1 + plots.OUT_FRAMES + 1
    assert max(durations) >= plots.HOLD_MS
    assert abs(sum(durations) - plots.LOOP_MS) <= 10 * n


def test_save_can_skip_animation(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    plots.save(fig, tmp_path, "still", animate=False)
    assert not (tmp_path / "still.gif").exists()


def _demo_fig():
    fig, (a1, a2) = plt.subplots(1, 2)
    a1.bar([0, 1], [5.0, -3.0])
    a1.bar([2], [2.0], bottom=[4.0])  # a waterfall step
    a1.axhline(0)
    a2.plot(np.arange(10), np.arange(10.0) ** 2)
    a2.scatter(np.arange(8), np.arange(8))
    a2.barh([0], [6.0])
    a2.barh([1], [2.0], left=[3.0])  # a stacked segment
    a2.hlines([1], [2.0], [7.0])
    return fig, a1, a2


def test_bars_grow_from_their_own_baseline():
    fig, a1, a2 = _demo_fig()
    at = plots._animator(fig)
    at(0.5)
    heights = [p.get_height() for p in a1.patches]
    assert heights == [2.5, -1.5, 1.0]
    assert a1.patches[2].get_y() == 2.0  # stacked pieces move with the bars below them
    assert a2.patches[0].get_width() == 3.0  # horizontal bars grow from the axis
    assert a2.patches[1].get_x() == 1.5 and a2.patches[1].get_width() == 1.0  # stays attached
    at(1.0)
    assert [p.get_height() for p in a1.patches] == [5.0, -3.0, 2.0]
    plt.close(fig)


def test_lines_draw_along_their_path_and_points_appear_in_order():
    fig, a1, a2 = _demo_fig()
    at = plots._animator(fig)
    xlim = a2.get_xlim()
    at(0.5)
    line = a2.lines[0]
    assert line.get_xdata()[-1] == 4.5  # ends part-way along, at an interpolated point
    assert len(a2.collections[0].get_offsets()) == 4
    seg = a2.collections[1].get_segments()[0]
    assert seg[0][0] == 2.0 and seg[1][0] == 4.5  # range grows from its start
    assert a2.get_xlim() == xlim  # axes stay fixed while data grows
    assert len(a1.lines[0].get_xdata()) == 2  # reference lines are not animated
    at(1.0)
    assert len(line.get_xdata()) == 10 and len(a2.collections[0].get_offsets()) == 8
    plt.close(fig)
