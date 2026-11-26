"""Build site/index.html from the fetched RBI yields (run scripts/fetch_data.py first)."""

from pathlib import Path

from yc import data
from yc.site import build_html


def main() -> None:
    w = data.yields()
    out = Path("site/index.html")
    out.parent.mkdir(exist_ok=True)
    out.write_text(build_html(w), encoding="utf-8")
    print(f"{out} ({out.stat().st_size / 1e3:.0f} kB), {len(w)} months, "
          f"{w.index[0]:%b %Y}-{w.index[-1]:%b %Y}")


if __name__ == "__main__":
    main()
