"""The PCA lab: one static Plotly page carrying RBI's month-end yields (yc.js does the PCA in the
browser)."""

import datetime
import json
import re
from pathlib import Path
from string import Template

import pandas as pd

from yc import data, plots

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
ICON = (
    "data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 16 16%22%3E"
    "%3Cpath d=%22M1 13c3-6 6-8 14-9%22 fill=%22none%22"
    " stroke=%22%232a78d6%22 stroke-width=%222%22 stroke-linecap=%22round%22/%3E"
    "%3C/svg%3E"
)
HERE = Path(__file__).parent
JS = (HERE / "yc.js").read_text(encoding="utf-8")
LAB = (HERE / "lab.html").read_text(encoding="utf-8")
PAGE = Template((HERE / "page.html").read_text(encoding="utf-8"))
REPO = "https://github.com/Dhirajdas18/yield-curve-pca"


def lab_data(w: pd.DataFrame) -> dict:
    """Months as 'YYYY-MM' and each maturity's yields in % (None where missing)."""
    return {"months": list(w.index.strftime("%Y-%m")),
            "y": {int(m): [None if pd.isna(v) else round(float(v), 4) for v in w[m]]
                  for m in w.columns}}


def lab_section(d: dict) -> str:
    colors = [plots.SERIES[0], plots.SERIES[1], plots.SERIES[2]]
    return (LAB.replace("$js", JS).replace("$data", json.dumps(d, separators=(",", ":")))
            .replace("$colors", json.dumps(colors)))


def method_section(d: dict) -> str:
    return """<ul class="plain">
<li>Data: month-end yields of Government of India dated securities at whole years to maturity
from 1 to 30, from RBI’s settled (SGL) trades, as published in the Handbook of Statistics on
Indian Economy. Each Handbook edition shows about four fiscal years; the page merges every edition
since 2008, taking the newest edition’s figure where they overlap.</li>
<li>PCA: the eigen-decomposition of the covariance of the chosen maturities’ yields (in basis
points), over the months in the chosen years that have every chosen maturity. On monthly changes,
only changes between consecutive months count. Each component’s sign is fixed so the level loads
positively, the slope rises with maturity and the curvature dips in the middle.</li>
<li>The paths chart projects each month onto each shape without removing the period’s average, so
on monthly changes its running total keeps the trend: a level path that ends lower means yields
fell over the period.</li>
<li>Everything is computed on the same months it describes. Research only, not investment
advice.</li>
</ul>"""


SECTIONS = [
    ("PCA lab", "Pick the years, the maturities and what to run PCA on. Research only, not "
     "investment advice.", lab_section),
    ("Method and caveats", "Where the numbers come from, and what they leave out.",
     method_section),
]


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def build_html(w: pd.DataFrame) -> str:
    d = lab_data(w)
    parts, nav = [], []
    for title, blurb, make in SECTIONS:
        sid = _slug(title)
        nav.append(f'<li><a href="#{sid}">{title}</a></li>')
        parts.append(f'<section id="{sid}">\n<h2>{title}</h2>\n<p>{blurb}</p>\n{make(d)}\n'
                     "</section>")
    dk, s = plots.DARK, plots.SERIES
    return PAGE.substitute(
        updated=datetime.date.today().strftime("%d %b %Y").lstrip("0"),
        first=f"{w.index[0]:%b %Y}", last=f"{w.index[-1]:%b %Y}", source=data.SOURCE,
        icon=ICON, font=FONT, repo=REPO,
        surface=plots.SURFACE, text=plots.TEXT, text2=plots.TEXT_2, grid=plots.GRID,
        accent=s[0], accent2=s[1], green=s[2], violet=s[6], muted=plots.MUTED,
        d_surface=dk[plots.SURFACE], d_text=dk[plots.TEXT], d_text2=dk[plots.TEXT_2],
        d_grid=dk[plots.GRID], d_accent=dk[s[0]], d_accent2=dk[s[1]], d_green=dk[s[2]],
        d_violet=dk[s[6]], d_muted="#6b6a64",
        nav="\n".join(nav), sections="\n".join(parts),
    )
