"""RBI's month-end yields of SGL transactions in Government of India dated securities, 1 to 30
years to maturity, from the Handbook of Statistics on Indian Economy (one web table per edition,
each covering about four fiscal years; the 2008 edition goes back to April 1996). Where editions
overlap, the newest edition's figure is used. Fetched into data/rbi/ (not committed)."""

import os
import re
import time
import urllib.request
from html import unescape
from pathlib import Path

import pandas as pd

PAGE = "https://www.rbi.org.in/scripts/PublicationsView.aspx?id={}"
INDEX = ("https://www.rbi.org.in/Scripts/AnnualPublications.aspx?head="
         "Handbook%20of%20Statistics%20on%20Indian%20Economy")
# Handbook editions' page ids for this table (2008, 2010, 2012, 2014, 2016, 2019, 2020, 2023,
# 2024, 2025, 2026), which together cover every month; fetch() also adds the newest edition
# listed on the Handbook index. Page ids rise with publication date, so sorting them puts the
# newest edition last.
EDITIONS = [10711, 12874, 14543, 15977, 17320, 19170, 19912, 21986, 22654, 23351, 24003]
TITLE = re.compile(r"Yield of SGL Transactions in Government Dated Securities for Various "
                   r"Maturities", re.I)
MONTHS = ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]
SOURCE = ("Reserve Bank of India, Handbook of Statistics on Indian Economy: Month-end Yield of SGL "
          "Transactions in Government Dated Securities for Various Maturities")


def data_dir() -> Path:
    return Path(os.environ.get("YC_DATA_DIR", "data")) / "rbi"


def _get(url: str, tries: int = 4) -> str:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", "replace")
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))
    raise RuntimeError("unreachable")


def _text(page: str) -> str:
    t = re.sub(r"<script.*?</script>|<style.*?</style>", " ", page, flags=re.S)
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", t)))


def parse(page: str) -> pd.DataFrame:
    """One edition's table -> rows (date = month-end, maturity in years, yield in %), skipping
    blanks ('-'). Rows are a fiscal year label, then for each maturity its number and 12 monthly
    values, April to March."""
    t = _text(page)
    m = TITLE.search(t)
    if not m:
        raise ValueError("table title not found")
    body = t[m.end():]
    end = re.search(r"Source\s*:|Note\s*:", body)
    tokens = (body[:end.start()] if end else body).split()
    rows, fy, i = [], None, 0
    while i < len(tokens):
        tok = tokens[i]
        if re.fullmatch(r"(19|20)\d\d-\d\d", tok):
            fy, i = int(tok[:4]), i + 1
            continue
        if fy is not None and re.fullmatch(r"\d{1,2}", tok) and i + 12 < len(tokens) + 1:
            vals = tokens[i + 1:i + 13]
            if len(vals) == 12 and all(re.fullmatch(r"-|\d+\.\d+", v) for v in vals):
                for k, v in enumerate(vals):
                    if v != "-":
                        y = fy if k < 9 else fy + 1
                        mth = (k + 4 - 1) % 12 + 1
                        rows.append((pd.Timestamp(y, mth, 1) + pd.offsets.MonthEnd(0), int(tok),
                                     float(v)))
                i += 13
                continue
        i += 1
    return pd.DataFrame(rows, columns=["date", "maturity", "yield"])


def latest_edition(index_html: str) -> int | None:
    m = re.search(r"PublicationsView\.aspx\?id=(\d+)>\s*Table \d+ : Month-end Yield of SGL "
                  r"Transactions in Government Dated", index_html, re.I)
    return int(m.group(1)) if m else None


def fetch() -> pd.DataFrame:
    """Download every edition, newest last, and save the merged table (newest edition wins)."""
    ids = list(EDITIONS)
    try:
        new = latest_edition(_get(INDEX))
        if new and new not in ids:
            ids.append(new)
    except OSError:
        pass
    parts = []
    for n, i in enumerate(sorted(ids)):
        d = parse(_get(PAGE.format(i)))
        parts.append(d.assign(edition=n))
        time.sleep(0.5)
    a = pd.concat(parts).sort_values("edition").drop_duplicates(["date", "maturity"], keep="last")
    months = pd.date_range(a["date"].min(), a["date"].max(), freq="ME")
    gap = months.difference(pd.DatetimeIndex(a["date"].unique()))
    if len(gap):
        raise ValueError(f"months missing from every edition: {list(gap.strftime('%b %Y'))}")
    out = data_dir()
    out.mkdir(parents=True, exist_ok=True)
    a.drop(columns="edition").to_parquet(out / "yields.parquet")
    return a


def yields(maturities=None) -> pd.DataFrame:
    """Month-end yields in % (rows: month-ends, columns: years to maturity)."""
    d = pd.read_parquet(data_dir() / "yields.parquet")
    w = d.pivot(index="date", columns="maturity", values="yield").sort_index()
    return w[list(maturities)] if maturities is not None else w
