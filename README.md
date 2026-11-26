# Yield Curve PCA

[![CI](https://github.com/Dhirajdas18/yield-curve-pca/actions/workflows/ci.yml/badge.svg)](https://github.com/Dhirajdas18/yield-curve-pca/actions/workflows/ci.yml)
[![PCA lab](https://github.com/Dhirajdas18/yield-curve-pca/actions/workflows/pages.yml/badge.svg)](https://dhirajdas18.github.io/yield-curve-pca/)

Principal component analysis of the Indian government bond yield curve from 1996 to 2026, on
RBI's month-end yields of Government of India dated securities from 1 to 30 years to maturity
(Handbook of Statistics on Indian Economy): the shapes the curve moves in, how stable they are,
hedging with them, what they predict, and how they compare with a Nelson-Siegel curve.

**Live PCA lab: [dhirajdas18.github.io/yield-curve-pca](https://dhirajdas18.github.io/yield-curve-pca/)**,
pick any years and maturities and see the level, slope and curvature shapes, how much of the
curve's movement each explains, and each one's path over time, computed in your browser.

## Features

- [x] PCA of monthly yield changes: level, slope and curvature, and how much each explains
- [ ] Stability of the shapes over time: rolling PCA through rate cycles
- [ ] Changes against levels, and how the maturities chosen change the answer
- [ ] Hedging a bond portfolio with the components, against duration alone, out of sample
- [ ] Does the curve predict? Slope and curvature against future rate changes, out of sample
- [ ] PCA against Nelson-Siegel: fit and stability
- [ ] Verdict: a curve-risk model and a regime input for the reference model

## Findings

### 1. Three shapes explained 95% of Indian yield moves; level alone 83%

<a href="docs/img/pca.png"><picture><source media="(prefers-reduced-motion: reduce)" srcset="docs/img/pca.png"><img src="docs/img/pca.gif" alt="Loadings of the level, slope and curvature components across 1 to 10 year maturities, and the share of yield movement each component explains in four samples" width="720"></picture></a>

RBI's month-end yields from settled (SGL) trades in Government of India dated securities, merged
from the Handbook editions since 2008 with the newest edition preferred; every month from April
1996 to June 2026 is covered. PCA is the eigen-decomposition of the covariance of monthly yield
changes in basis points, over the longest run of consecutive months with every chosen maturity:
1 to 10 years from November 1996 (356 changes), and 1, 2, 3, 5, 7, 10, 15, 20, 25 and 30 years
from December 2017 (103), when 30-year yields become complete. Signs are fixed so level loads
positively, slope rises with maturity and curvature dips in the middle. Nothing is forecast, so
nothing is out of sample here.

- **Level explained 83%.** On 1–10 years the first component explained 83.0% of monthly yield
  movement, the second (slope) 9.2% and the third (curvature) 3.3%: 95.4% together.
- **Short yields moved most.** The 1-year yield moved 47 bp a month (standard deviation) against
  30–33 bp from 3 to 10 years, so the level component leaned towards the short end (loading 0.41 at
  1 year against 0.28 at 10).
- **Three components rebuilt most months closely.** The error left after three components was 7.2
  bp a month (root mean square) against 33.9 bp of raw moves; the largest single miss was 81 bp.
- **Level mattered less after 2010.** Its share was 85.6% in 1996–2009 and 76.0% in 2010–2026, with
  slope rising from 8.3% to 11.9%.
- **On 1–30 years the third component was mostly the 10-year yield on its own.** Level 73.9%, slope
  13.0%, then 3.7% for a component loading −0.86 on the 10-year yield and under 0.3 on every other
  maturity.

## Quick start

```bash
uv sync
uv run python scripts/fetch_data.py   # RBI Handbook tables -> data/rbi/yields.parquet
uv run yc report pca                  # topics: pca
uv run python scripts/build_site.py   # PCA lab -> site/index.html
uv run pytest
```

The yields are downloaded from rbi.org.in, not committed.

## Design

- `src/yc/data.py`: fetches every Handbook edition's table of month-end SGL yields by maturity,
  parses it, merges editions (newest wins) and refuses a history with a missing month.
- `src/yc/pca.py`: PCA of yield moves with fixed signs for level, slope and curvature.
- `src/yc/yc.js`: the lab's browser copy of the PCA (Jacobi eigen-decomposition), tested against
  Python through Node.
- `src/yc/site.py`: the PCA lab, one static Plotly page carrying the monthly yields.
- `src/yc/reports.py`: README findings, reproducible with `yc report <topic>`.
- Conventions: yields in percent, changes in basis points, month-end to month-end; PCA on the
  covariance (not correlation) of the chosen maturities.
- Data: Reserve Bank of India, Handbook of Statistics on Indian Economy, "Month-end Yield of SGL
  Transactions in Government Dated Securities for Various Maturities" (earlier editions: "Yield of
  SGL Transactions in Government Dated Securities for Various Maturities").
- Charts: one validated palette; README charts are animated GIFs with a static fallback for
  reduced-motion users.

Research only, not investment advice. MIT licence.
