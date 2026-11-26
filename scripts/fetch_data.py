"""Download RBI's month-end G-sec yields by maturity into data/rbi/ (not committed)."""

from yc.data import fetch

if __name__ == "__main__":
    d = fetch()
    print(f"{len(d)} values, {d['date'].min():%b %Y}-{d['date'].max():%b %Y}, "
          f"maturities {d['maturity'].min()}-{d['maturity'].max()} -> data/rbi/yields.parquet")
