"""Command line: `yc report <topic>` builds a topic's charts and prints its numbers."""

import argparse
from pathlib import Path

OUT = Path("out")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="yc", description="Indian yield curve PCA")
    sub = parser.add_subparsers(dest="cmd", required=True)
    rep = sub.add_parser("report", help="build a topic's charts and print its numbers")
    rep.add_argument("topic")
    args = parser.parse_args(argv)

    from yc import plots, reports

    if args.topic not in reports.REPORTS:
        topics = ", ".join(reports.REPORTS) or "none yet"
        parser.error(f"unknown topic {args.topic!r}; topics: {topics}")
    plots.style()
    for key, value in reports.REPORTS[args.topic](OUT).items():
        print(f"{key}: {value}")
    print(f"charts -> {OUT.resolve()}")
    return 0
