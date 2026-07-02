#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


LINE_RE = re.compile(
    r"fps=(?P<fps>[0-9.]+)\s+"
    r"pre_us=(?P<pre_us>\d+)\s+"
    r"infer_us=(?P<infer_us>\d+)\s+"
    r"post_us=(?P<post_us>\d+)\s+"
    r"mask=(?P<mask>[0-9.]+)\s+"
    r"heap=(?P<heap>\d+)\s+"
    r"psram=(?P<psram>\d+)"
)


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    args = parser.parse_args()

    rows: list[dict[str, float]] = []
    for line in args.log.read_text(errors="replace").splitlines():
        match = LINE_RE.search(line)
        if not match:
            continue
        rows.append({key: float(value) for key, value in match.groupdict().items()})

    report = {
        "samples": len(rows),
        "fps": summarize([row["fps"] for row in rows]),
        "pre_us": summarize([row["pre_us"] for row in rows]),
        "infer_us": summarize([row["infer_us"] for row in rows]),
        "post_us": summarize([row["post_us"] for row in rows]),
        "mask_ratio": summarize([row["mask"] for row in rows]),
        "min_heap": min((row["heap"] for row in rows), default=None),
        "min_psram": min((row["psram"] for row in rows), default=None),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
