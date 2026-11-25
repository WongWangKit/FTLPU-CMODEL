#!/usr/bin/env python3
"""Phase1 MEM-space wrapper around the shared cycle scoreboard."""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
from typing import Any

from raw_macro_level1_scoreboard import run_comparison


def load_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def spatial_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_case: collections.Counter[str] = collections.Counter()
    per_coordinate: collections.Counter[tuple[str, int, int]] = collections.Counter()
    per_event: collections.Counter[str] = collections.Counter()
    for row in rows:
        if row.get("event_type") != "ARCH_EVENT":
            continue
        per_case[str(row["case_id"])] += 1
        per_event[str(row["signal"])] += 1
        coordinate = (str(row["hemisphere"]), int(row["mem_slice"]),
                      int(row["bank"]))
        if coordinate[1] >= 0 and coordinate[2] >= 0:
            per_coordinate[coordinate] += 1
    return {
        "architecture_rows": sum(per_case.values()),
        "per_case": dict(sorted(per_case.items())),
        "per_event": dict(sorted(per_event.items())),
        "per_coordinate": [
            {"hemisphere": coordinate[0], "mem_slice": coordinate[1],
             "bank": coordinate[2], "events": count}
            for coordinate, count in sorted(per_coordinate.items())
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmodel", required=True, type=pathlib.Path)
    parser.add_argument("--rtl", required=True, type=pathlib.Path)
    parser.add_argument("--rules", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()
    result = run_comparison(args.cmodel, args.rtl, args.rules)
    result["phase1_space"] = {
        "profile": args.profile,
        "cmodel": spatial_coverage(load_rows(args.cmodel)),
        "rtl": spatial_coverage(load_rows(args.rtl)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n",
                           encoding="utf-8")
    print(f"{result['status']} profile={args.profile} "
          f"differences={result['difference_count']}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
