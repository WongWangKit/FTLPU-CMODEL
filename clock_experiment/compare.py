#!/usr/bin/env python3
import csv
import sys
from pathlib import Path


BASE_FIELDS = [
    "half_clk", "cycle", "phase", "clk_level", "valid",
    "rst", "en", "din",
]
LOW_FIELDS = ["q1", "q2", "dout", "d1", "d2"]
HIGH_FIELDS = ["q1", "q2", "dout"]


def read_trace(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def context(rows, index, radius=2):
    start = max(0, index - radius)
    stop = min(len(rows), index + radius + 1)
    return rows[start:stop]


def main():
    if len(sys.argv) != 3:
        print("usage: compare.py <cpp_trace.csv> <rtl_trace.csv>", file=sys.stderr)
        return 2

    cpp_path = Path(sys.argv[1])
    rtl_path = Path(sys.argv[2])
    cpp_rows = read_trace(cpp_path)
    rtl_rows = read_trace(rtl_path)
    mismatch_count = 0
    first = None

    if len(cpp_rows) != len(rtl_rows):
        mismatch_count += 1
        first = (min(len(cpp_rows), len(rtl_rows)), "row_count",
                 str(len(cpp_rows)), str(len(rtl_rows)))

    for index, (cpp, rtl) in enumerate(zip(cpp_rows, rtl_rows)):
        fields = list(BASE_FIELDS)
        both_valid = cpp.get("valid") == "1" and rtl.get("valid") == "1"
        if cpp.get("phase") == "LOW":
            # d1/d2 are deterministic under the initial reset stimulus even
            # though the pre-reset q1/q2/dout values are intentionally invalid.
            fields += ["d1", "d2"]
            if both_valid:
                fields += ["q1", "q2", "dout"]
        elif both_valid:
            fields += HIGH_FIELDS

        for field in fields:
            cpp_value = cpp.get(field, "").strip().lower()
            rtl_value = rtl.get(field, "").strip().lower()
            if cpp_value != rtl_value:
                mismatch_count += 1
                if first is None:
                    first = (index, field, cpp_value, rtl_value)

    print(f"cpp_samples={len(cpp_rows)}")
    print(f"rtl_samples={len(rtl_rows)}")
    print(f"mismatches={mismatch_count}")

    if first is not None:
        index, field, cpp_value, rtl_value = first
        print(f"first_mismatch_row={index}")
        if index < len(cpp_rows):
            row = cpp_rows[index]
            print(f"first_mismatch_half_clk={row.get('half_clk')}")
            print(f"first_mismatch_cycle={row.get('cycle')}")
            print(f"first_mismatch_phase={row.get('phase')}")
        print(f"first_mismatch_field={field}")
        print(f"cpp_value={cpp_value}")
        print(f"rtl_value={rtl_value}")
        print("context:")
        cpp_ctx = context(cpp_rows, min(index, max(len(cpp_rows) - 1, 0)))
        rtl_ctx = context(rtl_rows, min(index, max(len(rtl_rows) - 1, 0)))
        for cpp_row, rtl_row in zip(cpp_ctx, rtl_ctx):
            print(f"  CPP {cpp_row}")
            print(f"  RTL {rtl_row}")
        return 1

    print("RESULT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
