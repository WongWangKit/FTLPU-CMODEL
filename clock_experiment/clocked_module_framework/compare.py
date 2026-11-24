#!/usr/bin/env python3
"""Exact C++/RTL trace comparison plus independent four-phase semantic audit."""

from __future__ import annotations

import csv
import sys
from pathlib import Path


def load(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"missing CSV header: {path}")
        rows = [{key: value.strip().lower() for key, value in row.items()} for row in reader]
        return reader.fieldnames, rows


def number(value: str) -> int | None:
    return None if value == "x" else int(value)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: compare.py <cmodel_trace.csv> <rtl_trace.csv>", file=sys.stderr)
        return 2
    c_header, c_rows = load(Path(sys.argv[1]))
    r_header, r_rows = load(Path(sys.argv[2]))
    failures: list[str] = []
    mismatches = 0

    if c_header != r_header:
        failures.append("CSV headers differ")
    if len(c_rows) != len(r_rows):
        failures.append(f"row count differs: cmodel={len(c_rows)} rtl={len(r_rows)}")
    for index, (c_row, r_row) in enumerate(zip(c_rows, r_rows)):
        for field in c_header:
            if c_row[field] != r_row[field]:
                mismatches += 1
                if mismatches <= 12:
                    failures.append(
                        f"mismatch row={index} field={field} cmodel={c_row[field]} rtl={r_row[field]}"
                    )

    phase1_state = [
        "src_q", "dst_q", "bank0_q", "bank1_q", "fifo_wr_ptr", "fifo_rd_ptr",
        "fifo_count", "fifo_rd_data", "fifo_rd_data_valid", "fifo_m0", "fifo_m1",
        "fifo_m2", "fifo_m3", "fifo_v0", "fifo_v1", "fifo_v2", "fifo_v3",
        "sram_rd_data", "sram_rd_valid", "sram_rd_initialized", "sram_m0", "sram_m1",
        "sram_m2", "sram_m3", "sram_i0", "sram_i1", "sram_i2", "sram_i3",
    ]
    next_fields = [field for field in c_header if "_next" in field or field.startswith("fifo_n_") or field.startswith("sram_n_")]
    full_cycles = empty_cycles = simultaneous = wr_wraps = rd_wraps = resets = 0
    sram_uninitialized_reads = sram_initialized_reads = 0

    if len(c_rows) % 4:
        failures.append("trace row count is not a multiple of four")
    for base in range(0, len(c_rows) - 3, 4):
        rows = c_rows[base : base + 4]
        cycle = base // 4
        if [row["phase"] for row in rows] != ["0", "1", "2", "3"]:
            failures.append(f"cycle {cycle}: illegal phase sequence")
            continue
        if [row["clk"] for row in rows] != ["0", "1", "1", "0"]:
            failures.append(f"cycle {cycle}: illegal clock levels")
        if any(row["cycle"] != str(cycle) for row in rows):
            failures.append(f"cycle {cycle}: cycle counter changed within four phases")

        p0, p1, p2, p3 = rows
        if p0["state_valid"] == "1":
            expected_wr = int(p0["fifo_wr_en"]) and int(p0["fifo_wr_ready"]) and not int(p0["rst"])
            expected_rd = int(p0["fifo_rd_en"]) and int(p0["fifo_rd_available"]) and not int(p0["rst"])
            if int(p0["fifo_wr_fire"]) != int(bool(expected_wr)):
                failures.append(f"cycle {cycle}: FIFO wr_fire contract")
            if int(p0["fifo_rd_fire"]) != int(bool(expected_rd)):
                failures.append(f"cycle {cycle}: FIFO rd_fire contract")
            full_cycles += int(p0["fifo_full"])
            empty_cycles += int(p0["fifo_empty"])
            simultaneous += int(p0["fifo_wr_en"] == "1" and p0["fifo_rd_en"] == "1")
            if p0["fifo_wr_ptr"] == "3" and p1["fifo_wr_ptr"] == "0":
                wr_wraps += 1
            if p0["fifo_rd_ptr"] == "3" and p1["fifo_rd_ptr"] == "0":
                rd_wraps += 1
        resets += int(p0["rst"] == "1")

        mappings = {
            "src_q": "src_next", "dst_q": "dst_next",
            "bank0_q": "bank0_next", "bank1_q": "bank1_next",
            "fifo_wr_ptr": "fifo_n_wr_ptr", "fifo_rd_ptr": "fifo_n_rd_ptr",
            "fifo_count": "fifo_n_count", "fifo_rd_data": "fifo_n_rd_data",
            "fifo_rd_data_valid": "fifo_n_rd_data_valid",
            "sram_rd_data": "sram_n_rd_data", "sram_rd_valid": "sram_n_rd_valid",
            "sram_rd_initialized": "sram_n_rd_initialized",
        }
        for current, prepared in mappings.items():
            if p1[current] != p0[prepared]:
                failures.append(f"cycle {cycle}: {current} did not commit prepared {prepared}")

        for field in phase1_state:
            if not (p1[field] == p2[field] == p3[field]):
                failures.append(f"cycle {cycle}: sequential field changed after posedge: {field}")
        for field in next_fields:
            if not (p0[field] == p1[field] == p2[field] == p3[field]):
                failures.append(f"cycle {cycle}: prepared Next changed after Phase 0: {field}")

        src = number(p2["src_q"])
        if src is not None:
            result16 = ((((src * 3) + 17) * 5 + 29) * 7 + 43) & 0xFFFF
            if number(p2["mac_result16"]) != result16 or number(p2["mac_result8"]) != (result16 & 0xFF):
                failures.append(f"cycle {cycle}: Phase 2 MAC did not consume new Source Current")
        if p1["sram_rd_valid"] == "1":
            if p1["sram_rd_initialized"] == "1":
                sram_initialized_reads += 1
            else:
                sram_uninitialized_reads += 1

    coverage = {
        "full_cycles": full_cycles,
        "empty_cycles": empty_cycles,
        "simultaneous_requests": simultaneous,
        "write_pointer_wraps": wr_wraps,
        "read_pointer_wraps": rd_wraps,
        "reset_cycles": resets,
        "sram_initialized_reads": sram_initialized_reads,
        "sram_uninitialized_reads": sram_uninitialized_reads,
    }
    for name, value in coverage.items():
        if value == 0:
            failures.append(f"coverage condition not reached: {name}")
    if resets < 2:
        failures.append("mid-traffic reset not covered")

    semantic_failures = len(failures) - min(mismatches, 12)
    print(f"cmodel_rows={len(c_rows)}")
    print(f"rtl_rows={len(r_rows)}")
    print(f"mismatches={mismatches}")
    for key, value in coverage.items():
        print(f"coverage_{key}={value}")
    for failure in failures[:40]:
        print(f"FAIL: {failure}")
    print(f"semantic_failures={max(0, semantic_failures)}")
    print(f"failures={len(failures)}")
    if mismatches == 0 and not failures:
        print("result=PASS")
        return 0
    print("result=FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

