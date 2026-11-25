#!/usr/bin/env python3
"""Compare the five MXM Local-iMEM/Fetch/Decoder/Context frontends."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from typing import Any

from raw_macro_level1_scoreboard import (
    TraceValidationError,
    run_comparison,
    write_reports,
)


STREAMS = ("load", "compute", "dequant", "param_load", "backend")
PRE_SIGNALS = (
    ("fetch_start_valid", "start_valid_i", 1, "DIRECT"),
    ("fetch_start_ready", "start_ready_o", 1, "DIRECT"),
    ("fetch_start_fire", "start_fire", 1, "DERIVED"),
    ("imem_request_valid", "imem_read_req_valid_o", 1, "DIRECT"),
    ("imem_request_ready", "imem_read_req_ready_i", 1, "DIRECT"),
    ("imem_request_fire", "imem_read_req_fire", 1, "DERIVED"),
    ("imem_request_address", "imem_read_req_address_o", 17, "DIRECT"),
    ("imem_request_last", "imem_read_req_last_o", 1, "DIRECT"),
    ("imem_response_valid", "imem_read_rsp_valid_i", 1, "DIRECT"),
    ("imem_response_ready", "imem_read_rsp_ready_o", 1, "DIRECT"),
    ("imem_response_fire", "imem_read_rsp_fire", 1, "DERIVED"),
    ("imem_response_last", "imem_read_rsp_last_i", 1, "DIRECT"),
    ("fetch_word_valid", "word_valid_o", 1, "DIRECT"),
    ("fetch_word_ready", "word_ready_i", 1, "DIRECT"),
    ("fetch_word_fire", "word_fire", 1, "DERIVED"),
    ("fetch_word_last", "word_last_o", 1, "DIRECT"),
    ("decoder_word_valid", "decoder_word_valid_i", 1, "DIRECT"),
    ("decoder_word_ready", "decoder_word_ready_o", 1, "DIRECT"),
    ("decoder_word_fire", "decoder_word_fire", 1, "DERIVED"),
    ("decoder_word_last", "decoder_word_last_i", 1, "DIRECT"),
    ("context_enqueue_valid", "context_enqueue_valid_i", 1, "DIRECT"),
    ("context_enqueue_ready", "context_enqueue_ready_o", 1, "DIRECT"),
    ("context_enqueue_fire", "context_enqueue_fire", 1, "DERIVED"),
    ("context_dequeue_valid", "context_dequeue_valid_o", 1, "DIRECT"),
    ("context_dequeue_ready", "context_dequeue_ready_i", 1, "DIRECT"),
    ("context_dequeue_fire", "context_dequeue_fire", 1, "DERIVED"),
)
POST_SIGNALS = (
    ("word_fifo_level", "word_fifo_level_o", 32, "DIRECT"),
    ("context_fifo_level", "context_fifo_level_o", 32, "DIRECT"),
    ("active_count", "active_count_o", 32, "DIRECT"),
    ("fetch_complete", "fetch_complete_q", 1, "DIRECT"),
    ("decoder_done", "decoder_done_o", 1, "DIRECT"),
    ("frontend_ready", "frontend_ready_o", 1, "DIRECT"),
    ("fault", "fault_o", 1, "DIRECT"),
)
EXPECTED_COLUMNS = [
    "cycle", "phase",
    *(f"{stream}_{signal[0]}" for stream in STREAMS
      for signal in (*PRE_SIGNALS, *POST_SIGNALS)),
]


def _logic(value: str, width: int, source: str) -> str:
    stripped = value.strip()
    if any(bit in "xXzZ" for bit in stripped):
        if len(stripped) != width or any(bit not in "xXzZ" for bit in stripped):
            raise TraceValidationError(
                f"{source}: invalid {width}-bit X/Z value {value!r}")
        return stripped.lower()
    try:
        integer = int(stripped, 10)
    except ValueError as error:
        raise TraceValidationError(
            f"{source}: invalid integer {value!r}") from error
    if integer < 0 or integer >= (1 << width):
        raise TraceValidationError(
            f"{source}: value {integer} exceeds {width} bits")
    return format(integer, f"0{width}b")


def normalize(source_path: pathlib.Path, source: str,
              destination: pathlib.Path) -> dict[str, Any]:
    with source_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise TraceValidationError(
                f"{source_path}: columns differ; expected={EXPECTED_COLUMNS} "
                f"actual={reader.fieldnames}")
        rows = list(reader)
    if len(rows) != 16:
        raise TraceValidationError(
            f"{source_path}: expected 16 cycles, got {len(rows)}")

    normalized: list[dict[str, Any]] = []

    def append(cycle: int, instance: str, signal: str, width: int,
               value: str, status: str, phase: str, sample_point: str,
               line: int) -> None:
        normalized.append({
            "schema": "ftlpu.mxm.full_path.c7.frontend.normalized.v1",
            "run_id": "mxm_full_path_c7_frontend",
            "case_id": "five_stream_two_word_frontend",
            "source": source,
            "global_cycle": cycle,
            "logical_cycle": cycle,
            "transport_tick": cycle,
            "phase": phase,
            "sample_point": sample_point,
            "event_sequence": len(normalized),
            "module": "lpu_mxm_local_imem_frontend",
            "instance": instance,
            "signal": signal,
            "width": width,
            "value": _logic(value, width,
                f"{source_path}:{line}:{instance}:{signal}"),
            "event_type": "SIGNAL_SAMPLE",
            "mapping_status": status,
            "source_ref": f"{source_path}:{line}",
        })

    for expected_cycle, row in enumerate(rows):
        line = expected_cycle + 2
        try:
            cycle = int(row["cycle"], 10)
        except ValueError as error:
            raise TraceValidationError(
                f"{source_path}:{line}: invalid cycle") from error
        if cycle != expected_cycle or row["phase"] != "BOTH":
            raise TraceValidationError(
                f"{source_path}:{line}: expected cycle {expected_cycle} BOTH")
        for instance in STREAMS:
            for csv_name, signal, width, status in PRE_SIGNALS:
                append(cycle, instance, signal, width,
                    row[f"{instance}_{csv_name}"], status,
                    "LOW", "PRE_POSEDGE", line)
            for csv_name, signal, width, status in POST_SIGNALS:
                append(cycle, instance, signal, width,
                    row[f"{instance}_{csv_name}"], status,
                    "HIGH", "POST_POSEDGE", line)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for record in normalized:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    return {"rows": len(rows), "signals_per_cycle": 165,
            "normalized_samples": len(normalized)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmodel", type=pathlib.Path, required=True)
    parser.add_argument("--rtl", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--rules", type=pathlib.Path,
        default=pathlib.Path(__file__).with_name(
            "mxm_full_path_c7_frontend_rules.json"))
    args = parser.parse_args(argv)
    try:
        normalized = args.output_dir / "normalized"
        summaries = {
            "CMODEL": normalize(args.cmodel, "CMODEL",
                normalized / "cmodel.jsonl"),
            "RTL": normalize(args.rtl, "RTL",
                normalized / "rtl.jsonl"),
        }
        result = run_comparison(normalized / "cmodel.jsonl",
            normalized / "rtl.jsonl", args.rules, None, 1_000_000)
        result["joint_verdict"] = (
            "MXM_C7_FRONTEND_CYCLE_PASS" if result["status"] == "PASS"
            else "MXM_C7_FRONTEND_CYCLE_FAIL")
        result["flat_trace_summary"] = summaries
        write_reports(result, args.output_dir)
        print(result["joint_verdict"])
        print(f"Compared: {result['compared_pairs']}")
        print(f"Matched: {result['matched_pairs']}")
        print(f"Differences: {result['difference_count']}")
        if result["first_difference"]:
            print(f"First difference: {result['first_difference']}")
        return 0 if result["status"] == "PASS" else 1
    except (OSError, ValueError, TraceValidationError) as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"MXM_C7_FRONTEND_CYCLE_FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
