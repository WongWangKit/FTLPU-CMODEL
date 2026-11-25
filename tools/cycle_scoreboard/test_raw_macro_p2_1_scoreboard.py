#!/usr/bin/env python3
"""Strict self-tests for the P2.1 Local-iMEM/Fetch scoreboard contract."""

from __future__ import annotations

import copy
import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from raw_macro_level1_scoreboard import TraceValidationError, run_comparison


ROOT = pathlib.Path(__file__).resolve().parent
RULES = ROOT / "raw_macro_p2_1_rules.json"
FIELDS = [
    "schema", "run_id", "case_id", "source", "global_cycle",
    "logical_cycle", "transport_tick", "phase", "sample_point",
    "event_sequence", "module", "instance", "signal", "width", "value",
    "event_type", "mapping_status", "source_ref",
]


def record(source: str, sequence: int, signal: str, value: str = "1") -> dict:
    return {
        "schema": "ftlpu.raw_macro.p2_1.trace.v1",
        "run_id": f"p2-1-selftest-{source.lower()}",
        "case_id": "P2.1-F01",
        "source": source,
        "global_cycle": sequence,
        "logical_cycle": -1,
        "transport_tick": sequence,
        "phase": "LOW",
        "sample_point": "PRE_POSEDGE",
        "event_sequence": sequence,
        "module": "architecture",
        "instance": "mem_queue_0",
        "signal": signal,
        "width": len(value),
        "value": value,
        "event_type": "ARCH_EVENT",
        "mapping_status": "SEMANTIC",
        "source_ref": "P2.1 scoreboard self-test",
    }


def trace(source: str) -> list[dict]:
    return [
        record(source, 0, "CASE_START"),
        record(source, 1, "START_FIRE"),
        record(source, 2, "REQUEST_FIRE"),
        record(source, 3, "RESPONSE_FIRE"),
        record(source, 4, "WORD_FIRE"),
        record(source, 5, "DECODER_WORD_FIRE"),
        record(source, 6, "CASE_END"),
    ]


def trace_with_descriptor_signals(
    source: str,
    *,
    start_valid: str = "1",
    base_address: str = "0011",
    word_count: str = "00010",
) -> list[dict]:
    rows = trace(source)
    for signal, width, value in (
        ("start_valid_i", 1, start_valid),
        ("start_ready_o", 1, "1"),
        ("start_base_address_i", 4, base_address),
        ("start_word_count_i", 5, word_count),
        ("start_fire", 1, start_valid),
    ):
        row = record(source, len(rows), signal, value)
        row["global_cycle"] = 7
        row["transport_tick"] = 7
        row["module"] = "lpu_mem_imem_fetch"
        row["event_type"] = "SIGNAL_SAMPLE"
        row["mapping_status"] = "DIRECT"
        row["width"] = width
        rows.append(row)
    return rows


def write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            assert set(row) == set(FIELDS)
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")


class P21ScoreboardSelfTest(unittest.TestCase):
    def compare(self, cmodel: list[dict], rtl: list[dict]) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            cmodel_path = root / "cmodel.jsonl"
            rtl_path = root / "rtl.jsonl"
            write_jsonl(cmodel_path, cmodel)
            write_jsonl(rtl_path, rtl)
            return run_comparison(cmodel_path, rtl_path, RULES)

    def test_identical_pass(self) -> None:
        self.assertEqual("PASS", self.compare(trace("CMODEL"), trace("RTL"))["status"])

    def test_cycle_shift_fails(self) -> None:
        rtl = trace("RTL")
        rtl[2]["global_cycle"] += 1
        self.assertEqual("cycle_mismatch",
            self.compare(trace("CMODEL"), rtl)["first_difference"]["category"])

    def test_value_mismatch_fails(self) -> None:
        rtl = trace("RTL")
        rtl[2]["value"] = "0"
        self.assertEqual("value_mismatch",
            self.compare(trace("CMODEL"), rtl)["first_difference"]["category"])

    def test_signal_rule_value_mismatch_fails(self) -> None:
        rtl = trace_with_descriptor_signals("RTL")
        descriptor = next(row for row in rtl if row["signal"] == "start_ready_o")
        descriptor["value"] = "0"
        self.assertEqual("value_mismatch",
            self.compare(trace_with_descriptor_signals("CMODEL"), rtl)
            ["first_difference"]["category"])

    def test_descriptor_payload_mismatch_when_valid_fails(self) -> None:
        rtl = trace_with_descriptor_signals("RTL", word_count="00100")
        self.assertEqual("value_mismatch",
            self.compare(trace_with_descriptor_signals("CMODEL"), rtl)
            ["first_difference"]["category"])

    def test_descriptor_payload_when_invalid_is_gated(self) -> None:
        cmodel = trace_with_descriptor_signals(
            "CMODEL", start_valid="0", base_address="0000", word_count="00000")
        rtl = trace_with_descriptor_signals(
            "RTL", start_valid="0", base_address="0011", word_count="00100")
        self.assertEqual("PASS", self.compare(cmodel, rtl)["status"])

    def test_xz_is_not_ignored(self) -> None:
        rtl = trace("RTL")
        rtl[2]["value"] = "x"
        self.assertEqual("xz_mismatch",
            self.compare(trace("CMODEL"), rtl)["first_difference"]["category"])

    def test_phase_mismatch_fails(self) -> None:
        rtl = trace("RTL")
        rtl[2]["phase"] = "HIGH"
        self.assertEqual("phase_mismatch",
            self.compare(trace("CMODEL"), rtl)["first_difference"]["category"])

    def test_missing_event_fails(self) -> None:
        rtl = trace("RTL")
        del rtl[3]
        for sequence, row in enumerate(rtl):
            row["event_sequence"] = sequence
        result = self.compare(trace("CMODEL"), rtl)
        self.assertEqual("FAIL", result["status"])

    def test_unobservable_is_excluded(self) -> None:
        cmodel = trace("CMODEL")
        rtl = trace("RTL")
        hidden = record("CMODEL", len(cmodel), "INTERNAL_ONLY")
        hidden["event_type"] = "STATE_SAMPLE"
        hidden["mapping_status"] = "UNOBSERVABLE"
        hidden["module"] = "cmodel_internal"
        cmodel.append(hidden)
        self.assertEqual("PASS", self.compare(cmodel, rtl)["status"])

    def test_duplicate_sequence_rejected(self) -> None:
        rtl = trace("RTL")
        rtl.insert(2, copy.deepcopy(rtl[1]))
        with self.assertRaises(TraceValidationError):
            self.compare(trace("CMODEL"), rtl)


if __name__ == "__main__":
    unittest.main(verbosity=2)
