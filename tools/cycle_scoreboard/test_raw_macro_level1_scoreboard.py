#!/usr/bin/env python3
"""Self-tests for the RAW Macro Level1 offline cycle scoreboard."""

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
RULES = ROOT / "raw_macro_level1_rules.json"
FIELDS = [
    "schema", "run_id", "case_id", "source", "global_cycle",
    "logical_cycle", "transport_tick", "phase", "sample_point",
    "event_sequence", "module", "instance", "signal", "width", "value",
    "event_type", "mapping_status", "source_ref",
]
EVENTS = [
    "CASE_START", "RAW_WORD_ACCEPTED", "PRIME_FIRE",
    "START", "MEM_ISSUE", "CASE_END",
]


def make_trace(source: str) -> list[dict[str, object]]:
    result = []
    for sequence, signal in enumerate(EVENTS):
        width = 64 if signal == "MEM_ISSUE" else 1
        value = "0" * (width - 1) + "1"
        result.append({
            "schema": "ftlpu.raw_macro.level1.trace.v1",
            "run_id": f"selftest-{source.lower()}",
            "case_id": "selftest_case",
            "source": source,
            "global_cycle": sequence,
            "logical_cycle": sequence - 3 if sequence >= 3 else -1,
            "transport_tick": sequence,
            "phase": "LOW",
            "sample_point": "PRE_POSEDGE",
            "event_sequence": sequence,
            "module": "raw_macro_level1",
            "instance": "raw_macro_level1",
            "signal": signal,
            "width": width,
            "value": value,
            "event_type": "ARCH_EVENT",
            "mapping_status": "SEMANTIC",
            "source_ref": "scoreboard self-test fixture",
        })
    return result


def write_jsonl(path: pathlib.Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            self_fields = set(record)
            if self_fields != set(FIELDS):
                raise AssertionError(f"bad fixture fields: {self_fields}")
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")


class ScoreboardSelfTest(unittest.TestCase):
    def compare(
        self,
        cmodel: list[dict[str, object]],
        rtl: list[dict[str, object]],
    ) -> dict[str, object]:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            cmodel_path = root / "cmodel.jsonl"
            rtl_path = root / "rtl.jsonl"
            write_jsonl(cmodel_path, cmodel)
            write_jsonl(rtl_path, rtl)
            return run_comparison(cmodel_path, rtl_path, RULES)

    def test_identical_pass(self) -> None:
        result = self.compare(make_trace("CMODEL"), make_trace("RTL"))
        self.assertEqual("PASS", result["status"])

    def test_cycle_shift_fails(self) -> None:
        rtl = make_trace("RTL")
        rtl[1]["global_cycle"] = 2
        result = self.compare(make_trace("CMODEL"), rtl)
        self.assertEqual("FAIL", result["status"])
        self.assertEqual("cycle_mismatch", result["first_difference"]["category"])

    def test_value_mismatch_fails(self) -> None:
        rtl = make_trace("RTL")
        rtl[1]["value"] = "0"
        result = self.compare(make_trace("CMODEL"), rtl)
        self.assertEqual("value_mismatch", result["first_difference"]["category"])

    def test_missing_event_fails(self) -> None:
        rtl = make_trace("RTL")
        del rtl[3]
        for sequence, record in enumerate(rtl):
            record["event_sequence"] = sequence
        result = self.compare(make_trace("CMODEL"), rtl)
        self.assertEqual("FAIL", result["status"])
        self.assertIn(
            result["first_difference"]["category"],
            {"event_order_mismatch", "missing_rtl"},
        )

    def test_width_mismatch_fails(self) -> None:
        rtl = make_trace("RTL")
        rtl[4]["width"] = 65
        rtl[4]["value"] = "0" + str(rtl[4]["value"])
        result = self.compare(make_trace("CMODEL"), rtl)
        self.assertEqual("width_mismatch", result["first_difference"]["category"])

    def test_xz_is_not_silent(self) -> None:
        rtl = make_trace("RTL")
        rtl[1]["value"] = "x"
        result = self.compare(make_trace("CMODEL"), rtl)
        self.assertEqual("xz_mismatch", result["first_difference"]["category"])

    def test_phase_mismatch_fails(self) -> None:
        rtl = make_trace("RTL")
        rtl[1]["phase"] = "HIGH"
        result = self.compare(make_trace("CMODEL"), rtl)
        self.assertEqual("phase_mismatch", result["first_difference"]["category"])

    def test_duplicate_sequence_fails_validation(self) -> None:
        rtl = make_trace("RTL")
        rtl.insert(2, copy.deepcopy(rtl[1]))
        with self.assertRaises(TraceValidationError):
            self.compare(make_trace("CMODEL"), rtl)


if __name__ == "__main__":
    unittest.main(verbosity=2)
