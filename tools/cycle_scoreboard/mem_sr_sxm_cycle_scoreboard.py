#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

REQUIRED = {
    "schema", "run_id", "case_id", "source", "global_cycle",
    "logical_cycle", "transport_tick", "phase", "sample_point",
    "event_sequence", "module", "instance", "signal", "width", "value",
    "event_type", "mapping_status", "source_ref",
}


def load(path: Path, expected_source: str):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        rows = list(reader)
    result = {}
    for row in rows:
        if row["source"] != expected_source:
            raise ValueError(f"{path}: source {row['source']} != {expected_source}")
        if row["event_type"] != "SIGNAL_SAMPLE" or row["mapping_status"] != "DIRECT":
            raise ValueError(f"{path}: non-comparable row {row}")
        key = (
            row["case_id"], int(row["global_cycle"]), row["phase"],
            row["sample_point"], row["module"], row["instance"],
            row["signal"], int(row["width"]),
        )
        if key in result:
            raise ValueError(f"{path}: duplicate key {key}")
        result[key] = row["value"].strip().lower()
    return result, len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmodel", type=Path, required=True)
    parser.add_argument("--rtl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--pass-verdict", required=True)
    args = parser.parse_args()

    cmodel, cmodel_rows = load(args.cmodel, "CMODEL")
    rtl, rtl_rows = load(args.rtl, "RTL")
    keys = sorted(set(cmodel) | set(rtl))
    differences = []
    matched = 0
    xz_samples = 0
    for key in keys:
        cv = cmodel.get(key)
        rv = rtl.get(key)
        if rv is not None:
            payload = rv[2:] if rv.startswith("0x") else rv
            if any(ch in payload for ch in "xz"):
                xz_samples += 1
        if cv == rv:
            matched += 1
        else:
            differences.append({"key": list(key), "cmodel": cv, "rtl": rv})

    verdict = args.pass_verdict if not differences else f"{args.stage}_CYCLE_FAIL"
    summary = {
        "stage": args.stage,
        "verdict": verdict,
        "status": "PASS" if not differences else "FAIL",
        "cmodel_rows": cmodel_rows,
        "rtl_rows": rtl_rows,
        "compared_pairs": len(keys),
        "matched_pairs": matched,
        "difference_count": len(differences),
        "rtl_xz_samples": xz_samples,
        "first_difference": differences[0] if differences else None,
    }
    if differences:
        first_key = tuple(differences[0]["key"])
        index = keys.index(first_key)
        summary["first_difference_context"] = [
            {"key": list(key), "cmodel": cmodel.get(key), "rtl": rtl.get(key)}
            for key in keys[max(0, index - 3):index + 4]
        ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scoreboard_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "scoreboard_summary.txt").write_text(
        f"{args.stage} MEM-SR-SXM Cycle Scoreboard\n"
        f"Verdict: {verdict}\nCompared: {len(keys)}\nMatched: {matched}\n"
        f"Differences: {len(differences)}\nRows: CModel={cmodel_rows} RTL={rtl_rows}\n"
        f"RTL X/Z samples: {xz_samples}\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    raise SystemExit(0 if not differences else 1)


if __name__ == "__main__":
    main()
