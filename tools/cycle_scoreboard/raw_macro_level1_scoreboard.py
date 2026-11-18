#!/usr/bin/env python3
"""Offline, rule-driven cycle scoreboard for RAW Macro Level1 JSONL traces."""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Any, Iterable


class TraceValidationError(RuntimeError):
    """The trace is malformed or violates the frozen Level1 schema."""


@dataclass(frozen=True)
class LoadedTrace:
    path: pathlib.Path
    source: str
    records: list[dict[str, Any]]
    sequence_mode: str


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _has_xz(value: str) -> bool:
    return any(bit in "xXzZ" for bit in value)


def _record_summary(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "line": record["_line"],
        "case_id": record["case_id"],
        "global_cycle": record["global_cycle"],
        "logical_cycle": record["logical_cycle"],
        "transport_tick": record["transport_tick"],
        "phase": record["phase"],
        "sample_point": record["sample_point"],
        "module": record["module"],
        "instance": record["instance"],
        "signal": record["signal"],
        "width": record["width"],
        "value": record["value"],
        "event_type": record["event_type"],
        "mapping_status": record["mapping_status"],
        "source_ref": record["source_ref"],
    }


def load_rules(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        rules = json.load(stream)
    if not isinstance(rules, dict) or "schema" not in rules:
        raise TraceValidationError(f"invalid rules file: {path}")
    return rules


def _validate_sequences(records: list[dict[str, Any]], path: pathlib.Path) -> str:
    sequence = [record["event_sequence"] for record in records]
    if sequence == list(range(len(sequence))):
        return "GLOBAL_CONTIGUOUS"

    groups: dict[tuple[Any, ...], list[int]] = collections.defaultdict(list)
    for record in records:
        bucket = (
            record["case_id"],
            record["global_cycle"],
            record["phase"],
            record["sample_point"],
        )
        groups[bucket].append(record["event_sequence"])
    for bucket, values in groups.items():
        if values != list(range(len(values))):
            raise TraceValidationError(
                f"{path}: non-contiguous or duplicate event_sequence in "
                f"sample bucket {bucket}: {values[:12]}"
            )
    return "PER_SAMPLE_CONTIGUOUS"


def load_trace(path: pathlib.Path, rules: dict[str, Any], expected_source: str) -> LoadedTrace:
    required = rules["required_fields"]
    required_set = set(required)
    allowed_types = set(rules["allowed_event_types"])
    allowed_mapping = set(rules["allowed_mapping_statuses"])
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise TraceValidationError(f"{path}:{line_number}: blank JSONL row")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise TraceValidationError(f"{path}:{line_number}: {error}") from error
            if not isinstance(record, dict):
                raise TraceValidationError(f"{path}:{line_number}: row is not an object")
            fields = set(record)
            if fields != required_set:
                missing = sorted(required_set - fields)
                extra = sorted(fields - required_set)
                raise TraceValidationError(
                    f"{path}:{line_number}: schema fields differ; "
                    f"missing={missing}, extra={extra}"
                )
            if record["schema"] != rules["schema"]:
                raise TraceValidationError(
                    f"{path}:{line_number}: schema {record['schema']!r} != {rules['schema']!r}"
                )
            if record["source"] != expected_source:
                raise TraceValidationError(
                    f"{path}:{line_number}: source {record['source']!r} != {expected_source!r}"
                )
            for field in (
                "schema", "run_id", "case_id", "source", "phase",
                "sample_point", "module", "instance", "signal", "value",
                "event_type", "mapping_status", "source_ref",
            ):
                if not isinstance(record[field], str):
                    raise TraceValidationError(
                        f"{path}:{line_number}: {field} is not a string"
                    )
            for field in (
                "global_cycle", "logical_cycle", "transport_tick",
                "event_sequence", "width",
            ):
                if not _is_int(record[field]):
                    raise TraceValidationError(
                        f"{path}:{line_number}: {field} is not an integer"
                    )
            if record["global_cycle"] < 0 or record["transport_tick"] < 0:
                raise TraceValidationError(f"{path}:{line_number}: negative physical time")
            if record["event_sequence"] < 0 or record["width"] <= 0:
                raise TraceValidationError(f"{path}:{line_number}: invalid sequence/width")
            if len(record["value"]) != record["width"]:
                raise TraceValidationError(
                    f"{path}:{line_number}: width={record['width']} but "
                    f"value has {len(record['value'])} bits"
                )
            if any(bit not in "01xXzZ" for bit in record["value"]):
                raise TraceValidationError(
                    f"{path}:{line_number}: value contains non-binary/non-XZ characters"
                )
            if record["event_type"] not in allowed_types:
                raise TraceValidationError(
                    f"{path}:{line_number}: invalid event_type {record['event_type']!r}"
                )
            if record["mapping_status"] not in allowed_mapping:
                raise TraceValidationError(
                    f"{path}:{line_number}: invalid mapping_status "
                    f"{record['mapping_status']!r}"
                )
            if record["mapping_status"] == "CONFLICT":
                raise TraceValidationError(
                    f"{path}:{line_number}: mapping_status CONFLICT is fatal"
                )
            record["_line"] = line_number
            record["_index"] = len(records)
            records.append(record)

    if not records:
        raise TraceValidationError(f"{path}: empty trace")
    sequence_mode = _validate_sequences(records, path)
    return LoadedTrace(path, expected_source, records, sequence_mode)


def _source_view(rule: dict[str, Any], source: str) -> dict[str, str]:
    return rule.get("source_views", {}).get(source, {})


def _view_matches(record: dict[str, Any], rule: dict[str, Any], source: str) -> bool:
    view = _source_view(rule, source)
    return (
        ("phase" not in view or record["phase"] == view["phase"])
        and ("sample_point" not in view
             or record["sample_point"] == view["sample_point"])
    )


def _lookup_signal(
    trace: LoadedTrace,
    record: dict[str, Any],
    signal: str,
    rule: dict[str, Any],
) -> list[dict[str, Any]]:
    view = _source_view(rule, trace.source)
    phase = view.get("phase", record["phase"])
    sample = view.get("sample_point", record["sample_point"])
    return [
        candidate for candidate in trace.records
        if candidate["case_id"] == record["case_id"]
        and candidate["global_cycle"] == record["global_cycle"]
        and candidate["phase"] == phase
        and candidate["sample_point"] == sample
        and candidate["instance"] == record["instance"]
        and candidate["signal"] == signal
    ]


def _gate_allows(
    trace: LoadedTrace,
    record: dict[str, Any],
    rule: dict[str, Any],
) -> bool:
    gate = rule.get("gate")
    if gate is None:
        return True
    candidates = _lookup_signal(trace, record, gate["signal"], rule)
    return any(candidate["value"] == gate["equals"] for candidate in candidates)


def _normalized_key(
    record: dict[str, Any],
    rule: dict[str, Any],
    occurrence: int,
) -> tuple[Any, ...]:
    module = rule.get("module_aliases", {}).get(
        record["module"], rule.get("canonical_module", record["module"])
    )
    if "canonical_module" in rule:
        module = rule["canonical_module"]
    instance = rule.get("canonical_instance", record["instance"])
    phase = rule.get("canonical_phase", record["phase"])
    sample = rule.get("canonical_sample_point", record["sample_point"])
    return (
        record["case_id"],
        record["global_cycle"],
        record["logical_cycle"],
        record["transport_tick"],
        phase,
        sample,
        module,
        instance,
        record["signal"],
        record["event_type"],
        occurrence,
    )


def _event_context(trace: LoadedTrace, record: dict[str, Any] | None) -> dict[str, Any]:
    if record is None:
        return {"previous": [], "next": []}
    index = record["_index"]
    previous = [
        _record_summary(item)
        for item in trace.records[max(0, index - 2):index]
    ]
    following = [
        _record_summary(item)
        for item in trace.records[index + 1:index + 3]
    ]
    return {"previous": previous, "next": following}


def _possible_cause(category: str) -> str:
    return {
        "cycle_mismatch": "pipeline latency, load cadence, PRIME completion, or reset epoch differs",
        "phase_mismatch": "the two monitors sampled different clock phases or state boundaries",
        "sample_point_mismatch": "pre-edge and post-edge observations were paired incorrectly",
        "value_mismatch": "the decoded value or handshake/state behavior differs",
        "xz_mismatch": "an X/Z reached an ungated comparable field; initialization or validity differs",
        "width_mismatch": "the physical field/container width differs",
        "missing_cmodel": "RTL produced a ruled event that CModel did not produce",
        "missing_rtl": "CModel produced a ruled event that RTL did not produce",
        "event_order_mismatch": "architecture events were emitted in a different order",
        "logical_cycle_mismatch": "the schedule timebase or START boundary differs",
        "transport_tick_mismatch": "transport progression differs from the common physical cycle",
        "module_mismatch": "the event was attributed to a different non-aliased module",
        "instance_mismatch": "the event was attributed to a different endpoint",
        "event_type_mismatch": "the same event was classified differently",
    }.get(category, "input, reset, mapping, or timing contract differs")


def _difference(
    category: str,
    cmodel: dict[str, Any] | None,
    rtl: dict[str, Any] | None,
    cmodel_trace: LoadedTrace,
    rtl_trace: LoadedTrace,
    rule_name: str,
) -> dict[str, Any]:
    anchor = cmodel if cmodel is not None else rtl
    assert anchor is not None
    return {
        "test": "raw_macro_level1_cycle_scoreboard",
        "status": "FAIL",
        "category": category,
        "rule": rule_name,
        "case_id": anchor["case_id"],
        "global_cycle": anchor["global_cycle"],
        "logical_cycle": anchor["logical_cycle"],
        "phase": anchor["phase"],
        "sample_point": anchor["sample_point"],
        "module": anchor["module"],
        "instance": anchor["instance"],
        "signal": anchor["signal"],
        "cmodel": _record_summary(cmodel),
        "rtl": _record_summary(rtl),
        "cmodel_context": _event_context(cmodel_trace, cmodel),
        "rtl_context": _event_context(rtl_trace, rtl),
        "possible_cause": _possible_cause(category),
    }


def _annotate_normalized(
    difference: dict[str, Any],
    module: str,
    instance: str,
    phase: str,
    sample_point: str,
    occurrence: int,
) -> dict[str, Any]:
    difference["normalized_module"] = module
    difference["normalized_instance"] = instance
    difference["normalized_phase"] = phase
    difference["normalized_sample_point"] = sample_point
    difference["event_occurrence"] = occurrence
    return difference


def _compare_architecture(
    cmodel: LoadedTrace,
    rtl: LoadedTrace,
    rules: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = rules["architecture"]
    event_names = set(config["events"])
    comparable = set(rules["comparable_mapping_statuses"])

    def selected(trace: LoadedTrace, case_id: str) -> list[dict[str, Any]]:
        return [
            record for record in trace.records
            if record["case_id"] == case_id
            and record["event_type"] == "ARCH_EVENT"
            and record["signal"] in event_names
            and record["mapping_status"] in comparable
        ]

    differences: list[dict[str, Any]] = []
    coverage = {"eligible": 0, "compared": 0, "matched": 0, "mismatched": 0}
    cases = sorted(
        {record["case_id"] for record in cmodel.records}
        | {record["case_id"] for record in rtl.records}
    )
    for case_id in cases:
        left = selected(cmodel, case_id)
        right = selected(rtl, case_id)
        coverage["eligible"] += len(left) + len(right)
        count = max(len(left), len(right))
        signal_occurrences: collections.Counter[str] = collections.Counter()
        for index in range(count):
            c_record = left[index] if index < len(left) else None
            r_record = right[index] if index < len(right) else None
            if c_record is None:
                differences.append(_annotate_normalized(_difference(
                    "missing_cmodel", None, r_record, cmodel, rtl, "architecture"
                ), config["canonical_module"], config["canonical_instance"],
                    config["phase"], config["sample_point"], index))
                coverage["mismatched"] += 1
                continue
            if r_record is None:
                differences.append(_annotate_normalized(_difference(
                    "missing_rtl", c_record, None, cmodel, rtl, "architecture"
                ), config["canonical_module"], config["canonical_instance"],
                    config["phase"], config["sample_point"], index))
                coverage["mismatched"] += 1
                continue
            occurrence = signal_occurrences[c_record["signal"]]
            signal_occurrences[c_record["signal"]] += 1
            coverage["compared"] += 1
            categories: list[str] = []
            if c_record["signal"] != r_record["signal"]:
                categories.append("event_order_mismatch")
            if c_record["global_cycle"] != r_record["global_cycle"]:
                categories.append("cycle_mismatch")
            if c_record["logical_cycle"] != r_record["logical_cycle"]:
                categories.append("logical_cycle_mismatch")
            if c_record["transport_tick"] != r_record["transport_tick"]:
                categories.append("transport_tick_mismatch")
            if c_record["phase"] != r_record["phase"]:
                categories.append("phase_mismatch")
            if c_record["sample_point"] != r_record["sample_point"]:
                categories.append("sample_point_mismatch")
            if c_record["width"] != r_record["width"]:
                categories.append("width_mismatch")
            if c_record["value"] != r_record["value"]:
                categories.append(
                    "xz_mismatch" if _has_xz(c_record["value"])
                    or _has_xz(r_record["value"]) else "value_mismatch"
                )
            if not categories:
                coverage["matched"] += 1
            else:
                difference = _annotate_normalized(_difference(
                    categories[0], c_record, r_record, cmodel, rtl,
                    "architecture"
                ), config["canonical_module"], config["canonical_instance"],
                    config["phase"], config["sample_point"], occurrence)
                difference["all_categories"] = categories
                differences.append(difference)
                coverage["mismatched"] += 1
    return differences, coverage


def _select_rule_records(
    trace: LoadedTrace,
    rule: dict[str, Any],
    coverage: dict[str, Any],
) -> list[dict[str, Any]]:
    signals = set(rule["signals"])
    event_types = set(rule["event_types"])
    statuses = set(rule["mapping_statuses"])
    source_modules = set(rule.get("source_modules", {}).get(trace.source, []))
    selected: list[dict[str, Any]] = []
    for record in trace.records:
        if record["signal"] not in signals:
            continue
        if record["event_type"] not in event_types:
            continue
        if record["mapping_status"] not in statuses:
            continue
        if source_modules and record["module"] not in source_modules:
            continue
        if not _view_matches(record, rule, trace.source):
            continue
        if not _gate_allows(trace, record, rule):
            coverage["gated_records"][trace.source] += 1
            if _has_xz(record["value"]):
                coverage["gated_xz_records"][trace.source] += 1
            continue
        selected.append(record)
    return selected


def _keyed_records(
    records: Iterable[dict[str, Any]], rule: dict[str, Any]
) -> dict[tuple[Any, ...], dict[str, Any]]:
    occurrences: collections.Counter[tuple[Any, ...]] = collections.Counter()
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        prefix = _normalized_key(record, rule, 0)[:-1]
        occurrence = occurrences[prefix]
        occurrences[prefix] += 1
        key = _normalized_key(record, rule, occurrence)
        result[key] = record
    return result


def _compare_signal_rules(
    cmodel: LoadedTrace,
    rtl: LoadedTrace,
    rules: dict[str, Any],
    coverage: dict[str, Any],
) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    for rule in rules["signal_rules"]:
        rule_stats = {
            "cmodel_eligible": 0,
            "rtl_eligible": 0,
            "compared": 0,
            "matched": 0,
            "mismatched": 0,
            "missing_cmodel": 0,
            "missing_rtl": 0,
        }
        c_records = _select_rule_records(cmodel, rule, coverage)
        r_records = _select_rule_records(rtl, rule, coverage)
        rule_stats["cmodel_eligible"] = len(c_records)
        rule_stats["rtl_eligible"] = len(r_records)
        c_keyed = _keyed_records(c_records, rule)
        r_keyed = _keyed_records(r_records, rule)
        for key in sorted(set(c_keyed) | set(r_keyed), key=repr):
            c_record = c_keyed.get(key)
            r_record = r_keyed.get(key)
            if c_record is None:
                difference = _annotate_normalized(_difference(
                    "missing_cmodel", None, r_record, cmodel, rtl, rule["name"]
                ), key[6], key[7], key[4], key[5], key[10])
                differences.append(difference)
                rule_stats["mismatched"] += 1
                rule_stats["missing_cmodel"] += 1
                continue
            if r_record is None:
                difference = _annotate_normalized(_difference(
                    "missing_rtl", c_record, None, cmodel, rtl, rule["name"]
                ), key[6], key[7], key[4], key[5], key[10])
                differences.append(difference)
                rule_stats["mismatched"] += 1
                rule_stats["missing_rtl"] += 1
                continue
            rule_stats["compared"] += 1
            category = None
            if c_record["width"] != r_record["width"]:
                category = "width_mismatch"
            elif c_record["value"] != r_record["value"]:
                category = (
                    "xz_mismatch" if _has_xz(c_record["value"])
                    or _has_xz(r_record["value"]) else "value_mismatch"
                )
            if category is None:
                rule_stats["matched"] += 1
            else:
                difference = _annotate_normalized(_difference(
                    category, c_record, r_record, cmodel, rtl, rule["name"]
                ), key[6], key[7], key[4], key[5], key[10])
                difference["all_categories"] = [category]
                differences.append(difference)
                rule_stats["mismatched"] += 1
        coverage["rules"][rule["name"]] = rule_stats
    return differences


def _mapping_coverage(trace: LoadedTrace) -> dict[str, Any]:
    counts = collections.Counter(record["mapping_status"] for record in trace.records)
    excluded_signals = sorted({
        record["signal"] for record in trace.records
        if record["mapping_status"] in {"UNMAPPED", "NA"}
    })
    return {
        "rows": len(trace.records),
        "sequence_mode": trace.sequence_mode,
        "mapping_status_counts": dict(sorted(counts.items())),
        "excluded_unmapped_na_rows": counts["UNMAPPED"] + counts["NA"],
        "excluded_unmapped_na_signals": excluded_signals,
        "xz_rows": sum(_has_xz(record["value"]) for record in trace.records),
    }


def _sort_differences(differences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(item: dict[str, Any]) -> tuple[Any, ...]:
        case_rank = {"raw_two_issue": 0, "raw_repeat4": 1}.get(item["case_id"], 99)
        rule_rank = 0 if item["rule"] == "architecture" else 1
        return (
            case_rank,
            item["global_cycle"],
            rule_rank,
            item["phase"],
            item["sample_point"],
            item["signal"],
            item["category"],
        )
    return sorted(differences, key=key)


def compare_traces(
    cmodel: LoadedTrace,
    rtl: LoadedTrace,
    rules: dict[str, Any],
    max_differences: int = 100,
) -> dict[str, Any]:
    coverage: dict[str, Any] = {
        "CMODEL": _mapping_coverage(cmodel),
        "RTL": _mapping_coverage(rtl),
        "gated_records": {"CMODEL": 0, "RTL": 0},
        "gated_xz_records": {"CMODEL": 0, "RTL": 0},
        "rules": {},
    }
    architecture_differences, architecture_coverage = _compare_architecture(
        cmodel, rtl, rules
    )
    coverage["architecture"] = architecture_coverage
    signal_differences = _compare_signal_rules(cmodel, rtl, rules, coverage)
    differences = _sort_differences(architecture_differences + signal_differences)
    compared = architecture_coverage["compared"] + sum(
        stats["compared"] for stats in coverage["rules"].values()
    )
    matched = architecture_coverage["matched"] + sum(
        stats["matched"] for stats in coverage["rules"].values()
    )
    status = "PASS" if not differences and compared > 0 else "FAIL"
    return {
        "test": "raw_macro_level1_cycle_scoreboard",
        "status": status,
        "schema": rules["schema"],
        "cmodel_trace": str(cmodel.path),
        "rtl_trace": str(rtl.path),
        "compared_pairs": compared,
        "matched_pairs": matched,
        "difference_count": len(differences),
        "first_difference": differences[0] if differences else None,
        "differences": differences[:max_differences],
        "differences_truncated": len(differences) > max_differences,
        "coverage": coverage,
    }


def _format_record(label: str, record: dict[str, Any] | None) -> list[str]:
    if record is None:
        return [f"{label}: <missing>"]
    return [
        f"{label}: value={record['value']} width={record['width']} "
        f"mapping={record['mapping_status']}",
        f"  module={record['module']} instance={record['instance']}",
        f"  source_ref={record['source_ref']}",
    ]


def text_report(result: dict[str, Any]) -> str:
    lines = [
        "RAW Macro Level1 Cycle Scoreboard",
        f"Test: {result['test']}",
        f"Status: {result['status']}",
        f"Compared pairs: {result.get('compared_pairs', 0)}",
        f"Matched pairs: {result.get('matched_pairs', 0)}",
        f"Differences: {result.get('difference_count', 0)}",
    ]
    first = result.get("first_difference")
    if first:
        lines.extend([
            "",
            "First difference:",
            f"  category={first['category']} rule={first['rule']}",
            f"  case={first['case_id']} cycle={first['global_cycle']} "
            f"logical={first['logical_cycle']}",
            f"  phase={first['phase']} sample={first['sample_point']}",
            f"  module={first['module']} instance={first['instance']} "
            f"signal={first['signal']}",
            f"  normalized module={first.get('normalized_module')} "
            f"instance={first.get('normalized_instance')} "
            f"phase={first.get('normalized_phase')} "
            f"sample={first.get('normalized_sample_point')} "
            f"occurrence={first.get('event_occurrence')}",
            f"  mismatch dimensions={','.join(first.get('all_categories', [first['category']]))}",
        ])
        lines.extend(_format_record("  CModel", first["cmodel"]))
        lines.extend(_format_record("  RTL", first["rtl"]))
        lines.append(f"  possible cause={first['possible_cause']}")
        for source in ("cmodel_context", "rtl_context"):
            context = first[source]
            lines.append(f"  {source} previous/next:")
            for direction in ("previous", "next"):
                summaries = context[direction]
                rendered = ", ".join(
                    f"{item['signal']}@C{item['global_cycle']}:{item['phase']}"
                    for item in summaries
                ) or "<none>"
                lines.append(f"    {direction}: {rendered}")
    coverage = result.get("coverage", {})
    if coverage:
        lines.extend(["", "Coverage:"])
        for source in ("CMODEL", "RTL"):
            item = coverage[source]
            lines.append(
                f"  {source}: rows={item['rows']} sequence={item['sequence_mode']} "
                f"xz_rows={item['xz_rows']} excluded_UNMAPPED_NA="
                f"{item['excluded_unmapped_na_rows']}"
            )
            lines.append(
                f"    excluded signals={','.join(item['excluded_unmapped_na_signals']) or '<none>'}"
            )
        lines.append(
            "  gated records: "
            f"CMODEL={coverage['gated_records']['CMODEL']} "
            f"RTL={coverage['gated_records']['RTL']}"
        )
        lines.append(
            "  gated X/Z records: "
            f"CMODEL={coverage['gated_xz_records']['CMODEL']} "
            f"RTL={coverage['gated_xz_records']['RTL']}"
        )
        for name, stats in coverage["rules"].items():
            lines.append(
                f"  {name}: C={stats['cmodel_eligible']} R={stats['rtl_eligible']} "
                f"compared={stats['compared']} matched={stats['matched']} "
                f"mismatch={stats['mismatched']}"
            )
    return "\n".join(lines) + "\n"


def write_reports(result: dict[str, Any], output_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "scoreboard_report.json"
    text_path = output_dir / "scoreboard_report.txt"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    text_path.write_text(text_report(result), encoding="utf-8")
    return text_path, json_path


def run_comparison(
    cmodel_path: pathlib.Path,
    rtl_path: pathlib.Path,
    rules_path: pathlib.Path,
    output_dir: pathlib.Path | None = None,
    max_differences: int = 100,
) -> dict[str, Any]:
    rules = load_rules(rules_path)
    cmodel = load_trace(cmodel_path, rules, "CMODEL")
    rtl = load_trace(rtl_path, rules, "RTL")
    result = compare_traces(cmodel, rtl, rules, max_differences)
    if output_dir is not None:
        text_path, json_path = write_reports(result, output_dir)
        result["text_report"] = str(text_path)
        result["json_report"] = str(json_path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmodel", type=pathlib.Path, required=True)
    parser.add_argument("--rtl", type=pathlib.Path, required=True)
    parser.add_argument(
        "--rules",
        type=pathlib.Path,
        default=pathlib.Path(__file__).with_name("raw_macro_level1_rules.json"),
    )
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--max-differences", type=int, default=100)
    arguments = parser.parse_args(argv)

    try:
        result = run_comparison(
            arguments.cmodel,
            arguments.rtl,
            arguments.rules,
            arguments.output_dir,
            arguments.max_differences,
        )
    except (OSError, json.JSONDecodeError, TraceValidationError) as error:
        failure = {
            "test": "raw_macro_level1_cycle_scoreboard",
            "status": "FAIL",
            "validation_error": str(error),
        }
        write_reports(failure, arguments.output_dir)
        print(text_report(failure), end="")
        return 2
    print(text_report(result), end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
