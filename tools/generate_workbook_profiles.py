"""Read the agreed workbook (never edit it) and emit schema-v2 hardware targets.

Requires openpyxl. Formula cells must have cached results from Excel.
The planning object is documentation, not a claim of implemented RTL features.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import openpyxl
from configure_mem_slices import resize_mem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "config")
    args = parser.parse_args()
    book = openpyxl.load_workbook(args.workbook, read_only=True, data_only=True)
    sheet = book["三档参数对照"]
    baseline = json.loads((Path(__file__).resolve().parents[1] /
                           "config/ftlpu-lpu32.json").read_text(encoding="utf-8"))
    cloud_label = sheet.cell(5, 3).value.split("（")[0].strip()
    if cloud_label not in ("LPU-MAX", "MAX", "LPU64"):
        raise ValueError(f"Unrecognized cloud profile in C5: {cloud_label}")
    max_profile = cloud_label != "LPU64"
    for col, name in ((3, "ftlpu-max" if max_profile else "ftlpu-lpu64-cloud"), (4, "ftlpu-lpu32-edge"),
                      (5, "ftlpu-lpu16-decode")):
        def value(row):
            result = sheet.cell(row, col).value
            if result is None:
                raise ValueError(f"Missing cached workbook value: {sheet.cell(row, col).coordinate}")
            return result
        if value(30) != baseline["stream_topology"]["bindings"]["mem"]["slices_per_group"]:
            raise ValueError("Workbook MEM group size must match the base fabric")
        if max_profile and (value(29) > 44 or value(14) not in (8, 16)):
            raise ValueError(f"{name}: expected at most 44 MEM slices per hemisphere and 8 or 16 lanes per tile")
        if max_profile and value(13) not in (2, 4, 8, 20):
            raise ValueError(f"{name}: tile count must be one of the workbook design points (2, 4, 8, 20)")
        config = resize_mem(baseline, value(29))
        config["target"]["name"] = name
        config["topology"].update(hemispheres=value(12), tiles_per_slice=value(13),
                                  lanes_per_tile=value(14))
        config["mem"].update(banks_per_slice=value(31),
                             rows_per_bank=value(32), bytes_per_lane=value(33))
        config["sr"]["registers_per_lane"] = value(43)
        config["c2c"].update(beat_bytes=value(44), streams_per_direction=value(46))
        for family in ("mem", "mxm", "sxm", "vxm", "c2c"):
            config["icu"][f"{family}_iq_depth"] = value(41)
        config["mxm"].update(instances_per_hemisphere=value(16), accum_contexts=value(21))
        # The IS column count must divide the physical supercell row count.
        column_limit = min(config["mxm"]["is_columns"], value(43) // 2 // value(14))
        config["mxm"]["is_columns"] = max(n for n in range(1, column_limit + 1) if value(13) % n == 0)
        config["vxm"]["alus"] = value(25)
        config["throughput"].update(mxms_per_hemisphere=value(16), mxm_weight_buffers=value(20))
        if value(15) != 32 or value(16) != 1:
            config["throughput"]["mxm_prefill_weight_streams"] = 8
        # Layout seeds, not a general tensor lifetime allocator. Keeping legacy
        # addresses 7936/1600 would already be out of bounds on the 1024-row target.
        config["memory"]["attention_mask_base_row"] = value(32) - 256
        config["memory"]["matmul_result_base_row"] = min(1600, value(32) // 2)
        # Row 53 is the device interface peak, row 54 the share granted to this
        # LPU, row 47 the on-chip RX entrance and row 55 the sustained
        # efficiency. Row 56 keeps the smaller of budget/entrance and applies
        # efficiency exactly once. Every input is stored so the reduction stays
        # auditable instead of collapsing into a single opaque number.
        ddr_effective_peak = min(value(54), value(47)) * value(55)
        config["external_memory"].pop("ddr_peak_bandwidth_mbytes_per_second", None)
        config["external_memory"].update(
            lpu_clock_mhz=value(26),
            ddr_interface_peak_bytes_per_second=round(value(53) * 1e9),
            ddr_budget_bytes_per_second=round(value(54) * 1e9),
            ddr_rx_entrance_bytes_per_second=round(value(47) * 1e9),
            ddr_scheduling_efficiency_percent=round(value(55) * 100),
            ddr_effective_peak_bytes_per_second=round(ddr_effective_peak * 1e9),
            ddr_read_latency_cycles=math.ceil(value(57) * value(26) / 1000),
            ddr_write_latency_cycles=math.ceil(value(58) * value(26) / 1000),
            ddr_read_latency_jitter_cycles=0, ddr_write_latency_jitter_cycles=0,
            ddr_request_queue_depth=value(60))
        lanes = value(13) * value(14)
        mxms = value(11) * value(12) * value(16)
        acc_kib = lanes * lanes * value(21) * value(22) / 8192
        banks = value(12) * value(29) * value(31)
        row_bytes = lanes * value(33)
        derived = {15: lanes, 17: mxms, 18: lanes, 19: lanes, 23: acc_kib,
                   24: mxms * acc_kib, 34: row_bytes,
                   35: banks * value(32) * row_bytes / 1048576, 39: value(11) * banks,
                   47: value(11) * value(12) * value(46) * value(44) * value(26) / 1000,
                   56: ddr_effective_peak}
        for row, expected in derived.items():
            if not math.isclose(value(row), expected):
                raise ValueError(f"{name}: workbook row {row} should be {expected}; recalculate the workbook")
        config["planning"] = {
            "source_workbook": args.workbook.name,
            "source_sha256": hashlib.sha256(args.workbook.read_bytes()).hexdigest(),
            "source_sheet": sheet.title,
            "source_column": sheet.cell(5, col).column_letter,
            "status": "experimental target; model-level validation required",
            "workbook_rows": {str(r): {"parameter": sheet.cell(r, 1).value,
                                       "value": value(r)} for r in range(7, 61)
                              if sheet.cell(r, col).value is not None},
            "unimplemented_requirements": [
                "RTL/PPA/FPGA acceptance and timing closure",
                "iMEM capacity from program peak and FPGA segmented loading",
                "external-memory capacity and device-type enforcement "
                "(capacity and type stay documentation only)"],
            "layout_note": "Mask starts at depth-256; result seed=min(1600,depth/2). These seeds do not prove tensor lifetime fit.",
            "timing_note": "No jitter specified in workbook; jitter=0. ddr_effective_peak_bytes_per_second is workbook row 56 (min of per-LPU budget and RX entrance, times efficiency) and is consumed directly as the model's sustained byte ceiling. ddr_read/write_latency_cycles and ddr_request_queue_depth are enforced by the DDR model, so the achievable rate also falls out of Little's law instead of being assumed.",
        }
        if value(14) == 16:
            config["planning"]["unimplemented_requirements"].append(
                "16-lane SXM row/stream encoding and matrix adapter before integrated ICU execution")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        dest = args.output_dir / (name + ".json")
        dest.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(dest)
    book.close()


if __name__ == "__main__":
    main()
