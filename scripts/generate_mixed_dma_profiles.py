"""Regenerate mixed-task CModel defaults without changing the decode/native targets.

These are standalone target JSON files: use the SAME file for SOFTWARE and CModel.
Only the target identity and the three DMA admission capacities may differ.
"""
import argparse
import json
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / "config"
POINTS = {
    "ftlpu-lpu16-mixed": ("ftlpu-lpu16-decode", 112, 128, 832),
    "ftlpu-lpu32-mixed": ("ftlpu-lpu32", 192, 128, 384),
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, (base, p, rt, q) in POINTS.items():
        obj = json.loads((CONFIG / f"{base}.json").read_text(encoding="utf-8-sig"))
        obj["target"]["name"] = name
        obj["external_memory"].update(dma_fifo_depth_vectors=rt, ddr_request_queue_depth=q)
        obj["program_load"]["response_fifo_depth"] = p
        path = CONFIG / f"{name}.json"
        if args.check:
            if json.loads(path.read_text(encoding="utf-8-sig")) != obj:
                parser.exit(1, f"{path}: stale profile\n")
        else:
            path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
