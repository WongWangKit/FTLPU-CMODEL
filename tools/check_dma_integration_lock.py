"""Check generated geometry of a mixed integration build against the reviewed capacity lock."""
import argparse,json,re
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("generated_header",type=Path)
args=p.parse_args()
src=args.generated_header.read_text(encoding="utf-8-sig")
lock=json.loads((Path(__file__).resolve().parents[1]/"config/dma-integration-v2.lock.json").read_text())
target=re.search(r'kTargetName\s*=\s*"([^"]+)"',src).group(1)
cfg=lock["profiles"][target]
def check(name,pattern,expected):
    m=re.search(pattern,src)
    if not m or int(m.group(1))!=expected:
        raise SystemExit(f"{target} {name}: expected {expected}, got {m.group(1) if m else 'missing'}")
def constant(name,expected):
    check(name,rf'\b{re.escape(name)}\s*=\s*(\d+)\s*;',expected)
constant('kProgramLoadResponseFifoDepth',cfg['P'])
constant('kDmaFifoDepthVectors',cfg['RT'])
check('Q',r'kC2cDmaSizing\s*\{\s*kProgramLoadResponseFifoDepth\s*,\s*kDmaFifoDepthVectors\s*,\s*(\d+)\s*\}',cfg['Q'])
for kind,words in cfg["imem"].items():constant('kIcu'+kind+'ImemDepth',words)
for kind,depth in lock["program_fifo"].items():constant('kProgramLoad'+kind+'FifoDepth',depth)
print(f"PASS {target}: P/RT/Q={cfg['P']}/{cfg['RT']}/{cfg['Q']}; iMEM={cfg['imem']}; program FIFO={lock['program_fifo']}")

