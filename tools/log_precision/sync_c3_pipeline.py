"""Synchronize only reviewed C3 changes, rejecting concurrent target edits."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2]
source=(ROOT/'rtl-log-work').resolve()
target=Path(r'C:\Users\zhaodong.feng\Desktop\LPU\FTLPU-VMODEL-develop').resolve()
backup=(ROOT/'build-log/c3-original-backup').resolve()
changes=json.loads((ROOT/'build-log/c3_pipeline_manifest.json').read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
for change in changes:
    src=(source/change['path']).resolve();dst=(target/change['path']).resolve()
    if not src.is_relative_to(source) or not dst.is_relative_to(target):
        raise SystemExit('Out-of-tree manifest entry')
    if digest(dst)!=change['old_sha256']:
        raise SystemExit('Concurrent target edit: '+change['path'])
    if digest(src)!=change['new_sha256']:
        raise SystemExit('Staging changed after review: '+change['path'])
for change in changes:
    src=source/change['path'];dst=target/change['path'];saved=backup/change['path']
    if dst.is_file():
        saved.parent.mkdir(parents=True,exist_ok=True)
        if not saved.exists():shutil.copy2(dst,saved)
    if change['new_sha256'] is None:
        # One explicitly listed, prevalidated obsolete serial-Log source file.
        dst.unlink()
    else:
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
for change in changes:
    if digest(target/change['path'])!=change['new_sha256']:
        raise SystemExit('Verification failed: '+change['path'])
print('C3 pipeline synced and hash-verified:',len(changes),'files.')
