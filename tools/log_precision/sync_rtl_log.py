"""Publish only the reviewed files; reject concurrent source modifications."""
from pathlib import Path
import hashlib,json,shutil

workspace=Path(__file__).resolve().parents[2]
staged=(workspace/'rtl-log-work').resolve()
target=Path(r'C:\Users\zhaodong.feng\Desktop\LPU\FTLPU-VMODEL-develop').resolve()
backup=workspace/'build-log/rtl-original-backup'
manifest=json.loads((workspace/'build-log/rtl_log_manifest.json').read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None

# Check every file before the first write, so a stale snapshot cannot partly
# overwrite newer user work. All destinations must remain inside the named repo.
for entry in manifest:
    dst=(target/entry['path']).resolve()
    src=(staged/entry['path']).resolve()
    dst.relative_to(target);src.relative_to(staged)
    if digest(dst)!=entry['original_sha256']:
        raise SystemExit('Concurrent modification; nothing written: '+str(dst))
    if digest(src)!=entry['new_sha256']:
        raise SystemExit('Staged file changed; nothing written: '+str(src))
for entry in manifest:
    dst=target/entry['path'];src=staged/entry['path']
    if dst.exists():
        saved=backup/entry['path'];saved.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(dst,saved)
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dst)
for entry in manifest:
    assert digest(target/entry['path'])==entry['new_sha256'],entry['path']
print('Synced and verified',len(manifest),'files to',target)
print('Original changed files backed up under',backup)
