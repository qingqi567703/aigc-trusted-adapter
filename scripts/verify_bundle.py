"""Standard-library preflight before installing an offline server environment."""

import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "offline/preparation_manifest.json").read_text(encoding="utf-8"))
if not manifest["ready"]:
    raise SystemExit("Preparation manifest is incomplete")

records = manifest["source_files"] + manifest["linux_wheels"] + manifest.get("input_files", [])
for name in ("sd15", "adapter", "clip"):
    records += json.loads((root / f"assets/models/{name}/download_manifest.json").read_text(encoding="utf-8"))["files"]
records.append(json.loads((root / "assets/metrics/manifest.json").read_text(encoding="utf-8")))
for item in records:
    path = root / item["path"].replace("\\", "/")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != item["sha256"]:
        raise SystemExit(f"Checksum mismatch {item['path']}")
print(f"Offline code/model/wheel checksums passed ({len(records)} files).")
