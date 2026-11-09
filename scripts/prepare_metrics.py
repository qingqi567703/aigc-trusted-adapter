import json
from pathlib import Path
from prepare_assets import ROOT, download, sha256

path = ROOT / "assets/metrics/inception-2015-12-05.pt"
url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt"
download(url, path)
(path.parent / "manifest.json").write_text(json.dumps({"source": url, "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size, "sha256": sha256(path)}, indent=2), encoding="utf-8")
print("Clean-FID Inception weights ready.", flush=True)
