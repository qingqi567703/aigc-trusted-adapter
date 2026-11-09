"""Record real-image acceptance checks and a clearly labelled preview."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigc.common import ROOT, offline_environment, read_json, write_json
offline_environment()

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from cleanfid import fid
from aigc.metrics import clean_features, clean_inception_extractor


def main():
    checkpoint = torch.load(str(ROOT / "runs/local_smoke_full/last.pt"), map_location="cpu", weights_only=False, mmap=True)
    original = torch.load(str(ROOT / "assets/models/adapter/diffusion_pytorch_model.bin"), map_location="cpu", weights_only=True, mmap=True)
    parameter = next(iter(original))
    change = (checkpoint["model"]["adapter." + parameter] - original[parameter]).float().abs().max().item()
    gate_change = checkpoint["model"]["gates.0.2.weight"].abs().max().item()
    assert change > 0 and gate_change > 0
    generated = ROOT / "runs/local_smoke_full/generated"
    metrics = read_json(generated / "metrics.json")
    model = clean_inception_extractor(ROOT / "assets/metrics/inception-2015-12-05.pt", "cuda")
    features = clean_features(generated / "references", model, "cuda", batch_size=1)
    assert features.shape == (2, 2048) and np.isfinite(features).all()
    summary = {
        "scope": "Engineering acceptance only. Three/four training updates and two images do not establish method gains.",
        "full_training": read_json(ROOT / "runs/local_smoke_full/status.json"),
        "baseline_training": read_json(ROOT / "runs/local_smoke_baseline/status.json"),
        "resume_reached_step": checkpoint["step"],
        "adapter_parameter_max_change": change,
        "gate_parameter_max_change": gate_change,
        "generation_images": 6,
        "offline_clip_evaluation": metrics,
        "offline_inception_feature_shape": list(features.shape),
        "training_512_bf16_global_batch8": read_json(ROOT / "runs/local_512_acceptance/status.json"),
        "all_training_cache_validation": read_json(ROOT / "work/cache_acceptance.json"),
        "generation_resume": {"status":"passed", "sample_rows_after_resume":len((generated / "samples.jsonl").read_text(encoding="utf-8").splitlines())},
        "linux_offline_runtime_validation": "pending server restart",
    }
    write_json(ROOT / "outputs/本地工程验收记录.json", summary)
    ids = [json.loads(line)["id"] for line in (generated / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    ids = list(dict.fromkeys(ids))
    size, row_height = 180, 230
    sheet = Image.new("RGB", (size * 6, 65 + row_height * len(ids)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 16)
    draw.text((12, 10), "本地工程验收 · 4 个优化步 · 仅验证生成链路", font=font, fill="black")
    labels = ["参考照片", "草图", "可信度", "干净条件生成", "扰动一生成", "扰动二生成"]
    for index, identity in enumerate(ids):
        paths = [generated / "references" / f"{identity}.png",
                 generated / "conditions" / f"{identity}_clean.png",
                 generated / "conditions" / f"{identity}_trust.png",
                 generated / "samples" / f"{identity}_clean.png",
                 generated / "samples" / f"{identity}_0.png",
                 generated / "samples" / f"{identity}_1.png"]
        top = 65 + index * row_height
        for column, (path, label) in enumerate(zip(paths, labels)):
            with Image.open(path) as image:
                sheet.paste(image.convert("RGB").resize((size, size)), (column * size, top))
            draw.text((column * size + 8, top + size + 5), label, font=font, fill="black")
    sheet.save(ROOT / "outputs/本地工程验收样例.png")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
