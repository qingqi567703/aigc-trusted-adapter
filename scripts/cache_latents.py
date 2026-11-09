"""Cache frozen encoders while retaining the VAE posterior distribution."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigc.common import cache_path, file_sha256, offline_environment, project_path, read_manifest, write_json
offline_environment()

import torch
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from aigc.data import image_tensor, sketch_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/train.jsonl")
    parser.add_argument("--model", default="assets/models/sd15")
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    model_path = project_path(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", variant="fp16", local_files_only=True)
    vae.to(device).eval().requires_grad_(False)
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", local_files_only=True)
    text_model = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", variant="fp16", local_files_only=True)
    text_model.to(device).eval().requires_grad_(False)
    records = read_manifest(args.manifest)
    if args.limit:
        records = records[:args.limit]
    metadata = {
        "resolution": args.resolution, "preprocessing": "RGB center fit lanczos; Canny 100/200; white strokes",
        "model_manifest_sha256": file_sha256(model_path / "download_manifest.json"),
        "vae_scaling_factor": float(vae.config.scaling_factor),
    }
    metadata_path = cache_path(records[0], args.resolution).parent / "cache_info.json"
    if metadata_path.exists():
        import json
        if json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
            raise RuntimeError("Cache preprocessing/model mismatch. Use a new cache directory.")
    write_json(metadata_path, metadata)
    with torch.no_grad():
        for start in range(0, len(records), args.batch_size):
            selected = records[start:start + args.batch_size]
            pending = [record for record in selected if not cache_path(record, args.resolution).exists()]
            if pending:
                rgb = torch.stack([image_tensor(project_path(record["image"]), args.resolution) for record in pending])
                posterior = vae.encode(rgb.to(device) * 2 - 1).latent_dist
                token_ids = tokenizer([record["caption"] for record in pending], padding="max_length", max_length=77,
                                      truncation=True, return_tensors="pt").input_ids
                means, logvars = posterior.mean.half().cpu(), posterior.logvar.half().cpu()
                texts = text_model(token_ids.to(device))[0].half().cpu()
                for index, record in enumerate(pending):
                    path = cache_path(record, args.resolution)
                    payload = {
                        "mean": means[index].clone(),
                        "logvar": logvars[index].clone(),
                        "text": texts[index].clone(),
                        "sketch": sketch_tensor(rgb[index]).half(),
                        "id": record["id"],
                        "caption": record["caption"],
                    }
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_name(path.name + ".tmp")
                    torch.save(payload, temporary)
                    temporary.replace(path)
            completed = min(start + args.batch_size, len(records))
            if completed % 100 < args.batch_size or completed == len(records):
                print(f"CACHE {completed}/{len(records)}", flush=True)


if __name__ == "__main__":
    main()
