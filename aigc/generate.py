import argparse
import json

from .common import file_sha256, offline_environment, project_path, read_json, read_manifest, seed_everything, write_json
offline_environment()

import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionAdapterPipeline
from PIL import Image
from .data import condition_pair, image_tensor, sketch_tensor
from .model import TrustedAdapter


def save_tensor(tensor, path):
    value = tensor.detach().cpu().clamp(0, 1)
    if value.shape[0] == 1:
        pixels = (value[0].numpy() * 255).round().astype(np.uint8)
    else:
        pixels = (value.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(pixels).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="data/manifests/dev.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.count < 1 or args.steps < 1:
        raise ValueError("count and steps must be positive")
    seed_everything(args.seed)
    saved = torch.load(str(project_path(args.checkpoint)), map_location="cpu", weights_only=False, mmap=True)
    config, variant = saved["config"], saved["variant"]
    output = project_path(args.output)
    generation_config = {"checkpoint_sha256": file_sha256(project_path(args.checkpoint)),
                         "manifest_sha256": file_sha256(project_path(args.manifest)),
                         "steps": args.steps, "seed": args.seed, "count": args.count}
    completed = set()
    if (output / "samples.jsonl").exists():
        if not args.resume:
            raise RuntimeError("Generation output already exists. Resume explicitly or use a new directory.")
        if read_json(output / "generation_config.json") != generation_config:
            raise ValueError("Generation resume configuration/checkpoint differs")
        for line in (output / "samples.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if not (output / row["sample"]).exists():
                raise FileNotFoundError(output / row["sample"])
            completed.add((row["id"], row["view"]))
    write_json(output / "generation_config.json", generation_config)
    model = TrustedAdapter(project_path(config["adapter_model"]), variant)
    model.load_state_dict(saved["model"])
    del saved
    model.eval().requires_grad_(False).to("cuda", dtype=torch.float16)
    pipeline = StableDiffusionAdapterPipeline.from_pretrained(
        project_path(config["sd_model"]), adapter=model.adapter, variant="fp16",
        torch_dtype=torch.float16, local_files_only=True
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.to("cuda")
    pipeline.enable_vae_slicing()
    pipeline.set_progress_bar_config(disable=True)
    active_trust = [None]
    hook = model.adapter.register_forward_hook(lambda module, inputs, features: model.gate_features(features, active_trust[0]))
    for directory in ("samples", "references", "conditions"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    records = read_manifest(args.manifest)[:args.count]
    with (output / "samples.jsonl").open("a", encoding="utf-8") as stream:
        for record in records:
            rgb = image_tensor(project_path(record["image"]), config["resolution"])
            clean = sketch_tensor(rgb)
            first, second, trust, corruption = condition_pair(clean, args.seed * 1000003 + record["id"])
            save_tensor(rgb, output / "references" / f"{record['id']}.png")
            save_tensor(clean, output / "conditions" / f"{record['id']}_clean.png")
            save_tensor(trust, output / "conditions" / f"{record['id']}_trust.png")
            active_trust[0] = trust.unsqueeze(0).cuda().half()
            for view, condition in (("clean", clean), ("0", first), ("1", second)):
                if (record["id"], view) in completed:
                    continue
                save_tensor(condition, output / "conditions" / f"{record['id']}_{view}.png")
                supplied = condition
                if variant in ("premask", "full", "no_pair", "fixed_gate"):
                    supplied = supplied * trust
                if model.adapter.config.in_channels == 3:
                    supplied = supplied.repeat(3, 1, 1)
                generator = torch.Generator("cuda").manual_seed(args.seed + record["id"])
                with torch.inference_mode():
                    result = pipeline(record["caption"], image=supplied.unsqueeze(0), height=config["resolution"],
                                      width=config["resolution"], num_inference_steps=args.steps,
                                      guidance_scale=7.5, generator=generator)
                relative = f"samples/{record['id']}_{view}.png"
                result.images[0].save(output / relative)
                row = {"id": record["id"], "caption": record["caption"], "view": view, "sample": relative,
                       "variant": variant, "corruption": corruption, "steps": args.steps,
                       "seed": args.seed + record["id"], "safety_flag": bool(result.nsfw_content_detected[0]) if result.nsfw_content_detected else False}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                print(json.dumps(row, ensure_ascii=False), flush=True)
    hook.remove()


if __name__ == "__main__":
    main()
