import argparse
import json
import signal
import time
from pathlib import Path

from .common import ROOT, cache_path, file_sha256, offline_environment, project_path, read_json, seed_everything, write_json
offline_environment()

import torch
from accelerate import Accelerator
from diffusers import DDPMScheduler, UNet2DConditionModel
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .data import CachedImages
from .model import TrustedAdapter, VARIANTS, pair_consistency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/course.json")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    for field in ("resolution", "global_batch", "micro_batch", "max_steps", "checkpoint_every", "max_runtime_seconds"):
        if config[field] <= 0:
            raise ValueError(f"{field} must be positive")
    if config["mixed_precision"] not in ("fp16", "bf16"):
        raise ValueError("Supported mixed_precision values are fp16 and bf16")
    if config["resolution"] % 8 or config["learning_rate"] <= 0 or config["pair_weight"] < 0:
        raise ValueError("Invalid resolution, learning rate, or pair weight")
    output = project_path(args.output)
    state_path = output / "last.pt"
    if state_path.exists() and not args.resume:
        raise RuntimeError("Output already contains a checkpoint. Resume explicitly or use a new directory.")
    world_size = int(__import__("os").environ.get("WORLD_SIZE", 1))
    divisor = world_size * config["micro_batch"]
    if config["global_batch"] % divisor:
        raise ValueError("global_batch must be divisible by GPU count × micro_batch")
    accumulation = config["global_batch"] // divisor
    accelerator = Accelerator(mixed_precision=config["mixed_precision"], gradient_accumulation_steps=accumulation)
    seed_everything(args.seed)
    device = accelerator.device
    weight_dtype = torch.float16 if config["mixed_precision"] == "fp16" else torch.bfloat16
    unet = UNet2DConditionModel.from_pretrained(project_path(config["sd_model"]), subfolder="unet", variant="fp16", local_files_only=True)
    unet.to(device=device, dtype=weight_dtype).eval().requires_grad_(False)
    model = TrustedAdapter(project_path(config["adapter_model"]), args.variant)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.01, foreach=False)
    scheduler = DDPMScheduler.from_pretrained(project_path(config["sd_model"]), subfolder="scheduler", local_files_only=True)
    data = CachedImages(config["manifest"], config["resolution"], args.seed)
    cache_metadata = read_json(cache_path(data.records[0], config["resolution"]).parent / "cache_info.json")
    if cache_metadata["resolution"] != config["resolution"] or cache_metadata["model_manifest_sha256"] != file_sha256(project_path(config["sd_model"]) / "download_manifest.json"):
        raise ValueError("Cached encoder metadata differs from the current resolution/model")
    step = 0
    previous_wall_seconds = 0.0
    if args.resume:
        if not state_path.exists():
            raise FileNotFoundError(state_path)
        saved = torch.load(str(state_path), map_location="cpu", weights_only=False, mmap=True)
        if saved["variant"] != args.variant or saved["seed"] != args.seed:
            raise ValueError("Resume variant/seed mismatch")
        for field in ("resolution", "global_batch", "manifest", "learning_rate", "pair_weight",
                      "mixed_precision", "sd_model", "adapter_model"):
            if config[field] != saved["config"][field]:
                raise ValueError(f"Resume changes frozen field {field}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        step = saved["step"]
        previous_wall_seconds = saved.get("wall_seconds", 0.0)
        if "wall_seconds" not in saved and (output / "status.json").exists():
            previous_wall_seconds = read_json(output / "status.json").get("wall_seconds", 0.0)
        del saved
    model, optimizer = accelerator.prepare(model, optimizer)
    output.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        write_json(output / "config.json", {**config, "variant": args.variant, "seed": args.seed,
                    "gpu_count": accelerator.num_processes, "gradient_accumulation_steps": accumulation,
                    "torch": torch.__version__, "device": torch.cuda.get_device_name()})
    interrupted = [False]
    def stop(signum, frame):
        interrupted[0] = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()

    def save():
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            payload = {"model": accelerator.unwrap_model(model).state_dict(), "optimizer": optimizer.state_dict(),
                       "step": step, "variant": args.variant, "seed": args.seed, "config": config,
                       "wall_seconds": previous_wall_seconds + time.monotonic() - started}
            if accelerator.scaler is not None:
                payload["scaler"] = accelerator.scaler.state_dict()
            temporary = state_path.with_name("last.tmp")
            torch.save(payload, temporary)
            temporary.replace(state_path)
        accelerator.wait_for_everyone()

    if args.resume and accelerator.scaler is not None:
        saved = torch.load(str(state_path), map_location="cpu", weights_only=False, mmap=True)
        if "scaler" in saved:
            accelerator.scaler.load_state_dict(saved["scaler"])
        del saved

    def denoise(latents, timesteps, text, *states):
        # These residuals enter the frozen UNet but retain gradients to the adapter.
        return unet(latents, timesteps, encoder_hidden_states=text,
                    down_intrablock_additional_residuals=list(states)).sample

    optimizer.zero_grad(set_to_none=True)
    status = "complete"
    while step < config["max_steps"]:
        stop_flag = torch.tensor(int(interrupted[0] or time.monotonic() - started > config["max_runtime_seconds"]), device=device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(stop_flag, op=torch.distributed.ReduceOp.MAX)
        if stop_flag.item():
            status = "interrupted_or_runtime_limit"
            break
        losses = []
        step_start = time.monotonic()
        for micro in range(accumulation):
            position = step * config["global_batch"] + micro * divisor + accelerator.process_index * config["micro_batch"]
            positions = list(range(position, position + config["micro_batch"]))
            cached, first, second, trust = data.batch(positions, args.variant)
            mean, logvar, text = (cached[key].to(device, dtype=weight_dtype) for key in ("mean", "logvar", "text"))
            posterior_samples, noises, times = [], [], []
            for index, item_position in enumerate(positions):
                generator = torch.Generator(device=device).manual_seed(args.seed * 1000003 + item_position)
                epsilon = torch.randn(mean[index].shape, generator=generator, device=device, dtype=weight_dtype)
                posterior_samples.append(mean[index] + (0.5 * logvar[index]).exp() * epsilon)
                noises.append(torch.randn(mean[index].shape, generator=generator, device=device, dtype=weight_dtype))
                times.append(torch.randint(0, scheduler.config.num_train_timesteps, (), generator=generator, device=device))
            latent = torch.stack(posterior_samples) * cache_metadata["vae_scaling_factor"]
            noise, timesteps = torch.stack(noises), torch.stack(times)
            noisy = scheduler.add_noise(latent, noise, timesteps)
            if scheduler.config.prediction_type == "epsilon":
                target = noise
            elif scheduler.config.prediction_type == "v_prediction":
                target = scheduler.get_velocity(latent, noise, timesteps)
            else:
                raise ValueError("Unsupported diffusion prediction target")
            conditions = torch.cat((first, second)).to(device)
            reliability = torch.cat((trust, trust)).to(device)
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    states = model(conditions, reliability)
                    noisy_pair, time_pair, text_pair = (torch.cat((item, item)) for item in (noisy, timesteps, text))
                    if config["checkpoint_unet"]:
                        predicted = checkpoint(denoise, noisy_pair, time_pair, text_pair, *states, use_reentrant=False)
                    else:
                        predicted = denoise(noisy_pair, time_pair, text_pair, *states)
                    first_pred, second_pred = predicted.chunk(2)
                    diffusion = 0.5 * (F.mse_loss(first_pred.float(), target.float()) + F.mse_loss(second_pred.float(), target.float()))
                    pair = pair_consistency(first_pred, second_pred, trust.to(device))
                    pair_weight = config["pair_weight"] if args.variant in ("full", "fixed_gate") else 0.0
                    loss = diffusion + pair_weight * pair
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                losses.append(torch.stack((diffusion.detach(), pair.detach(), loss.detach())))
        step += 1
        summary = accelerator.gather(torch.stack(losses).mean(0)).view(-1, 3).mean(0).cpu().tolist()
        if accelerator.is_main_process:
            row = {"step": step, "diffusion_loss": summary[0], "pair_loss": summary[1], "loss": summary[2],
                   "seconds": time.monotonic() - step_start, "peak_vram_MiB": torch.cuda.max_memory_allocated() / 2**20}
            with (output / "train.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if step % config["checkpoint_every"] == 0 and step < config["max_steps"]:
            save()
    save()
    if accelerator.is_main_process:
        write_json(output / "status.json", {"status": status, "step": step, "target_steps": config["max_steps"],
                   "wall_seconds": previous_wall_seconds + time.monotonic() - started,
                   "session_wall_seconds": time.monotonic() - started,
                   "peak_vram_MiB": torch.cuda.max_memory_allocated() / 2**20})
    accelerator.end_training()


if __name__ == "__main__":
    main()
