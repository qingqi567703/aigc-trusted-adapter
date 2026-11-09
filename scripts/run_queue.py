"""Run independent jobs only on explicitly allocated Linux GPU IDs."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigc.common import ROOT, project_path, read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", required=True, help="Comma-separated IDs allocated by the school")
    parser.add_argument("--gpus-per-run", type=int, default=1)
    parser.add_argument("--config", default="configs/course.json")
    parser.add_argument("--max-hours", type=float, default=72)
    parser.add_argument("--seeds", default="42,43")
    parser.add_argument("--phase", choices=("train", "generate", "evaluate"), default="train")
    parser.add_argument("--sample-count", type=int, default=1200)
    args = parser.parse_args()
    if os.name != "posix":
        raise RuntimeError("The school queue launcher targets Linux. Use the module entry points for local Windows checks.")
    if args.gpus_per_run < 1 or args.max_hours <= 0 or args.sample_count < 32:
        raise ValueError("GPU group size and runtime must be positive; distribution evaluation needs at least 32 images")
    gpu_ids = args.gpus.split(",")
    if any(not value.strip() for value in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids) or len(gpu_ids) % args.gpus_per_run:
        raise ValueError("Provide unique GPU IDs in complete per-run groups")
    groups = [gpu_ids[start:start + args.gpus_per_run] for start in range(0, len(gpu_ids), args.gpus_per_run)]
    seeds = [int(value) for value in args.seeds.split(",")]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Training seeds must be unique")
    jobs = [(variant, seed) for seed in seeds for variant in ("baseline", "augmented", "premask", "full")]
    jobs += [(variant, seeds[0]) for variant in ("no_pair", "fixed_gate")]
    ledger_path = ROOT / "runs/queue_ledger.json"
    ledger = read_json(ledger_path) if ledger_path.exists() else {"occupied_seconds": 0.0, "events": []}
    stop_requested = [False]
    def stop(signum, frame):
        stop_requested[0] = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    running = []
    previous = time.monotonic()
    failed = False
    try:
        while jobs or running:
            now = time.monotonic()
            if running:
                ledger["occupied_seconds"] += now - previous
            previous = now
            write_json(ledger_path, ledger)
            if stop_requested[0] or ledger["occupied_seconds"] >= args.max_hours * 3600:
                break
            for entry in list(running):
                code = entry["process"].poll()
                if code is not None:
                    entry["log"].close()
                    groups.append(entry["gpus"])
                    running.remove(entry)
                    ledger["events"].append({"job": entry["name"], "phase": args.phase, "exit_code": code})
                    completed_training = True
                    if args.phase == "train":
                        status_path = ROOT / "runs/course" / entry["name"] / "status.json"
                        completed_training = status_path.exists() and read_json(status_path).get("status") == "complete"
                    if code != 0 or not completed_training:
                        failed = True
                        jobs.clear()
            while groups and jobs:
                variant, seed = jobs.pop(0)
                name = f"{variant}_s{seed}"
                folder = ROOT / "runs/course" / name
                samples = folder / "generated"
                folder.mkdir(parents=True, exist_ok=True)
                if args.phase == "train":
                    status = folder / "status.json"
                    if status.exists() and read_json(status).get("status") == "complete":
                        continue
                    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                               f"--nproc_per_node={args.gpus_per_run}", "--module", "aigc.train",
                               "--config", args.config, "--variant", variant, "--seed", str(seed),
                               "--output", str(folder)]
                    if (folder / "last.pt").exists():
                        command.append("--resume")
                elif args.phase == "generate":
                    command = [sys.executable, "-m", "aigc.generate", "--checkpoint", str(folder / "last.pt"),
                               "--manifest", "data/manifests/test.jsonl", "--output", str(samples),
                               "--count", str(args.sample_count)]
                    if (samples / "samples.jsonl").exists():
                        command.append("--resume")
                else:
                    if (samples / "metrics.json").exists():
                        continue
                    command = [sys.executable, "-m", "aigc.evaluate", "--samples", str(samples), "--distribution"]
                group = groups.pop(0)
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = ",".join(group)
                log = (folder / f"{args.phase}_console.log").open("a", encoding="utf-8")
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                running.append({"name": name, "process": process, "log": log, "gpus": group})
                print("START", name, args.phase, group, flush=True)
            time.sleep(5)
    finally:
        for entry in running:
            if entry["process"].poll() is None:
                os.killpg(entry["process"].pid, signal.SIGINT)
        for entry in running:
            try:
                entry["process"].wait(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(entry["process"].pid, signal.SIGTERM)
                entry["process"].wait(timeout=30)
            entry["log"].close()
        write_json(ledger_path, ledger)
    if running or failed:
        raise SystemExit("Queue stopped before all requested jobs completed. Check logs and saved checkpoints.")
    print("Requested queue completed.", flush=True)


if __name__ == "__main__":
    main()
