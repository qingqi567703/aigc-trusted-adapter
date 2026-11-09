"""Collect completed runs without inventing missing seeds or results."""

import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigc.common import ROOT, project_path, read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs/course")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    rows = []
    for folder in sorted(project_path(args.runs).glob("*")):
        metric_file = folder / "generated/metrics.json"
        if not metric_file.is_file():
            continue
        config = read_json(folder / "config.json")
        status = read_json(folder / "status.json")
        metrics = read_json(metric_file)
        generation = read_json(folder / "generated/generation_config.json")
        rows.append({"run":folder.name, "variant":config["variant"], "seed":config["seed"],
                     "resolution":config["resolution"], "global_batch":config["global_batch"],
                     "learning_rate":config["learning_rate"], "train_manifest":config["manifest"],
                     "sd_model":config["sd_model"], "adapter_model":config["adapter_model"],
                     "generation_steps":generation["steps"], "generation_seed":generation["seed"],
                     "test_manifest_sha256":generation["manifest_sha256"],
                     "steps":status["step"], "train_status":status["status"],
                     "train_wall_seconds":status["wall_seconds"], "peak_vram_MiB":status["peak_vram_MiB"],
                     **{key:metrics.get(key) for key in ("n_images","anchor_f1","anchor_precision","anchor_recall",
                          "fake_follow_lift","clip_cosine","fid_clean","kid_clean")}})
    if not rows:
        raise SystemExit("No evaluated course runs found. No result files written.")
    protocol_fields = ("resolution", "global_batch", "steps", "n_images", "learning_rate", "train_manifest",
                       "sd_model", "adapter_model", "generation_steps", "generation_seed", "test_manifest_sha256")
    protocols = {tuple(row[field] for field in protocol_fields) for row in rows}
    if len(protocols) != 1 or any(row["train_status"] != "complete" for row in rows):
        raise ValueError("Run protocols differ or some training runs are incomplete. Review before aggregating.")
    output = project_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "course_results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for variant in sorted({row["variant"] for row in rows}):
        subset = [row for row in rows if row["variant"] == variant]
        if len({row["seed"] for row in subset}) != len(subset):
            raise ValueError(f"Duplicate seeds for {variant}")
        summary[variant] = {"training_seeds":[row["seed"] for row in subset]}
        for metric in ("anchor_f1","fake_follow_lift","clip_cosine","fid_clean","kid_clean"):
            values = [row[metric] for row in subset if row[metric] is not None]
            summary[variant][metric] = {"mean":float(np.mean(values)) if values else None,
                                       "seed_std":float(np.std(values, ddof=1)) if len(values)>1 else None,
                                       "n_seeds":len(values)}
    write_json(output / "course_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
