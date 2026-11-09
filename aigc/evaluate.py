import argparse
import json

from .common import offline_environment, project_path, write_json
offline_environment()

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def edges(path):
    # Pillow handles Unicode Windows paths consistently, unlike cv2.imread.
    with Image.open(path) as source:
        pixels = np.asarray(source.convert("L"))
    return cv2.Canny(pixels, 100, 200) > 0


def dilate(mask, radius=2):
    return cv2.dilate(mask.astype(np.uint8), np.ones((2 * radius + 1,) * 2, np.uint8)) > 0


def anchor_scores(prediction, target, trusted):
    predicted, expected = prediction & trusted, target & trusted
    precision = (predicted & dilate(expected)).sum() / max(predicted.sum(), 1)
    recall = (expected & dilate(predicted)).sum() / max(expected.sum(), 1)
    return {"anchor_precision": float(precision), "anchor_recall": float(recall),
            "anchor_f1": float(2 * precision * recall / max(precision + recall, 1e-12))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--clip-model", default="assets/models/clip")
    parser.add_argument("--distribution", action="store_true")
    args = parser.parse_args()
    root = project_path(args.samples)
    rows = [json.loads(line) for line in (root / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(project_path(args.clip_model), local_files_only=True).eval().to(device)
    processor = CLIPProcessor.from_pretrained(project_path(args.clip_model), local_files_only=True)
    diagnostics = []
    for row in rows:
        if row["view"] == "clean":
            continue
        identity, view = row["id"], row["view"]
        clean = np.asarray(Image.open(root / "conditions" / f"{identity}_clean.png")) > 127
        condition = np.asarray(Image.open(root / "conditions" / f"{identity}_{view}.png")) > 127
        trust_values = np.asarray(Image.open(root / "conditions" / f"{identity}_trust.png"))
        trusted = trust_values == 255
        trusted = ~dilate(~trusted)
        fake = condition & ~dilate(clean)
        prediction = edges(root / row["sample"])
        clean_prediction = edges(root / "samples" / f"{identity}_clean.png")
        fake_coverage = (fake & dilate(prediction)).sum() / max(fake.sum(), 1)
        reference_coverage = (fake & dilate(clean_prediction)).sum() / max(fake.sum(), 1)
        with Image.open(root / row["sample"]) as source:
            inputs = processor(text=[row["caption"]], images=source.convert("RGB"), return_tensors="pt",
                               padding=True, truncation=True, max_length=77)
        with torch.inference_mode():
            output = model(**{key: value.to(device) for key, value in inputs.items()})
            cosine = torch.nn.functional.cosine_similarity(output.image_embeds, output.text_embeds).item()
        diagnostics.append({"id": identity, "view": view, **anchor_scores(prediction, clean, trusted),
                            "has_fake_strokes": bool(fake.any()), "fake_follow_lift": float(fake_coverage-reference_coverage),
                            "clip_cosine": cosine, "safety_flag": row["safety_flag"],
                            "corruption": row["corruption"], "low_trust": round(float(trust_values.min())/255, 2)})
    summary = {"n_images": len({row["id"] for row in diagnostics}), "n_generated_corrupt_views": len(diagnostics)}
    if not diagnostics:
        raise ValueError("No generated intervention views to evaluate")
    for metric in ("anchor_precision", "anchor_recall", "anchor_f1", "clip_cosine"):
        summary[metric] = float(np.mean([row[metric] for row in diagnostics]))
    fake_rows = [row for row in diagnostics if row["has_fake_strokes"]]
    summary["fake_follow_lift"] = float(np.mean([row["fake_follow_lift"] for row in fake_rows])) if fake_rows else None
    summary["n_fake_stroke_views"] = len(fake_rows)
    summary["safety_flag_count"] = sum(row["safety_flag"] for row in diagnostics)
    summary["by_condition"] = {}
    for key in sorted({(row["corruption"], row["low_trust"]) for row in diagnostics}):
        subset = [row for row in diagnostics if (row["corruption"], row["low_trust"]) == key]
        summary["by_condition"][str(key)] = {"n_views": len(subset), "anchor_f1": float(np.mean([row["anchor_f1"] for row in subset])),
                                          "clip_cosine": float(np.mean([row["clip_cosine"] for row in subset]))}
    # Resample image IDs, keeping two views of the same source image together.
    summary["image_bootstrap_95ci"] = {}
    random = np.random.default_rng(20261108)
    for metric in ("anchor_f1", "clip_cosine", "fake_follow_lift"):
        eligible = fake_rows if metric == "fake_follow_lift" else diagnostics
        ids = sorted({row["id"] for row in eligible})
        if len(ids) >= 2:
            values = np.array([np.mean([row[metric] for row in eligible if row["id"] == identity]) for identity in ids])
            means = values[random.integers(0, len(values), size=(1000, len(values)))].mean(1)
            summary["image_bootstrap_95ci"][metric] = np.quantile(means, [0.025, 0.975]).tolist()
    if args.distribution:
        if summary["n_images"] < 32:
            raise ValueError("At least 32 images required even for preliminary distribution metrics")
        import shutil
        from cleanfid import fid
        from .metrics import clean_features, clean_inception_extractor
        folder = root / "distribution_view0"
        folder.mkdir(exist_ok=True)
        for row in rows:
            if row["view"] == "0":
                shutil.copy2(root / row["sample"], folder / f"{row['id']}.png")
        # An explicit local extractor prevents clean-fid's implicit /tmp download.
        extractor = clean_inception_extractor(project_path("assets/metrics/inception-2015-12-05.pt"), device)
        first = clean_features(folder, extractor, device)
        second = clean_features(root / "references", extractor, device)
        summary["fid_clean"] = float(fid.frechet_distance(first.mean(0), np.cov(first, rowvar=False),
                                                        second.mean(0), np.cov(second, rowvar=False)))
        np.random.seed(20261108)
        summary["kid_clean"] = float(fid.kernel_distance(first, second, num_subsets=20,
                                                         max_subset_size=min(100, summary["n_images"])))
    write_json(root / "diagnostics.json", diagnostics)
    write_json(root / "metrics.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
