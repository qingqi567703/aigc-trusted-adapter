import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from .common import cache_path, read_manifest


def image_tensor(path, resolution):
    with Image.open(path) as source:
        image = ImageOps.fit(source.convert("RGB"), (resolution, resolution), method=Image.Resampling.LANCZOS)
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255


def sketch_tensor(rgb):
    pixels = (rgb.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    return torch.from_numpy(cv2.Canny(gray, 100, 200).copy()).unsqueeze(0).float() / 255


def condition_pair(clean, seed, kind="mixed"):
    """Two interventions share a target, trust map, region, and noise scale."""
    rng = np.random.default_rng(seed)
    height, width = clean.shape[-2:]
    trust = np.ones((height, width), dtype=np.float32)
    patch_h = int(rng.integers(max(8, height // 5), max(9, height // 2)))
    patch_w = int(rng.integers(max(8, width // 5), max(9, width // 2)))
    top = int(rng.integers(0, height - patch_h + 1))
    left = int(rng.integers(0, width - patch_w + 1))
    trust[top:top + patch_h, left:left + patch_w] = float(rng.choice([0.0, 0.15, 0.35, 0.55]))
    selected = kind if kind != "mixed" else str(rng.choice(["strokes", "shift", "erase"]))
    outputs = []
    clean_array = clean[0].numpy()
    for _ in range(2):
        canvas = clean_array.copy()
        patch = canvas[top:top + patch_h, left:left + patch_w].copy()
        if selected == "erase":
            patch *= (rng.random(patch.shape) > 0.55)
        elif selected == "shift":
            patch[:] = np.roll(patch, (int(rng.integers(-12, 13)), int(rng.integers(-12, 13))), axis=(0, 1))
        elif selected == "strokes":
            for _ in range(6):
                start = (int(rng.integers(patch_w)), int(rng.integers(patch_h)))
                end = (int(rng.integers(patch_w)), int(rng.integers(patch_h)))
                cv2.line(patch, start, end, color=1.0, thickness=int(rng.integers(1, 4)))
        else:
            raise ValueError(f"Unknown corruption {selected}")
        canvas[top:top + patch_h, left:left + patch_w] = patch
        outputs.append(torch.from_numpy(canvas.copy()).unsqueeze(0))
    return outputs[0], outputs[1], torch.from_numpy(trust).unsqueeze(0), selected


class CachedImages:
    def __init__(self, manifest, resolution, seed):
        self.records = read_manifest(manifest)
        if not self.records:
            raise ValueError("Empty data manifest")
        self.resolution = resolution
        self.seed = seed
        self._epoch = None
        self._order = None

    def sample(self, position, variant):
        epoch, offset = divmod(position, len(self.records))
        if epoch != self._epoch:
            self._order = np.random.default_rng(self.seed + epoch).permutation(len(self.records))
            self._epoch = epoch
        record = self.records[int(self._order[offset])]
        path = cache_path(record, self.resolution)
        cached = torch.load(path, map_location="cpu", weights_only=True)
        if cached["caption"] != record["caption"]:
            raise ValueError("Cached caption differs from the data manifest")
        clean = cached["sketch"].float()
        first, second, trust, _ = condition_pair(clean, self.seed * 1000003 + position)
        if variant == "baseline":
            first, second, trust = clean, clean, torch.ones_like(trust)
        return cached, first, second, trust

    def batch(self, positions, variant):
        samples = [self.sample(position, variant) for position in positions]
        cached = {key: torch.stack([sample[0][key] for sample in samples]) for key in ("mean", "logvar", "text")}
        return cached, *(torch.stack([sample[index] for sample in samples]) for index in (1, 2, 3))
