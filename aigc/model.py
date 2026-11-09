import torch
from torch import nn
from torch.nn import functional as F
from diffusers import T2IAdapter

VARIANTS = ("baseline", "augmented", "premask", "full", "no_pair", "fixed_gate")


class TrustedAdapter(nn.Module):
    def __init__(self, path, variant):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant}")
        self.variant = variant
        self.adapter = T2IAdapter.from_pretrained(path, local_files_only=True)
        self.gates = nn.ModuleList()
        if variant in ("full", "no_pair"):
            for channels in self.adapter.config.channels:
                gate = nn.Sequential(nn.Conv2d(channels + 1, 16, 1), nn.SiLU(), nn.Conv2d(16, 1, 1))
                nn.init.zeros_(gate[-1].weight)
                nn.init.zeros_(gate[-1].bias)
                self.gates.append(gate)

    def forward(self, sketch, trust):
        if self.variant in ("premask", "full", "no_pair", "fixed_gate"):
            sketch = sketch * trust
        channels = self.adapter.config.in_channels
        if sketch.shape[1] != channels:
            if sketch.shape[1] == 1 and channels == 3:
                sketch = sketch.repeat(1, 3, 1, 1)
            else:
                raise ValueError("Unexpected adapter input channels")
        return self.gate_features(self.adapter(sketch), trust)

    def gate_features(self, features, trust):
        if self.variant in ("full", "no_pair", "fixed_gate"):
            modified = []
            for index, feature in enumerate(features):
                reliability = F.interpolate(trust, size=feature.shape[-2:], mode="area")
                scale = reliability
                if self.variant != "fixed_gate":
                    delta = self.gates[index](torch.cat((feature, reliability.to(feature.dtype)), dim=1))
                    scale = reliability * (1 + 0.5 * torch.tanh(delta))
                modified.append(feature * scale.to(feature.dtype))
            features = modified
        return features


def pair_consistency(first, second, trust, margin=2):
    reliable = F.interpolate(trust, size=first.shape[-2:], mode="area") >= 0.999
    unknown = (~reliable).float()
    expanded = F.max_pool2d(unknown, kernel_size=2 * margin + 1, stride=1, padding=margin)
    region = 1 - expanded
    numerator = ((first.float() - second.float()).square() * region).sum(dim=(1, 2, 3))
    denominator = region.sum(dim=(1, 2, 3)) * first.shape[1]
    return (numerator / denominator.clamp_min(1)).mean()
