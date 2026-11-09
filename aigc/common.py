import hashlib
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def offline_environment():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(ROOT / "work/hf_cache")
    os.environ["TORCH_HOME"] = str(ROOT / "assets/torch")


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path):
    return json.loads(project_path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_manifest(path):
    return [json.loads(line) for line in project_path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cache_path(record, resolution):
    return ROOT / "data/cache" / str(resolution) / (str(record["id"]) + ".pt")
