"""Download pinned model files and COCO subsets with resumable transfers."""

import argparse
import concurrent.futures
import hashlib
import json
import random
import shutil
import threading
import time
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LOCAL = threading.local()


def session():
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
        LOCAL.session.trust_env = False
    return LOCAL.session


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, path, expected_size=None, expected_sha=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and (expected_size is None or path.stat().st_size == expected_size):
        if expected_sha is None or sha256(path) == expected_sha:
            return path
    partial = path.with_name(path.name + ".part")
    for attempt in range(6):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with session().get(url, headers=headers, stream=True, timeout=(20, 90)) as response:
                if response.status_code == 416 and expected_size == offset:
                    pass
                else:
                    response.raise_for_status()
                    append = response.status_code == 206 and offset > 0
                    if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise RuntimeError("Unexpected byte range")
                    if response.status_code == 206 and expected_size is None:
                        total = response.headers.get("Content-Range", "").rsplit("/", 1)[-1]
                        if total.isdigit():
                            expected_size = int(total)
                    with partial.open("ab" if append else "wb") as stream:
                        for chunk in response.iter_content(1024 * 1024):
                            if chunk:
                                stream.write(chunk)
            if expected_size is not None and partial.stat().st_size != expected_size:
                raise RuntimeError(f"Incomplete file {path.name}")
            if expected_sha and sha256(partial) != expected_sha:
                partial.unlink()
                raise RuntimeError(f"Checksum mismatch for {path.name}")
            partial.replace(path)
            return path
        except (requests.RequestException, OSError, RuntimeError):
            if attempt == 5:
                raise
            time.sleep(min(2 ** attempt, 15))


def choose_weights(files, prefix):
    candidates = [name for name in files if name.startswith(prefix) and name.endswith((".bin", ".safetensors"))]
    candidates = [name for name in candidates if "non_ema" not in name and "flax" not in name]
    if not candidates:
        raise RuntimeError(f"No weights for {prefix}")
    return min(candidates, key=lambda name: ("fp16" not in name, not name.endswith(".safetensors"), len(name), name))


def model(repo, destination, endpoint, prefixes):
    response = session().get(f"{endpoint}/api/models/{repo}?blobs=true", timeout=40)
    response.raise_for_status()
    info = response.json()
    revision = info["sha"]
    siblings = {item["rfilename"]: item for item in info["siblings"]}
    selected = {name for name in siblings if name.endswith((".json", ".txt", ".model")) and not name.startswith("images/")}
    if prefixes:
        selected = {name for name in selected if name.startswith(tuple(prefixes)) or name in ("model_index.json", "README.md")}
        selected.update(choose_weights(siblings, prefix) for prefix in prefixes if prefix in ("unet/", "vae/", "text_encoder/", "safety_checker/"))
    else:
        selected.add(choose_weights(siblings, ""))
    if "README.md" in siblings:
        selected.add("README.md")
    records = []
    for name in sorted(selected):
        metadata = siblings[name]
        lfs = metadata.get("lfs") or {}
        expected = lfs.get("sha256")
        path = ROOT / destination / name
        print(f"MODEL {repo} {name}", flush=True)
        download(f"{endpoint}/{repo}/resolve/{revision}/{name}", path, metadata.get("size"), expected)
        records.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path), "upstream_sha256": expected})
    manifest = {"repo": repo, "revision": revision, "download_endpoint": endpoint, "files": records}
    target = ROOT / destination / "download_manifest.json"
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def pinned_model(name, endpoint):
    source = ROOT / f"docs/resources/models/{name}.json"
    metadata = json.loads(source.read_text(encoding="utf-8"))
    destination = ROOT / f"assets/models/{name}"
    for item in metadata["files"]:
        relative = Path(item["path"].replace("\\", "/"))
        path = ROOT / relative
        filename = path.relative_to(destination).as_posix()
        print(f"PINNED {metadata['repo']} {filename}", flush=True)
        download(f"{endpoint}/{metadata['repo']}/resolve/{metadata['revision']}/{filename}",
                 path, item["bytes"], item["sha256"])
    shutil.copyfile(source, destination / "download_manifest.json")
    return metadata


def prepare_models(endpoint):
    if all((ROOT / f"docs/resources/models/{name}.json").exists() for name in ("sd15", "adapter", "clip")):
        manifests = [pinned_model(name, endpoint) for name in ("sd15", "adapter", "clip")]
        target = ROOT / "assets/manifests/models.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(manifests, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    manifests = []
    manifests.append(model("stable-diffusion-v1-5/stable-diffusion-v1-5", "assets/models/sd15", endpoint,
                           ["unet/", "vae/", "text_encoder/", "tokenizer/", "scheduler/", "safety_checker/", "feature_extractor/"]))
    manifests.append(model("TencentARC/t2iadapter_sketch_sd15v2", "assets/models/adapter", endpoint, []))
    manifests.append(model("openai/clip-vit-base-patch32", "assets/models/clip", endpoint, []))
    target = ROOT / "assets/manifests/models.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifests, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_data(train_count, dev_count, test_count, workers):
    frozen = ROOT / "docs/resources/splits"
    if (train_count, dev_count, test_count) == (8000, 400, 1200) and all((frozen/f"{split}.jsonl").is_file() for split in ("train","dev","test")):
        records = []
        for split in ("train","dev","test"):
            source = frozen/f"{split}.jsonl"
            target = ROOT/f"data/manifests/{split}.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            records.extend(json.loads(line) for line in source.read_text(encoding="utf-8").splitlines())
        if (frozen/"smoke.jsonl").exists():
            shutil.copyfile(frozen/"smoke.jsonl", ROOT/"data/manifests/smoke.jsonl")
        checksums_path = ROOT/"docs/resources/images_sha256.json"
        checksums = json.loads(checksums_path.read_text(encoding="utf-8")) if checksums_path.exists() else {}
        def fetch_pinned(item):
            path = ROOT/item["image"]
            url = "https://s3.amazonaws.com/images.cocodataset.org/" + item["source_split"] + "/" + path.name
            expected = checksums.get(item["image"], {})
            for attempt in range(2):
                download(url, path, expected.get("bytes"), expected.get("sha256"))
                try:
                    with Image.open(path) as image:
                        image.load()
                    return
                except OSError:
                    if attempt:
                        raise
                    path.unlink()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for count, _ in enumerate(executor.map(fetch_pinned, records), 1):
                if count % 100 == 0 or count == len(records):
                    print(f"PINNED COCO {count}/{len(records)}", flush=True)
        return
    annotations = ROOT / "data/coco/annotations"
    caption_hashes = {
        "train": "4b62086319480e0739ef390d04084515defb9c213ff13605a036061e33314317",
        "val": "afe3b30e403dd7f228e2373023abbd60042a6e10ec6874d3652df034d289ebb9",
    }
    for split, digest in caption_hashes.items():
        name = f"captions_{split}2017.json"
        print("ANNOTATIONS", name, flush=True)
        download("https://hf-mirror.com/datasets/merve/coco/resolve/main/annotations/" + name,
                 annotations / name, expected_sha=digest)
    selections = {}
    for split in ("train", "val"):
        content = json.loads((annotations / f"captions_{split}2017.json").read_text(encoding="utf-8"))
        captions = {}
        for item in sorted(content["annotations"], key=lambda item: item["id"]):
            captions.setdefault(item["image_id"], item["caption"])
        images = sorted(content["images"], key=lambda item: item["id"])
        random.Random(20261108).shuffle(images)
        groups = {"train": images[:train_count]} if split == "train" else {"dev": images[:dev_count], "test": images[dev_count:dev_count + test_count]}
        for group, chosen in groups.items():
            selections[group] = [
                {"id": item["id"], "image": f"data/coco/images/{split}2017/{item['file_name']}",
                 "caption": captions[item["id"]], "source_split": split + "2017"}
                for item in chosen
            ]
    records = [item for values in selections.values() for item in values]
    for split, values in selections.items():
        path = ROOT / f"data/manifests/{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in values), encoding="utf-8")
    def fetch(item):
        path = ROOT / item["image"]
        url = "https://s3.amazonaws.com/images.cocodataset.org/" + item["source_split"] + "/" + path.name
        for attempt in range(2):
            download(url, path)
            try:
                with Image.open(path) as image:
                    image.load()
                break
            except OSError:
                if attempt:
                    raise
                path.unlink()
        return item
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for count, _ in enumerate(executor.map(fetch, records), 1):
            if count % 100 == 0 or count == len(records):
                print(f"COCO {count}/{len(records)}", flush=True)
    print("COCO manifests completed.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("models", "data"))
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--train-count", type=int, default=8000)
    parser.add_argument("--dev-count", type=int, default=400)
    parser.add_argument("--test-count", type=int, default=1200)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.phase == "models":
        prepare_models(args.endpoint.rstrip("/"))
    else:
        prepare_data(args.train_count, args.dev_count, args.test_count, args.workers)


if __name__ == "__main__":
    main()
