"""Verify local preparation and optionally create a secret-free deployment tar."""

import argparse
import hashlib
import json
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def check_wheels(wheels):
    versions = {"torch": Version("2.2.2")}
    for wheel in wheels:
        name, version, _, _ = parse_wheel_filename(wheel.name)
        versions[canonicalize_name(name)] = version
    requirements = []
    def read_requirements(path):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line.startswith("-r "):
                read_requirements(ROOT / line[3:].strip())
            elif line:
                requirements.append(("project", Requirement(line)))
    read_requirements(ROOT / "requirements-linux-cp310.txt")
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            metadata_path = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
            metadata = Parser().parsestr(archive.read(metadata_path).decode("utf-8"))
            requirements.extend((wheel.name, Requirement(value)) for value in metadata.get_all("Requires-Dist", []))
    environment = default_environment()
    environment.update({"os_name": "posix", "sys_platform": "linux", "platform_system": "Linux",
                        "platform_machine": "x86_64", "python_version": "3.10", "python_full_version": "3.10.14",
                        "implementation_name": "cpython", "platform_python_implementation": "CPython", "extra": ""})
    errors = []
    for source, requirement in requirements:
        if requirement.marker and not requirement.marker.evaluate(environment):
            continue
        version = versions.get(canonicalize_name(requirement.name))
        if version is None or version not in requirement.specifier:
            errors.append(f"Offline dependency {source} requires {requirement}; available={version}")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", action="store_true")
    parser.add_argument("--archive-name", default="aigc_course_2026_v2.tar.gz")
    args = parser.parse_args()
    problems, records = [], []
    for name in ("sd15", "adapter", "clip"):
        manifest = ROOT / f"assets/models/{name}/download_manifest.json"
        if not manifest.exists():
            problems.append(f"Missing model manifest {name}")
            continue
        for item in json.loads(manifest.read_text(encoding="utf-8"))["files"]:
            path = ROOT / item["path"].replace("\\", "/")
            if not path.exists() or digest(path) != item["sha256"]:
                problems.append(f"Missing or invalid model file {item['path']}")
    required = ROOT / "assets/metrics/inception-2015-12-05.pt"
    if not required.is_file():
        problems.append("Missing Inception metric weights")
    else:
        metric_record = json.loads((required.parent / "manifest.json").read_text(encoding="utf-8"))
        if digest(required) != metric_record["sha256"]:
            problems.append("Invalid Inception metric checksum")
    split_ids = {}
    for split in ("train", "dev", "test"):
        path = ROOT / f"data/manifests/{split}.jsonl"
        if not path.is_file():
            problems.append(f"Missing {split} manifest")
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        split_ids[split] = {row["id"] for row in rows}
        missing = [row["image"] for row in rows if not (ROOT / row["image"]).is_file()]
        if missing:
            problems.append(f"{split} is missing {len(missing)} image files")
        if split == "train":
            missing_cache = sum(not (ROOT / f"data/cache/512/{row['id']}.pt").is_file() for row in rows)
            if missing_cache:
                problems.append(f"Training 512px cache is missing {missing_cache} entries")
    if len(split_ids) == 3:
        for first, second in (("train", "dev"), ("train", "test"), ("dev", "test")):
            if split_ids[first] & split_ids[second]:
                problems.append(f"Split overlap {first}/{second}")
    wheels = list((ROOT / "offline/wheelhouse").glob("*.whl"))
    if len(wheels) < 27:
        problems.append(f"Linux wheelhouse incomplete ({len(wheels)} wheels)")
    for wheel in wheels:
        if "win_" in wheel.name:
            problems.append(f"Windows wheel in Linux bundle {wheel.name}")
    problems.extend(check_wheels(wheels))
    for path in sorted((ROOT / "offline/wheelhouse").glob("*.whl")):
        records.append({"path":path.relative_to(ROOT).as_posix(), "bytes":path.stat().st_size, "sha256":digest(path)})
    input_records = []
    for folder in ("data/manifests", "data/coco/annotations", "data/coco/images", "data/cache"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and path.suffix not in (".tmp", ".part"):
                input_records.append({"path":path.relative_to(ROOT).as_posix(),
                                      "bytes":path.stat().st_size, "sha256":digest(path)})
    result = {"ready":not problems,"problems":problems,"split_counts":{key:len(value) for key,value in split_ids.items()},
              "linux_wheels":records,
              "input_files":input_records,
              "server_base_requirement":"Python 3.10 + preinstalled torch 2.2.2/CUDA 12.1; matching torchvision is bundled",
              "linux_install_status":"validated on school RTX4090 Linux with Python3.10.14 / Torch2.2.2 / CUDA12.1; see docs/results/server_full_acceptance.json",
              "bundle_generation":2,
              "provenance_note":"Core modules match the frozen formal experiment; documentation and postprocessing updated after evaluation."}
    result["source_files"] = []
    for source in ("aigc", "scripts", "configs", "docs"):
        for path in sorted((ROOT / source).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                result["source_files"].append({"path":path.relative_to(ROOT).as_posix(), "sha256":digest(path)})
    for name in ("README.md", "requirements.txt", "requirements-linux-cp310.txt", "requirements-report.txt", ".gitignore"):
        result["source_files"].append({"path":name, "sha256":digest(ROOT/name)})
    (ROOT / "offline").mkdir(exist_ok=True)
    (ROOT / "offline/preparation_manifest.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({key:value for key,value in result.items() if key not in ("linux_wheels", "input_files", "source_files")},ensure_ascii=False,indent=2))
    if problems:
        raise SystemExit(1)
    if args.bundle:
        if Path(args.archive_name).name!=args.archive_name or not args.archive_name.endswith('.tar.gz'):
            raise ValueError('Archive name must be a simple .tar.gz filename')
        destination = ROOT / 'offline' / args.archive_name
        temporary = destination.with_name(destination.name + ".part")
        allowed = ["aigc","scripts","configs","docs","assets","data/manifests","data/coco/annotations","data/coco/images","data/cache",
                   "offline/wheelhouse","offline/preparation_manifest.json","README.md","requirements.txt","requirements-linux-cp310.txt","requirements-report.txt",".gitignore"]
        with tarfile.open(temporary,"w:gz",compresslevel=1) as archive:
            for name in allowed:
                path = ROOT / name
                if path.exists():
                    archive.add(path,arcname="aigc_course_2026/"+name,
                                filter=lambda item: None if "__pycache__" in item.name or item.name.endswith((".pyc",".part")) else item)
        temporary.replace(destination)
        result={"path":destination.relative_to(ROOT).as_posix(),"bytes":destination.stat().st_size,"sha256":digest(destination)}
        (ROOT / "offline/bundle_manifest.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
        print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
