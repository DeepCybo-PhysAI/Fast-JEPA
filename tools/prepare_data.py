#!/usr/bin/env python3
"""Prepare the four fixed benchmark datasets; never starts training."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from urllib.parse import quote
from urllib.request import urlopen, build_opener, ProxyHandler, install_opener

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "pusht": ("quentinll/lewm-pusht", "pusht_expert_train.h5.zst", "pusht_expert_train.h5"),
    "cube": ("quentinll/lewm-cube", "cube_single_expert.tar.zst", "ogbench/cube_single_expert.h5"),
    "reacher": ("quentinll/lewm-reacher", "reacher.tar.zst", "reacher.h5"),
    "tworoom": ("quentinll/lewm-tworooms", "tworoom.tar.zst", "tworoom.h5"),
}


def log(message):
    print(message, flush=True)


def run(command, **kwargs):
    process = subprocess.Popen(command, **kwargs)
    try:
        code = process.wait()
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    if code:
        raise subprocess.CalledProcessError(code, command)


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def get_json(url):
    for attempt in range(3):
        try:
            with urlopen(url, timeout=30) as response:
                return json.load(response)
        except Exception:
            if attempt == 2:
                raise
            log(f"Metadata request failed; retry {attempt + 1}/2")
            time.sleep(2 * (attempt + 1))


def source_manifest(task, data_dir, endpoint):
    repo, filename, _ = TASKS[task]
    saved = data_dir / f"{task}-source.json"
    local_copy = REPO_ROOT / "data" / "manifests" / f"{task}-source.json"
    if saved.exists() or local_copy.exists():
        manifest = json.loads((saved if saved.exists() else local_copy).read_text())
    else:
        info = get_json(f"{endpoint}/api/datasets/{repo}")
        revision = info["sha"]
        entries = get_json(f"{endpoint}/api/datasets/{repo}/tree/{revision}")
        entry = next(item for item in entries if item["path"] == filename)
        manifest = {"repo": repo, "revision": revision, "file": entry}
    if manifest["repo"] != repo or manifest["file"]["path"] != filename:
        raise RuntimeError(f"Source manifest does not match {task}")
    if not re.fullmatch(r"[0-9a-f]{40,64}", manifest["revision"]):
        raise RuntimeError("Invalid dataset revision")
    if not re.fullmatch(r"[0-9a-f]{64}", manifest["file"]["lfs"]["oid"]):
        raise RuntimeError("Missing or invalid archive SHA256")
    if manifest["file"]["size"] <= 0:
        raise RuntimeError("Invalid archive size")
    write_json(saved, manifest)
    return manifest


def verify_archive(path, manifest):
    expected_size = manifest["file"]["size"]
    if path.stat().st_size != expected_size:
        raise RuntimeError(f"Incomplete archive: {path}; expected {expected_size} bytes")
    log(f"SHA256 verification: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != manifest["file"]["lfs"]["oid"]:
        raise RuntimeError(f"SHA256 mismatch: {path}. Keep or rename this file before retrying; it was not deleted.")


def download(manifest, data_dir, args):
    archive = data_dir / manifest["file"]["path"]
    if archive.exists():
        verify_archive(archive, manifest)
        return archive
    partial = archive.with_name(archive.name + ".partial")
    aria_control = partial.with_name(partial.name + ".aria2")
    method = args.downloader
    if method == "auto":
        method = "aria2" if shutil.which("aria2c") else "curl"
    executable = "aria2c" if method == "aria2" else "curl"
    if not shutil.which(executable):
        raise RuntimeError(f"{executable} is unavailable; install it or choose --downloader curl")
    if method == "curl" and aria_control.exists():
        raise RuntimeError("This partial file has aria2 state; resume with --downloader aria2, not curl.")
    size = manifest["file"]["size"]
    if partial.exists() and partial.stat().st_size > size:
        raise RuntimeError(f"Partial file is larger than the archive: {partial}; rename it before retrying")
    if not (partial.exists() and partial.stat().st_size == size and not aria_control.exists()):
        url = (f"{args.endpoint}/datasets/{manifest['repo']}/resolve/"
               f"{manifest['revision']}/{quote(archive.name)}?download=true")
        log(f"Download: {archive.name} ({size / 1e9:.2f} GB), method={method}")
        if method == "aria2":
            command = [executable, "--continue=true", "--auto-file-renaming=false",
                       "--file-allocation=none", f"--max-connection-per-server={args.connections}",
                       f"--split={args.connections}", "--min-split-size=8M",
                       "--connect-timeout=20", "--timeout=120", "--max-tries=5", "--retry-wait=5",
                       "--summary-interval=30", "--console-log-level=warn",
                       f"--dir={data_dir}", f"--out={partial.name}", url]
        else:
            command = [executable, "--fail", "--location", "--retry", "5", "--retry-all-errors",
                       "--continue-at", "-", "--connect-timeout", "20", "--speed-limit", "1024",
                       "--speed-time", "120", "--output", str(partial), url]
        if args.no_proxy:
            if method == "aria2":
                command[1:1] = ["--all-proxy=", "--http-proxy=", "--https-proxy="]
            else:
                command[1:1] = ["--noproxy", "*"]
        run(command)
    verify_archive(partial, manifest)
    os.replace(partial, archive)
    return archive


def inspect_hdf5(path):
    import h5py
    import hdf5plugin  # noqa: F401; register archive compression filters
    with h5py.File(path, "r") as dataset:
        for key in ("action", "ep_len", "ep_offset"):
            if key not in dataset:
                raise RuntimeError(f"Missing HDF5 column {key}: {path}")
        return {"frames": int(dataset["action"].shape[0]),
                "episodes": int(dataset["ep_len"].shape[0]),
                "columns": list(dataset.keys())}


def extract(archive, destination):
    if destination.exists():
        log(f"Reusing existing HDF5: {destination}")
        return inspect_hdf5(destination)
    import zstandard
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    log(f"Extract: {archive.name} -> {destination}")
    with archive.open("rb") as compressed, zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
        if archive.name.endswith(".tar.zst"):
            found = False
            with tarfile.open(fileobj=stream, mode="r|") as container:
                for member in container:
                    if PurePosixPath(member.name).name != destination.name:
                        continue
                    if not member.isfile() or found:
                        raise RuntimeError(f"Unexpected or duplicate HDF5 member: {member.name}")
                    source = container.extractfile(member)
                    if source is None:
                        raise RuntimeError(f"Cannot read tar member: {member.name}")


                    with source, partial.open("wb") as output:
                        shutil.copyfileobj(source, output, 8 * 1024 * 1024)
                    if partial.stat().st_size != member.size:
                        raise RuntimeError("Truncated tar member")
                    found = True
            if not found:
                raise RuntimeError(f"{destination.name} was not found in {archive}")
        else:
            with partial.open("wb") as output:
                shutil.copyfileobj(stream, output, 8 * 1024 * 1024)
    metadata = inspect_hdf5(partial)
    os.replace(partial, destination)
    return metadata


def prepare_lance(data_dir, python, expected_frames):
    destination = data_dir / "pusht_expert_train.lance"
    if not destination.exists():
        staging = Path(tempfile.mkdtemp(prefix=".pusht-convert-", dir=data_dir))
        output_name = f"{staging.name}/pusht_expert_train.lance"
        environment = dict(os.environ, STABLEWM_HOME=str(data_dir.parent))
        log("Convert PushT HDF5 -> Lance (CPU/data preparation only)")
        run([python, "-m", "stable_worldmodel.cli", "convert", "pusht_expert_train", output_name,
             "--source-format", "hdf5", "--dest-format", "lance"], cwd=REPO_ROOT, env=environment)
        os.rename(staging / destination.name, destination)

        try:
            staging.rmdir()
        except OSError:
            pass
    import lance
    dataset = lance.dataset(str(destination))
    if dataset.count_rows() != expected_frames:
        raise RuntimeError(f"Lance/HDF5 frame-count mismatch: {destination}")
    for name in ("pixels", "action", "proprio", "state"):
        if name not in dataset.schema.names:
            raise RuntimeError(f"Missing Lance column {name}: {destination}")
    log(f"PushT Lance verified: {expected_frames} rows")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=["all", *TASKS], default=["all"])
    parser.add_argument("--data-root", type=Path,
                        default=Path(os.environ.get("LOCAL_DATASET_DIR", os.environ.get("STABLEWM_HOME", REPO_ROOT / "cache"))))
    parser.add_argument("--endpoint", default="https://huggingface.co",
                        help="Dataset/API host; e.g. https://hf-mirror.com")
    parser.add_argument("--no-proxy", action="store_true",
                        help="Connect directly for metadata and downloads; overrides local proxy settings")
    parser.add_argument("--downloader", choices=["auto", "curl", "aria2"], default="auto")
    parser.add_argument("--connections", type=int, default=8, help="aria2 connections per file (1-16)")
    parser.add_argument("--download-only", action="store_true", help="Download and verify archives only")
    parser.add_argument("--skip-convert", action="store_true", help="Extract HDF5 but skip PushT Lance conversion")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without network, downloads or conversion")
    parser.add_argument("--convert-only", action="store_true", help="Only convert an existing PushT HDF5; no download")
    args = parser.parse_args()
    if args.no_proxy:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            os.environ.pop(key, None)
        install_opener(build_opener(ProxyHandler({})))
    args.endpoint = args.endpoint.rstrip("/")
    if not args.endpoint.startswith("https://"):
        parser.error("--endpoint must use HTTPS")
    if not 1 <= args.connections <= 16:
        parser.error("--connections must be between 1 and 16")
    if args.convert_only:
        args.tasks = ["pusht"]
    selected = list(TASKS) if "all" in args.tasks else list(dict.fromkeys(args.tasks))
    data_dir = args.data_root.expanduser().resolve() / "datasets"
    log(f"Data directory: {data_dir}")
    for task in selected:
        repo, filename, destination = TASKS[task]
        log(f"{task}: {repo}/{filename} -> {destination}")
    if args.dry_run:
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    if args.convert_only:
        metadata = inspect_hdf5(data_dir / "pusht_expert_train.h5")
        prepare_lance(data_dir, sys.executable, metadata["frames"])
        return
    for task in selected:
        with (data_dir / f".{task}-prepare.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"Another {task} preparation is running") from None
            manifest = source_manifest(task, data_dir, args.endpoint)
            log(f"Preparing {task}; pinned dataset revision {manifest['revision']}")
            archive = download(manifest, data_dir, args)
            record = {"task": task, "revision": manifest["revision"], "archive_sha256": manifest["file"]["lfs"]["oid"],
                      "archive_verified": True, "hdf5_ready": False, "lance_ready": False}
            if not args.download_only:
                record.update(extract(archive, data_dir / TASKS[task][2]))
                record["hdf5_ready"] = True
                if task == "pusht" and not args.skip_convert:
                    prepare_lance(data_dir, sys.executable, record["frames"])
                    record["lance_ready"] = True
            write_json(data_dir / f"{task}-prepared.json", record)
            log(f"Completed {task}")
    log("Selected datasets prepared. No training or evaluation was started.")


def on_terminate(signum, frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, on_terminate)
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted; partial downloads are preserved for resume.")
        sys.exit(130)
    except Exception as error:
        log(f"ERROR: {error}")
        sys.exit(1)
