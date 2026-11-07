#!/usr/bin/env python3
"""Fetch the official benchmark bundle in Colab and extract it defensively.

No data is downloaded merely by importing this module. Hugging Face helpers are
optional dependencies, loaded only for a network fetch. Downloads use the SDK's
normal transport, authentication, and TLS verification. The archive is checked
against the Hub's LFS digest when available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


REPO_ID = "NPULH/LLMRouterBench"
ARCHIVE_NAMES = ("bench-release.tar.gz", "bench.tar.gz")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract_archive(
    archive: str | Path,
    destination: str | Path,
    *,
    max_bytes: int,
    max_files: int = 100_000,
) -> Path:
    """Atomically extract only regular files/directories below a new directory.

    Refuses absolute/traversing paths, symlinks, hardlinks, special files, empty
    archives, duplicate members, excessive declared sizes, and existing output.
    The temporary extraction is removed on failure. tarfile's data filter adds
    its own ownership/permission and path protections on Python 3.12+.
    """
    archive, destination = Path(archive), Path(destination)
    if not isinstance(max_bytes, int) or max_bytes < 1 or max_files < 1:
        raise ValueError("Extraction limits must be positive")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite extraction destination: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".tine-extract-", dir=destination.parent))
    try:
        with tarfile.open(archive, mode="r:gz") as bundle:
            members = []
            names = set()
            total_size = 0
            for member in bundle:
                relative = PurePosixPath(member.name)
                if relative.is_absolute() or ".." in relative.parts or "\\" in member.name:
                    raise ValueError(f"Unsafe archive path: {member.name}")
                if not (member.isfile() or member.isdir()):
                    raise ValueError(f"Archive member is not a regular file/directory: {member.name}")
                name = str(relative)
                if name in names:
                    raise ValueError(f"Duplicate archive member: {member.name}")
                names.add(name)
                if member.size < 0:
                    raise ValueError("Negative archive member size")
                total_size += member.size
                if total_size > max_bytes or len(names) > max_files:
                    raise ValueError("Archive exceeds configured extraction limits")
                members.append(member)
            if not members or not any(member.isfile() for member in members):
                raise ValueError("Archive contains no regular files")
            bundle.extractall(temporary, members=members, filter="data")
        # A concurrent creator must never have their directory replaced.
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Extraction destination appeared: {destination}")
        temporary.rename(destination)
        return destination
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def find_bench_dir(extracted: str | Path) -> Path:
    """Locate the one documented bench root; fail rather than guess among runs."""
    extracted = Path(extracted)
    candidates = [path for path in extracted.rglob("bench") if path.is_dir()]
    if extracted.name == "bench":
        candidates.append(extracted)
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise ValueError(f"Expected one extracted bench directory, found {len(candidates)}")
    return candidates[0]


def select_archive(repo_files: list[str]) -> str:
    """Select an officially listed supported filename, never a guessed URL."""
    for basename in ARCHIVE_NAMES:
        matches = [name for name in repo_files if PurePosixPath(name).name == basename]
        if len(matches) > 1:
            raise ValueError(f"Multiple upstream {basename} files; select source revision explicitly")
        if matches:
            return matches[0]
    raise ValueError(f"Official dataset does not list a supported archive: {ARCHIVE_NAMES}")


def fetch_benchmark(
    output_dir: str | Path,
    *,
    revision: str = "main",
    max_extracted_gb: float = 20,
) -> dict:
    """Resolve official immutable revision, download bundle, and record provenance."""
    if not 0 < max_extracted_gb <= 1000:
        raise ValueError("max_extracted_gb must be positive and at most 1000")
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as error:
        raise RuntimeError("Install huggingface_hub in the Colab runtime first") from error
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    info = api.repo_info(repo_id=REPO_ID, repo_type="dataset", revision=revision)
    resolved_revision = info.sha
    if not isinstance(resolved_revision, str) or not resolved_revision:
        raise ValueError("Official dataset revision could not be resolved")
    filenames = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset", revision=resolved_revision)
    filename = select_archive(filenames)
    file_info = api.get_paths_info(
        repo_id=REPO_ID, repo_type="dataset", paths=filename, revision=resolved_revision,
    )
    lfs = getattr(file_info[0], "lfs", None) if file_info else None
    expected_checksum = lfs.sha256 if lfs is not None else None
    archive = Path(hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset", filename=filename,
        revision=resolved_revision, local_dir=output_dir / "archive",
    ))
    checksum = sha256_file(archive)
    if expected_checksum is not None and checksum != expected_checksum:
        raise ValueError(f"Archive SHA-256 mismatch for {filename} at {resolved_revision}")
    extracted = output_dir / "extracted"
    manifest_path = output_dir / "manifest.json"
    if extracted.exists():
        if not manifest_path.is_file():
            raise FileExistsError("Existing extracted directory has no matching Tine manifest")
        previous = json.loads(manifest_path.read_text())
        if previous.get("archive_sha256") != checksum or previous.get("source_revision") != resolved_revision:
            raise FileExistsError("Existing extraction belongs to a different archive/revision; use a new output directory")
        bench = find_bench_dir(extracted)
    else:
        safe_extract_archive(archive, extracted, max_bytes=int(max_extracted_gb * 1024**3))
        try:
            bench = find_bench_dir(extracted)
        except ValueError:
            # This directory was created by this call, so failure can be retried
            # with a corrected source without touching pre-existing user files.
            shutil.rmtree(extracted)
            raise
    manifest = {
        "source_repo": REPO_ID, "source_revision": resolved_revision,
        "archive_filename": filename, "archive_path": str(archive),
        "archive_sha256": checksum, "expected_archive_sha256": expected_checksum,
        "archive_size_bytes": archive.stat().st_size,
        "extracted_dir": str(extracted), "bench_dir": str(bench),
        "max_extracted_gb": max_extracted_gb,
        "checksum_note": (
            "Verified against the Hugging Face LFS SHA-256 for the pinned revision"
            if expected_checksum is not None
            else "Computed local SHA-256 for provenance because LFS metadata has no digest"
        ),
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-extracted-gb", type=float, default=20)
    args = parser.parse_args()
    print(json.dumps(fetch_benchmark(
        args.output_dir, revision=args.revision, max_extracted_gb=args.max_extracted_gb,
    ), indent=2))


if __name__ == "__main__":
    main()
