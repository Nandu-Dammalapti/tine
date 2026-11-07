import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest


_spec = importlib.util.spec_from_file_location(
    "fetch_benchmark", Path(__file__).resolve().parents[1] / "scripts" / "fetch_benchmark.py"
)
fetch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch)


def _archive(path, members):
    with tarfile.open(path, "w:gz") as bundle:
        for name, kind, content in members:
            member = tarfile.TarInfo(name)
            member.type = kind
            if kind == tarfile.REGTYPE:
                member.size = len(content)
                bundle.addfile(member, io.BytesIO(content))
            else:
                member.linkname = content.decode()
                bundle.addfile(member)
    return path


def test_extracts_safe_bundle_atomically_and_finds_bench(tmp_path):
    archive = _archive(tmp_path / "bundle.tar.gz", [
        ("results/bench/coding/test/model/20260101_000000.json", tarfile.REGTYPE, b"{}"),
    ])
    destination = tmp_path / "extracted"
    fetch.safe_extract_archive(archive, destination, max_bytes=1000)
    assert fetch.find_bench_dir(destination) == destination / "results" / "bench"
    assert len(fetch.sha256_file(archive)) == 64
    with pytest.raises(FileExistsError):
        fetch.safe_extract_archive(archive, destination, max_bytes=1000)


@pytest.mark.parametrize("name,kind", [
    ("../escape", tarfile.REGTYPE), ("/tmp/escape", tarfile.REGTYPE),
    ("link", tarfile.SYMTYPE), ("hardlink", tarfile.LNKTYPE),
    ("device", tarfile.CHRTYPE), ("directory\\escape", tarfile.REGTYPE),
])
def test_rejects_unsafe_members_and_removes_temporary_output(tmp_path, name, kind):
    archive = _archive(tmp_path / "bundle.tar.gz", [(name, kind, b"target")])
    destination = tmp_path / "extracted"
    with pytest.raises(ValueError):
        fetch.safe_extract_archive(archive, destination, max_bytes=1000)
    assert not destination.exists()
    assert not list(tmp_path.glob(".tine-extract-*"))


def test_size_limit_and_duplicate_members_fail_before_extraction(tmp_path):
    archive = _archive(tmp_path / "large.tar.gz", [("a", tarfile.REGTYPE, b"abc")])
    with pytest.raises(ValueError, match="limits"):
        fetch.safe_extract_archive(archive, tmp_path / "large", max_bytes=2)
    archive = _archive(tmp_path / "duplicates.tar.gz", [
        ("a", tarfile.REGTYPE, b"a"), ("a", tarfile.REGTYPE, b"b"),
    ])
    with pytest.raises(ValueError, match="Duplicate"):
        fetch.safe_extract_archive(archive, tmp_path / "duplicates", max_bytes=10)


def test_selects_only_unique_listed_official_filename():
    assert fetch.select_archive(["README.md", "bench.tar.gz", "bench-release.tar.gz"]) == "bench-release.tar.gz"
    with pytest.raises(ValueError):
        fetch.select_archive(["a/bench-release.tar.gz", "b/bench-release.tar.gz"])
    with pytest.raises(ValueError):
        fetch.select_archive(["different.tar.gz"])


def test_fetch_records_immutable_revision_and_reuses_matching_extraction(tmp_path, monkeypatch):
    archive = _archive(tmp_path / "official.tar.gz", [
        ("results/bench/livecodebench/test/model/run.json", tarfile.REGTYPE, b"{}"),
    ])
    calls = []

    class FakeApi:
        def repo_info(self, **kwargs):
            calls.append(("info", kwargs))
            return SimpleNamespace(sha="a" * 40)

        def list_repo_files(self, **kwargs):
            calls.append(("list", kwargs))
            return ["bench-release.tar.gz"]

    def download(**kwargs):
        calls.append(("download", kwargs))
        return str(archive)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        HfApi=FakeApi, hf_hub_download=download,
    ))
    output = tmp_path / "output"
    first = fetch.fetch_benchmark(output, revision="release")
    second = fetch.fetch_benchmark(output, revision="release")
    assert first == second
    assert json.loads((output / "manifest.json").read_text()) == first
    assert first["source_revision"] == "a" * 40
    assert first["archive_sha256"] == fetch.sha256_file(archive)
    assert Path(first["bench_dir"]).is_dir()
    assert all(kwargs["revision"] == "a" * 40 for kind, kwargs in calls if kind in {"list", "download"})
    (output / "manifest.json").write_text(json.dumps({"archive_sha256": "wrong"}))
    with pytest.raises(FileExistsError, match="different archive"):
        fetch.fetch_benchmark(output)
