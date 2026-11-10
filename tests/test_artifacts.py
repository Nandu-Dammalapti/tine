import os
from pathlib import Path

import pytest

from tine.artifacts import experiment_output


def test_publication_keeps_user_files_and_replaces_the_complete_run(tmp_path):
    output = tmp_path / "run"
    output.mkdir(mode=0o700)
    (output / "summary.json").write_text("old summary")
    (output / "report.md").write_text("old report")
    (output / "notes.txt").write_text("user notes")
    with experiment_output(output) as staged:
        assert (output / "summary.json").read_text() == "old summary"
        (staged / "summary.json").write_text("new summary")
        (staged / "report.md").write_text("new report")
    assert (output / "summary.json").read_text() == "new summary"
    assert (output / "report.md").read_text() == "new report"
    assert (output / "notes.txt").read_text() == "user notes"
    assert output.stat().st_mode & 0o777 == 0o700
    assert not list(tmp_path.glob(".run.previous-*"))


def test_failed_generation_preserves_previous_run_and_allows_retry(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "report.md").write_text("original")
    with pytest.raises(OSError, match="generation"):
        with experiment_output(output) as staged:
            (staged / "report.md").write_text("incomplete")
            raise OSError("generation failed")
    assert (output / "report.md").read_text() == "original"
    with experiment_output(output) as staged:
        (staged / "report.md").write_text("retry succeeded")
    assert (output / "report.md").read_text() == "retry succeeded"


@pytest.mark.parametrize("rollback_fails", [False, True])
def test_failed_publication_keeps_previous_results(tmp_path, monkeypatch, rollback_fails):
    output = tmp_path / "run"
    output.mkdir()
    (output / "report.md").write_text("original")
    replace = os.replace

    def fail_publication(source, target):
        if Path(source).name == "staged":
            raise OSError("publish failed")
        if rollback_fails and Path(source).name.startswith(".run.previous-"):
            raise OSError("restore failed")
        return replace(source, target)

    monkeypatch.setattr("tine.artifacts.os.replace", fail_publication)
    with pytest.raises(OSError):
        with experiment_output(output) as staged:
            (staged / "report.md").write_text("replacement")
    if rollback_fails:
        backup, = tmp_path.glob(".run.previous-*")
        assert (backup / "report.md").read_text() == "original"
    else:
        assert (output / "report.md").read_text() == "original"


def test_concurrent_writer_is_rejected_without_touching_active_run(tmp_path):
    output = tmp_path / "run"
    with experiment_output(output) as staged:
        (staged / "report.md").write_text("first writer")
        with pytest.raises(RuntimeError, match="Another experiment"):
            with experiment_output(output):
                pytest.fail("Concurrent writer entered")
    assert (output / "report.md").read_text() == "first writer"


def test_artifact_symlinks_are_rejected(tmp_path):
    outside = tmp_path / "outside"
    outside.write_text("preserve me")
    output = tmp_path / "run"
    output.mkdir()
    (output / "report.md").symlink_to(outside)
    with pytest.raises(ValueError, match="regular file"):
        with experiment_output(output):
            pytest.fail("Symbolic artifact accepted")
    assert outside.read_text() == "preserve me"


def test_output_dot_segments_resolve_before_staging(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "child").mkdir()
    with experiment_output(output / "child" / "..") as staged:
        assert output not in staged.parents
        (staged / "report.md").write_text("complete")
    assert (output / "report.md").read_text() == "complete"
    assert (output / "child").is_dir()


def test_output_directory_symlink_is_rejected(tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(destination, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic links"):
        with experiment_output(alias):
            pytest.fail("Output symlink accepted")
    assert alias.is_symlink()


def test_output_cannot_replace_the_current_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="current working directory"):
        with experiment_output("."):
            pytest.fail("Current directory replacement accepted")
    assert Path.cwd() == tmp_path
