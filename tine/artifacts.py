"""Publish complete offline experiment directories with rollback on failure."""

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import shutil
import tempfile


RUN_FILES = frozenset({
    "summary.json", "calibration.json", "excluded.json", "splits.csv",
    "decisions.csv", "curves.csv", "metrics.csv", "group-metrics.csv",
    "cost-quality.png", "report.md", "synthetic-outcomes.jsonl",
    "routing-latency.json",
})


def atomic_write_text(path, contents):
    """Replace one UTF-8 output file after encoding and writing succeed."""
    encoded = contents.encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"Output file must exclude symbolic links: {path}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            shutil.copymode(path, temporary)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def experiment_output(destination):
    """Stage a run and replace its directory only after generation succeeds.

    A sibling advisory lock serializes cooperating writers. It stays on disk so
    all processes lock the same inode, including after a writer exits. Unrelated
    files are copied into the replacement directory, preserving user additions.
    """
    destination = Path(destination)
    if destination.is_symlink():
        raise ValueError("Experiment output must be a directory, excluding symbolic links")
    destination = destination.resolve()
    if not destination.name:
        raise ValueError("Experiment output must be below a filesystem root")
    if Path.cwd().resolve().is_relative_to(destination):
        raise ValueError("Experiment output cannot contain the current working directory; choose a dedicated output folder")
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_name(f".{destination.name}.tine.lock")
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another experiment is writing to {destination}") from None
        if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
            raise ValueError("Experiment output must be a directory, excluding symbolic links")
        for name in RUN_FILES:
            target = destination / name
            if target.is_symlink() or (target.exists() and not target.is_file()):
                raise ValueError(f"Experiment artifact must be a regular file: {target}")
        with tempfile.TemporaryDirectory(prefix=f".{destination.name}.tine-", dir=destination.parent) as temporary:
            workspace = Path(temporary)
            staged = workspace / "staged"
            staged.mkdir(mode=0o700)
            if destination.exists():
                for source in destination.iterdir():
                    if source.name in RUN_FILES:
                        continue
                    target = staged / source.name
                    if source.is_dir() and not source.is_symlink():
                        shutil.copytree(source, target, symlinks=True)
                    else:
                        shutil.copy2(source, target, follow_symlinks=False)
            yield staged
            backup = None
            if destination.exists():
                shutil.copymode(destination, staged)
                # Keep the previous run outside the temporary workspace so a
                # failed rollback cannot cause cleanup to delete the only copy.
                backup = Path(tempfile.mkdtemp(prefix=f".{destination.name}.previous-", dir=destination.parent))
                try:
                    os.replace(destination, backup)
                except BaseException:
                    backup.rmdir()
                    raise
            try:
                os.replace(staged, destination)
            except BaseException:
                if backup is not None:
                    try:
                        os.replace(backup, destination)
                    except OSError as error:
                        raise OSError(f"Could not restore output; previous run is preserved at {backup}") from error
                raise
            if backup is not None:
                shutil.rmtree(backup)
