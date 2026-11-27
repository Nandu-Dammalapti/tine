# Strips carriage-return progress-bar overwrites (tqdm noise) from Colab logs, keeping the final text per line.
# Source: Gemini scheduled action, 2026-11-26 08:35
# Experimental; not imported by the tine package.
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def collapse_carriage_returns(text: str) -> List[str]:
    """Resolve \\r terminal overwrites so only the final visible text per line remains."""
    cleaned_lines: List[str] = []
    # Split on "\n" only: str.splitlines() also breaks on "\r", which would keep every progress frame.
    for raw_line in text.replace("\r\n", "\n").split("\n"):
        if "\r" in raw_line:
            segments = [seg.strip() for seg in raw_line.split("\r") if seg.strip()]
            if segments:
                cleaned_lines.append(segments[-1])
        else:
            stripped = raw_line.strip()
            if stripped:
                cleaned_lines.append(stripped)
    return cleaned_lines


def clean_colab_progress_log(
    input_log: Union[str, Path],
    output_log: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Clean progress-bar overwrite noise from a Colab log file and write a readable log."""
    in_path = Path(input_log)
    out_path = Path(output_log) if output_log else in_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Read bytes: read_text() uses universal newlines and would turn every "\r" into "\n".
    raw_text = in_path.read_bytes().decode("utf-8", errors="ignore")
    raw_line_count = len(raw_text.replace("\r\n", "\n").split("\n"))
    cleaned_lines = collapse_carriage_returns(raw_text)

    out_path.write_text(
        "\n".join(cleaned_lines) + ("\n" if cleaned_lines else ""),
        encoding="utf-8",
    )

    return {
        "source_file": in_path.name,
        "output_file": out_path.name,
        "raw_lines": raw_line_count,
        "cleaned_lines": len(cleaned_lines),
        "lines_removed": max(0, raw_line_count - len(cleaned_lines)),
    }
