from __future__ import annotations
from typing import TYPE_CHECKING

from pathlib import Path
from collections.abc import Container

from sj_ai_utils.datasets.esic_v1.file_type import FILE_TYPE

if TYPE_CHECKING:
    pass


def select_file_from_dir(dir: Path, file_type: str) -> Path:
    """Search for a specific file type in the given directory.

    Args:
        dir (Path): The directory to search in.
        file_type (str): The type of file to search for. Valid types are:
            - "txt": Manual translation by the interpreter.
            - "vert+ts": Contains start time, end time, original word, symbol-included word,
              sentence segment number, and other tags.
            - "o": Refined sentence with special tags removed.
            - "o+ts": Refined sentence with special tags removed and timestamps.
            - "v": Lowercase, punctuation removed, numbers as words, tags removed,
              incomplete utterances included.
            - "pv": Includes uppercase and punctuation, numbers as numbers, tags removed,
              incomplete utterances included.
            - "mp4": Original video.

    Raises:
        ValueError: If the file type is invalid.
        FileNotFoundError: If the file is not found.

    Returns:
        Path: The path to the found file.
    """

    if file_type not in FILE_TYPE:
        raise ValueError(
            f"Invalid file type: {file_type}. Valid types are: {', '.join(FILE_TYPE.keys())}"
        )

    file_path = dir / FILE_TYPE[file_type]["file"]
    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}. Expected file for type '{file_type}'."
        )

    return file_path


def search_dirs(source: Path, excludes: Container[str] = []) -> list[Path]:
    if not source.exists():
        raise FileNotFoundError(f"Source path {source} does not exist.")
    if not source.is_dir():
        raise ValueError(f"Expected a directory for source, but got a file: {source}")

    data_dirs = []
    for dirpath in (p for p in source.rglob("*") if p.is_dir()):
        if str(dirpath.absolute()) in excludes:
            continue
        if (dirpath / "en.OS.man-diar.mp4").exists():
            data_dirs.append(dirpath)
    return data_dirs


__all__ = [
    "select_file_from_dir",
    "search_dirs",
]
