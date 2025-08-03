from __future__ import annotations
from typing import TYPE_CHECKING

from pathlib import Path
from collections.abc import Container

from sj_ai_utils.datasets.libri_speech_asr_corpus.file_type import FILE_TYPE

if TYPE_CHECKING:
    pass


def select_file_from_dir(dir: Path, file_type: str) -> Path | list[Path]:
    if file_type not in FILE_TYPE:
        raise ValueError(
            f"Invalid file type: {file_type}. Valid types are: {', '.join(FILE_TYPE.keys())}"
        )

    files = list(dir.glob(FILE_TYPE[file_type]["file"]))
    if FILE_TYPE[file_type]["multiple"]:
        return files

    if len(files) == 0:
        raise FileNotFoundError(
            f"File not found: {files}. Expected file for type '{file_type}'."
        )
    elif len(files) > 1:
        raise ValueError(
            f"Expected one file for type '{file_type}', found {len(files)}: {files}"
        )
    return files[0]


def search_dirs(source: Path, excludes: Container[str] = []) -> list[Path]:
    if not source.exists():
        raise FileNotFoundError(f"Source path {source} does not exist.")
    if not source.is_dir():
        raise ValueError(f"Expected a directory for source, but got a file: {source}")

    data_dirs = []
    for dirpath in (p for p in source.rglob("*") if p.is_dir()):
        if str(dirpath.absolute()) in excludes:
            continue
        trans_files = list(dirpath.glob("*.trans.txt"))
        if len(trans_files) == 0:
            continue
        elif len(trans_files) > 1:
            raise ValueError(
                f"Expected one trans file in {dirpath}, found {len(trans_files)}"
            )
        data_dirs.append(dirpath)

    return data_dirs


__all__ = [
    "select_file_from_dir",
    "search_dirs",
]
