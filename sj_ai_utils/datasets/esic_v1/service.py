from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from pathlib import Path
from typing import Generator
from functools import lru_cache
from collections.abc import Container

from sj_utils.audio import load_audio_from_mp4
from sj_utils.typing import deprecated
from sj_utils.string import normalize_text_only_en
from sj_ai_utils.datasets.esic_v1.file_type import FILE_TYPE, MP4, VERBATIM, ORTO

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


@lru_cache(maxsize=4196)
def load_mp4_to_audio(mp4, sr):
    return load_audio_from_mp4(mp4, sr=sr)


@deprecated()
def load_data(
    data_paths: list[Path],
    sr: int,
    sample_size: int = -1,
    rng: np.random.Generator | np.random.RandomState | None = None,
) -> Generator[tuple[np.ndarray, str, str, Path], None, None]:
    if sample_size < 0:
        sample_size = len(data_paths)
    if rng is None:
        data_paths = data_paths[:sample_size]
    else:
        data_paths = rng.choice(data_paths, size=sample_size, replace=False)

    for path in data_paths:
        mp4 = select_file_from_dir(path, MP4)
        audio = load_mp4_to_audio(mp4, sr)[0]
        txt = select_file_from_dir(path, ORTO)
        txt = txt.read_text(encoding="utf-8")
        key = Path(*path.parts[-2:])
        _id = normalize_text_only_en(str(key))[-255:]

        yield audio, _id, txt, key


def load_data_v2(
    data: list[dict[str, Path]],
    sr: int,
    sample_size: int = -1,
    rng: np.random.Generator | np.random.RandomState | None = None,
) -> Generator[tuple[np.ndarray, str, str, Path], None, None]:
    if sample_size < 0:
        sample_size = len(data)
    if rng is None:
        data = data[:sample_size]
    else:
        data = rng.choice(data, size=sample_size, replace=False)

    for d in data:
        x, y = d["X"], d["Y"]
        audio = load_mp4_to_audio(x, sr)[0]
        txt = y.read_text(encoding="utf-8")
        key = Path(*x.parts[-3:-1])
        _id = normalize_text_only_en(str(key))[-255:]

        yield audio, _id, txt, key


__all__ = [
    "select_file_from_dir",
    "search_dirs",
    "load_mp4_to_audio",
    "load_data",
    "load_data_v2",
]
