from __future__ import annotations
from typing import TYPE_CHECKING

import librosa
import numpy as np

from pathlib import Path
from collections.abc import Container
from typing import Generator
from functools import lru_cache

from sj_ai_utils.datasets.libri_speech_asr_corpus.file_type import FILE_TYPE, Y
from sj_utils.typing import deprecated

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


@lru_cache(maxsize=4196)
def load_flac(flac: Path, sr: int) -> tuple[np.ndarray, int]:
    return librosa.load(flac, sr=sr)


def trans_txt_to_key_data(trans: Path) -> list[tuple[str, Path]]:
    result = []
    with trans.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            uid, *words = line.rstrip().split()
            sent = " ".join(words)
            result.append((uid, sent))
    return result


@deprecated()
def load_data(
    data_paths: list[Path],
    sr: int,
    sample_size: int = -1,
    rng: np.random.Generator | np.random.RandomState | None = None,
) -> Generator[tuple[np.ndarray, str, str, Path], None, None]:
    if sample_size < 0:
        sample_size = len(data_paths)
    if rng is None or sample_size == len(data_paths):
        data_paths = data_paths[:sample_size]
    else:
        data_paths = rng.choice(data_paths, size=sample_size, replace=False)

    for path in data_paths:
        trans_txt = select_file_from_dir(path, Y)
        key_data = trans_txt_to_key_data(trans_txt)
        for _id, data in key_data:
            flac = path / f"{_id}.flac"
            if not flac.exists():
                raise FileNotFoundError(f"FLAC file {flac} does not exist.")

            audio, _ = load_flac(flac, sr)
            key = Path(*flac.parts[-3:-1]) / flac.stem
            yield audio, _id, data, key


def load_data_v2(
    data: list[dict[str, Path]],
    sr: int,
    sample_size: int = -1,
    rng: np.random.Generator | np.random.RandomState | None = None,
) -> Generator[tuple[np.ndarray, str, str, Path], None, None]:
    if sample_size < 0:
        sample_size = len(data)
    if rng is None or sample_size == len(data):
        data = data[:sample_size]
    else:
        data = rng.choice(data, size=sample_size, replace=False)

    for d in data:
        _, y = d["X"], d["Y"]
        key_data = trans_txt_to_key_data(y)
        path = y.parent
        for _id, d in key_data:
            flac = path / f"{_id}.flac"
            if not flac.exists():
                raise FileNotFoundError(f"FLAC file {flac} does not exist.")

            audio, _ = load_flac(flac, sr)
            key = Path(*flac.parts[-3:-1]) / flac.stem
            yield audio, _id, data, key


__all__ = [
    "select_file_from_dir",
    "search_dirs",
    "load_flac",
    "trans_txt_to_key_data",
    "load_data",
    "load_data_v2",
]
