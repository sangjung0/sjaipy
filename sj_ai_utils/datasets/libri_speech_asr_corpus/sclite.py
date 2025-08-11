from __future__ import annotations
from typing import TYPE_CHECKING

import librosa
import numpy as np

from pathlib import Path
from typing import Callable, Generator
from functools import lru_cache

from sj_ai_utils.evaluator.sclite_utils import TRNFormat
from sj_ai_utils.datasets.libri_speech_asr_corpus.file_type import X, Y
from sj_ai_utils.datasets.libri_speech_asr_corpus.service import select_file_from_dir

if TYPE_CHECKING:
    pass


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


def load_data(
    data_paths: list[Path], sr: int
) -> Generator[tuple[np.ndarray, str, str, Path], None, None]:
    for path in data_paths:
        if not path.exists():
            raise FileNotFoundError(f"Data path {path} does not exist.")
        if not path.is_dir():
            raise ValueError(f"Data path {path} should be a directory.")

        key_data = trans_txt_to_key_data(path)
        for key, data in key_data:
            flac = path / f"{key}.flac"
            if not flac.exists():
                raise FileNotFoundError(f"FLAC file {flac} does not exist.")

            audio, _ = load_flac(flac, sr)
            yield audio, key, data, flac


def generate_ref_and_hyp(
    data_paths: list[Path],
    transcriber: Callable[[np.ndarray, Path], str],
    normalizer: Callable[[str], str] = lambda x: x,
    max_count: int = -1,
    sr: int = 16_000,
) -> dict[str, dict[str, list[TRNFormat]]]:
    if any(not p.exists() for p in data_paths):
        raise ValueError("One or more data paths do not exist.")

    result_ref = []
    result_hyp = []
    for i, (audio, key, y, path) in enumerate(load_data(data_paths, sr=sr)):
        if max_count != -1 and i >= max_count:
            break

        txt = normalizer(y)
        ref = TRNFormat(id=key, text=txt)

        pred = transcriber(audio, path)
        pred = normalizer(pred)
        hyp = TRNFormat(id=key, text=pred)

        result_ref.append(ref)
        result_hyp.append(hyp)

    return result_ref, result_hyp


__all__ = ["generate_ref_and_hyp"]
