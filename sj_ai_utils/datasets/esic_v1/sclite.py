from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from pathlib import Path
from typing import Callable, Generator
from functools import lru_cache

from sj_utils.audio import load_audio_from_mp4
from sj_ai_utils.evaluator.sclite_utils import TRNFormat
from sj_ai_utils.datasets.esic_v1.service import select_file_from_dir
from sj_ai_utils.datasets.esic_v1.file_type import MP4, VERBATIM, ORTO

if TYPE_CHECKING:
    pass


@lru_cache(maxsize=4196)
def load_mp4_to_audio(mp4, sr):
    return load_audio_from_mp4(mp4, sr=sr)


def load_data(
    data_paths: list[Path], sr: int
) -> Generator[tuple[np.ndarray, str, str, Path], None, None]:
    for path in data_paths:
        key = str(path.absolute())[:255]
        mp4 = select_file_from_dir(path, MP4)
        audio = load_mp4_to_audio(mp4, sr)[0]
        txt = select_file_from_dir(path, ORTO)
        txt = txt.read_text(encoding="utf-8")

        yield audio, key, txt, path


def generate_ref_and_hyp(
    data_paths: list[Path],
    transcriber: Callable[[np.ndarray, Path], str],
    normalizer: Callable[[str], str] = lambda x: x,
    max_count: int = -1,
    sr: int = 16_000,
) -> tuple[list[TRNFormat], list[TRNFormat]]:
    if any(not p.exists() or not p.is_dir() for p in data_paths):
        raise FileNotFoundError("One or more data paths do not exist.")

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


__all__ = ["generate_ref_and_hyp", "load_data"]
