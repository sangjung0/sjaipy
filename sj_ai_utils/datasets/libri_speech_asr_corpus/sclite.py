from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from pathlib import Path
from typing import Callable

from sj_ai_utils.datasets.sclite import generate_ref_and_hyp as grah
from sj_ai_utils.evaluator.sclite_utils import TRNFormat
from sj_ai_utils.datasets.libri_speech_asr_corpus.service import load_data

if TYPE_CHECKING:
    pass


def generate_ref_and_hyp(
    data_paths: list[Path],
    transcriber: Callable[[np.ndarray, Path], str],
    normalizer: Callable[[str], str] = lambda x: x,
    sr: int = 16_000,
    size: int = -1,
    rng: np.random.Generator | np.random.RandomState | None = None,
) -> dict[str, dict[str, list[TRNFormat]]]:
    return grah(
        data_paths=data_paths,
        transcriber=transcriber,
        data_loader=load_data,
        normalizer=normalizer,
        sr=sr,
        size=size,
        rng=rng,
    )


__all__ = ["generate_ref_and_hyp"]
