from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from pathlib import Path
from typing import Callable, Generator
from datasets import Dataset

from sj_ai_utils.evaluator.sclite_utils import TRNFormat

if TYPE_CHECKING:
    pass


def generate_ref_and_hyp(
    data_paths: Dataset,
    transcriber: Callable[[np.ndarray, Path], str],
    data_loader: Callable[
        [Dataset, int, int, np.random.Generator | np.random.RandomState],
        Generator[tuple[np.ndarray, str, str, Path], None, None],
    ],
    normalizer: Callable[[str], str] = lambda x: x,
    sr: int = 16_000,
    size: int = -1,
    rng: np.random.Generator | np.random.RandomState | None = None,
) -> dict[str, dict[str, list[TRNFormat]]]:
    result_ref = []
    result_hyp = []
    for audio, key, y, path in data_loader(
        data_paths, sr=sr, sample_size=size, rng=rng
    ):
        txt = normalizer(y)
        ref = TRNFormat(id=key, text=txt)

        pred = transcriber(audio, path)
        pred = normalizer(pred)
        hyp = TRNFormat(id=key, text=pred)

        result_ref.append(ref)
        result_hyp.append(hyp)

    return result_ref, result_hyp


__all__ = ["generate_ref_and_hyp"]
