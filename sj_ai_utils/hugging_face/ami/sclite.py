from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from pathlib import Path
from typing import Callable
from datasets import Dataset

from sj_ai_utils.evaluator.sclite_utils import TRNFormat, generate_ref_and_hyp as grah
from sj_ai_utils.hugging_face.ami.service import load_data

if TYPE_CHECKING:
    pass


def generate_ref_and_hyp(
    datasets: Dataset,
    transcriber: Callable[[np.ndarray, Path], str],
    normalizer: Callable[[str], str] = lambda x: x,
    sr: int = 16_000,
    size: int = -1,
    rng: np.random.Generator | np.random.RandomState | None = None,
) -> dict[str, dict[str, list[TRNFormat]]]:
    return grah(
        data_paths=datasets,
        transcriber=transcriber,
        data_loader=load_data,
        normalizer=normalizer,
        sr=sr,
        size=size,
        rng=rng,
    )


__all__ = ["generate_ref_and_hyp"]
