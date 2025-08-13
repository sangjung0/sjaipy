from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from pathlib import Path
from typing import Generator
from datasets import Dataset, Audio

if TYPE_CHECKING:
    pass


def load_data(
    dataset: Dataset, sr: int
) -> Generator[tuple[np.ndarray, str, str, Path], None, None]:

    dataset.cast_column("audio", Audio(sampling_rate=sr))

    for data in dataset:
        key = data["audio_id"]
        audio = data["audio"]["array"]
        y = data["text"]
        path = Path(data["audio"]["path"])

        yield audio, key, y, path


__all__ = ["load_data"]
