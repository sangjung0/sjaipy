from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from pathlib import Path
from typing import Generator
from datasets import Dataset, Audio

if TYPE_CHECKING:
    pass


def load_data(
    dataset: Dataset,
    sr: int,
    sample_size: int = -1,
    rng: np.random.Generator | np.random.RandomState | None = None,
) -> Generator[tuple[np.ndarray, str, str, Path], None, None]:
    dataset.cast_column("audio", Audio(sampling_rate=sr))

    if sample_size < 0:
        sample_size = len(dataset)
    if rng is None:
        dataset = dataset.select(range(sample_size))
    else:
        index = rng.choice(len(dataset), size=sample_size, replace=False)
        dataset = dataset.select(index)

    for data in dataset:
        _id = data["audio_id"][-255:]
        audio = data["audio"]["array"]
        y = data["text"]
        path = Path(data["audio"]["path"])
        key = Path(*path.parts[-2:-1]) / path.stem

        yield audio, _id, y, key


__all__ = ["load_data"]
