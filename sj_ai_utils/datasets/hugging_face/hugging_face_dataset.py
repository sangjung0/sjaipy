from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from datasets import Dataset as DT
from abc import ABC

from sj_ai_utils.datasets.dataset import Dataset

if TYPE_CHECKING:
    pass


class HuggingFaceDataset(Dataset, ABC):
    def __init__(self, dataset: DT, sr: int):
        self._dataset = dataset
        self._sr = sr

    def _get_construct_args(self) -> dict:
        return {"sr": self._sr}

    def sample(
        self,
        sample_size: int,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ) -> "HuggingFaceDataset":
        if sample_size <= 0:
            sample_size = len(self._dataset)
        if sample_size > len(self._dataset):
            sample_size = min(sample_size, len(self._dataset))

        if rng is None or sample_size == len(self._dataset):
            dataset = self._dataset.select(range(sample_size))
        else:
            index = rng.choice(len(self._dataset), size=sample_size, replace=False)
            dataset = self._dataset.select(index)

        args = self._get_construct_args()
        return type(self)._construct(dataset, **args)


__all__ = ["HuggingFaceDataset"]
