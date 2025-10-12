from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from typing_extensions import override, Self
from datasets import Dataset as DT
from abc import ABC

from sj_ai_utils.datasets.dataset import Dataset, Task

if TYPE_CHECKING:
    pass


class HuggingFaceDataset(Dataset, ABC):
    def __init__(self, dataset: DT, sr: int, task: tuple[Task]):
        super().__init__(sr, task)
        self._dataset = dataset

    def _get_construct_args(self) -> dict:
        args = super()._get_construct_args()
        args["dataset"] = self._dataset
        return args

    @override
    def __len__(self):
        return len(self._dataset)

    @override
    def _sample(
        self,
        size: int,
        start: int = 0,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ) -> Self:
        if rng is None or size == len(self) - start:
            return self.slice(start, start + size)
        else:
            indices = range(len(self))[start:]
            index = rng.choice(indices, size=size, replace=False)
            dataset = self._dataset.select(index)

        args = self._get_construct_args()
        args["dataset"] = dataset
        return type(self)(**args)

    @override
    def slice(
        self, start: int | None = None, stop: int | None = None, step: int | None = None
    ) -> Self:
        indices = range(len(self._dataset))[start:stop:step]
        dataset = self._dataset.select(indices)

        args = self._get_construct_args()
        args["dataset"] = dataset
        return type(self)(**args)


__all__ = ["HuggingFaceDataset"]
