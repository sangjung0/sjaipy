from __future__ import annotations
from typing import TYPE_CHECKING

import librosa
import numpy as np

from typing import Sequence
from typing_extensions import override, Self
from datasets import Dataset as DT
from abc import ABC

from sjaipy.datasets.dataset import Dataset, Task

if TYPE_CHECKING:
    pass


class HuggingFaceDataset(Dataset, ABC):
    def __init__(self, dataset: DT, sr: int, task: tuple[Task, ...]):
        super().__init__(sr, task)
        self._dataset = dataset
        self._original_sr = sr

    @Dataset.args.getter
    @override
    def args(self) -> dict:
        return {
            **super().args,
            "dataset": self._dataset,
        }

    @Dataset.length.getter
    @override
    def length(self) -> int:
        return len(self._dataset)

    @override
    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "dataset": self._dataset.to_dict(),
        }

    @override
    def select(self, indices: Sequence[int]) -> Self:
        dataset = self._dataset.select(indices)

        args = self.args
        args["dataset"] = dataset
        return type(self)(**args)

    @override
    def slice(
        self, start: int | None = None, stop: int | None = None, step: int | None = None
    ) -> Self:
        indices = range(len(self._dataset))[start:stop:step]
        dataset = self._dataset.select(indices)

        args = self.args
        args["dataset"] = dataset
        return type(self)(**args)

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

        args = self.args
        args["dataset"] = dataset
        return type(self)(**args)

    @staticmethod
    @override
    def from_dict(data: dict) -> Self:
        data["dataset"] = DT.from_dict(data["dataset"])
        return HuggingFaceDataset(**data)

    def _resample_audio(self, audio: np.ndarray) -> np.ndarray:
        if self._sr != self._original_sr:
            audio = librosa.resample(
                audio, orig_sr=self._original_sr, target_sr=self._sr
            )
        return audio


__all__ = ["HuggingFaceDataset"]
