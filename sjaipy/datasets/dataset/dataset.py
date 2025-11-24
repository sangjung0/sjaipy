from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from abc import ABC, abstractmethod
from typing import Generator, Any, overload, Sequence
from typing_extensions import Self

if TYPE_CHECKING:
    from sjaipy.datasets.dataset.aliases import Task
    from sjaipy.datasets.dataset.sample import Sample
    from sjaipy.datasets.dataset.concat_dataset import ConcatDataset


class Dataset(ABC):
    def __init__(self, sr: int, task: tuple[Task, ...] = ("asr",)):
        self._sr = sr
        self.task = task

    @property
    def sr(self) -> int:
        return self._sr

    @sr.setter
    def sr(self, value: int):
        self._sr = value

    @property
    def args(self) -> dict:
        return {"sr": self._sr, "task": self.task}

    @property
    @abstractmethod
    def length(self) -> int: ...

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __iter__(self) -> Generator[Sample, Any, None]:
        yield from self.iter()

    @overload
    def __getitem__(self, key: int) -> Sample: ...
    @overload
    def __getitem__(self, key: slice) -> Self: ...
    @overload
    def __getitem__(self, key: Sequence[int]) -> Self: ...
    def __getitem__(self, key: int | slice | Sequence[int]) -> Sample | Self:
        return self.getitem(key)

    @overload
    def __add__(self, other: Self) -> "ConcatDataset": ...
    @overload
    def __add__(self, other: "ConcatDataset") -> "ConcatDataset": ...
    def __add__(self, other: Self | "ConcatDataset") -> "ConcatDataset":
        return self.concat(other)

    def __len__(self) -> int:
        return self.length

    def iter(self) -> Generator[Sample, Any, None]:
        for idx in range(len(self)):
            yield self.get(idx)

    @overload
    def getitem(self, key: int) -> Sample: ...
    @overload
    def getitem(self, key: slice) -> Self: ...
    @overload
    def getitem(self, key: Sequence[int]) -> Self: ...
    def getitem(self, key: int | slice | Sequence[int]) -> Sample | Self:
        if isinstance(key, slice):
            return self.slice(start=key.start, stop=key.stop, step=key.step)
        elif isinstance(key, Sequence):
            return self.select(key)
        elif isinstance(key, int):
            n = len(self)
            if key < 0:
                key += n
            if not (0 <= key < n):
                raise IndexError("Index out of range")
            return self.get(key)
        else:
            raise TypeError("Invalid key type")

    @overload
    def concat(self, other: Self) -> "ConcatDataset": ...
    @overload
    def concat(self, other: "ConcatDataset") -> "ConcatDataset": ...
    def concat(self, other: Self | "ConcatDataset") -> "ConcatDataset":
        from sjaipy.datasets.dataset.concat_dataset import ConcatDataset

        if isinstance(other, ConcatDataset):
            return ConcatDataset([self] + other._datasets)
        elif isinstance(other, Dataset):
            return ConcatDataset([self, other])
        else:
            raise TypeError("Invalid type for concatenation")

    def to_dict(self) -> dict:
        return {
            "sr": self._sr,
            "task": self.task,
        }

    @overload
    def sample(self, size: int) -> Self: ...
    @overload
    def sample(self, size: int, start: int) -> Self: ...
    @overload
    def sample(self, size: int, start: int, rng: np.random.Generator) -> Self: ...
    def sample(
        self,
        size: int,
        start: int = 0,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ) -> Self:
        if start < 0 or start >= len(self):
            raise IndexError("Invalid start index")
        elif size <= 0:
            size = len(self) - start
        else:
            size = min(size, len(self) - start)
        return self._sample(size=size, start=start, rng=rng)

    def samples_to_list(self) -> list[Sample]:
        return list(self.iter())

    @abstractmethod
    def select(self, indices: Sequence[int]) -> Self: ...

    @abstractmethod
    def slice(
        self, start: int | None = None, stop: int | None = None, step: int | None = None
    ) -> Self: ...

    @abstractmethod
    def get(self, idx: int) -> Sample: ...

    @abstractmethod
    def _sample(
        self,
        size: int,
        start: int = 0,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ) -> Self: ...

    @staticmethod
    @abstractmethod
    def from_dict(data: dict) -> Self:
        raise NotImplementedError


__all__ = ["Dataset"]
