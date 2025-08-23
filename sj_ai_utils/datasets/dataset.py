import numpy as np

from pathlib import Path
from abc import ABC, abstractmethod
from typing import Generator, Any


class Dataset(ABC):
    def __init__(self, sr: int):
        self._sr = sr

    def _get_construct_args(self):
        return {"sr": self._sr}

    @property
    def sample_rate(self):
        return self._sr

    @sample_rate.setter
    def sample_rate(self, sr: int):
        self._sr = sr

    def __iter__(self) -> Generator[tuple[str, np.ndarray, str], Any, None]:
        for idx in range(len(self)):
            yield self.get_item(idx)

    def __getitem__(self, key: int | slice):
        if isinstance(key, slice):
            return self.slice(start=key.start, stop=key.stop, step=key.step)
        elif isinstance(key, int):
            n = len(self)
            if key < 0:
                key += n
            if not (0 <= key < n):
                raise IndexError("Index out of range")
            return self.get_item(key)
        else:
            raise TypeError("Invalid key type")

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def _sample(
        self,
        size: int,
        start: int = 0,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ) -> "Dataset": ...

    def sample(
        self,
        size: int,
        start: int = 0,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ) -> "Dataset":
        if start < 0 or start >= len(self):
            raise IndexError("Invalid start index")
        elif size <= 0:
            size = len(self) - start
        else:
            size = min(size, len(self) - start)
        return self._sample(size=size, start=start, rng=rng)

    @abstractmethod
    def slice(
        self, start: int | None = None, stop: int | None = None, step: int | None = None
    ) -> "Dataset": ...

    @abstractmethod
    def get_item(self, idx: int) -> tuple[str, np.ndarray, str]: ...


__all__ = ["Dataset"]
