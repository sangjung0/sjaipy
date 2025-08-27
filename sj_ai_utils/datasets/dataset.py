import numpy as np

from abc import ABC, abstractmethod
from typing import Generator, Any

from sj_utils.typing import override


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

    def __add__(self, other: "Dataset") -> "ConcatDataset":
        if isinstance(other, ConcatDataset):
            return ConcatDataset([self] + other._datasets)
        elif isinstance(other, Dataset):
            return ConcatDataset([self, other])
        else:
            raise TypeError("Invalid type for concatenation")

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


class ConcatDataset(Dataset):
    def __init__(self, datasets: list[Dataset]):
        if len(datasets) == 0:
            raise ValueError("At least one dataset is required")
        sr = datasets[0].sample_rate
        if any(ds.sample_rate != sr for ds in datasets):
            raise ValueError("All datasets must have the same sample rate")

        super().__init__(sr=sr)
        self._datasets = datasets
        self._cum_lens: list[int] = []
        curr_len = 0
        for ds in datasets:
            self._cum_lens.append(curr_len)
            curr_len += len(ds)

    def _get_construct_args(self):
        args = super()._get_construct_args()
        args["datasets"] = self._datasets
        return args

    def __len__(self) -> int:
        return self._cum_lens[-1]

    @override
    def _sample(
        self,
        size: int,
        start: int = 0,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ):
        if rng is not None:
            raise NotImplementedError("Random sampling is not implemented")
        return self.slice(start=start, stop=start + size)

    @override
    def slice(
        self, start: int | None = None, stop: int | None = None, step: int | None = None
    ) -> "ConcatDataset":
        start = start if start else 0
        stop = stop if stop else len(self)
        step = step if step else 1

        if start < 0:
            raise IndexError("Invalid start index")
        if stop > len(self):
            raise IndexError("Invalid stop index")

        new_datasets = []
        for ds in self._datasets:
            if start >= stop:
                break
            d_stop = min(len(ds), stop - start)
            new_datasets.append(ds.slice(start=0, stop=d_stop, step=step))
        return ConcatDataset(new_datasets)

    @override
    def get_item(self, idx: int) -> tuple[str, np.ndarray, str]:
        start = 0
        for ds in self._datasets:
            d_idx = idx - start
            if d_idx < len(ds):
                return ds.get_item(d_idx)
            start += len(ds)
        raise IndexError("Index out of range")


__all__ = ["Dataset", "ConcatDataset"]
