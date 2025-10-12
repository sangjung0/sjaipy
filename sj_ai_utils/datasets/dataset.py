import numpy as np

from abc import ABC, abstractmethod
from typing import Generator, Any, Literal
from typing_extensions import override, Self
from dataclasses import dataclass

Task = Literal["asr", "diarization"]


@dataclass(frozen=True, slots=True)
class Sample:
    id: str
    audio: np.ndarray
    Y: dict[str, Any]


class Dataset(ABC):
    def __init__(self, sr: int, task: tuple[Task] = ("asr",)):
        self.sr = sr
        self.task = task

    def _get_construct_args(self):
        return {"sr": self.sr, "task": self.task}

    def __iter__(self) -> Generator[Sample, Any, None]:
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

    def __add__(self, other: Self) -> "ConcatDataset":
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
    ) -> Self: ...

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

    @abstractmethod
    def slice(
        self, start: int | None = None, stop: int | None = None, step: int | None = None
    ) -> Self: ...

    @abstractmethod
    def get_item(self, idx: int) -> Sample: ...


class ConcatDataset(Dataset):
    def __init__(self, datasets: list[Dataset]):
        if len(datasets) == 0:
            raise ValueError("At least one dataset is required")
        sr = datasets[0].sr
        if any(ds.sr != sr for ds in datasets):
            raise ValueError("All datasets must have the same sample rate")
        task = tuple(sorted(datasets[0].task))
        if any(tuple(sorted(ds.task)) != task for ds in datasets):
            raise ValueError("All datasets must have the same task")

        super().__init__(sr=sr, task=task)
        self._datasets = datasets

    def _get_construct_args(self):
        args = super()._get_construct_args()
        args["datasets"] = self._datasets
        return args

    def __len__(self) -> int:
        return sum(len(ds) for ds in self._datasets)

    @override
    def __add__(self, other: Dataset) -> Self:
        if isinstance(other, ConcatDataset):
            return ConcatDataset(self._datasets + other._datasets)
        elif isinstance(other, Dataset):
            return ConcatDataset(self._datasets + [other])
        else:
            raise TypeError("Invalid type for concatenation")

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
    ) -> Self:
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
    def get_item(self, idx: int) -> Sample:
        start = 0
        for ds in self._datasets:
            d_idx = idx - start
            if d_idx < len(ds):
                return ds.get_item(d_idx)
            start += len(ds)
        raise IndexError("Index out of range")


__all__ = ["Dataset", "ConcatDataset", "Sample", "Task"]
