import numpy as np

from abc import ABC, abstractmethod
from typing import Generator, Any, Literal, overload, Sequence
from typing_extensions import override, Self
from dataclasses import dataclass, field

Task = Literal["asr", "diarization"]


@dataclass(frozen=True, slots=True)
class Sample:
    id: str = field(compare=True, hash=True, repr=True)
    audio: np.ndarray = field(compare=False, hash=False, repr=False)
    _Y: dict[str, Any] = field(compare=False, hash=False, repr=False)

    @property
    def ASR(self) -> str:
        if "asr" not in self._Y:
            raise AttributeError("ASR label is not available in this sample")
        return self._Y["asr"]

    @property
    def diarization(self) -> list[dict[str, Any]]:
        if "diarization" not in self._Y:
            raise AttributeError("Diarization label is not available in this sample")
        return self._Y["diarization"]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "audio": self.audio.tolist(),
            "_Y": self._Y,
        }

    @staticmethod
    def from_dict(data: dict) -> "Sample":
        return Sample(
            id=data["id"],
            audio=np.array(data["audio"]),
            _Y=data["_Y"],
        )


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


class ConcatDataset(Dataset):
    def __init__(
        self,
        datasets: list[Dataset],
        sr: int | None = None,
        task: tuple[Task, ...] | None = None,
    ):
        if len(datasets) == 0:
            raise ValueError("At least one dataset is required")

        sr = sr or datasets[0]._sr
        if any(ds._sr != sr for ds in datasets):
            raise ValueError("All datasets must have the same sample rate")

        task = tuple(sorted(task)) if task else tuple(sorted(datasets[0].task))
        if any(tuple(sorted(ds.task)) != task for ds in datasets):
            raise ValueError("All datasets must have the same task")

        super().__init__(sr=sr, task=task)
        self._datasets = datasets

    @Dataset.args.getter
    @override
    def args(self):
        return {**super().args, "datasets": self._datasets}

    @Dataset.length.getter
    @override
    def length(self) -> int:
        return sum(len(ds) for ds in self._datasets)

    @override
    def concat(self, other: Dataset | Self) -> Self:
        if isinstance(other, ConcatDataset):
            return ConcatDataset(self._datasets + other._datasets)
        elif isinstance(other, Dataset):
            return ConcatDataset(self._datasets + [other])
        else:
            raise TypeError("Invalid type for concatenation")

    @override
    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "datasets": [ds.to_dict() for ds in self._datasets],
            "module": [ds.__class__.__module__ for ds in self._datasets],
            "qualname": [ds.__class__.__qualname__ for ds in self._datasets],
        }

    @override
    def select(self, indices: Sequence[int]) -> Self:
        selected_datasets = []
        start = 0
        for ds in self._datasets:
            end = start + len(ds)
            if ds_indices := [i - start for i in indices if start <= i < end]:
                selected_datasets.append(ds.select(ds_indices))
            start = end
        return ConcatDataset(selected_datasets)

    @override
    def slice(
        self, start: int | None = None, stop: int | None = None, step: int | None = None
    ) -> Self:
        start = start if start is not None else 0
        stop = stop if stop is not None else len(self)
        step = step if step is not None else 1

        if start < 0:
            raise IndexError("Negative start index is not supported")
        if stop < start:
            raise ValueError("Stop index must be greater than or equal to start index")
        if step <= 0:
            raise ValueError("Step must be a positive integer")

        return self.select(list(range(start, stop, step)))

    @override
    def get(self, idx: int) -> Sample:
        start = 0
        for ds in self._datasets:
            d_idx = idx - start
            if d_idx < len(ds):
                return ds.get(d_idx)
            start += len(ds)
        raise IndexError("Index out of range")

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

    @staticmethod
    @override
    def from_dict(data: dict) -> Self:
        import sys
        from importlib import import_module
        from functools import reduce

        datasets = []
        for ds, module, qual in zip(data["datasets"], data["module"], data["qualname"]):
            if module in ("__main__", "__mp_main__"):
                m = sys.modules[module]
            else:
                m = import_module(module)

            T = reduce(getattr, qual.split("."), m)
            if not issubclass(T, Dataset):
                raise TypeError(f"{T} is not a subclass of Dataset")
            datasets.append(T.from_dict(ds))

        return ConcatDataset(
            datasets=datasets,
            sr=data["sr"],
            task=tuple(data["task"]),
        )


__all__ = ["Dataset", "ConcatDataset", "Sample", "Task"]
