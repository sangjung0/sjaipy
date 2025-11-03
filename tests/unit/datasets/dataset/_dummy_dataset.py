import numpy as np

from typing import Sequence
from typing_extensions import override

from sjaipy.datasets import Dataset, Sample


class _DummyDataset(Dataset):
    def __init__(self, samples: list[Sample], sr: int, task: tuple[str, ...]):
        super().__init__(sr, task)
        self.samples = samples

    @Dataset.args.getter
    def args(self) -> dict:
        return {
            **super().args,
            "samples": self.samples,
        }

    @Dataset.length.getter
    def length(self) -> int:
        return len(self.samples)

    @override
    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "samples": [s.to_dict() for s in self.samples],
        }

    @override
    def select(self, indices: Sequence[int]) -> "_DummyDataset":
        samples = [self.samples[i] for i in indices]
        args = self.args
        args["samples"] = samples
        return _DummyDataset(**args)

    @override
    def slice(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> "_DummyDataset":
        samples = self.samples[start:stop:step]
        args = self.args
        args["samples"] = samples
        return _DummyDataset(**args)

    @override
    def get(self, idx: int) -> Sample:
        if idx < 0 or idx >= len(self.samples):
            raise IndexError("Index out of range")
        return self.samples[idx]

    @override
    def _sample(
        self,
        size: int,
        start: int = 0,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ) -> "_DummyDataset":
        if rng is None or size == len(self) - start:
            sampled_samples = self.samples[start : start + size]
        else:
            indices = rng.choice(range(start, len(self)), size=size, replace=False)
            sampled_samples = [self.samples[i] for i in indices]

        args = self.args
        args["samples"] = sampled_samples
        return _DummyDataset(**args)

    @staticmethod
    @override
    def from_dict(data: dict) -> "_DummyDataset":
        samples = [Sample.from_dict(s) for s in data["samples"]]
        return _DummyDataset(
            samples=samples,
            sr=data["sr"],
            task=tuple(data["task"]),
        )
