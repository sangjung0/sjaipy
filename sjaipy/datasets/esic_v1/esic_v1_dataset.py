import warnings
import numpy as np

from pathlib import Path
from typing_extensions import override, Self
from typing import Sequence

from sjpy.audio import load_audio_from_mp4
from sjpy.string import normalize_text_only_en
from sjpy.file.json import JsonSaver, load_json
from sjaipy.datasets.dataset import Dataset, Sample

DEFAULT_SAMPLE_RATE = 16_000


class ESICv1Dataset(Dataset):
    def __init__(self, X: list[Path], Y: list[Path], sr: int = DEFAULT_SAMPLE_RATE):
        if len(X) != len(Y):
            raise ValueError("X and Y must have the same length")
        super().__init__(sr, task=("asr",))
        self._X = X
        self._Y = Y

    @Dataset.args.getter
    @override
    def args(self) -> dict:
        return {
            **super().args,
            "X": self._X,
            "Y": self._Y,
        }

    @Dataset.length.getter
    @override
    def length(self) -> int:
        return len(self._X)

    @override
    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "X": [str(x) for x in self._X],
            "Y": [str(y) for y in self._Y],
        }

    @override
    def select(self, indices: Sequence[int]) -> Self:
        return ESICv1Dataset(
            [self._X[i] for i in indices], [self._Y[i] for i in indices], self._sr
        )

    @override
    def slice(
        self, start: int | None = None, stop: int | None = None, step: int | None = None
    ) -> Self:
        return ESICv1Dataset(
            self._X[start:stop:step], self._Y[start:stop:step], self._sr
        )

    @override
    def get(self, idx: int) -> Sample:
        x, y = self._X[idx], self._Y[idx]
        audio = load_audio_from_mp4(x, self._sr)[0]
        txt = y.read_text(encoding="utf-8")
        _id = normalize_text_only_en(str(Path(*x.parts[-3:-1])))[-255:]
        return Sample(id=_id, audio=audio, _Y={"asr": txt})

    def save(self, path: Path, description="ESICv1Dataset"):
        JsonSaver(description).save(self.to_dict(), path)

    @override
    def _sample(
        self,
        size: int,
        start: int = 0,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ) -> Self:
        if rng is None or size == len(self._X) - start:
            return self.slice(start, start + size)
        else:
            data = list(zip(self._X[start:], self._Y[start:]))
            data = rng.choice(data, size=size, replace=False)
            X, Y = zip(*data)
            return ESICv1Dataset(X, Y, self._sr)

    @staticmethod
    @override
    def from_dict(data: dict) -> Self:
        return ESICv1Dataset(
            [Path(x) for x in data["X"]], [Path(y) for y in data["Y"]], sr=data["sr"]
        )

    @staticmethod
    def load(path: Path):
        _, data = load_json(path)
        return ESICv1Dataset.from_dict(data)


if __name__ != "__main__":
    warnings.warn(
        "[INFO] ESICv1Dataset 화자분리 데이터 없음", category=UserWarning, stacklevel=2
    )

__all__ = ["ESICv1Dataset"]
