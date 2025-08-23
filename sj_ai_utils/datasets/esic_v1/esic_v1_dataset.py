import numpy as np

from pathlib import Path
from functools import cached_property

from sj_utils.audio import load_audio_from_mp4
from sj_utils.string import normalize_text_only_en
from sj_utils.file.json import JsonSaver, load_json
from sj_utils.typing import override
from sj_ai_utils.datasets.dataset import Dataset

DEFAULT_SAMPLE_RATE = 16_000


class ESICv1Dataset(Dataset):
    def __init__(self, X: list[Path], Y: list[Path], sr: int = DEFAULT_SAMPLE_RATE):
        if len(X) != len(Y):
            raise ValueError("X and Y must have the same length")
        super().__init__(sr)
        self._X = X
        self._Y = Y

    @cached_property
    def __len__(self):
        return len(self._X)

    @override
    def _get_construct_args(self):
        args = super()._get_construct_args()
        args["X"] = self._X
        args["Y"] = self._Y
        return args

    @override
    def _sample(
        self,
        size: int,
        start: int = 0,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ) -> "ESICv1Dataset":
        if rng is None or size == len(self._X) - start:
            return self.slice(start, start + size)
        else:
            data = list(zip(self._X[start:], self._Y[start:]))
            data = rng.choice(data, size=size, replace=False)
            X, Y = zip(*data)
            return ESICv1Dataset(X, Y, self._sr)

    @override
    def slice(
        self, start: int | None = None, stop: int | None = None, step: int | None = None
    ) -> "ESICv1Dataset":
        return ESICv1Dataset(
            self._X[start:stop:step], self._Y[start:stop:step], self._sr
        )

    @override
    def get_item(self, idx: int) -> tuple[str, np.ndarray, str]:
        x, y = self._X[idx], self._Y[idx]
        audio = load_audio_from_mp4(x, self._sr)[0]
        txt = y.read_text(encoding="utf-8")
        _id = normalize_text_only_en(str(Path(*x.parts[-2:])))[-255:]
        return _id, audio, txt

    def save(self, path: Path, description="ESICv1Dataset"):
        data = {
            "X": [str(x) for x in self._X],
            "Y": [str(y) for y in self._Y],
            "sr": self._sr,
        }
        JsonSaver(description).save(data, path)

    @classmethod
    def load(path: Path):
        _, data = load_json(path)
        data["X"] = [Path(x) for x in data["X"]]
        data["Y"] = [Path(y) for y in data["Y"]]
        return ESICv1Dataset(**data)


__all__ = ["ESICv1Dataset"]
