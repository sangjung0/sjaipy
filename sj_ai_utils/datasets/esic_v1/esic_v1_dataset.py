import numpy as np

from pathlib import Path

from sj_utils.audio import load_audio_from_mp4
from sj_utils.string import normalize_text_only_en
from sj_utils.file.json import JsonSaver, load_json
from sj_ai_utils.datasets.dataset import Dataset

DEFAULT_SAMPLE_RATE = 16_000


class ESICv1Dataset(Dataset):
    def __init__(self, X: list[Path], Y: list[Path], sr: int = DEFAULT_SAMPLE_RATE):
        self._X = X
        self._Y = Y
        self._sr = sr

    def __iter__(self):
        for x, y in zip(self._X, self._Y):
            audio = load_audio_from_mp4(x, self._sr)[0]
            txt = y.read_text(encoding="utf-8")
            _id = normalize_text_only_en(str(Path(*x.parts[-2:])))[-255:]
            yield _id, audio, txt

    def sample(
        self,
        sample_size: int,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ) -> "ESICv1Dataset":
        if sample_size <= 0:
            sample_size = len(self._X)
        else:
            sample_size = min(sample_size, len(self._X))

        data = list(zip(self._X, self._Y))
        if rng is None or sample_size == len(self._X):
            data = data[:sample_size]
        else:
            data = rng.choice(data, size=sample_size, replace=False)

        X, Y = zip(*data)
        return ESICv1Dataset(X, Y, self._sr)

    def save(self, path: Path, description="ESICv1Dataset"):
        data = {
            "X": [str(x) for x in self._X],
            "Y": [str(y) for y in self._Y],
            "sr": self._sr,
        }
        JsonSaver(description).save(data, path)

    @staticmethod
    def load(path: Path):
        _, data = load_json(path)
        data["X"] = [Path(x) for x in data["X"]]
        data["Y"] = [Path(y) for y in data["Y"]]
        return ESICv1Dataset(**data)


__all__ = ["ESICv1Dataset"]
