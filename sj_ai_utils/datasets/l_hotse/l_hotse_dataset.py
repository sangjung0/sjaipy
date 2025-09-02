from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from lhotse import RecordingSet, SupervisionSet
from functools import lru_cache
from typing_extensions import override

from sj_ai_utils.datasets.dataset import Dataset

if TYPE_CHECKING:
    pass


class LHotseDataset(Dataset):
    def __init__(
        self,
        recording_set: RecordingSet,
        supervision_set: SupervisionSet,
        sr: int,
        X: list[tuple[int, int]] = [],
    ):
        self._recording_set = recording_set
        self._supervision_set = supervision_set
        self._sr = sr
        if not X:
            self._X = [
                (c, idx)
                for idx in range(len(recording_set))
                for c in recording_set[idx].channel_ids
            ]
        else:
            self._X = X

    @override
    def __len__(self):
        return len(self._X)

    @override
    def _sample(
        self,
        size: int,
        start: int = 0,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ) -> "LHotseDataset":
        if rng is None or size == len(self) - start:
            return self.slice(start=start, stop=start + size)
        else:
            X = rng.choice(self._X[start:], size=size, replace=False)
            return LHotseDataset(
                self._recording_set, self._supervision_set, sr=self._sr, X=list(X)
            )

    @override
    def slice(
        self, start: int | None = None, stop: int | None = None, step: int | None = None
    ) -> "Dataset":
        X = self._X[start:stop:step]
        return LHotseDataset(
            self._recording_set, self._supervision_set, sr=self._sr, X=X
        )

    @override
    @lru_cache(maxsize=128)
    def get_item(self, idx: int):
        channel, r_idx = self._X[idx]
        rec = self._recording_set[r_idx]
        rec.resample(self._sr)
        rid = rec.id
        wav = rec.load_audio(channels=channel)
        assert len(wav) == 1, "wav must be mono"

        segs = [
            s
            for s in self._supervision_set
            # 이부분은 추후 수정 필요할 듯 채널 탐지 확실하지 않음
            if s.recording_id == rid
            and (
                s.channel == channel
                or (isinstance(s.channel, list) and channel in s.channel)
            )
        ]
        segs.sort(key=lambda s: s.start)
        y = " ".join([s.text for s in segs])

        _id = (rid + "_" + str(channel))[-255:]
        return _id, wav[0], y


__all__ = ["LHotseDataset"]
