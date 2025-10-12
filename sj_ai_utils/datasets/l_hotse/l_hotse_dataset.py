from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from lhotse import RecordingSet, SupervisionSet
from functools import lru_cache
from typing_extensions import override

from sj_ai_utils.datasets.dataset import Dataset, Sample, Task

if TYPE_CHECKING:
    pass


class LHotseDataset(Dataset):
    def __init__(
        self,
        recording_set: RecordingSet,
        supervision_set: SupervisionSet,
        sr: int,
        task: tuple[Task],
        X: list[tuple[int, int]] = [],
    ):
        super().__init__(sr, task)
        self.recording_set = recording_set.resample(sampling_rate=sr)
        self.supervision_set = supervision_set
        if not X:
            self.__X = [
                (c, idx)
                for idx in range(len(self.recording_set))
                for c in self.recording_set[idx].channel_ids
            ]
        else:
            self.__X = X

    @override
    def __len__(self):
        return len(self.__X)

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
            X = rng.choice(self.__X[start:], size=size, replace=False)
            return LHotseDataset(
                self.recording_set, self.supervision_set, sr=self.sr, X=list(X)
            )

    @override
    def slice(
        self, start: int | None = None, stop: int | None = None, step: int | None = None
    ) -> "Dataset":
        X = self.__X[start:stop:step]
        return LHotseDataset(self.recording_set, self.supervision_set, sr=self.sr, X=X)

    @override
    @lru_cache(maxsize=128)
    def get_item(self, idx: int):
        channel, r_idx = self.__X[idx]
        rec = self.recording_set[r_idx]
        rid = rec.id
        wav = rec.load_audio(channels=channel)
        assert len(wav) == 1, "wav must be mono"

        segs = [
            s
            for s in self.supervision_set
            if s.recording_id == rid
            and (
                s.channel == channel
                or (isinstance(s.channel, list) and channel in s.channel)
            )
        ]
        segs.sort(key=lambda s: s.start)
        result = {}
        if "asr" in self.task:
            result["asr"] = " ".join([s.text for s in segs])
        if "diarization" in self.task:
            result["diarization"] = [
                {"start": s.start, "end": s.end, "label": s.speaker} for s in segs
            ]

        return Sample(
            id=(rid + "_" + str(channel))[-255:],
            audio=wav[0],
            Y=result,
        )


__all__ = ["LHotseDataset"]
