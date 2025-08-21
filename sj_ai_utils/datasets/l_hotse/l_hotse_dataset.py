from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from lhotse import RecordingSet, SupervisionSet

from sj_ai_utils.datasets.dataset import Dataset
from sj_utils.typing import override

if TYPE_CHECKING:
    pass


class LHotseDataset(Dataset):
    def __init__(
        self, recording_set: RecordingSet, supervision_set: SupervisionSet, sr: int
    ):
        self._recording_set = recording_set
        self._supervision_set = supervision_set
        self._sr = sr

    def __iter__(self):
        for rec in self._recording_set:
            rid = rec.id
            rec.resample(self._sr)
            sources = [rec.load_audio(channels=c) for c in rec.channel_ids]

            for wav, channel_id in zip(sources, rec.channel_ids):
                assert len(wav) == 1, "wav must be mono"
                segs = [
                    s
                    for s in self._supervision_set
                    if s.recording_id == rid and s.channel == channel_id
                ]
                segs.sort(key=lambda s: s.start)
                y = " ".join([s.text for s in segs])

                _id = (rid + "_" + str(channel_id))[-255:]
                yield _id, wav[0], y

    @override
    def sample(
        self,
        sample_size: int,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ) -> "LHotseDataset":
        if sample_size <= 0:
            sample_size = len(self._recording_set)
        else:
            sample_size = min(sample_size, len(self._recording_set))

        if rng is None or sample_size == len(self._recording_set):
            indices = range(sample_size)
        else:
            indices = rng.choice(
                len(self._recording_set), size=sample_size, replace=False
            )

        recording_set = RecordingSet.from_recordings(
            self._recording_set[i] for i in indices
        )

        return LHotseDataset(recording_set, self._supervision_set, sr=self._sr)


__all__ = ["LHotseDataset"]
