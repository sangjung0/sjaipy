from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from typing import Generator
from pathlib import Path
from lhotse import RecordingSet, SupervisionSet

if TYPE_CHECKING:
    pass


def load_data(
    datasets: tuple[RecordingSet, SupervisionSet],
    sr: int,
    sample_size: int = -1,
    rng: np.random.Generator | np.random.RandomState | None = None,
) -> Generator[tuple[np.ndarray, str, str, Path], None, None]:
    recording_set, supervision_set = datasets
    if sample_size < 0:
        sample_size = len(recording_set)
    if rng is None or sample_size == len(recording_set):
        indices = range(sample_size)
    else:
        indices = rng.choice(len(recording_set), size=sample_size, replace=False)

    for idx in indices:
        rec = recording_set[idx]
        rid = rec.id
        rec.resample(sr)  # resample to 16kHz
        wav = rec.load_audio(channels=0)

        assert len(wav) == 1, "wav must be mono"

        segs = [s for s in supervision_set if s.recording_id == rid]
        segs.sort(key=lambda s: s.start)

        y = " ".join([s.text for s in segs])

        yield wav[0], rid, y, rid


__all__ = ["load_data"]
