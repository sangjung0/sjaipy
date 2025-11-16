from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from lhotse import RecordingSet, SupervisionSet, SupervisionSegment, Recording
from typing import Sequence
from typing_extensions import override, Self

from sjaipy.datasets.dataset import Dataset, Sample, Task

if TYPE_CHECKING:
    pass


class LHotseDataset(Dataset):
    def __init__(
        self,
        recordings: list[tuple[Recording, int]],
        segments: list[list[SupervisionSegment]],
        sr: int,
        task: tuple[Task, ...],
    ):
        if len(recordings) != len(segments):
            raise ValueError("Mismatched lengths between recordings and segments")

        super().__init__(sr, task)
        self.recordings = recordings
        self.segments = segments

    @Dataset.sr.setter
    @override
    def sr(self, value: int):
        if value != self._sr:
            self.recordings = [(rec.resample(value), ch) for rec, ch in self.recordings]
            self._sr = value

    @Dataset.args.getter
    @override
    def args(self) -> dict:
        return {
            **super().args,
            "recordings": self.recordings,
            "segments": self.segments,
        }

    @Dataset.length.getter
    @override
    def length(self) -> int:
        return len(self.recordings)

    @override
    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "recordings": [(r.to_dict(), ch) for r, ch in self.recordings],
            "segments": [[s.to_dict() for s in ss] for ss in self.segments],
        }

    @override
    def select(self, indices: Sequence[int]) -> "LHotseDataset":
        return LHotseDataset(
            **{
                **self.args,
                "recordings": [self.recordings[i] for i in indices],
                "segments": [self.segments[i] for i in indices],
            }
        )

    @override
    def slice(
        self, start: int | None = None, stop: int | None = None, step: int | None = None
    ) -> Self:
        return LHotseDataset(
            **{
                **self.args,
                "recordings": self.recordings[start:stop:step],
                "segments": self.segments[start:stop:step],
            }
        )

    @override
    def get(self, idx: int):
        rec, channel = self.recordings[idx]
        rid = rec.id

        def load_audio() -> np.ndarray:
            wav = rec.load_audio(channels=channel)
            assert len(wav) == 1, "wav must be mono"
            return wav[0]

        segments = self.segments[idx]
        result = {}
        if "asr" in self.task:
            result["asr"] = " ".join([s.text for s in segments])
        if "diarization" in self.task:
            result["diarization"] = [
                {"start": s.start, "end": s.end, "label": s.speaker} for s in segments
            ]

        return Sample(
            id=(rid + "_" + str(channel))[-255:],
            load_audio=load_audio,
            Y=result,
        )

    @staticmethod
    def from_recording_supervision(
        recording_set: RecordingSet,
        supervision_set: SupervisionSet,
        sr: int,
        task: tuple[Task, ...],
    ) -> "LHotseDataset":
        recordings = []
        segments = []

        for rec in recording_set:
            rec = rec if rec.sampling_rate == sr else rec.resample(sr)
            for c in rec.channel_ids:
                rec_segments = list(
                    supervision_set.find(recording_id=rec.id, channel=c)
                )
                # if len(rec_segments) == 0:
                #     continue
                recordings.append((rec, c))
                segments.append(rec_segments)

        return LHotseDataset(
            recordings=recordings,
            segments=segments,
            sr=sr,
            task=task,
        )

    # def _generate_metadata(self) -> list[_Metadata]:
    #     metadata = {
    #         (rec.id, c): _Metadata(rid=rec.id, channel=c, index=idx, segments=[])
    #         for idx, rec in enumerate(self.recording_set)
    #         for c in rec.channel_ids
    #     }

    #     for s in self.supervision_set:
    #         rid = s.recording_id
    #         if isinstance(s.channel, list):
    #             for c in s.channel:
    #                 metadata[(rid, c)].segments.append(s)
    #         else:
    #             metadata[(rid, s.channel)].segments.append(s)

    #     for key in metadata:
    #         metadata[key].segments.sort(key=lambda s: s.start)

    #     return [value for value in metadata.values()]

    @override
    def _sample(
        self,
        size: int,
        start: int = 0,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ) -> "LHotseDataset":
        if rng is None or size == len(self) - start:
            return self.slice(start=start, stop=start + size)
        idxs = rng.choice(np.arange(start, len(self)), size=size, replace=False)
        recs = [self.recordings[i] for i in idxs]
        segs = [self.segments[i] for i in idxs]
        return LHotseDataset(
            **{**self.args, "recordings": list(recs), "segments": list(segs)}
        )

    @staticmethod
    @override
    def from_dict(data: dict) -> Self:
        return LHotseDataset(
            recordings=[(Recording.from_dict(r), ch) for (r, ch) in data["recordings"]],
            segments=[
                [SupervisionSegment.from_dict(s) for s in ss] for ss in data["segments"]
            ],
            sr=data["sr"],
            task=tuple(data["task"]),
        )


__all__ = ["LHotseDataset"]
