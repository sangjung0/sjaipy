from typing import Iterable
from faster_whisper.transcribe import Segment

from sjaipy.evaluator.sclite_utils import TRNFormat


def segments_to_sclite_trn(id: str, segments: Iterable[Segment]) -> TRNFormat:
    return TRNFormat(
        id=id,
        text=segments_to_text(segments),
    )


def segments_to_text(segments: Iterable[Segment]) -> str:
    return " ".join(
        segment.text.strip() for segment in segments if segment.text.strip()
    )


__all__ = [
    "segments_to_sclite_trn",
    "segments_to_text",
]
