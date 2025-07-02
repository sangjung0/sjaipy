from typing import Iterable
from faster_whisper.transcribe import Segment

from sj_ai_utils.evaluator.sclite_utils import TRNFormat


def segments_to_sclite_trn(id: str, segments: Iterable[Segment]) -> TRNFormat:
    return TRNFormat(
        id=id,
        text=" ".join(
            segment.text.strip() for segment in segments if segment.text.strip()
        ),
    )


__all__ = [
    "segments_to_sclite_trn",
]
