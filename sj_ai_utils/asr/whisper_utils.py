from typing import Iterable
from faster_whisper.transcribe import Segment


def segments_to_sclite_trn(id: str, segments: Iterable[Segment]):
    return f"{' '.join([segment.text.strip() for segment in segments if segment.text.strip()]).upper()} ({id})"
