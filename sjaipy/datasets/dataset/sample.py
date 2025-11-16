import numpy as np

from typing import Any, Callable
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Sample:
    id: str = field(compare=True, hash=True, repr=True)
    load_audio: np.ndarray | Callable[[], np.ndarray] = field(
        compare=False, hash=False, repr=False
    )
    Y: dict[str, Any] = field(compare=False, hash=False, repr=False)

    @property
    def audio(self) -> np.ndarray:
        return self.load_audio() if callable(self.load_audio) else self.load_audio

    @property
    def ASR(self) -> str:
        if "asr" not in self.Y:
            raise AttributeError("ASR label is not available in this sample")
        return self.Y["asr"]

    @property
    def diarization(self) -> list[dict[str, Any]]:
        if "diarization" not in self.Y:
            raise AttributeError("Diarization label is not available in this sample")
        return self.Y["diarization"]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "audio": self.audio.tolist(),
            "Y": self.Y,
        }

    @staticmethod
    def from_dict(data: dict) -> "Sample":
        return Sample(
            id=data["id"],
            load_audio=np.array(data["audio"]),
            Y=data["Y"],
        )


__all__ = ["Sample"]
