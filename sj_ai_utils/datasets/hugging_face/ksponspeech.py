from __future__ import annotations
from typing import TYPE_CHECKING

import warnings
import numpy as np

from typing_extensions import override
from functools import lru_cache

from sj_utils.string import normalize_text_only_en
from sj_ai_utils.datasets.dataset import Sample, Task
from sj_ai_utils.datasets.hugging_face.dataset_loader import DatasetLoader
from sj_ai_utils.datasets.hugging_face.hugging_face_dataset import HuggingFaceDataset

if TYPE_CHECKING:
    pass


DEFAULT_PATH = "DragonLine/ksponspeech"
DEFAULT_CONFIG_NAME = "default"
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_TASK = ("asr",)


class KSPonSpeechDataset(HuggingFaceDataset):
    @override
    def get(self, idx: int) -> tuple[str, np.ndarray, str]:
        data = self._dataset[idx]
        _id = normalize_text_only_en(data["audio"]["path"])[-255:]
        audio = data["audio"]["array"]
        audio = self._resample_audio(audio).astype(np.float32)
        txt = data["transcripts"]
        return Sample(id=_id, audio=audio, _Y={"asr": txt})


class KSPonSpeech(DatasetLoader):
    def __init__(self, path=DEFAULT_PATH):
        super().__init__(path)

    def split_names(self, config_name: str = DEFAULT_CONFIG_NAME) -> list[str]:
        return super().split_names(config_name)

    @lru_cache(maxsize=1)
    def train(
        self,
        config_name: str = DEFAULT_CONFIG_NAME,
        sr: int = DEFAULT_SAMPLE_RATE,
        task: tuple[Task, ...] = DEFAULT_TASK,
        **kwargs,
    ):
        return KSPonSpeechDataset(
            super().load(config_name, "train", **kwargs), sr, task
        )

    @lru_cache(maxsize=1)
    def valid(
        self,
        config_name: str = DEFAULT_CONFIG_NAME,
        sr: int = DEFAULT_SAMPLE_RATE,
        task: tuple[Task, ...] = DEFAULT_TASK,
        **kwargs,
    ):
        return KSPonSpeechDataset(
            super().load(config_name, "valid", **kwargs), sr, task
        )

    @lru_cache(maxsize=1)
    def test(
        self,
        config_name: str = DEFAULT_CONFIG_NAME,
        sr: int = DEFAULT_SAMPLE_RATE,
        task: tuple[Task, ...] = DEFAULT_TASK,
        **kwargs,
    ):
        return KSPonSpeechDataset(super().load(config_name, "test", **kwargs), sr, task)


if __name__ != "__main__":
    warnings.warn(
        "[INFO] KSPonSpeechDataset 화자분리 데이터 없음, 오디오가 연속적이지 않고 세그먼트로 나눠져 있음",
        category=UserWarning,
        stacklevel=2,
    )


__all__ = ["KSPonSpeech", "KSPonSpeechDataset"]
