from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from typing_extensions import override
from functools import lru_cache
from datasets import Dataset

from sj_utils.string import normalize_text_only_en
from sj_ai_utils.datasets.dataset import Dataset, Task, Sample
from sj_ai_utils.datasets.hugging_face.dataset_loader import DatasetLoader
from sj_ai_utils.datasets.hugging_face.hugging_face_dataset import HuggingFaceDataset

if TYPE_CHECKING:
    pass


DEFAULT_PATH = "edinburghcstr/ami"
DEFAULT_CONFIG_NAME = "sdm"
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_TASK = ("asr",)


class AMIDataset(HuggingFaceDataset):
    def __init__(self, dataset: Dataset, sr: int, task: tuple[Task]):
        super().__init__(dataset, sr, task)
        print("[INFO] 오디오가 연속적이지 않고 세그먼트로 나눠져 있음")

    @override
    def get_item(self, idx: int) -> tuple[str, np.ndarray, str]:
        data = self._dataset[idx]
        _id = normalize_text_only_en(data["audio_id"])[-255:]
        audio = data["audio"]["array"].astype(np.float32)
        result = {}
        if "asr" in self.task:
            result["asr"] = data["text"]
        if "diarization" in self.task:
            result["diarization"] = [
                {
                    "start": data["begin_time"],
                    "end": data["end_time"],
                    "label": data["speaker_id"],
                }
            ]
        return Sample(id=_id, audio=audio, Y=result)


class AMI(DatasetLoader):
    def __init__(self, path=DEFAULT_PATH):
        super().__init__(path)

    def split_names(self, config_name: str = DEFAULT_CONFIG_NAME) -> list[str]:
        return super().split_names(config_name)

    @lru_cache(maxsize=1)
    def train(
        self,
        config_name: str = DEFAULT_CONFIG_NAME,
        sr: int = DEFAULT_SAMPLE_RATE,
        task: list[Task] = DEFAULT_TASK,
        **kwargs,
    ):
        return AMIDataset(super().load(config_name, "train", **kwargs), sr, task)

    @lru_cache(maxsize=1)
    def validation(
        self,
        config_name: str = DEFAULT_CONFIG_NAME,
        sr: int = DEFAULT_SAMPLE_RATE,
        task: list[Task] = DEFAULT_TASK,
        **kwargs,
    ):
        return AMIDataset(super().load(config_name, "validation", **kwargs), sr, task)

    @lru_cache(maxsize=1)
    def test(
        self,
        config_name: str = DEFAULT_CONFIG_NAME,
        sr: int = DEFAULT_SAMPLE_RATE,
        task: list[Task] = DEFAULT_TASK,
        **kwargs,
    ):
        return AMIDataset(super().load(config_name, "test", **kwargs), sr, task)


__all__ = ["AMI", "AMIDataset"]
