from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from functools import lru_cache
from datasets import Dataset

from sj_ai_utils.datasets.hugging_face.dataset_loader import DatasetLoader
from sj_ai_utils.datasets.hugging_face.hugging_face_dataset import HuggingFaceDataset

if TYPE_CHECKING:
    pass


DEFAULT_PATH = "edinburghcstr/ami"
DEFAULT_CONFIG_NAME = "ihm"
DEFAULT_SAMPLE_RATE = 16_000


class AMIDataset(HuggingFaceDataset):
    def __init__(self, dataset: Dataset, sr: int = DEFAULT_SAMPLE_RATE):
        super().__init__(dataset, sr)

    def __iter__(self):
        for data in self._dataset:
            _id = data["audio_id"][-255:]
            audio = data["audio"]["array"].astype(np.float32)
            txt = data["text"]
            yield _id, audio, txt


class AMI(DatasetLoader):
    def __init__(self, path=DEFAULT_PATH):
        super().__init__(path)

    def split_names(self, config_name: str = DEFAULT_CONFIG_NAME) -> list[str]:
        return super().split_names(config_name)

    @lru_cache(maxsize=1)
    def train(self, config_name: str = DEFAULT_CONFIG_NAME, **kwargs):
        return AMIDataset(super().load(config_name, "train", **kwargs))

    @lru_cache(maxsize=1)
    def validation(self, config_name: str = DEFAULT_CONFIG_NAME, **kwargs):
        return AMIDataset(super().load(config_name, "validation", **kwargs))

    @lru_cache(maxsize=1)
    def test(self, config_name: str = DEFAULT_CONFIG_NAME, **kwargs):
        return AMIDataset(super().load(config_name, "test", **kwargs))


__all__ = ["AMI", "AMIDataset"]
