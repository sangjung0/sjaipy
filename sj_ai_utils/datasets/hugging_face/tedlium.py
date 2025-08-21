from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from functools import lru_cache
from datasets import Dataset

from sj_ai_utils.datasets.hugging_face.hugging_face_dataset import HuggingFaceDataset
from sj_ai_utils.datasets.hugging_face.dataset_loader import DatasetLoader
from sj_utils.string import normalize_text_only_en

if TYPE_CHECKING:
    pass

DEFAULT_PATH = "LIUM/tedlium"
DEFAULT_CONFIG_NAME = "release1"
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_IGNORE_SET = ["ignore_time_segment_in_scoring"]


class TedliumDataset(HuggingFaceDataset):
    def __init__(
        self,
        dataset: Dataset,
        sr: int = DEFAULT_SAMPLE_RATE,
        ignore_set: list[str] = DEFAULT_IGNORE_SET,
    ):
        super().__init__(dataset, sr)
        self._ignore_set = ignore_set

    def _get_construct_args(self):
        args = super()._get_construct_args()
        args["ignore_set"] = self._ignore_set
        return args

    def __iter__(self):
        for data in self._dataset:
            _id = normalize_text_only_en(data["id"][-255:])
            audio = data["audio"]["array"].astype(np.float32)
            txt = data["text"]

            if txt in self._ignore_set:
                continue
            yield _id, audio, txt


class Tedlium(DatasetLoader):
    def __init__(self, path=DEFAULT_PATH):
        super().__init__(path)

    def split_names(self, config_name: str = DEFAULT_CONFIG_NAME) -> list[str]:
        return super().split_names(config_name)

    @lru_cache(maxsize=1)
    def train(self, config_name: str = DEFAULT_CONFIG_NAME, **kwargs):
        return TedliumDataset(super().load(config_name, "train", **kwargs))

    @lru_cache(maxsize=1)
    def validation(self, config_name: str = DEFAULT_CONFIG_NAME, **kwargs):
        return TedliumDataset(super().load(config_name, "validation", **kwargs))

    @lru_cache(maxsize=1)
    def test(self, config_name: str = DEFAULT_CONFIG_NAME, **kwargs):
        return TedliumDataset(super().load(config_name, "test", **kwargs))


__all__ = ["Tedlium", "TedliumDataset"]
