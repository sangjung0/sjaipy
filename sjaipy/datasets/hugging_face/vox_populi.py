from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from typing_extensions import override
from functools import lru_cache
from datasets import Dataset

from sjaipy.datasets.hugging_face.hugging_face_dataset import HuggingFaceDataset
from sjaipy.datasets.hugging_face.dataset_loader import DatasetLoader
from sjaipy.datasets.dataset import Task
from sjpy.string import normalize_text_only_en

if TYPE_CHECKING:
    pass

DEFAULT_PATH = "facebook/voxpopuli"
DEFAULT_CONFIG_NAME = "en"
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_TASK = ("asr",)


class VoxPopuliDataset(HuggingFaceDataset):
    def __init__(self, dataset: Dataset, sr: int, task: tuple[Task]):
        super().__init__(dataset, sr, task)

    @override
    def get(self, idx: int) -> tuple[str, np.ndarray, str]:
        data = self._dataset[idx]
        _id = normalize_text_only_en(data["audio_id"])[-255:]
        audio = data["audio"]["array"]
        audio = self._resample_audio(audio).astype(np.float32)
        txt = data["raw_text"]
        return _id, audio, txt


class VoxPopuli(DatasetLoader):
    def __init__(self, path=DEFAULT_PATH):
        raise Exception("deprecated")
        super().__init__(path)

    def split_names(
        self,
        config_name: str = DEFAULT_CONFIG_NAME,
        sr: int = DEFAULT_SAMPLE_RATE,
        task: tuple[Task] = DEFAULT_TASK,
    ) -> list[str]:
        if self._path == DEFAULT_PATH:
            print(f"warning: facebook/voxpopuli can't be loaded with split_names")
            if config_name == "en_accented":
                return ["test"]
            else:
                return ["train", "validation", "test"]
        return super().split_names(config_name, sr, task)

    @lru_cache(maxsize=1)
    def train(
        self,
        config_name: str = DEFAULT_CONFIG_NAME,
        sr: int = DEFAULT_SAMPLE_RATE,
        task: tuple[Task] = DEFAULT_TASK,
        **kwargs,
    ):
        if self._path == DEFAULT_PATH and config_name == "en_accented":
            raise ValueError(
                "facebook/voxpopuli en_accented config does not have a train split"
            )
        return VoxPopuliDataset(super().load(config_name, "train", **kwargs), sr, task)

    @lru_cache(maxsize=1)
    def validation(
        self,
        config_name: str = DEFAULT_CONFIG_NAME,
        sr: int = DEFAULT_SAMPLE_RATE,
        task: tuple[Task] = DEFAULT_TASK,
        **kwargs,
    ):
        if self._path == DEFAULT_PATH and config_name == "en_accented":
            raise ValueError(
                "facebook/voxpopuli en_accented config does not have a validation split"
            )
        return VoxPopuliDataset(
            super().load(config_name, "validation", **kwargs), sr, task
        )

    @lru_cache(maxsize=1)
    def test(
        self,
        config_name: str = DEFAULT_CONFIG_NAME,
        sr: int = DEFAULT_SAMPLE_RATE,
        task: tuple[Task] = DEFAULT_TASK,
        **kwargs,
    ):
        return VoxPopuliDataset(super().load(config_name, "test", **kwargs), sr, task)


__all__ = ["VoxPopuliDataset", "VoxPopuli"]
