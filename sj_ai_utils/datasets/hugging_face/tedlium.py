from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from typing_extensions import override
from functools import lru_cache
from datasets import Dataset

from sj_ai_utils.datasets.hugging_face.hugging_face_dataset import HuggingFaceDataset
from sj_ai_utils.datasets.hugging_face.dataset_loader import DatasetLoader
from sj_ai_utils.datasets.dataset import Task, Sample
from sj_utils.string import normalize_text_only_en

if TYPE_CHECKING:
    pass

DEFAULT_PATH = "LIUM/tedlium"
DEFAULT_CONFIG_NAME = "release1"
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_IGNORE_SET = set(("ignore_time_segment_in_scoring", "inter_segment_gap"))
DEFAULT_TASK = ("asr",)


class TedliumDataset(HuggingFaceDataset):
    def __init__(
        self, dataset: Dataset, sr: int, task: list[Task], ignore_set: set[str]
    ):
        super().__init__(dataset, sr, task)
        print("[WARN] 오디오가 연속적이지 않고 세그먼트로 나눠져 있음.")
        self._ignore_set = ignore_set

    @override
    def _get_construct_args(self):
        args = super()._get_construct_args()
        args["ignore_set"] = self._ignore_set
        return args

    @override
    def get_item(self, idx: int) -> tuple[str, np.ndarray, str]:
        data = self._dataset[idx]
        _id = normalize_text_only_en(data["id"])[-255:]
        audio = data["audio"]["array"].astype(np.float32)

        result = {}
        if "asr" in self.task:
            txt = data["text"]
            if txt in self._ignore_set:
                txt = ""
            result["asr"] = txt
        if "diarization" in self.task:
            diarization = []
            if data["speaker_id"] not in self._ignore_set:
                diarization.append(
                    {"start": 0, "end": len(audio), "label": data["speaker_id"]}
                )
            result["diarization"] = diarization

        return Sample(_id, audio, result)


class Tedlium(DatasetLoader):
    def __init__(self, path=DEFAULT_PATH):
        super().__init__(path)

    def split_names(self, config_name: str = DEFAULT_CONFIG_NAME) -> list[str]:
        return super().split_names(config_name)

    @lru_cache(maxsize=1)
    def train(
        self,
        config_name: str = DEFAULT_CONFIG_NAME,
        sr: int = DEFAULT_SAMPLE_RATE,
        task: tuple[Task] = DEFAULT_TASK,
        ignore_set: set[str] = DEFAULT_IGNORE_SET,
        **kwargs,
    ):
        return TedliumDataset(
            super().load(config_name, "train", **kwargs), sr, task, ignore_set
        )

    @lru_cache(maxsize=1)
    def validation(
        self,
        config_name: str = DEFAULT_CONFIG_NAME,
        sr: int = DEFAULT_SAMPLE_RATE,
        task: tuple[Task] = DEFAULT_TASK,
        ignore_set: set[str] = DEFAULT_IGNORE_SET,
        **kwargs,
    ):
        return TedliumDataset(
            super().load(config_name, "validation", **kwargs), sr, task, ignore_set
        )

    @lru_cache(maxsize=1)
    def test(
        self,
        config_name: str = DEFAULT_CONFIG_NAME,
        sr: int = DEFAULT_SAMPLE_RATE,
        task: tuple[Task] = DEFAULT_TASK,
        ignore_set: set[str] = DEFAULT_IGNORE_SET,
        **kwargs,
    ):
        return TedliumDataset(
            super().load(config_name, "test", **kwargs), sr, task, ignore_set
        )


__all__ = ["Tedlium", "TedliumDataset"]
