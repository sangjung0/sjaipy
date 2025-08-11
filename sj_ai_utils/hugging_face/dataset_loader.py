from __future__ import annotations
from typing import TYPE_CHECKING

from functools import lru_cache, cached_property
from datasets import (
    load_dataset,
    get_dataset_config_names,
    get_dataset_split_names,
    Dataset,
    DownloadConfig,
)

if TYPE_CHECKING:
    pass


class DatasetLoader:
    def __init__(self, path: str):
        self._path = path

    @cached_property
    def config_names(self) -> list[str]:
        return get_dataset_config_names(self._path)

    @lru_cache(maxsize=32)
    def split_names(self, config_name: str) -> list[str]:
        if config_name not in self.config_names:
            raise ValueError(
                f"Config name '{config_name}' is not valid. Available configs: {self.config_names}"
            )
        return get_dataset_split_names(self._path, config_name)

    def load(
        self, config_name: str, split: str, local_files_only: bool = False
    ) -> Dataset:
        if config_name not in self.config_names:
            raise ValueError(
                f"Config name '{config_name}' is not valid. Available configs: {self.config_names}"
            )
        if split not in self.split_names(config_name):
            raise ValueError(
                f"Split '{split}' is not valid for config '{config_name}'. Available splits: {self.split_names(config_name)}"
            )
        return load_dataset(
            self._path,
            config_name,
            split=split,
            download_config=DownloadConfig(local_files_only=local_files_only),
        )


__all__ = ["DatasetLoader"]
