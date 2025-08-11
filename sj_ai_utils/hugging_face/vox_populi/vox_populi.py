from __future__ import annotations
from typing import TYPE_CHECKING

from functools import lru_cache

from sj_ai_utils.hugging_face import DatasetLoader

if TYPE_CHECKING:
    pass

DEFAULT_PATH = "facebook/voxpopuli"
DEFAULT_CONFIG_NAME = "en"


class VoxPopuli(DatasetLoader):
    def __init__(self, path=DEFAULT_PATH):
        super().__init__(path)

    def split_names(self, config_name: str = DEFAULT_CONFIG_NAME) -> list[str]:
        if self._path == DEFAULT_PATH:
            print(f"warning: facebook/voxpopuli can't be loaded with split_names")
            if config_name == "en_accented":
                return ["test"]
            else:
                return ["train", "validation", "test"]
        return super().split_names(config_name)

    @lru_cache(maxsize=1)
    def train(self, config_name: str = DEFAULT_CONFIG_NAME, **kwargs):
        if self._path == DEFAULT_PATH and config_name == "en_accented":
            raise ValueError(
                "facebook/voxpopuli en_accented config does not have a train split"
            )
        return super().load(config_name, "train", **kwargs)

    @lru_cache(maxsize=1)
    def validation(self, config_name: str = DEFAULT_CONFIG_NAME, **kwargs):
        if self._path == DEFAULT_PATH and config_name == "en_accented":
            raise ValueError(
                "facebook/voxpopuli en_accented config does not have a validation split"
            )
        return super().load(config_name, "validation", **kwargs)

    @lru_cache(maxsize=1)
    def test(self, config_name: str = DEFAULT_CONFIG_NAME, **kwargs):
        return super().load(config_name, "test", **kwargs)


__all__ = ["VoxPopuli"]
