from __future__ import annotations
from typing import TYPE_CHECKING

from functools import lru_cache

from sj_ai_utils.hugging_face import DatasetLoader

if TYPE_CHECKING:
    pass

DEFAULT_PATH = "LIUM/tedlium"
DEFAULT_CONFIG_NAME = "release1"


class Tedlium(DatasetLoader):
    def __init__(self, path=DEFAULT_PATH):
        super().__init__(path)

    def split_names(self, config_name: str = DEFAULT_CONFIG_NAME) -> list[str]:
        return super().split_names(config_name)

    @lru_cache(maxsize=1)
    def train(self, config_name: str = DEFAULT_CONFIG_NAME, **kwargs):
        return super().load(config_name, "train", **kwargs)

    @lru_cache(maxsize=1)
    def validation(self, config_name: str = DEFAULT_CONFIG_NAME, **kwargs):
        return super().load(config_name, "validation", **kwargs)

    @lru_cache(maxsize=1)
    def test(self, config_name: str = DEFAULT_CONFIG_NAME, **kwargs):
        return super().load(config_name, "test", **kwargs)


__all__ = ["Tedlium"]
