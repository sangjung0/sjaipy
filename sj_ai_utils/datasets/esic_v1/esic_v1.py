from __future__ import annotations
from typing import TYPE_CHECKING

from pathlib import Path
from functools import lru_cache

from sj_ai_utils.datasets.esic_v1.service import search_dirs, select_file_from_dir
from sj_ai_utils.datasets.esic_v1.file_type import VERBATIM, MP4

if TYPE_CHECKING:
    pass

DEV = "v1.1/dev"
DEV2 = "v1.1/dev2"
TEST = "v1.1/test"


class ESICv1:
    def __init__(self, root: Path):
        self.__root = root

    @lru_cache(maxsize=1)
    def dev_dirs(
        self, post_path: Path | str = DEV, excludes: tuple[str] = ()
    ) -> list[Path]:
        return search_dirs(self.__root / post_path, excludes=excludes)

    @lru_cache(maxsize=1)
    def dev2_dirs(
        self, post_path: Path | str = DEV2, excludes: tuple[str] = ()
    ) -> list[Path]:
        return search_dirs(self.__root / post_path, excludes=excludes)

    @lru_cache(maxsize=1)
    def test_dirs(
        self, post_path: Path | str = TEST, excludes: tuple[str] = ()
    ) -> list[Path]:
        return search_dirs(self.__root / post_path, excludes=excludes)

    def all_dirs(self, excludes: tuple[str] = ()) -> list[Path]:
        return (
            self.dev_dirs(excludes=excludes)
            + self.dev2_dirs(excludes=excludes)
            + self.test_dirs(excludes=excludes)
        )

    def __generate_items(
        self, dirs: list[Path], source_file_type: str, truth_file_type: str
    ) -> list[dict[str, Path]]:
        items = []
        for d in dirs:
            x = select_file_from_dir(d, source_file_type)
            y = select_file_from_dir(d, truth_file_type)
            items.append({"X": x, "Y": y})
        return items

    @lru_cache(maxsize=1)
    def dev(
        self,
        post_path: Path | str = DEV,
        source_file_type: str = MP4,
        truth_file_type: str = VERBATIM,
        excludes: tuple[str] = (),
    ) -> list[dict[str, Path]]:
        return self.__generate_items(
            self.dev_dirs(post_path, excludes=excludes),
            source_file_type,
            truth_file_type,
        )

    @lru_cache(maxsize=1)
    def dev2(
        self,
        post_path: Path | str = DEV2,
        source_file_type: str = MP4,
        truth_file_type: str = VERBATIM,
        excludes: tuple[str] = (),
    ) -> list[dict[str, Path]]:
        return self.__generate_items(
            self.dev2_dirs(post_path, excludes=excludes),
            source_file_type,
            truth_file_type,
        )

    @lru_cache(maxsize=1)
    def test(
        self,
        post_path: Path | str = TEST,
        source_file_type: str = MP4,
        truth_file_type: str = VERBATIM,
        excludes: tuple[str] = (),
    ) -> list[dict[str, Path]]:
        return self.__generate_items(
            self.test_dirs(post_path, excludes=excludes),
            source_file_type,
            truth_file_type,
        )


__all__ = ["ESICv1"]
