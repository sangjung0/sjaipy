from __future__ import annotations
from typing import TYPE_CHECKING

from pathlib import Path
from functools import lru_cache

from sj_ai_utils.datasets.libri_speech_asr_corpus.service import (
    search_dirs,
    select_file_from_dir,
)
from sj_ai_utils.datasets.libri_speech_asr_corpus.file_type import X, Y

if TYPE_CHECKING:
    pass

DEV_CLEAN = "dev-clean"
DEV_OTHER = "dev-other"
TEST_CLEAN = "test-clean"
TEST_OTHER = "test-other"
TRAIN_CLEAN_100 = "train-clean-100"
TRAIN_CLEAN_360 = "train-clean-360"
TRAIN_OTHER = "train-other-500"


class LibriSpeechASRCorpus:
    def __init__(self, root: Path):
        self.__root = root

    @lru_cache(maxsize=1)
    def train_clean_100_dirs(
        self, post_path: Path | str = TRAIN_CLEAN_100, excludes: tuple[str] = ()
    ) -> list[Path]:
        return search_dirs(self.__root / post_path, excludes=excludes)

    @lru_cache(maxsize=1)
    def train_clean_360_dirs(
        self, post_path: Path | str = TRAIN_CLEAN_360, excludes: tuple[str] = ()
    ) -> list[Path]:
        return search_dirs(self.__root / post_path, excludes=excludes)

    @lru_cache(maxsize=1)
    def train_other_dirs(
        self, post_path: Path | str = TRAIN_OTHER, excludes: tuple[str] = ()
    ) -> list[Path]:
        return search_dirs(self.__root / post_path, excludes=excludes)

    @lru_cache(maxsize=1)
    def dev_clean_dirs(
        self, post_path: Path | str = DEV_CLEAN, excludes: tuple[str] = ()
    ) -> list[Path]:
        return search_dirs(self.__root / post_path, excludes=excludes)

    @lru_cache(maxsize=1)
    def dev_other_dirs(
        self, post_path: Path | str = DEV_OTHER, excludes: tuple[str] = ()
    ) -> list[Path]:
        return search_dirs(self.__root / post_path, excludes=excludes)

    @lru_cache(maxsize=1)
    def test_clean_dirs(
        self, post_path: Path | str = TEST_CLEAN, excludes: tuple[str] = ()
    ) -> list[Path]:
        return search_dirs(self.__root / post_path, excludes=excludes)

    @lru_cache(maxsize=1)
    def test_other_dirs(
        self, post_path: Path | str = TEST_OTHER, excludes: tuple[str] = ()
    ) -> list[Path]:
        return search_dirs(self.__root / post_path, excludes=excludes)

    def all_dirs(self, excludes: tuple[str] = ()) -> list[Path]:
        return (
            self.train_clean_100_dirs(excludes=excludes)
            + self.train_clean_360_dirs(excludes=excludes)
            + self.train_other_dirs(excludes=excludes)
            + self.dev_clean_dirs(excludes=excludes)
            + self.dev_other_dirs(excludes=excludes)
            + self.test_clean_dirs(excludes=excludes)
            + self.test_other_dirs(excludes=excludes)
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
    def train_clean_100(
        self,
        post_path: Path | str = TRAIN_CLEAN_100,
        source_file_type: str = X,
        truth_file_type: str = Y,
        excludes: tuple[str] = (),
    ) -> list[dict[str, Path]]:
        return self.__generate_items(
            self.train_clean_100_dirs(post_path, excludes=excludes),
            source_file_type,
            truth_file_type,
        )

    @lru_cache(maxsize=1)
    def train_clean_360(
        self,
        post_path: Path | str = TRAIN_CLEAN_360,
        source_file_type: str = X,
        truth_file_type: str = Y,
        excludes: tuple[str] = (),
    ) -> list[dict[str, Path]]:
        return self.__generate_items(
            self.train_clean_360_dirs(post_path, excludes=excludes),
            source_file_type,
            truth_file_type,
        )

    @lru_cache(maxsize=1)
    def train_other(
        self,
        post_path: Path | str = TRAIN_OTHER,
        source_file_type: str = X,
        truth_file_type: str = Y,
        excludes: tuple[str] = (),
    ) -> list[dict[str, Path]]:
        return self.__generate_items(
            self.train_other_dirs(post_path, excludes=excludes),
            source_file_type,
            truth_file_type,
        )

    @lru_cache(maxsize=1)
    def dev_clean(
        self,
        post_path: Path | str = DEV_CLEAN,
        source_file_type: str = X,
        truth_file_type: str = Y,
        excludes: tuple[str] = (),
    ) -> list[dict[str, Path]]:
        return self.__generate_items(
            self.dev_clean_dirs(post_path, excludes=excludes),
            source_file_type,
            truth_file_type,
        )

    @lru_cache(maxsize=1)
    def dev_other(
        self,
        post_path: Path | str = DEV_OTHER,
        source_file_type: str = X,
        truth_file_type: str = Y,
        excludes: tuple[str] = (),
    ) -> list[dict[str, Path]]:
        return self.__generate_items(
            self.dev_other_dirs(post_path, excludes=excludes),
            source_file_type,
            truth_file_type,
        )

    @lru_cache(maxsize=1)
    def test_clean(
        self,
        post_path: Path | str = TEST_CLEAN,
        source_file_type: str = X,
        truth_file_type: str = Y,
        excludes: tuple[str] = (),
    ) -> list[dict[str, Path]]:
        return self.__generate_items(
            self.test_clean_dirs(post_path, excludes=excludes),
            source_file_type,
            truth_file_type,
        )

    @lru_cache(maxsize=1)
    def test_other(
        self,
        post_path: Path | str = TEST_OTHER,
        source_file_type: str = X,
        truth_file_type: str = Y,
        excludes: tuple[str] = (),
    ) -> list[dict[str, Path]]:
        return self.__generate_items(
            self.test_other_dirs(post_path, excludes=excludes),
            source_file_type,
            truth_file_type,
        )


__all__ = [
    "trans_txt_to_sclite_trn",
    "generate_all_ref_and_hyp_file",
    "make_ref_and_hyp",
    "search_all_data",
]
