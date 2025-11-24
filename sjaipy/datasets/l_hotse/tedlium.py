from pathlib import Path
from lhotse import RecordingSet, SupervisionSet
from lhotse.recipes.tedlium import download_tedlium, prepare_tedlium

from sjaipy.datasets.l_hotse.l_hotse_dataset import LHotseDataset
from sjaipy.datasets.dataset import Task

DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_TASK = ("asr",)


class TedliumDataset(LHotseDataset):
    pass


class Tedlium:
    def __init__(self, path: Path, prepare_path: Path | None = None):
        self.__path = path
        self.__prepare_out = prepare_path or path / ".prepare"

    def download(self, **kwargs) -> Path:
        return download_tedlium(target_dir=self.__path, **kwargs)

    def prepare(self, **kwargs) -> dict[str, dict[str, RecordingSet | SupervisionSet]]:
        return prepare_tedlium(
            self.__path / "TEDLIUM_release-3", output_dir=self.__prepare_out, **kwargs
        )

    def __load_set(
        self, set_name: str, sr: int, task: tuple[Task, ...]
    ) -> TedliumDataset:
        recording_set = RecordingSet.from_file(
            self.__prepare_out / f"tedlium_recordings_{set_name}.jsonl.gz"
        )
        supervision_set = SupervisionSet.from_file(
            self.__prepare_out / f"tedlium_supervisions_{set_name}.jsonl.gz"
        )
        return TedliumDataset.from_recording_supervision(
            recording_set, supervision_set, sr, task
        )

    def load_train(
        self,
        sr: int = DEFAULT_SAMPLE_RATE,
        task: tuple[Task, ...] = DEFAULT_TASK,
    ) -> TedliumDataset:
        return self.__load_set("train", sr, task)

    def load_dev(
        self, sr: int = DEFAULT_SAMPLE_RATE, task: tuple[Task, ...] = DEFAULT_TASK
    ) -> TedliumDataset:
        return self.__load_set("dev", sr, task)

    def load_test(
        self, sr: int = DEFAULT_SAMPLE_RATE, task: tuple[Task, ...] = DEFAULT_TASK
    ) -> TedliumDataset:
        return self.__load_set("test", sr, task)


__all__ = ["Tedlium", "TedliumDataset"]
