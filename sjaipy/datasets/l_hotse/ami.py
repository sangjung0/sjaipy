from pathlib import Path
from lhotse import RecordingSet, SupervisionSet
from lhotse.recipes.ami import prepare_ami, download_ami

from sjaipy.datasets.l_hotse.l_hotse_dataset import LHotseDataset
from sjaipy.datasets.dataset import Task

DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_TASK = ("asr",)


class AMI:
    def __init__(self, path: Path):
        self.__path = path
        self.__prepare_out = path / ".prepare"

    def download(self, mic="ihm", **kwargs) -> Path:
        return download_ami(target_dir=self.__path, mic=mic, **kwargs)

    def prepare(
        self, mic="ihm", **kwargs
    ) -> dict[str, dict[str, RecordingSet | SupervisionSet]]:
        return prepare_ami(
            self.__path, output_dir=self.__prepare_out, mic=mic, **kwargs
        )

    def __load_set(
        self, mic: str, set_name: str, sr: int, task: tuple[Task, ...]
    ) -> LHotseDataset:
        recording_set = RecordingSet.from_file(
            self.__prepare_out / f"ami-{mic}_recordings_{set_name}.jsonl.gz"
        )
        supervision_set = SupervisionSet.from_file(
            self.__prepare_out / f"ami-{mic}_supervisions_{set_name}.jsonl.gz"
        )
        return LHotseDataset(recording_set, supervision_set, sr=sr, task=task)

    def load_train_ihm(
        self, sr: int = DEFAULT_SAMPLE_RATE, task: tuple[Task, ...] = DEFAULT_TASK
    ) -> LHotseDataset:
        return self.__load_set("ihm", "train", sr=sr, task=task)

    def load_dev_ihm(
        self, sr: int = DEFAULT_SAMPLE_RATE, task: tuple[Task, ...] = DEFAULT_TASK
    ) -> LHotseDataset:
        return self.__load_set("ihm", "dev", sr=sr, task=task)

    def load_test_ihm(
        self, sr: int = DEFAULT_SAMPLE_RATE, task: tuple[Task, ...] = DEFAULT_TASK
    ) -> LHotseDataset:
        return self.__load_set("ihm", "test", sr=sr, task=task)

    def load_train_sdm(
        self, sr: int = DEFAULT_SAMPLE_RATE, task: tuple[Task, ...] = DEFAULT_TASK
    ) -> LHotseDataset:
        return self.__load_set("sdm", "train", sr=sr, task=task)

    def load_dev_sdm(
        self, sr: int = DEFAULT_SAMPLE_RATE, task: tuple[Task, ...] = DEFAULT_TASK
    ) -> LHotseDataset:
        return self.__load_set("sdm", "dev", sr=sr, task=task)

    def load_test_sdm(
        self, sr: int = DEFAULT_SAMPLE_RATE, task: tuple[Task, ...] = DEFAULT_TASK
    ) -> LHotseDataset:
        return self.__load_set("sdm", "test", sr=sr, task=task)


__all__ = ["AMI"]
