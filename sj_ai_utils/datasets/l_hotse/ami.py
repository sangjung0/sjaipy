from pathlib import Path
from lhotse import RecordingSet, SupervisionSet
from lhotse.recipes.ami import prepare_ami, download_ami

from sj_ai_utils.datasets.l_hotse.l_hotse_dataset import LHotseDataset

DEFAULT_SAMPLE_RATE = 16_000


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

    def __load_set(self, mic: str, set_name: str) -> LHotseDataset:
        recording_set = RecordingSet.from_file(
            self.__prepare_out / f"ami-{mic}_recordings_{set_name}.jsonl.gz"
        )
        supervision_set = SupervisionSet.from_file(
            self.__prepare_out / f"ami-{mic}_supervisions_{set_name}.jsonl.gz"
        )
        return LHotseDataset(recording_set, supervision_set, sr=DEFAULT_SAMPLE_RATE)

    def load_train_ihm(self) -> LHotseDataset:
        return self.__load_set("ihm", "train")

    def load_dev_ihm(self) -> LHotseDataset:
        return self.__load_set("ihm", "dev")

    def load_test_ihm(self) -> LHotseDataset:
        return self.__load_set("ihm", "test")

    def load_train_sdm(self) -> LHotseDataset:
        return self.__load_set("sdm", "train")

    def load_dev_sdm(self) -> LHotseDataset:
        return self.__load_set("sdm", "dev")

    def load_test_sdm(self) -> LHotseDataset:
        return self.__load_set("sdm", "test")
