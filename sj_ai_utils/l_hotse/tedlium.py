from pathlib import Path
from lhotse import RecordingSet, SupervisionSet
from lhotse.recipes.tedlium import download_tedlium, prepare_tedlium


class Tedlium:
    def __init__(self, path: Path):
        self.__path = path
        self.__prepare_out = path / ".prepare"

    def download(self, **kwargs) -> Path:
        return download_tedlium(target_dir=self.__path, **kwargs)

    def prepare(self, **kwargs) -> dict[str, dict[str, RecordingSet | SupervisionSet]]:
        return prepare_tedlium(self.__path, output_dir=self.__prepare_out, **kwargs)

    def __load_set(self, set_name: str) -> tuple[RecordingSet, SupervisionSet]:
        recording_set = RecordingSet.from_file(
            self.__prepare_out / f"tedlium_recordings_{set_name}.jsonl.gz"
        )
        supervision_set = SupervisionSet.from_file(
            self.__prepare_out / f"tedlium_supervisions_{set_name}.jsonl.gz"
        )
        return recording_set, supervision_set

    def load_train(self) -> tuple[RecordingSet, SupervisionSet]:
        return self.__load_set("train")

    def load_dev(self) -> tuple[RecordingSet, SupervisionSet]:
        return self.__load_set("dev")

    def load_test(self) -> tuple[RecordingSet, SupervisionSet]:
        return self.__load_set("test")
