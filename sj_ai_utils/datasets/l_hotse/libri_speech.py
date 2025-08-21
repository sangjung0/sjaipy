from pathlib import Path
from lhotse import RecordingSet, SupervisionSet
from lhotse.recipes.librispeech import download_librispeech, prepare_librispeech

from sj_ai_utils.datasets.l_hotse.l_hotse_dataset import LHotseDataset

DEFAULT_SAMPLE_RATE = 16_000


class LibriSpeech:
    def __init__(self, path: Path):
        self.__path = path
        self.__prepare_out = path / ".prepare"

    def download(self, dataset_parts: str = "librispeech", **kwargs) -> Path:
        return download_librispeech(
            target_dir=self.__path, dataset_parts=dataset_parts, **kwargs
        )

    def prepare(self, **kwargs) -> dict[str, dict[str, RecordingSet | SupervisionSet]]:
        # mini-librispeech 에서도 동작하는지 확인 안함
        return prepare_librispeech(
            self.__path / "LibriSpeech", output_dir=self.__prepare_out, **kwargs
        )

    def __load_set(self, set_name: str) -> LHotseDataset:
        recording_set = RecordingSet.from_file(
            self.__prepare_out / f"librispeech_recordings_{set_name}.jsonl.gz"
        )
        supervision_set = SupervisionSet.from_file(
            self.__prepare_out / f"librispeech_supervisions_{set_name}.jsonl.gz"
        )
        return LHotseDataset(recording_set, supervision_set, sr=DEFAULT_SAMPLE_RATE)

    def load_train_clean_100(self) -> LHotseDataset:
        return self.__load_set("train-clean-100")

    def load_train_clean_360(self) -> LHotseDataset:
        return self.__load_set("train-clean-360")

    def load_train_other_500(self) -> LHotseDataset:
        return self.__load_set("train-other-500")

    def load_dev_clean(self) -> LHotseDataset:
        return self.__load_set("dev-clean")

    def load_dev_other(self) -> LHotseDataset:
        return self.__load_set("dev-other")

    def load_test_clean(self) -> LHotseDataset:
        return self.__load_set("test-clean")

    def load_test_other(self) -> LHotseDataset:
        return self.__load_set("test-other")
