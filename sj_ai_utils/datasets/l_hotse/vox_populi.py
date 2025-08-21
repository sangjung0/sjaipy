from pathlib import Path
from lhotse import RecordingSet, SupervisionSet
from lhotse.recipes.voxpopuli import download_voxpopuli, prepare_voxpopuli

from sj_ai_utils.datasets.l_hotse.l_hotse_dataset import LHotseDataset

DEFAULT_SAMPLE_RATE = 16_000


class VoxPopuli:
    def __init__(self, path: Path):
        self.__path = path
        self.__prepare_out = path / ".prepare"

    def download(self, subset="en") -> Path:
        return download_voxpopuli(target_dir=self.__path, subset=subset)

    def prepare(self, **kwargs) -> dict[str, dict[str, RecordingSet | SupervisionSet]]:
        return prepare_voxpopuli(self.__path, output_dir=self.__prepare_out, **kwargs)

    def __load_set(self, set_name: str, subset, lang) -> LHotseDataset:
        recording_set = RecordingSet.from_file(
            self.__prepare_out
            / f"voxpopuli-{subset}-{lang}_recordings_{set_name}.jsonl.gz"
        )
        supervision_set = SupervisionSet.from_file(
            self.__prepare_out
            / f"voxpopuli-{subset}-{lang}_supervisions_{set_name}.jsonl.gz"
        )
        return LHotseDataset(recording_set, supervision_set)

    def load_train_asr_en(self):
        return self.__load_set("train", "asr", "en")

    def load_dev_asr_en(self):
        return self.__load_set("dev", "asr", "en")

    def load_test_asr_en(self):
        return self.__load_set("test", "asr", "en")
