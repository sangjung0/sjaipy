import warnings

from pathlib import Path
from lhotse import RecordingSet, SupervisionSet
from lhotse.recipes.voxpopuli import download_voxpopuli, prepare_voxpopuli

from sjaipy.datasets.l_hotse.l_hotse_dataset import LHotseDataset
from sjaipy.datasets.dataset import Task

DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_TASK = ("asr",)


class VoxPopuliDataset(LHotseDataset):
    pass


class VoxPopuli:
    def __init__(self, path: Path, prepare_path: Path | None = None):
        self.__path = path
        self.__prepare_out = prepare_path or path / ".prepare"

    def download(self, subset="en") -> Path:
        return download_voxpopuli(target_dir=self.__path, subset=subset)

    def prepare(self, **kwargs) -> dict[str, dict[str, RecordingSet | SupervisionSet]]:
        return prepare_voxpopuli(self.__path, output_dir=self.__prepare_out, **kwargs)

    def __load_set(
        self, set_name: str, subset: str, lang: str, sr: int, task: tuple[Task, ...]
    ) -> VoxPopuliDataset:
        recording_set = RecordingSet.from_file(
            self.__prepare_out
            / f"voxpopuli-{subset}-{lang}_recordings_{set_name}.jsonl.gz"
        )
        supervision_set = SupervisionSet.from_file(
            self.__prepare_out
            / f"voxpopuli-{subset}-{lang}_supervisions_{set_name}.jsonl.gz"
        )
        return VoxPopuliDataset.from_recording_supervision(
            recording_set, supervision_set, sr, task
        )

    def load_train_asr_en(
        self, sr: int = DEFAULT_SAMPLE_RATE, task: tuple[Task, ...] = DEFAULT_TASK
    ) -> VoxPopuliDataset:
        return self.__load_set("train", "asr", "en", sr, task)

    def load_dev_asr_en(
        self, sr: int = DEFAULT_SAMPLE_RATE, task: tuple[Task, ...] = DEFAULT_TASK
    ) -> VoxPopuliDataset:
        return self.__load_set("dev", "asr", "en", sr, task)

    def load_test_asr_en(
        self, sr: int = DEFAULT_SAMPLE_RATE, task: tuple[Task, ...] = DEFAULT_TASK
    ) -> VoxPopuliDataset:
        return self.__load_set("test", "asr", "en", sr, task)


if __name__ != "__main__":
    warnings.warn(
        "[INFO] VoxPopuli 오디오와 출력이 일치하지 않음. 출력이 나눠져 있음.",
        category=UserWarning,
        stacklevel=2,
    )


__all__ = ["VoxPopuli", "VoxPopuliDataset"]
