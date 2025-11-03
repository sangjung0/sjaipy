import pytest

from pathlib import Path
from typing_extensions import override

from sjaipy.datasets import Dataset, Sample, Task
from sjaipy.datasets.l_hotse import LibriSpeech, LHotseDataset

from tests.unit.datasets.dataset._mixin_dataset_test import _MixinDatasetTest


PATH = "/workspaces/dev/.datasets/libri_speech"


class TestLibriSpeech(_MixinDatasetTest):
    @pytest.fixture
    def libri_speech(self) -> LibriSpeech:
        return LibriSpeech(Path(PATH))

    @pytest.fixture(
        params=(
            "load_train_clean_100",
            "load_train_clean_360",
            "load_train_other_500",
            "load_dev_clean",
            "load_dev_other",
            "load_test_clean",
            "load_test_other",
        )
    )
    def dataset(
        self, libri_speech: LibriSpeech, request: pytest.FixtureRequest
    ) -> LHotseDataset:
        return getattr(libri_speech, request.param)(task=("asr", "diarization"))

    @pytest.fixture
    def sample_rate(self, dataset: Dataset):
        return dataset.sr

    @pytest.fixture
    def task(self, dataset: Dataset):
        return dataset.task

    @pytest.fixture
    def samples(self, dataset: Dataset):
        return [sample for sample in dataset]

    @override
    def test_get(self, dataset: Dataset, samples: list[Sample]):
        for i in range(len(samples)):
            assert dataset.get(i) == samples[i]
        dataset.get(-1)
        with pytest.raises(IndexError):
            dataset.get(len(samples))

    @override
    def test_task_diarization(
        self, dataset: Dataset, samples: list[Sample], task: Task
    ):
        return  # LibriSpeech does not support diarization task
