import pytest

from pathlib import Path
from typing_extensions import override

from sjaipy.datasets import Dataset, Sample
from sjaipy.datasets.l_hotse import VoxPopuli, LHotseDataset

from tests.unit.datasets.dataset._mixin_dataset_test import _MixinDatasetTest


PATH = "/workspaces/dev/.datasets/vox_populi"


class TestVoxPopuli(_MixinDatasetTest):
    @pytest.fixture
    def vox_populi(self) -> VoxPopuli:
        return VoxPopuli(Path(PATH))

    @pytest.fixture(
        params=(
            "load_train_asr_en",
            "load_dev_asr_en",
            "load_test_asr_en",
        )
    )
    def dataset(
        self, vox_populi: VoxPopuli, request: pytest.FixtureRequest
    ) -> LHotseDataset:
        return getattr(vox_populi, request.param)(task=("asr", "diarization"))

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
