import pytest

from pathlib import Path
from typing_extensions import override

from sjaipy.datasets import Dataset, Sample
from sjaipy.datasets.l_hotse import AMI, LHotseDataset

from tests.unit.datasets.dataset._mixin_dataset_test import _MixinDatasetTest


PATH = "/workspaces/dev/.datasets/ami"


class TestAMI(_MixinDatasetTest):
    @pytest.fixture
    def ami(self) -> AMI:
        return AMI(Path(PATH))

    @pytest.fixture(
        params=(
            "load_train_ihm",
            "load_dev_ihm",
            "load_test_ihm",
            # "load_train_sdm",
            # "load_dev_sdm",
            # "load_test_sdm",
        )
    )
    def dataset(self, ami: AMI, request: pytest.FixtureRequest) -> LHotseDataset:
        return getattr(ami, request.param)(task=("asr", "diarization"))

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
