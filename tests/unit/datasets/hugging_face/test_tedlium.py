import pytest

from sjaipy.datasets import Dataset
from sjaipy.datasets.hugging_face import AMI, HuggingFaceDataset

from tests.unit.datasets.dataset._mixin_dataset_test import _MixinDatasetTest


class TestAMI(_MixinDatasetTest):
    @pytest.fixture
    def ami(self) -> AMI:
        return AMI()

    @pytest.fixture(params=("train", "validation", "test"))
    def dataset(self, ami: AMI, request: pytest.FixtureRequest) -> HuggingFaceDataset:
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
