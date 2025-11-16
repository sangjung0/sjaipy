import pytest

from pathlib import Path
from typing_extensions import override

from sjaipy.datasets import Dataset, Sample
from sjaipy.datasets.l_hotse import Tedlium, LHotseDataset

from tests.unit.datasets.dataset._mixin_dataset_test import _MixinDatasetTest


PATH = "/workspaces/dev/.datasets/tedlium"


class TestTedlium(_MixinDatasetTest):
    @pytest.fixture
    def tedlium(self) -> Tedlium:
        return Tedlium(Path(PATH))

    @pytest.fixture(
        params=(
            "load_train",
            "load_dev",
            "load_test",
        )
    )
    def dataset(
        self, tedlium: Tedlium, request: pytest.FixtureRequest
    ) -> LHotseDataset:
        return getattr(tedlium, request.param)(task=("asr", "diarization"))

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
