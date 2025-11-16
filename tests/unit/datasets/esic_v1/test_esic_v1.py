import pytest

from pathlib import Path
from typing_extensions import override

from sjaipy.datasets import Dataset, Sample, Task
from sjaipy.datasets.esic_v1 import ESICv1, ESICv1Dataset

from tests.unit.datasets.dataset._mixin_dataset_test import _MixinDatasetTest


PATH = "/workspaces/dev/.datasets/ESIC-v1.1"


class TestESICv1(_MixinDatasetTest):
    @pytest.fixture
    def esic_v1(self) -> ESICv1:
        return ESICv1(Path(PATH))

    @pytest.fixture(
        params=(
            "dev",
            "dev2",
            "test",
        )
    )
    def dataset(self, esic_v1: ESICv1, request: pytest.FixtureRequest) -> ESICv1Dataset:
        return getattr(esic_v1, request.param)()

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
    def test_task_diarization(
        self, dataset: Dataset, samples: list[Sample], task: Task
    ):
        return  # ESICv1 does not support diarization task

    @override
    def test_get(self, dataset: Dataset, samples: list[Sample]):
        for i in range(len(samples)):
            assert dataset.get(i) == samples[i]
        dataset.get(-1)
        with pytest.raises(IndexError):
            dataset.get(len(samples))
