import pytest
import numpy as np

from typing_extensions import override

from sjaipy.datasets import Sample, Dataset, ConcatDataset

from tests.unit.datasets.dataset._dummy_dataset import _DummyDataset
from tests.unit.datasets.dataset.test_dataset import TestDataset


class TestConcatDataset(TestDataset):
    @pytest.fixture
    def samples(self):
        return [
            Sample(
                id=str(i),
                load_audio=np.array([i]),
                Y={"asr": f"text_{i}", "diarization": f"dia_{i}"},
            )
            for i in range(50)
        ]

    @pytest.fixture
    def sample_rate(self):
        return 16000

    @pytest.fixture
    def task(self):
        return ("asr", "diarization")

    @pytest.fixture
    def subset(self, samples: list[Sample], sample_rate: int, task: tuple[str, ...]):
        subset_1 = _DummyDataset(samples=samples[:25], sr=sample_rate, task=task)
        subset_2 = _DummyDataset(samples=samples[25:], sr=sample_rate, task=task)
        return [subset_1, subset_2]

    @pytest.fixture
    def dataset(self, subset: list[Dataset]):
        return ConcatDataset(datasets=subset)

    @override
    def test_args(
        self,
        dataset: Dataset,
        sample_rate: int,
        task: tuple[str, ...],
        subset: list[Dataset],
    ):
        super().test_args(dataset, sample_rate, task)
        assert dataset.args["datasets"] == subset

    @override
    def test__getitem__(self, dataset: Dataset, samples: list[Sample]):
        # int
        for i in range(len(samples)):
            assert dataset[i] == samples[i]

        # slice
        sl = slice(len(samples) // 5, len(samples) // 2, 2)
        sliced_dataset = dataset[sl]
        assert isinstance(sliced_dataset, Dataset)
        assert sliced_dataset.samples_to_list() == samples[sl]

        # fancy indexing
        indices = [i for i in range(0, len(samples), 5)]
        indexed_dataset = dataset[indices]
        assert isinstance(indexed_dataset, Dataset)
        assert indexed_dataset.samples_to_list() == [samples[i] for i in indices]

    @override
    def test_slice(
        self,
        dataset: Dataset,
        samples: list[Sample],
    ):
        sl = slice(len(samples) // 5, len(samples) // 2, 2)
        sliced_dataset = dataset.slice(sl.start, sl.stop, sl.step)
        validate_dataset = dataset[sl]
        assert isinstance(sliced_dataset, Dataset)
        assert sliced_dataset.samples_to_list() == validate_dataset.samples_to_list()

    @override
    def test_sample(
        self,
        dataset: Dataset,
        samples: list[Sample],
    ):
        # without rng
        size = len(samples) // 5
        start = np.random.randint(0, len(samples) - size)
        sampled_dataset = dataset.sample(size=size, start=start)
        assert isinstance(sampled_dataset, Dataset)
        assert sampled_dataset.samples_to_list() == samples[start : start + size]

        with pytest.raises(IndexError):
            dataset.sample(size=size, start=-1)
        with pytest.raises(IndexError):
            dataset.sample(size=size, start=len(samples) + 1)
