import pytest
import numpy as np

from sjaipy.datasets import Dataset, Sample, ConcatDataset

from tests.unit.datasets.dataset._dummy_dataset import _DummyDataset


class _MixinDatasetTest:
    # need dataset, sample_rate, task, samples

    def test_sample_rate(self, dataset: Dataset, sample_rate: int):
        assert dataset.sr == sample_rate

        test_sr = 8000
        dataset.sr = test_sr
        assert dataset.sr == test_sr
        dataset.sr = sample_rate  # reset

    def test_args(self, dataset: Dataset, sample_rate: int, task: tuple[str, ...]):
        args = dataset.args
        assert args["sr"] == sample_rate
        assert args["task"] == task

        with pytest.raises(AttributeError):
            dataset.args = {}

    def test_length(self, dataset: Dataset, samples: list[Sample]):
        assert dataset.length == len(samples)

        with pytest.raises(AttributeError):
            dataset.length = 100

    def test_task_asr(self, dataset: Dataset, samples: list[Sample], task: tuple[str, ...]):
        assert dataset.task == task

        test_task = ("asr",)
        dataset.task = test_task
        sample = dataset[0]
        assert sample.ASR == samples[0].ASR
        assert dataset.task == test_task

    def test_task_diarization(self, dataset: Dataset, samples: list[Sample], task: tuple[str, ...]):
        assert dataset.task == task

        test_task = ("diarization",)
        dataset.task = test_task
        sample = dataset[0]
        assert sample.diarization == samples[0].diarization
        assert dataset.task == test_task

    def test__iter__(self, dataset: Dataset, samples: list[Sample]):
        for i, sample in enumerate(dataset):
            assert sample == samples[i]

    def test__getitem__(self, dataset: Dataset, samples: list[Sample]):
        # int
        for i in range(len(samples)):
            assert dataset[i] == samples[i]

        # slice
        sl = slice(-len(samples) // 3, len(samples) // 2, 2)
        sliced_dataset = dataset[sl]
        assert isinstance(sliced_dataset, Dataset)
        assert sliced_dataset.samples_to_list() == samples[sl]

        # fancy indexing
        indices = [i for i in range(0, len(samples), 5)]
        indexed_dataset = dataset[indices]
        assert isinstance(indexed_dataset, Dataset)
        assert indexed_dataset.samples_to_list() == [samples[i] for i in indices]

    def test__add__(
        self,
        dataset: Dataset,
        samples: list[Sample],
        sample_rate: int,
        task: tuple[str, ...],
    ):
        # __add__
        other = _DummyDataset(
            samples=[
                Sample(
                    id=str(i),
                    load_audio=np.array([i]),
                    Y={"asr": f"text_{i}", "diarization": f"dia_{i}"},
                )
                for i in range(50, 100)
            ],
            sr=sample_rate,
            task=task,
        )

        ## other: Dataset
        concat_dataset = dataset + other
        assert isinstance(concat_dataset, Dataset)
        assert len(concat_dataset) == len(samples) + len(other.samples)
        assert concat_dataset.samples_to_list() == samples + other.samples

        ## other: ConcatDataset
        concat_other = ConcatDataset([other])
        concat_dataset = dataset + concat_other
        assert isinstance(concat_dataset, Dataset)
        assert len(concat_dataset) == len(samples) + len(other.samples)
        assert concat_dataset.samples_to_list() == samples + other.samples

    def test__len__(self, dataset: Dataset, samples: list[Sample]):
        assert len(dataset) == len(samples)

    def test_to_dict(self, dataset: Dataset, sample_rate: int, task: tuple[str, ...]):
        d = dataset.to_dict()
        assert d["sr"] == sample_rate
        assert d["task"] == task

    def test_from_dict(
        self,
        dataset: Dataset,
        samples: list[Sample],
        sample_rate: int,
        task: tuple[str, ...],
    ):
        d = dataset.to_dict()
        new_dataset = type(dataset).from_dict(d)
        assert isinstance(new_dataset, type(dataset))
        assert new_dataset.sr == sample_rate
        assert new_dataset.task == task
        assert new_dataset.samples_to_list() == samples

    def test_samples_to_list(self, dataset: Dataset, samples: list[Sample]):
        assert dataset.samples_to_list() == samples

    def test_select(
        self,
        dataset: Dataset,
        samples: list[Sample],
    ):
        indices = [i for i in range(0, len(samples), 5)]
        selected_dataset = dataset.select(indices)
        validate_dataset = dataset[indices]
        assert isinstance(selected_dataset, Dataset)
        assert selected_dataset.samples_to_list() == validate_dataset.samples_to_list()

    def test_slice(
        self,
        dataset: Dataset,
        samples: list[Sample],
    ):
        sl = slice(-len(samples) // 3, len(samples) // 2, 2)
        sliced_dataset = dataset.slice(sl.start, sl.stop, sl.step)
        validate_dataset = dataset[sl]
        assert isinstance(sliced_dataset, Dataset)
        assert sliced_dataset.samples_to_list() == validate_dataset.samples_to_list()

    def test_get(self, dataset: Dataset, samples: list[Sample]):
        for i in range(len(samples)):
            dataset.get(i) == samples[i]
        with pytest.raises(IndexError):
            dataset.get(len(samples))
        with pytest.raises(IndexError):
            dataset.get(-1)

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

        # with rng
        rng = np.random.default_rng(seed=42)
        size = len(samples) // 5
        start = np.random.randint(0, len(samples) - size)
        sampled_dataset = dataset.sample(size=size, start=start, rng=rng)
        assert isinstance(sampled_dataset, Dataset)
        assert len(sampled_dataset) == size

    def test_sample_identity(self, dataset: Dataset):
        assert len(dataset) == len(set(sample.id for sample in dataset))
