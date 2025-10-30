import pytest
import numpy as np

from typing_extensions import override
from typing import Sequence

from sjaipy.datasets import Sample, Dataset
from sjaipy.datasets.dataset import ConcatDataset


# dummy
class DummyDataset(Dataset):
    def __init__(self, samples: list[Sample], sr: int, task: tuple[str, ...]):
        super().__init__(sr, task)
        self.samples = samples

    @Dataset.args.getter
    def args(self) -> dict:
        return {
            **super().args,
            "samples": self.samples,
        }

    @Dataset.length.getter
    def length(self) -> int:
        return len(self.samples)

    @override
    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "samples": [s.to_dict() for s in self.samples],
        }

    @override
    def select(self, indices: Sequence[int]) -> "DummyDataset":
        samples = [self.samples[i] for i in indices]
        args = self.args
        args["samples"] = samples
        return DummyDataset(**args)

    @override
    def slice(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> "DummyDataset":
        samples = self.samples[start:stop:step]
        args = self.args
        args["samples"] = samples
        return DummyDataset(**args)

    @override
    def get(self, idx: int) -> Sample:
        if idx < 0 or idx >= len(self.samples):
            raise IndexError("Index out of range")
        return self.samples[idx]

    @override
    def _sample(
        self,
        size: int,
        start: int = 0,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ) -> "DummyDataset":
        if rng is None or size == len(self) - start:
            sampled_samples = self.samples[start : start + size]
        else:
            indices = rng.choice(range(start, len(self)), size=size, replace=False)
            sampled_samples = [self.samples[i] for i in indices]

        args = self.args
        args["samples"] = sampled_samples
        return DummyDataset(**args)

    @staticmethod
    @override
    def from_dict(data: dict) -> "DummyDataset":
        samples = [Sample.from_dict(s) for s in data["samples"]]
        return DummyDataset(
            samples=samples,
            sr=data["sr"],
            task=tuple(data["task"]),
        )


class TestSample:
    @pytest.fixture
    def original(self):
        return Sample(
            id="test_id", audio=np.array([0.0, 1.0, -1.0]), _Y={"diarization": "world"}
        )

    @pytest.fixture
    def original_copy(self):
        return Sample(
            id="test_id",
            audio=np.array([0.0, 1.0, -1.0]),
            _Y={"asr": "hello", "diarization": "world"},
        )

    @pytest.fixture
    def other(self):
        return Sample(id="different_id", audio=np.array([0.0, 1.0]), _Y={"asr": "hi"})

    def test_equality(self, original: Sample, original_copy: Sample, other: Sample):
        assert original == original_copy
        assert original != other

    def test_hash(self, original: Sample, original_copy: Sample, other: Sample):
        assert hash(original) == hash(original_copy)
        assert hash(original) != hash(other)

    def test_ASR(self, original: Sample, other: Sample):
        assert other.ASR == "hi"
        with pytest.raises(AttributeError):
            original.ASR

    def test_diarization(self, original: Sample, other: Sample):
        assert original.diarization == "world"
        with pytest.raises(AttributeError):
            other.diarization


class MixinDatasetTest:
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

    def test_task(self, dataset: Dataset, samples: list[Sample], task: tuple[str, ...]):
        assert dataset.task == task

        test_task = ("asr",)
        dataset.task = test_task
        sample = dataset[0]
        assert sample.ASR == samples[0].ASR
        assert dataset.task == test_task

        test_task = ("diarization",)
        dataset.task = test_task
        sample = dataset[0]
        assert sample.diarization == samples[0].diarization
        assert dataset.task == test_task

        dataset.task = task  # reset

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
        other = DummyDataset(
            samples=[
                Sample(
                    id=str(i),
                    audio=np.array([i]),
                    _Y={"asr": f"text_{i}", "diarization": f"dia_{i}"},
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
            assert dataset.get(i) == samples[i]
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

    def test_sample_identity(self, all_dataset: Dataset):
        assert len(all_dataset) == len(set(sample.id for sample in all_dataset))


class TestDataset(MixinDatasetTest):
    @pytest.fixture
    def samples(self):
        return [
            Sample(
                id=str(i),
                audio=np.array([i]),
                _Y={"asr": f"text_{i}", "diarization": f"dia_{i}"},
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
    def dataset(self, samples: list[Sample], sample_rate: int, task: tuple[str, ...]):
        return DummyDataset(samples=samples, sr=sample_rate, task=task)

    @pytest.fixture
    def all_dataset(self):
        return DummyDataset(
            samples=[
                Sample(
                    id=str(i),
                    audio=np.array([i]),
                    _Y={"asr": f"text_{i}", "diarization": f"dia_{i}"},
                )
                for i in range(100)
            ],
            sr=16000,
            task=("asr", "diarization"),
        )


class TestConcatDataset(TestDataset):
    @pytest.fixture
    def samples(self):
        return [
            Sample(
                id=str(i),
                audio=np.array([i]),
                _Y={"asr": f"text_{i}", "diarization": f"dia_{i}"},
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
        subset_1 = DummyDataset(samples=samples[:25], sr=sample_rate, task=task)
        subset_2 = DummyDataset(samples=samples[25:], sr=sample_rate, task=task)
        return [subset_1, subset_2]

    @pytest.fixture
    def dataset(self, subset: list[Dataset]):
        return ConcatDataset(datasets=subset)

    @pytest.fixture
    def all_dataset(self):
        return DummyDataset(
            samples=[
                Sample(
                    id=str(i),
                    audio=np.array([i]),
                    _Y={"asr": f"text_{i}", "diarization": f"dia_{i}"},
                )
                for i in range(50)
            ],
            sr=16000,
            task=("asr", "diarization"),
        )

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
