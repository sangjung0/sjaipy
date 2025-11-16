import pytest
import numpy as np

from sjaipy.datasets import Sample

from tests.unit.datasets.dataset._dummy_dataset import _DummyDataset
from tests.unit.datasets.dataset._mixin_dataset_test import _MixinDatasetTest


class TestDataset(_MixinDatasetTest):
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
    def dataset(self, samples: list[Sample], sample_rate: int, task: tuple[str, ...]):
        return _DummyDataset(samples=samples, sr=sample_rate, task=task)
