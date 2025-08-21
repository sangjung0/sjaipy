import numpy as np

from abc import ABC, abstractmethod


class Dataset(ABC):
    @abstractmethod
    def sample(
        self,
        sample_size: int,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ) -> "Dataset": ...


__all__ = ["Dataset"]
