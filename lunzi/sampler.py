import abc
import numpy as np


class BaseSampler(abc.ABC):
    @abc.abstractmethod
    def __iter__(self):
        pass

    def __len__(self):
        raise NotImplementedError


class BatchSampler(BaseSampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def _iterator(self):
        indices = np.arange(self.dataset.size, dtype=np.int32)
        if self.shuffle:
            np.random.shuffle(indices)
        index = 0
        while index < self.dataset.size:
            end = min(index + self.batch_size, self.dataset.size)
            yield self.dataset[indices[index:end]]
            index = end

    def __iter__(self):
        return self._iterator()

    def __len__(self):
        return self.dataset.size // self.batch_size

