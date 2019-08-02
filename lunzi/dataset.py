import numpy as np
import numpy.lib.recfunctions
from collections import namedtuple


class BaseDataset(object):
    def __getitem__(self, item):
        raise NotImplementedError

    def apply(self, transforms):
        return TransformedDataset(self, transforms)

    @property
    def size(self):
        raise NotImplementedError


class Dataset(np.recarray, BaseDataset):
    def __init__(self, dtype, size):
        super().__init__()
        self.resize(size)

    @staticmethod
    def fromdict(**arrays):
        array_list = [np.asarray(x) for x in arrays.values()]
        rank = 0
        for rank in range(len(array_list[0].shape) + 1):
            ok = True
            for array in array_list:
                if len(array.shape) <= rank or array.shape[rank] != array_list[0].shape[rank]:
                    ok = False
                    break
            if not ok:
                break

        dtypes = []
        for name, array in zip(arrays.keys(), array_list):
            dtypes.append((name, (array.dtype, array.shape[rank:])))
        return np.rec.fromarrays(array_list, dtypes).view(Dataset)

    def __new__(cls, dtype, size):
        return np.recarray.__new__(cls, size, dtype=dtype)

    def to_dict(self):
        return {name: self[name] for name in self.dtype.names}

    def to_torch(self):
        import torch
        return {name: torch.tensor(self[name].copy()) for name in self.dtype.names}

    @property
    def size(self):
        return len(self)

    def sample(self, size, indices=None):
        if indices is None:
            indices = np.random.randint(0, self.size, size=size)
        return self[indices]

    def append_fields(self, names, data, dtype=None):
        if isinstance(names, str) and names in self.dtype.names:
            self[names] = data
            return self
        return np.lib.recfunctions.append_fields(self, names, data, dtype, usemask=False).view(Dataset)

    def drop_fields(self, names):
        return np.lib.recfunctions.drop_fields(self, names, usemask=False).view(Dataset)

    def iterator(self, batch_size, shuffle=True):
        indices = np.arange(self.size, dtype=np.int32)
        if shuffle:
            np.random.shuffle(indices)
        index = 0
        while index < self.size:
            end = min(index + batch_size, self.size)
            yield self[indices[index:end]]
            index = end

    def sample_iterator(self, batch_size, n_iters=0):
        while True:
            yield self.sample(batch_size)
            n_iters = max(n_iters - 1, -1)
            if n_iters == 0:
                break


class TransformedDataset(BaseDataset):
    def __init__(self, dataset: BaseDataset, transform):
        if isinstance(dataset, TransformedDataset):
            self._dataset = dataset._dataset
            self._transforms = dataset._transforms + [transform]
        else:
            self._dataset = dataset
            self._transforms = [transform]

    def __getitem__(self, items):
        items = self._dataset[items]
        for transform in self._transforms:
            items = transform(items)
        return items

    @property
    def size(self):
        return self._dataset.size


class ExtendableDataset(Dataset):
    """
        Overallocation can be supported, by making examinations before
        each `append` and `extend`.
    """

    def __init__(self, dtype, max_size, verbose=False):
        super().__init__(dtype, max_size)
        self.max_size = max_size
        self._index = 0
        self._buf_size = 0
        self._len = 0
        self._buf_size = max_size

    def __new__(cls, dtype, max_size):
        return np.recarray.__new__(cls, max_size, dtype=dtype)

    @property
    def size(self):
        return self._len

    def reserve(self, size):
        cur_size = max(self._buf_size, 1)
        while cur_size < size:
            cur_size *= 2
        if cur_size != self._buf_size:
            self.resize(cur_size)

    def clear(self):
        self._index = 0
        self._len = 0
        return self

    def append(self, item):
        self[self._index] = item
        self._index = (self._index + 1) % self.max_size
        self._len = min(self._len + 1, self.max_size)
        return self

    def extend(self, items):
        n_new = len(items)
        if n_new > self.max_size:
            items = items[-self.max_size:]
            n_new = self.max_size

        n_tail = self.max_size - self._index
        if n_new <= n_tail:
            self[self._index:self._index + n_new] = items
        else:
            n_head = n_new - n_tail
            self[self._index:] = items[:n_tail]
            self[:n_head] = items[n_tail:]

        self._index = (self._index + n_new) % self.max_size
        self._len = min(self._len + n_new, self.max_size)
        return self
