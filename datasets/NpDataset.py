import numpy as np
from torch.utils.data import Dataset
import torch


class NpDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, transform=None, target_transform=None, return_indices=False,
                 store_as_tensor=False):
        """
        A dataset wrapper for numpy array style datasets
        :param X: The X data of shape [n_examples, ...]
        :param y: The targets of shape [n_examples, ...]
        :param transform: Optional transform to apply to features
        :param target_transform: Optional transform to apply to targets
        """
        if store_as_tensor:
            self.X = torch.Tensor(X)
            self.y = torch.Tensor(y).long()
        else:
            self.X = X
            self.y = y
        self.transform = transform
        self.target_transform = target_transform
        self.return_indices = return_indices

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        y = self.y[index]

        # optionally transform data
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)

        # optionally return indices
        if self.return_indices:
            return x, y, index
        else:
            return x, y


class SequentialNpDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len, stride, transform=None, target_transform=None,
                 return_indices=False, store_as_tensor=False, sequential_targets=True):
        """
        A dataset wrapper for numpy array style datasets
        :param X: The X data of shape [n_examples, ...]
        :param y: The targets of shape [n_examples, ...]
        :param seq_len: The length of each subsequence to use
        :param stride: The distance between the start of each sequence
        :param transform: Optional transform to apply to features
        :param target_transform: Optional transform to apply to targets
        """
        if store_as_tensor:
            self.X = torch.Tensor(X)
            self.y = torch.Tensor(y).long()
        else:
            self.X = X
            self.y = y
        self.transform = transform
        self.target_transform = target_transform
        self.return_indices = return_indices
        self.sequential_targets = sequential_targets
        self.seq_len = seq_len
        self.stride = stride
        self._ex_len = len(X[0])
        self._seqs_per_example = int((self._ex_len - self.seq_len) / self.stride + 1)

    def __len__(self):
        return len(self.X) * self._seqs_per_example

    def __getitem__(self, index):
        ex = int(index // self._seqs_per_example)
        offset = index % self._seqs_per_example

        x = self.X[ex, offset*self.stride:(offset*self.stride + self.seq_len)]
        if self.sequential_targets:
            y = self.y[ex].repeat(self.seq_len)
        else:
            y = self.y[ex]

        # optionally transform data
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)

        # optionally return indices
        if self.return_indices:
            return x, y, index
        else:
            return x, y