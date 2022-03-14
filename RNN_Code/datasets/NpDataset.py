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


class PreprocessedNpDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, wndsze=1, clipping=False, sample_size=100,
                 sample_type='stratified', transform=None, target_transform=None, store_as_tensor=False):
        """
        A np dataset with common prepocessing functions.

        :param X: input ndarray of shape N, L, H
        :param y: target ndarray of shape N,
        :param wndsze: window size for temporal preprocessing (ie average every 2 points)
        :param clipping: optionally cut the dataset in half
        :param sample_size: random sample size from the dataset
        :param sample_type type of sample. 'stratified' for sample every few points, 'random' for a random sample from
            the sequence. 'none' for no sampling
        :param transform: transform to be applied to the feature vectors
        :param target_transform: transform to be applied to targets (y)
        :param store_as_tensor: store as a pytorch tensor
        """
        self.store_as_tensor = store_as_tensor
        if store_as_tensor:
            self.X = torch.Tensor(X)
        else:
            self.X = X
        self.y = torch.Tensor(y).long()
        self.wndsze = wndsze
        self.clipping = clipping
        self.sample_size = sample_size
        self.sample_type = sample_type
        self.transform = transform
        self.target_transform = target_transform

        # first, clipping
        if clipping:
            N, L, H = self.X.shape
            self.X = self.X[:, :int(L // 2), :]

        # next, temporal averaging
        N, L, H = self.X.shape
        X = self.X.reshape(N, int(L // wndsze), int(wndsze), H)
        self.X = X.mean(axis=2)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        # load x and y
        x, y = self.X[index], self.y[index]

        # take a sample
        if self.sample_type == "stratified":
            L, H = x.shape
            L2 = int(L // self.sample_size)
            x = x.reshape(L2, self.sample_size, H)
            s = np.random.choice(self.sample_size, size=L2)
            x = x[np.arange(L2), s]
        elif self.sample_type == "random":
            L, H = x.shape
            s = np.random.choice(L, size=self.sample_size, replace=False)
            s.sort()
            x = x[s]
        elif self.sample_type == "none":
            # do nothing
            pass
        else:
            raise NotImplementedError("unknown sample type")

        # transform
        if self.transform is not None:
            x = self.transform(x)
        # target transform
        if self.target_transform is not None:
            y = self.target_transform(y)

        return torch.Tensor(x), y

