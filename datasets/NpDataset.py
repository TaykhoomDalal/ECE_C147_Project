import numpy as np
from torch.utils.data import Dataset


class NpDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, transform=None, target_transform=None, return_indices=False):
        """
        A dataset wrapper for numpy array style datasets
        :param X: The X data of shape [n_examples, ...]
        :param y: The targets of shape [n_examples, ...]
        :param transform: Optional transform to apply to features
        :param target_transform: Optional transform to apply to targets
        """
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