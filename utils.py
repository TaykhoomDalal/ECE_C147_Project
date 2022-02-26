import numpy as np
import os


def load_data(data_root='data/'):
    """
    Loads the project data as a dict of np arrays

    :param data_root: the source where the np arrays are located
    :return: data - a dict mapping the data name to the np array
    """

    data = {}
    for name in os.listdir(data_root):
        ns = name.split('.')
        if ns[-1] == 'npy' and len(ns) == 2:
            data[ns[0]] = np.load(os.path.join(data_root, name))

    return data

