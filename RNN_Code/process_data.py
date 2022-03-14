from utils import load_data
import numpy as np
import os

data_root = 'data/'
normalized_root = 'normalized/'

# load data
data = load_data(os.path.join(data_root, 'project/'))

# reshape data
data['X_train_valid'] = np.transpose(data['X_train_valid'], axes=(0, 2, 1))
data['X_test'] = np.transpose(data['X_test'], axes=(0, 2, 1))
data['person_train_valid'] = np.squeeze(data['person_train_valid'])
data['person_test'] = np.squeeze(data['person_test'])

# normalize data per channel
mean = np.mean(data['X_train_valid'], axis=(0, 1))
std = np.std(data['X_train_valid'], axis=(0, 1))
data_norm = {
    "X_train_valid": (data['X_train_valid'] - mean) / std,
    "y_train_valid": data['y_train_valid'],
    "person_train_valid": data['person_train_valid'],
    "X_test": (data['X_test'] - mean) / std,
    "y_test": data['y_test'],
    "person_test": data['person_test']
}

# save data in 'data/'
for key, val in data.items():
    np.save(os.path.join(data_root, key), val)

# create a normalized dir
try:
    os.mkdir(os.path.join(data_root, normalized_root))
except FileExistsError as e:
    pass

# save data in normalized
norm_dir = os.path.join(data_root, normalized_root)
for key, val in data_norm.items():
    np.save(os.path.join(norm_dir, key), val)