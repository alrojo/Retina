import numpy as np
import cPickle as pickle
from sklearn.cross_validation import StratifiedShuffleSplit

import data

# We have to take into account the eye pairing. We don't want the left and the
# right eye of the same patient to be in different subsets.
# Luckily the images are nicely ordered in pairs.
# data.labels_train[::2] gives the left images
# data.labels_train[1::2] gives the right images

idcs_train, idcs_valid = iter(StratifiedShuffleSplit(data.labels_train[::2], n_iter=1, test_size=0.2, random_state=42)).next()

idcs_train = np.concatenate([idcs_train * 2, idcs_train * 2 + 1]) # left + right
idcs_valid = np.concatenate([idcs_valid * 2, idcs_valid * 2 + 1]) # left + right


# assert that no patient is in both train and validation subsets
patients_train = set([int(f.split("_")[0]) for f in data.filenames_train[idcs_train]])
patients_valid = set([int(f.split("_")[0]) for f in data.filenames_train[idcs_valid]])

assert len(patients_train & patients_valid) == 0 # empty intersection


with open("data/split.pkl", 'w') as f:
    pickle.dump({
            'train': idcs_train,
            'valid': idcs_valid,
        }, f, pickle.HIGHEST_PROTOCOL)
