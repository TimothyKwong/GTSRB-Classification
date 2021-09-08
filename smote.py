import os
import numpy as np

class SMOTE():
    def __init__(self, threshold, train_dir, val_dir):
        self.threshold = threshold
        self.train_dir = train_dir
        self.val_dir = val_dir

    # oversamples minority classes
    def get_samples(self):
        path_train = [f.path for f in os.scandir(self.train_dir) if f.is_dir()]
        per_folder = [ [(os.path.join(path_train[i], j), i) for j in os.listdir(path_train[i])] for i in range(len(path_train)) ]

        #sample_per_folder_ = np.sum(self.sample_dist, 1)
        sample_per_folder = [ len(per_folder[i]) for i in range(len(path_train)) ]
        diff_per_folder = np.max(sample_per_folder) - sample_per_folder
        idx = [np.random.choice(sample_per_folder[i], diff_per_folder[i]) for i in range( len(sample_per_folder) ) if diff_per_folder[i] >= self.threshold]

        oversamples = [[per_folder[i][j] for j in idx[i]] for i in range( len(sample_per_folder) )]
        return oversamples

#smote=SMOTE()