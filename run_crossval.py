#! /usr/bin/python

import numpy as np
import os
import sys
from master import run_worker


if __name__ == '__main__':
    file_location = os.path.realpath(os.path.join(
        os.getcwd(), os.path.dirname(__file__)))
    sys.path.append(file_location)

    n_folds = 4
    fold_idx = range(n_folds)
    fold_idx_str = [str(s + 1) + '_4' for s in fold_idx]
    n_type = 'rgb'
    parent_dir = n_type + '_crossval'
    for idx in fold_idx_str:
        worker_dir = os.path.join(file_location, parent_dir, n_type + '_'+idx)
        run_worker(worker_dir, 'worker', fold_idx=idx)
