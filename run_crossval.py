#! /usr/bin/python

import numpy as np
import os
import sys
from master import run_worker
import time

if __name__ == '__main__':
    file_location = os.path.realpath(os.path.join(
        os.getcwd(), os.path.dirname(__file__)))
    sys.path.append(file_location)

    n_folds = 4
    fold_idx = range(n_folds)
    fold_idx_str = [str(s + 1) + '_4' for s in fold_idx]
    # fold_idx_str = ['4_4']
    n_type = 'rgbhha_early_nyurgb'
    parent_dir = n_type + '_crossval'
    cross_val_dir = os.path.join(file_location, parent_dir)
    print cross_val_dir
    if not os.path.isdir(cross_val_dir):
        os.mkdir(cross_val_dir)
    base_lr = 1e-10
    net_type = 'rgbhha2_early'
    net_init = 'NYU_rgb'
    print '-'*20
    for idx in fold_idx_str:
        worker_dir = os.path.join(cross_val_dir, n_type + '_'+idx)
        print worker_dir
        job_id = run_worker(worker_dir, 'worker', fold_idx=idx)
        print 'job id = ', job_id
        time.sleep(15)  # my attempt to fix no cuda capable device error
        # (when there are cuda capable devices)
