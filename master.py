#! /usr/bin/python
"""
trip trainer, designed to work with any modality

By Sean McMahpn

"""
# import caffe
import numpy as np
import os
import sys
from os.path import expanduser
import argparse
import tempfile
import subprocess

# add '../' directory to path for importing score.py, surgery.py and
# pycaffe layer
# for engine: 1 CAFFE 2 CUDNN
file_location = os.path.realpath(os.path.join(
    os.getcwd(), os.path.dirname(__file__)))
sys.path.append(file_location[:file_location.rfind('/')])
home_dir = expanduser("~")
# User Input
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='gpu')
parser.add_argument('--working_dir', default='rgb_1')
args = parser.parse_args()


def run_worker(working_dir):
    worker_file = os.path.join(file_location, 'worker.bash')
    if not os.path.isfile(worker_file):
        Exception("Could not find solve_any.py at {}".format(worker_file))
    subprocess.call(os.path.basename(worker_file) + ' ' + working_dir)

if __name__ == '__main__':
    # run workers (maximum jobs 5?)
    for ii in range(4):
        run_worker()

    # check in on workes, deleting and adding as needed
    # do this infinitely or for certain time period?
    while(time <= time_limit):
        # check status (train loss hasn't exploded) every 500 iterations
        # create plots of all running jobs
        # overview of performances of all jobs with params
        # check for completed jobs

        # delete bad training runs

        # run new batch of jobs so 5 are running

        # repeat

    while(jobs_running):
        # monitor existing jobs cancel if needed,
        # do no creaet any new jobs
