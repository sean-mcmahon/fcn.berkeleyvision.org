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


def run_worker(number_workers, working_dir):
    worker_file = os.path.join(file_location, 'worker.bash')
    if not os.path.isfile(worker_file):
        Exception("Could not find solve_any.py at {}".format(worker_file))
    # job_name = assign_worker_id(working_dir)
    qsub_call = "qsub -v TRAIN_DIR={} {}".format(working_dir, worker_file)
    try:
        subprocess.call(qsub_call, shell=True)
    except:
        print '****\nError submitting worker job with command \n', qsub_call
        print '****'
        print "Error message:", sys.exc_info()[0]
        raise

    return number_workers+1

def assign_worker_id(working_dir):
    # modify bash script with a desired name and then return that name
    pass

def check_worker(worker_dir):
    # check on loss over last 500 iterations

    # create/update plots of training

    # return status -> continue training, cancel training or finished training
    pass

def del_worker(number_workers, job_id):
    # deletes worker runnig
    # need to get the hpc job ID to cancel

    pass
    # return number_workers - 1

if __name__ == '__main__':
    jobs_running = False
    time = 5
    time_limit = 4
    num_workers = 0
    # run workers (maximum jobs 5?)
    directories = ['rgb_1', 'rgb_2']
    for directory in directories:
        num_workers = run_worker(num_workers, directory)
    print num_workers, 'workers running!'
    subprocess.call('qstat -u n8307628', shell=True)

    # check in on workes, deleting and adding as needed
    # do this infinitely or for certain time period?
    while(time <= time_limit):
        pass
        # check status (train loss hasn't exploded) every 500 iterations
        # create plots of all running jobs
        # overview of performances of all jobs with params
        # check for completed jobs

        # delete bad training runs

        # run new batch of jobs so 5 are running

        # repeat

    while(jobs_running):
        pass
        # monitor existing jobs cancel if needed,
        # do no create any new jobs
