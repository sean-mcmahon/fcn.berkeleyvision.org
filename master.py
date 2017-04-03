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
import glob
import re

# add '../' directory to path for importing score.py, surgery.py and
# pycaffe layer
# for engine: 1 CAFFE 2 CUDNN
file_location = os.path.realpath(os.path.join(
    os.getcwd(), os.path.dirname(__file__)))
sys.path.append(file_location)
import vis_finetune
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
    qsub_call = "qsub -v MY_TRAIN_DIR={} {}".format(working_dir, worker_file)
    try:
        subprocess.call(qsub_call, shell=True)
    except:
        print '****\nError submitting worker job with command \n', qsub_call
        print '****'
        print "Error message:", sys.exc_info()[0]
        raise

    return number_workers + 1


def check_worker(num_workers, worker_dir):
    logfilename = os.path.basename(
        glob.glob(os.path.join(worker_dir, '*.log'))[0])
    # Get job ID
    job_id = os.path.basename(glob.glob(os.path.join(worker_dir, '*.txt'))[0])
    a = subprocess.check_output(['qstat', job_id])
    a_s = a.split("\n")
    status = a_s[2][62]
    if status == 'R':
        # Job is running
        # this will fail (no matplotlib on HPC)
        vis_finetune.main(os.path.join(worker_dir, logfilename))
        with open(logfilename, 'r') as logfile:
            log = logfile.read()

        # get the first loss value and compare against the last 5
        # if more than 1 of the last 5 are above half cancel
        loss_0_pattern = r"Iteration 0, loss = (?P<loss_val>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
        init_loss = float(re.findall(loss_0_pattern, log)
                          [0])  # should be 1 match
        loss_pattern = r"Iteration (?P<iter_num>\d+), loss = (?P<loss_val>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
        all_losses = re.findall(loss_pattern, log)
        last_5 = all_losses[-5, :]
        last_5 = [float(i) for i in last_5]
        if np.sum(last_5 > (init_loss / 2)) > 1:
            # more than 1 of the last 5 losses greater than half initial loss
            num_workers = del_worker(num_workers, job_id)
            worker_status = 'del'
            print 'Deleted job {} (ID: {})'.format(os.path.basename(worker_dir),
                                                   job_id)
        else:
            worker_status = 'deployed'
    elif status == 'Q':
        # job qued return
        worker_status = 'deployed'
    elif status == 'F':
        # this will fail (no matplotlib on HPC)
        vis_finetune.main(os.path.join(worker_dir, logfilename))
        worker_status = 'finished'
    else:
        print 'Unexpected status ', status, 'for job', job_id
        worker_status = 'deployed'

    return num_workers, worker_status


def del_worker(number_workers, job_id):
    # deletes worker runnig
    # need to get the hpc job ID to cancel
    qsub_call = "qdel {}.pbs".format(job_id)
    try:
        subprocess.call(qsub_call, shell=True)
    except:
        print '****\nError submitting worker job with command \n', qsub_call
        print '****'
        print "Error message:", sys.exc_info()[0]
        raise
    return number_workers - 1

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
        to_remove = []
        for worker_dir in directories:
            # check status (train loss hasn't exploded)
            # create plots of all running jobs
            num_workers, worker_status = check_worker(num_workers, worker_dir)
            if worker_status == 'deployed':
                pass
            elif worker_status == 'del':
                to_remove.append(worker_dir)
                # create new worker
            elif worker_status == 'finished':
                to_remove.append(worker_dir)
                # create new worker
                pass
            else:
                Exception('Unkown worker status returned %s.' % worker_status)

        # overview of performances of all jobs with params
        # check for completed jobs

        # delete bad training runs

        # run new batch of jobs so 5 are running

        # repeat

    while(num_workers > 0):
        break
        # monitor existing jobs cancel if needed,
        # do no create any new jobs
