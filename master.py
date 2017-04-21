#! /usr/bin/python
"""
trip trainer, designed to work with any modality

By Sean McMahon

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
import time

# add '../' directory to path for importing score.py, surgery.py and
# pycaffe layer
# for engine: 1 CAFFE 2 CUDNN
file_location = os.path.realpath(os.path.join(
    os.getcwd(), os.path.dirname(__file__)))
sys.path.append(file_location)
try:
    import vis_finetune
except ImportError:
    print sys.exc_info()[0]
    raise

home_dir = expanduser("~")
# User Input
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='gpu')
parser.add_argument('--working_dir', default='rgb_1')
args = parser.parse_args()


def append_dir_to_txt(dir_txt, dir_name):
    with open(dir_txt, "a") as myfile:
        myfile.write(dir_name + '\n')
    myfile.close()


def run_worker(work_dir):
    worker_file = os.path.join(file_location, 'worker_live.bash')
    if not os.path.isfile(worker_file):
        raise Exception(
            "Could not find solve_any.py at {}".format(worker_file))
    if not os.path.isdir(work_dir):
        os.mkdir(work_dir)
    qsub_call = "qsub -v MY_TRAIN_DIR={} {}".format(work_dir, worker_file)
    try:
        jobid_ = subprocess.check_output(qsub_call, shell=True)
    except:
        print '****\nError submitting worker job with command \n', qsub_call
        print '****'
        print "Error message:", sys.exc_info()[0]
        raise

    return jobid_


def check_job_status(job_id):
    a = subprocess.check_output('qstat -x ' + job_id, shell=True)
    a_s = a.split("\n")
    status = a_s[2][62]
    return status


def check_worker(id_, worker_dir):

    # Get job ID
    status = check_job_status(id_)
    if status == 'R':
        # Job is running
        try:
            logfilename = glob.glob(os.path.join(worker_dir, '*.log'))[0]
        except IndexError:
            print '*** \nError finding logfilenames at', os.path.join(worker_dir,
                                                                      '*.log')
            print '***'
            return 'deployed'
        try:
            with open(logfilename, 'r') as logfile:
                log = logfile.read()
        except:
            print 'logfilename = {}.'.format(logfilename)
            subprocess.call('ls -l %s' %
                            os.path.dirname(logfilename), shell=True)
            print sys.exc_info()[0]
            raise
        # Get loss values
        loss_0_pattern = r"Iteration 0, loss = (?P<loss_val>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
        loss_0_match = re.findall(loss_0_pattern, log)
        loss_pattern = r"Iteration (?P<iter_num>\d+), loss = (?P<loss_val>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
        all_losses = re.findall(loss_pattern, log)
        # Check for insufficient number of iterations
        if len(all_losses) < 6 or loss_0_match is None or not loss_0_match:
            # print 'Insufficient train iterations to perform check.',
            '\n{}\n'.format(all_losses)
            return 'deployed'
        try:
            init_loss = float(loss_0_match[0][0])  # should be 1 match
        except TypeError:
            print 'loss_0_match[0]:', loss_0_match[0]
            print 'shape loss_0_match: ', np.shape(loss_0_match)
            print "Error msg: ", sys.exc_info()[0]
            raise
        vis_finetune.main((logfilename,),
                          printouts=False)
        # print 'check_worker:: init_loss=', init_loss
        last_5 = [float(i[1]) for i in all_losses[-5:]]
        # print 'last 5 losses = ', all_losses[-5:][0:1]
        if np.sum(last_5 > (init_loss / 2)) > 1:
            # more than 1 of the last 5 losses greater than half initial loss
            print 'Deleting: last 5 losses = ', all_losses[-5:][0:1]
            worker_status = 'del'
            print 'job {} (ID: {})'.format(os.path.basename(worker_dir),
                                           job_id)
        else:
            worker_status = 'deployed'
    elif status == 'Q':
        # job qued return
        worker_status = 'deployed'
    elif status == 'F':
        try:
            logfilename = glob.glob(os.path.join(worker_dir, '*.log'))[0]
        except IndexError:
            print '*** \nError finding logfilenames at', os.path.join(worker_dir,
                                                                      '*.log')
            print '***'
            return 'deployed'
        vis_finetune.main((logfilename,),
                          printouts=False)

        worker_status = 'finished'
    else:
        print '++++\nUnexpected status ', status, 'for job', job_id,  \
            '\ndir: ', worker_dir, '\n++++'
        worker_status = 'deployed'

    return worker_status


def del_worker(job_id):
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

if __name__ == '__main__':
    # TODO write a bash script executing this parsing in a different session foldre
    # and run time
    jobs_running = False
    intialising_workers = True
    # session_folder = os.path.join(file_location, 'rgb_workers')
    session_folder = '/home/n8307628/Fully-Conv-Network/Resources' + \
        '/FCN_paramsearch/rgb_workers'
    if not os.path.isdir(session_folder):
        os.mkdir(session_folder)
    workers_name = os.path.join(session_folder, 'rgb_2_')
    directories = []
    worker_ids = []
    print '---- master creating workers ----'
    for directory_num in range(3):
        dir_name = workers_name + str(directory_num)
        while(os.path.isdir(dir_name)):
            directory_num += 1
            dir_name = workers_name + str(directory_num)
        print 'Creatng worker @ ', dir_name
        directories.append(dir_name)
        job_id = run_worker(dir_name)
        print 'job_id: ', job_id
        worker_ids.append(job_id)
    print len(worker_ids), 'workers running!'
    subprocess.call('qstat -u n8307628', shell=True)
    print 'directory_num=', directory_num

    dir_txt = os.path.join(session_folder, 'directories.txt')
    thefile = open(dir_txt, 'w')
    for item in directories:
        thefile.write("%s\n" % item)
    thefile.close()

    # wait for job to start and training to initialise
    waiting_for_init = True
    print 'waiting for a job to start...'
    while (waiting_for_init):
        for job_id in worker_ids:
            status = check_job_status(job_id)
            if status == 'R':
                waiting_for_init = False
            elif status == 'Q':
                pass
            else:
                print 'unforseen job status -', status

        time.sleep(5)

    # check in on workes, deleting and adding as needed
    # do this infinitely or for certain time period?
    timeout = time.time() + 60 * 60 * 72  # 1 minute
    print '---- master: checking on workers ----'
    while(time.time() < timeout):
        to_remove = []
        id_to_remove = []
        # subprocess.call('qstat -u n8307628', shell=True)
        print '\n--- Directories in use:\n', directories
        for worker_dir, job_id in zip(directories, worker_ids):
            # check status (train loss hasn't exploded)
            # create plots of all running jobs
            # print '-- Checking dir {}, job_id={}'.format(worker_dir, job_id)
            worker_status = check_worker(job_id, worker_dir)
            # print '-- After check, status:',  worker_status
            if worker_status == 'deployed':
                pass
            elif worker_status == 'del':
                to_remove.append(worker_dir)
                id_to_remove.append(job_id)
                del_worker(job_id)
            elif worker_status == 'finished':
                to_remove.append(worker_dir)
                id_to_remove.append(job_id)
                # del_worker(job_id)
                pass
            else:
                Exception('Unkown worker status returned %s.' % worker_status)
                raise
        # remove deleted or finshed jobs from list and run new ones
        if len(to_remove) != len(id_to_remove):
            print 'to_remove:', to_remove
            print 'id_to_remove', to_remove
            Exception('Dir and IDs of jobs to remove do not align:')
            raise

        # Remove finished workers from lists and spawn new workers
        for item_dir, item_id in zip(to_remove, id_to_remove):
            directories.remove(item_dir)
            worker_ids.remove(item_id)
            directory_num += 1
            dir_name = workers_name + str(directory_num)
            job_id = run_worker(dir_name)
            directories.append(dir_name)
            worker_ids.append(job_id)
            append_dir_to_txt(dir_txt, dir_name)
        # print '-- directories after deleting and adding:\n', directories,
        # '\n'

        if len(directories) != len(worker_ids):
            print 'directories', directories
            print 'worker_ids', worker_ids
            Exception(
                'Number of workers does not equal number of worker directories')
            raise
        time.sleep(120)

    print '\n---- master waiting for jobs to finish ----\n'
    try:
        while(len(worker_ids) > 0):
            worker_status = check_worker(job_id, worker_dir)
            # print '-- After check, status:',  worker_status
            if worker_status == 'deployed':
                pass
            elif worker_status == 'del':
                to_remove.append(worker_dir)
                id_to_remove.append(job_id)
                del_worker(job_id)
            elif worker_status == 'finished':
                to_remove.append(worker_dir)
                id_to_remove.append(job_id)
                # del_worker(job_id)
            else:
                Exception('Unkown worker status returned %s.' % worker_status)
                raise
            for item_dir, item_id in zip(to_remove, id_to_remove):
                directories.remove(item_dir)
                worker_ids.remove(item_id)
            # monitor existing jobs cancel if needed,
            # do no create any new jobs
    except:
        print '\n---- master deleting workers ----\n'
        for worker_dir, job_id in zip(directories, worker_ids):
            print 'Deleting \nworker_dir:', worker_dir
            print 'job_id:', job_id
            del_worker(job_id)
        raise(Exception(sys.exc_info()[0]))
    for worker_dir, job_id in zip(directories, worker_ids):
        print 'Deleting \nworker_dir:', worker_dir
        print 'job_id:', job_id
        del_worker(job_id)
