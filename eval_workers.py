#! /usr/bin/python
"""
cstrip validate general

"""
# import caffe
import numpy as np
import os
import sys
import imp
import argparse
import glob
import re
import score
import click

home_dir = os.path.expanduser("~")
if 'n8307628' in home_dir:
    caffe_root = home_dir + '/Fully-Conv-Network/Resources/caffe'
elif 'sean' in home_dir:
    caffe_root = home_dir + '/src/caffe'
filename, path, desc = imp.find_module('caffe', [caffe_root + '/python/'])
caffe = imp.load_module('caffe', filename, path, desc)


def run_test(solver, data_set, output_dir):
    score.seg_tests(solver, output_dir,
                    data_set, layer='score')


def parse_val(logfilename):

    with open(logfilename, 'r') as files:
        logfile = files.read()

    # TODO check if splitting string across lines does not void literal string
    val_acc_pattern = r"Iteration (?P<iter_num>\d+) val trip accuracy"" (?P<accuracy>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    val_loss_pattern = r"Iteration (?P<iter_num>\d+) val set loss = (?P<loss_val>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"

    v_l = float(5e8)
    iter_l = -1
    v_a = 0
    iter_a = -1
    for r_a, r_l in zip(re.findall(val_acc_pattern, logfile),
                        re.findall(val_loss_pattern, logfile)):
        iteration_acc = int(r_a[0])
        accuracy = float(r_a[1])
        if accuracy > v_a:
            v_a = accuracy
            iter_a = iteration_acc

        iteration_loss = int(r_l[0])
        loss = float(r_l[1])
        if loss < v_l:
            v_l = loss
            iter_l = iteration_loss
    if 'r_a' not in locals() or 'r_l' not in locals():
        print 'No matches for logfile: ', logfilename
        return {'val_loss': [0, 0], 'val_acc': [0, 0],
                'logfile': logfilename}

    # print 'iter acc {}, accuracy {}\n iter loss {}, loss {}'.format(
    #     iteration_acc, accuracy, iteration_loss, loss)

    if iter_a == -1 or iter_l == -1:
        raise(Exception(
            ("Max acc or min loss not found! (iter_a {}, iter_l {}) "
             "\n- Logfile {}").format(iter_a,
                                      iter_l, logfilename)))

    results_dict = {'val_loss': [iter_l, v_l], 'val_acc': [iter_a, v_a],
                    'logfile': logfilename}

    return results_dict


def sort_n_write(results, sort_key):
    sort_dict = sorted(results, key=lambda k: k[sort_key])

    print 'sort_n_write:: sort_dict[0:4]: \n', sort_dict[0:4]

    with open('top_{}.txt'.format(sort_key), 'w') as myfile:
        for res in sort_dict[0:4]:
            # TODO check formatting of this string, source of error!
            res_str = ("Best accuracy {} @ iter {}. "
                       "Best Loss {} @ iter {}. "
                       "Filename {}").format(round(res['val_acc'][1], 3),
                                             res['val_acc'][0],
                                             round(res['val_loss'][1], 3),
                                             res['val_loss'][0],
                                             res['logfile'])
            print 'sort_n_write:: adding string: \n', res_str
            myfile.write(res_str)
    return sort_dict


def main(worker_parent_dir):
    # find logfile in sub_dir, if none look in dir
    sub_dirs = next(os.walk(worker_parent_dir))[1]
    logfiles = []
    results_list = []
    for sub_dir in sub_dirs:
        filenames = glob.glob(os.path.join(
            worker_parent_dir, sub_dir, '*.log'))
        for filename in filenames:
            logfiles.append(filename)
    for count, log in enumerate(logfiles):
        print 'parsing log {}/{}'.format(count, len(logfiles))
        results = parse_val(log)
        results_list.append(results)
    sort_res_acc = sort_n_write(results, 'val_acc')
    sort_res_loss = sort_n_write(results, 'val_loss')

    return sort_res_acc, sort_res_loss


@click.command()
@click.argument('logfile', nargs=-1, type=click.Path(exists=True))
def main_cl(logfile):
    print 'processing:\n{}'.format(logfile)
    if len(logfile) == 1:
        main(logfile)
        # TODO search sub directories for other worker files
        pass
    else:
        for log in logfile:
            print 'processing {}'.format(log)
            res = parse_val(log)
            # TODO add sorting to get top 5 results
            print 'results', res

if __name__ == '__main__':
    # main_cl()
    parent_dir = '/home/sean/hpc-home/Fully-Conv-Network/Resources/FCN_paramsearch/rgb_workers'

    main(parent_dir)

    # loop over worker jobs

    # find best performance (val loss or val accuracy)
    # read this from text file
    # write put it into text file or append to folder name?

    # save top 5 into text file

    # run top 5 on test set
