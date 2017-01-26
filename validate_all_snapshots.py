#! /usr/bin/python
"""
cstrip validate general

"""
# import caffe
import numpy as np
import os
import sys
from os.path import expanduser
import imp
import argparse
import glob
import score


def add_slash(mystring):
    if mystring.endswith('/'):
        return mystring
    else:
        return mystring + '/'


def find_between(s, first, last):
    # from:
    # http://stackoverflow.com/questions/3368969/find-string-between-two-substrings
    try:
        start = s.index(first) + len(first)
        end = s.index(last, start)
        return s[start:end]
    except ValueError:
        return ""


def find_between_r(s, first, last):
    # from:
    # http://stackoverflow.com/questions/3368969/find-string-between-two-substrings
    try:
        start = s.rindex(first) + len(first)
        end = s.rindex(last, start)
        return s[start:end]
    except ValueError:
        return ""
# add '../' directory to path for importing score.py, surgery.py and
# pycaffe layer
file_location = os.path.realpath(os.path.join(
    os.getcwd(), os.path.dirname(__file__)))
sys.path.append(file_location)
home_dir = expanduser("~")

# User Input
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='cpu')
parser.add_argument('--iteration', default=8000)
parser.add_argument('--test_type', default='val')
parser.add_argument('--network_dir', default='cstrip-fcn32s-hha')
parser.add_argument('--snapshot_filter', default='val')
args = parser.parse_args()
iteration = args.iteration
network_dir = args.network_dir
network_dir = add_slash(network_dir)  # ensure slash present
snapshot_filter = args.snapshot_filter
print 'This is the general validation script!'

# import support functions
if 'n8307628' in home_dir:
    caffe_root = home_dir + '/Fully-Conv-Network/Resources/caffe'
    weight_dir = home_dir + '/Fully-Conv-Network/Resources/FCN_models/' \
        + network_dir
elif 'sean' in home_dir:
    caffe_root = home_dir + '/src/caffe'
    weight_dir = home_dir + '/hpc-home/Fully-Conv-Network/Resources/FCN_models/' + network_dir
filename, path, desc = imp.find_module('caffe', [caffe_root + '/python/'])
caffe = imp.load_module('caffe', filename, path, desc)
if 'g' in args.mode or 'G' in args.mode:
    caffe.set_mode_gpu()
    print '-- GPU Mode -- {}'.format(args.mode)
elif 'c' in args.mode or 'C' in args.mode:
    caffe.set_mode_cpu()
    print '-- CPU Mode -- {}'.format(args.mode)
else:
    caffe.set_mode_gpu()
    print '==============='
    print 'No Mode (CPU or GPU) Given'
    print '-- GPU Mode Chosen --'
    print '==============='

# init
train_set = None
if args.test_type == 'val':
    solver = caffe.SGDSolver(file_location + '/' +
                             network_dir + 'solver.prototxt')
    test_set = np.loadtxt(file_location + '/data/cs-trip/val.txt', dtype=str)
    train_set = np.loadtxt(
        file_location + '/data/cs-trip/train.txt', dtype=str)
# elif args.test_type == 'test':
#     solver = caffe.SGDSolver(file_location + '/' +
#                              network_dir + 'solver_test.prototxt')
#     test_set = np.loadtxt(file_location + '/data/cs-trip/test.txt', dtype=str)
else:
    print 'Incorrect test_type given {}; expecting "val" or "test"'.format(args.test_type)
    raise

print '----------------------------------------'
print 'Validating all iterations of Network {}'.format(network_dir)
print '\nTest_type is', args.test_type, '    '
print '----------------------------------------'

snapshot_dir = glob.glob(os.path.join(weight_dir, '*napshot*'))
caffemodel_files = glob.glob(os.path.join(
    snapshot_dir[0], snapshot_filter + '_iter_*' + '*.caffemodel'))
# new_weight = os.path.join(
#     snapshot_dir[0], 'secondTrain_lowerLR_iter_14000.caffemodel')

for weight_file in caffemodel_files:
    iteration = int(find_between_r(weight_file, '_iter_', '.caffemodel'))
    print 'Network weights used: {} iter {}'.format(
        os.path.basename(weight_file), iteration)
    solver.net.copy_from(weight_file)
    if train_set:
        print '>>>> Training Set {} <<<<'.format(iteration)
        # solver.net.CopyTrainedLayersFromBinaryProto(new_weight) # this wont
        # work - method not passed to python
        score.do_seg_tests(solver.net, iteration, None,
                           train_set, layer='score')

    print '>>>> Validation Set {} <<<<'.format(iteration)
    score.seg_tests(solver, file_location + '/' + network_dir +
                    args.test_type + '_images', test_set, layer='score')

print '\n(python) Test Network: {} \n'.format(network_dir)
