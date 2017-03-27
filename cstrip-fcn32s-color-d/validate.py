#! /usr/bin/python
"""
cstrip validate Color-DEPTH only

"""
# import caffe
import numpy as np
import os
import sys
from os.path import expanduser
import imp
import argparse
import glob


def add_slash(mystring):
    if mystring.endswith('/'):
        return mystring
    else:
        return mystring + '/'

# add '../' directory to path for importing score.py, surgery.py and
# pycaffe layer
file_location = os.path.realpath(os.path.join(
    os.getcwd(), os.path.dirname(__file__)))
sys.path.append(file_location[:file_location.rfind('/')])
home_dir = expanduser("~")

# User Input
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='cpu')
parser.add_argument('--iteration', default=8000)
parser.add_argument('--test_type', default='val')
parser.add_argument('--network_dir', default='cstrip-fcn32s-color-d')
parser.add_argument('--snapshot_filter', default='train')
args = parser.parse_args()
iteration = args.iteration
network_dir = args.network_dir
network_dir = add_slash(network_dir)  # ensure slash present
snapshot_filter = args.snapshot_filter
print 'This is the colour-Depth validation!'

# import support functions
if 'n8307628' in home_dir:
    caffe_root = home_dir + '/Fully-Conv-Network/Resources/caffe'
    weight_dir = home_dir + '/Fully-Conv-Network/Resources/FCN_models/' + network_dir
    snapshot_dir = glob.glob(weight_dir + '*napshot*')
    weights = snapshot_dir[0] + '/' + snapshot_filter + \
        '_iter_' + str(iteration) + '.caffemodel'
elif 'sean' in home_dir:
    caffe_root = home_dir + '/src/caffe'
    weight_dir = home_dir + '/hpc-home/Fully-Conv-Network/Resources/FCN_models/' + network_dir
    snapshot_dir = glob.glob(weight_dir + '*napshot*')
    weights = snapshot_dir[0] + '/_iter_' + str(iteration) + '.caffemodel'
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
# caffe.set_device(1)
import score

# init
logFilenames = glob.glob(file_location + '/logs/FCNcolorDepth_train*')
if args.test_type == 'val':
    solver = caffe.SGDSolver(file_location + '/solver.prototxt')
    test_set = np.loadtxt(file_location[:file_location.rfind(
        '/')] + '/data/cs-trip/val.txt', dtype=str)
elif args.test_type == 'test':
    solver = caffe.SGDSolver(file_location + '/solver_test.prototxt')
    test_set = np.loadtxt(file_location[:file_location.rfind(
        '/')] + '/data/cs-trip/test.txt', dtype=str)
elif args.test_type == 'train':
    solver = caffe.SGDSolver(
        file_location + '/solver_test-trainingSet.prototxt')
    test_set = np.loadtxt(file_location[:file_location.rfind(
        '/')] + '/data/cs-trip/train.txt', dtype=str)
elif args.test_type == 'test_nd':
    solver = caffe.SGDSolver(file_location + '/solver_test_nd.prototxt')
    test_set = np.loadtxt(file_location[:file_location.rfind(
        '/')] + '/data/cs-trip/test.txt', dtype=str)
else:
    print 'Incorrect test_type given {}; expecting "train", "val" or "test"'.format(args.test_type)
    raise
solver.net.copy_from(weights)
print '-- test_type is', args.test_type, ' --'
weight_name = os.path.basename(weights)
print '-- network (colorDepth) weights used: {}'.format(weight_name)
print 'For more details on taining view log file(s):'
match_found = False
for filename in logFilenames:
    if weight_name in open(filename).read():
        print os.path.basename(filename)
        match_found = True
if not match_found:
    print 'Error, no logfile found for {}'.format(weight_name)

print '\n>>>> Validation <<<<\n'
score.seg_tests(solver, file_location + '/' + args.test_type +
                '_images', test_set, layer='score', dataL='color')
print 'WARNING if your getting 0 trip acc and no trip predicitons, \
check the name of the upscore layers in trainval, val and test.prototxt. \
 Some networks load with "upscore_trip" and others just "upscore"'
