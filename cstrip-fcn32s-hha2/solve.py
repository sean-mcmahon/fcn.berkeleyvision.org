#! /usr/bin/python
"""
cstrip HHA2 only

"""
# import caffe
import numpy as np
import os
import sys
from os.path import expanduser
import imp
import argparse

# add '../' directory to path for importing score.py, surgery.py and
# pycaffe layer
file_location = os.path.realpath(os.path.join(
    os.getcwd(), os.path.dirname(__file__)))
sys.path.append(file_location[:file_location.rfind('/')])
home_dir = expanduser("~")

# User Input
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='gpu')
parser.add_argument('--solver_type', default='standard')
args = parser.parse_args()
solver_type = args.solver_type
print 'This is the HHA2 only solver!'

# import support functions
if 'n8307628' in home_dir:
    caffe_root = home_dir + '/Fully-Conv-Network/Resources/caffe'
    weights = home_dir + \
        '/Fully-Conv-Network/Resources/FCN_models/pretrained_weights/nyud-fcn32s-hha-heavy.caffemodel'
elif 'sean' in home_dir:
    caffe_root = home_dir + '/src/caffe'
    weights = home_dir + '/hpc-home/Fully-Conv-Network/Resources/FCN_models/pretrained_weights/nyud-fcn32s-hha-heavy.caffemodel'
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
import surgery
import score

# init
if solver_type == 'standard':
    solver = caffe.SGDSolver(file_location + '/solver.prototxt')
elif solver_type=='adam' or solver_type=='Adam':
    print '++++++++++++++++++\n Using Adam Solver \n++++++++++++++++++'
    solver = caffe.SGDSolver(file_location + '/solver_adam.prototxt')
else:
    solver = caffe.SGDSolver(file_location + '/solver.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
print 'performing surgery on {}'.format(interp_layers)
surgery.interp(solver.net, interp_layers)  # calc deconv filter weights

# scoring
val = np.loadtxt(file_location[:file_location.rfind(
    '/')] + '/data/cs-trip/val.txt', dtype=str)

for _ in range(50):
    print '------------------------------'
    print 'Running solver.step iter {}'.format(_)
    print '------------------------------'
    solver.step(2000)
    # filter_1 = solver.net.params['conv1_1'][0].data
    # print 'layer: conv1_1 len {}, shape {}, values {}'.format(len(filter_1), np.shape(filter_1), np.unique(filter_1))
    # filter_2 = solver.net.params['conv1_2'][0].data
    # print 'layer: conv1_2 len {}, shape {}, values {}'.format(len(filter_2), np.shape(filter_2), np.unique(filter_2))
    # score_fr_trip = solver.net.params['score_fr_trip'][0].data
    # print 'layer: score_fr_trip len {}, shape {}, values {}'.format(len(score_fr_trip), np.shape(score_fr_trip), np.unique(score_fr_trip))
    # if getting issues on HPC try
    # export MKL_CBWR=AUTO
    # and 'export CUDA_VISIBLE_DEVICES=1'
    # print '\n>>>> Validation <<<<\n'
    # score.seg_tests(solver, False, val, layer='score')
