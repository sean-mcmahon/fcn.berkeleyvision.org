#! /usr/bin/python
"""
cstrip Depth only

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
args = parser.parse_args()
print 'This is the Depth only solver!'

# import support functions
if 'n8307628' in home_dir:
    caffe_root = home_dir + '/Fully-Conv-Network/Resources/caffe'
    weights = home_dir + '/Fully-Conv-Network/Resources/FCN_models/pretrained_weights' + \
        '/nyud-fcn32s-hha-heavy.caffemodel'
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
print 'initialising depth only from Depth network, using weights' + \
    ' {}'.format(weights)
# no pre-trained depth network, hha pretrain is closest thing
hha_net_arch = file_location[:file_location.rfind(
    '/')] + '/cstrip-fcn32s-hha/test.prototxt'
hha_net = caffe.Net(hha_net_arch, weights,
                    caffe.TEST)
solver = caffe.SGDSolver(file_location + '/solver.prototxt')
surgery.transplant(solver.net, hha_net)  # copy whatever weights i can accros

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
print 'performing surgery on {}'.format(interp_layers)
surgery.interp(solver.net, interp_layers)  # calc deconv filter weights
# print 'copying hha params from conv1_1 (hha)  ->  conv1_1 (depth)'
# solver.net.params['conv1_1_bgrd'][0].data[:, :3] = base_net.params[
#     'conv1_1'][0].data
# solver.net.params['conv1_1_bgrd'][0].data[:, 3] = np.mean(base_net.params[
#     'conv1_1'][0].data, axis=1)
# solver.net.params['conv1_1_bgrd'][1].data[...] = base_net.params[
#     'conv1_1'][1].data
# del base_net

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
