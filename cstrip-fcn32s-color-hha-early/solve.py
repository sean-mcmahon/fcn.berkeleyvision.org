#! /usr/bin/python
"""
cstrip color HHA Early Fusion

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
parser.add_argument('--mode', default='CPU')
parser.add_argument('--pretrain_hha', default='False')
args = parser.parse_args()
pretrain_hha = False
if args.pretrain_hha == "True" or args.pretrain_hha == "true":
    pretrain_hha = True
print 'This is the colour-HHA Early Fusion solver!'

# import support functions
if 'n8307628' in home_dir:
    caffe_root = home_dir + '/Fully-Conv-Network/Resources/caffe'
    weights = home_dir + \
        '/Fully-Conv-Network/Resources/FCN_models/cstrip-fcn32s-color/colorSnapshot/_iter_2000.caffemodel'
    weights_hha = home_dir + \
        '/Fully-Conv-Network/Resources/FCN_models/cstrip-fcn32s-hha/HHAsnapshot/secondTrain_lowerLR_iter_2000.caffemodel'
elif 'sean' in home_dir:
    caffe_root = home_dir + '/src/caffe'
    weights = home_dir + '/hpc-home/Fully-Conv-Network/Resources/FCN_models/cstrip-fcn32s-color/colorSnapshot/_iter_2000.caffemodel'
    weights_hha = home_dir + '/hpc-home/Fully-Conv-Network/Resources/FCN_models/cstrip-fcn32s-hha/HHAsnapshot/secondTrain_lowerLR_iter_2000.caffemodel'
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
    print '-- GPU Mode Chosen -- {}'.format(args.mode)
    print '==============='
import surgery
import score

# init
print 'Using colour weights from {}'.format(weights)
base_net_color_arch = file_location[:file_location.rfind(
    '/')] + '/cstrip-fcn32s-color/test.prototxt'
base_net_color = caffe.Net(base_net_color_arch, weights,
                           caffe.TEST)
print 'Using HHA weights from {}'.format(weights_hha)
base_net_hha_arch = file_location[:file_location.rfind(
    '/')] + '/cstrip-fcn32s-hha/val.prototxt'
base_net_hha = caffe.Net(base_net_hha_arch, weights_hha,
                         caffe.TEST)
solver = caffe.SGDSolver(file_location + '/solver.prototxt')
# copy weights to solver network
surgery.transplant(solver.net, base_net_color)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
print 'performing surgery on {}'.format(interp_layers)
surgery.interp(solver.net, interp_layers)  # calc deconv filter weights
# Copy weights from color network into color-depth network (I think)
# Array sizes (for colour-hha early fusion only):
# conv1_1_bgrhha[0] shape (64, 6, 3, 3)
#  conv1_1_bgr[0] shape (64, 3, 3, 3)
# conv1_1_bgrhha[1] shape (64,)
#  conv1_1_bgr[1] shape (64,)

print 'copying color params from conv1_1  ->  conv1_1_bgrhha'
solver.net.params['conv1_1_bgrhha'][0].data[:, :3] = base_net_color.params[
    'conv1_1'][0].data
solver.net.params['conv1_1_bgrhha'][0].data[:, 3] = np.mean(base_net_color.params[
    'conv1_1'][0].data, axis=1)
solver.net.params['conv1_1_bgrhha'][1].data[...] = base_net_color.params[
    'conv1_1'][1].data  # copies the bias's

if (pretrain_hha):
    print 'copying HHA params from conv1_1  ->  conv1_1_bgrhha'
    solver.net.params['conv1_1_bgrhha'][0].data[:, 3:6] = base_net_hha.params[
        'conv1_1'][0].data
    solver.net.params['conv1_1_bgrhha'][0].data[:, 5] = np.mean(base_net_hha.params[
        'conv1_1'][0].data, axis=1)
# solver.net.params['conv1_1_bgrhha'][1].data[...] = base_net_hha.params[
#     'conv1_1'][1].data # copies the bias's

# print '\n----'  # to determine conv1 blob dimensions
# print 'conv1_1_bgrhha[0] shape {} \n conv1_1_bgr[0] shape {}'.format(
#     np.shape(solver.net.params['conv1_1_bgrhha'][0].data),
#     np.shape(base_net_color.params['conv1_1'][0].data))
# print 'conv1_1_bgrhha[1] shape {} \n conv1_1_bgr[1] shape {}'.format(
#     np.shape(solver.net.params['conv1_1_bgrhha'][1].data),
#     np.shape(base_net_color.params['conv1_1'][1].data))
# print '\n----'


del base_net_color, base_net_hha

# scoring
val = np.loadtxt(file_location[:file_location.rfind('/')] +
                 '/data/cs-trip/val.txt',
                 dtype=str)

for _ in range(50):
    print '------------------------------'
    print 'Running solver.step iter {}'.format(_)
    print '------------------------------'
    solver.step(2000)
    # filter_1 = solver.net.params['conv1_1_bgrd'][0].data
    # print 'layer: conv1_1_bgrd len {}, shape {}, values {}'.format(len(filter_1), np.shape(filter_1), np.unique(filter_1))
    # filter_2 = solver.net.params['conv1_2'][0].data
    # print 'layer: conv1_2 len {}, shape {}, values {}'.format(len(filter_2), np.shape(filter_2), np.unique(filter_2))
    # score_fr_trip = solver.net.params['score_fr_trip'][0].data
    # print 'layer: score_fr_trip len {}, shape {}, values {}'.format(len(score_fr_trip), np.shape(score_fr_trip), np.unique(score_fr_trip))
    # upscore_trip = solver.net.params['upscore_trip'][0].data
    # print 'layer: upscore_trip len {}, shape {}, values {}'.format(len(upscore_trip), np.shape(upscore_trip), np.unique(upscore_trip))
    # break

    # if getting issues on HPC try
    # export MKL_CBWR=AUTO
    # and 'export CUDA_VISIBLE_DEVICES=1'
    # print '\n>>>> Validation <<<<\n'
    # score.seg_tests(solver, False, val, layer='score')
