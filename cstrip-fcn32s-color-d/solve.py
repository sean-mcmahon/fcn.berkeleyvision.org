#! /usr/bin/python
"""
cstrip color DEPTH

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
parser.add_argument('--pretrain_depth_conv1_1', default='False')
parser.add_argument('--pretrain_network', default='RGB_CS')
args = parser.parse_args()
pretrain_depth_conv1_1 = False
if args.pretrain_depth_conv1_1 == "True" or args.pretrain_depth_conv1_1 == "true":
    pretrain_depth_conv1_1 = True
pretrain_network = args.pretrain_network
print 'This is the colour-DEPTH solver!'

# import support functions
if 'n8307628' in home_dir:
    caffe_root = home_dir + '/Fully-Conv-Network/Resources/caffe'
    weights = home_dir + \
        '/Fully-Conv-Network/Resources/FCN_models'
    weights_depth = home_dir + \
        '/Fully-Conv-Network/Resources/FCN_models/cstrip-fcn32s-depth/DepthSnapshot/negOneNull_mean_sub_iter_8000.caffemodel'
elif 'sean' in home_dir:
    caffe_root = home_dir + '/src/caffe'
    weights = home_dir + '/hpc-home/Fully-Conv-Network/Resources/FCN_models'
    weights_depth = home_dir + \
        '/Fully-Conv-Network/Resources/FCN_models/cstrip-fcn32s-depth/DepthSnapshot/negOneNull_mean_sub_iter_8000.caffemodel'
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
if pretrain_network == "RGB_CS":
    weights = os.path.join(
        weights, 'cstrip-fcn32s-color/colorSnapshot/_iter_2000.caffemodel')
    print 'Pretrain: Springfield Construction Colour (RGB) weights'
elif pretrain_network == "RGB_NYU":
    weights = os.path.join(
        weights, 'pretrained_weights/nyud-fcn32s-color-heavy.caffemodel')
    print 'Pretrain: NYUv2 colour weights'
elif pretrain_network == "Depth_CS":
    weights = os.path.join(
        weights,
        'cstrip-fcn32s-depth/DepthSnapshot/negOneNull_mean_sub_iter_8000.caffemodel')
    print 'Pretrain: Springfield Construction Depth weights'
else:
    raise Exception(
        'Invalid network pretrain network specified ({})'.format(pretrain_network))

# init
print 'Using colour weights from {}'.format(weights)
base_net_arch = file_location[:file_location.rfind(
    '/')] + '/cstrip-fcn32s-color/test.prototxt'
base_net = caffe.Net(base_net_arch, weights,
                     caffe.TEST)

print 'Using Depth weights from {}'.format(weights_depth)
base_net_depth_arch = file_location[:file_location.rfind(
    '/')] + '/cstrip-fcn32s-depth/val.prototxt'
base_net_depth = caffe.Net(base_net_depth_arch, weights_depth,
                           caffe.TEST)

solver = caffe.SGDSolver(file_location + '/solver.prototxt')
surgery.transplant(solver.net, base_net)  # copy weights to solver network

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
print 'performing surgery on {}'.format(interp_layers)
surgery.interp(solver.net, interp_layers)  # calc deconv filter weights
# Copy weights from color network into color-depth network (I think)
print 'copying color params from conv1_1  ->  conv1_1_bgrd'
solver.net.params['conv1_1_bgrd'][0].data[:, :3] = base_net.params[
    'conv1_1'][0].data
solver.net.params['conv1_1_bgrd'][0].data[:, 3] = np.mean(base_net.params[
    'conv1_1'][0].data, axis=1)
solver.net.params['conv1_1_bgrd'][1].data[...] = base_net.params[
    'conv1_1'][1].data

if (pretrain_depth_conv1_1):
    print 'copying Depth params from conv1_1  ->  conv1_1_bgrd'
    depth_filters = base_net_depth.params['conv1_1'][0].data
    solver.net.params['conv1_1_bgrd'][0].data[:, 3] = np.squeeze(depth_filters)
    # solver.net.params['conv1_1_bgrd'][0].data[:, 3] = np.mean(base_net_depth.params[
    #     'conv1_1'][0].data, axis=1)
del base_net, base_net_depth

# scoring
val = np.loadtxt(file_location[:file_location.rfind('/')] +
                 '/data/cs-trip/val.txt',
                 dtype=str)
score_layer = 'score'
for _ in range(10):
    print '------------------------------'
    print 'Running solver.step iter {}'.format(_)
    print '------------------------------'
    solver.step(1000)
    score.seg_loss_tests(solver, val, layer=score_layer)
    # if getting issues on HPC try
    # export MKL_CBWR=AUTO
    # and 'export CUDA_VISIBLE_DEVICES=1'
print 'Colour-depth early'
