#! /usr/bin/python
"""
cstrip Color and HHA2 Eltwise late fusion

"""
# import caffe
import numpy as np
import os
import sys
from os.path import expanduser
import imp
import argparse


def fusion_solver(train_net_path, test_net_path, file_location):
    s = caffe_pb2.SolverParameter()
    s.train_net = train_net_path
    s.test_net.append(test_net_path)
    s.test_interval = 999999999  # do not invoke tests here
    s.test_iter.append(654)
    s.max_iter = 300000
    s.base_lr = 5e-13
    s.lr_policy = 'fixed'
    s.gamma = 0.1
    s.average_loss = 20
    s.momentum = 0.99
    s.iter_size = 1
    s.weight_decay = 0.0005
    s.display = 20
    s.snapshot = 1000

    snapshot_dir = file_location + '/fusionSnapshot/secondTrain'
    if not os.path.isdir(snapshot_dir):
        os.mkdir(snapshot_dir)
    s.snapshot_prefix = snapshot_dir
    s.test_initialization = False
    return s

# add '../' directory to path for importing score.py, surgery.py and
# pycaffe layer
file_location = os.path.realpath(os.path.join(
    os.getcwd(), os.path.dirname(__file__)))
sys.path.append(file_location[:file_location.rfind('/')])
file_parent_dir = file_location[:file_location.rfind('/')]
home_dir = expanduser("~")

# User Input
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='GPU')
parser.add_argument('--fusion_type', default='mixDCNN')
args = parser.parse_args()
print 'This is the colour-HHA2 summation solver!'
fusion_type = args.fusion_type

# import support functions
if 'n8307628' in home_dir:
    caffe_root = home_dir + '/Fully-Conv-Network/Resources/caffe'
elif 'sean' in home_dir:
    caffe_root = home_dir + '/src/caffe'
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


color_weights = file_parent_dir + \
    '/cstrip-fcn32s-color/colorSnapshot/_iter_2000.caffemodel'
color_proto = file_parent_dir + '/cstrip-fcn32s-color/test.prototxt'
hha2_weights = file_parent_dir + \
    '/cstrip-fcn32s-hha2/HHA2snapshot/secondTrain_lowerLR_iter_2000.caffemodel'
hha2_proto = file_parent_dir + '/cstrip-fcn32s-hha2/test.prototxt'
if fusion_type == 'sum':
    print '------\n Loading sum fusion approach \n------'
    solver = caffe.SGDSolver(file_location + '/solver.prototxt')
elif 'mix' in fusion_type or fusion_type == 'mixDCNN':
    print '------\n Loading mixDCNN fusion approach \n------'
    solver = caffe.SGDSolver(file_location + '/solver_mix.prototxt')
else:
    print 'unrecognised/no fusion approach specified,',
    'loading standard summation approach'
    solver = caffe.SGDSolver(file_location + '/solver.prototxt')

# surgeries
color_net = caffe.Net(color_proto, color_weights, caffe.TEST)
surgery.transplant(solver.net, color_net, suffix='color')
del color_net

hha2_net = caffe.Net(hha2_proto, hha2_weights, caffe.TEST)
surgery.transplant(solver.net, hha2_net, suffix='hha2')
del hha2_net

interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
print 'performing surgery on {}'.format(interp_layers)
surgery.interp(solver.net, interp_layers)

# scoring
val = np.loadtxt(
    file_location[:file_location.rfind('/')] + '/data/cs-trip/val.txt',
    dtype=str)
score.seg_tests(solver, False, val, layer='score')

for _ in range(20):
    score.seg_loss_tests(solver, val, layer='score')
    print '------------------------------'
    print 'Running solver.step iter {}'.format(_)
    print '------------------------------'
    solver.step(250)
score.seg_loss_tests(solver, val, layer='score')
print '(python) color-hha2 fusion, fusion_type', fusion_type
