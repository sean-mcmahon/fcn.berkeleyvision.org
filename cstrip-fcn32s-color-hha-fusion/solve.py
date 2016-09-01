#! /usr/bin/python
"""
by Sean McMahon

"""
# import caffe
import numpy as np
import os
import sys
from os.path import expanduser
import imp
import argparse
file_location = os.path.realpath(os.path.join(
    os.getcwd(), os.path.dirname(__file__)))
sys.path.append(file_location[:file_location.rfind('/')])
import surgery
import score
from PIL import Image
from nets import convFusionNet


def fusion_solver(train_net_path, test_net_path, file_location):
    s = caffe_pb2.SolverParameter()
    s.train_net = train_net_path
    s.test_net.append(test_net_path)
    s.test_interval = 999999999  # do not invoke tests here
    s.test_iter.append(654)
    s.max_iter(3000)
    s.base_lr = 1e-12
    s.lr_policy = 'fixed'
    s.gamma = 0.1
    s.stepsize = 5000
    s.momentum = 9
    s.weight_decay = 0.0005
    s.display(20)
    s.snapshot = 1000
    s.snapshot_prefix = file_location + '/colourHHAsnapshot/fusion_train'
    return s


file_parent_dir = file_location[:file_location.rfind('/')]
home_dir = expanduser("~")
# import support functions
if 'n8307628' in home_dir:
    caffe_root = home_dir + '/Fully-Conv-Network/Resources/caffe'
elif 'sean' in home_dir:
    caffe_root = home_dir + '/src/caffe'
filename, path, desc = imp.find_module('caffe', [caffe_root + '/python/'])
caffe = imp.load_module('caffe', filename, path, desc)
from caffe.proto import caffe_pb2
from caffe import layers, params
from caffe.coord_map import crop

# User Input
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='CPU')
args = parser.parse_args()

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

data_split = 'val'
# Create fusion_test prototxt files
val_net_path = file_location + '/fusion_val.prototxt'
train_net_path = file_location + '/fusion_train.prototxt'
hdf5_filename = os.path.join(file_location, data_split + '_hdf5.txt')
val_hdf5s = np.loadtxt(hdf5_filename, dtype=str)
batchSize = len(val_hdf5s)
val_imgs = np.loadtxt(
    file_parent_dir + '/data/cs-trip/{}.txt'.format(data_split), dtype=str)
with open(val_net_path, 'w') as f:
    f.write(str(convFusionNet(os.path.join(file_location, 'testhdf5.txt'), batchSize)))

# Create and load solver
solver_path = file_location + '/fusion_solver.prototxt'
with open(solver_path, 'w') as f:
    f.write(str(fusion_solver(train_net_path, val_net_path)))
solver = caffe.SGDSolver(file_location + '/fusion_solver.prototxt')



# Net surgery, filling the deconvolution layer
interp_layers = [k for k in fusion_fcn.params.keys() if 'up' in k]
print 'performing surgery on {}'.format(interp_layers)
surgery.interp(fusion_fcn, interp_layers)


score.do_seg_tests(fusion_fcn, 0, os.path.join(
    file_location, data_split + '_images'), val_imgs, layer='score', gt='label')
