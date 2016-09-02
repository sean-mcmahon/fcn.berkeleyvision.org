#! /usr/bin/python
"""
by Sean McMahon

"""
import numpy as np
import imp
import os
import sys
from os.path import expanduser
import argparse
import glob
file_location = os.path.realpath(os.path.join(
    os.getcwd(), os.path.dirname(__file__)))
sys.path.append(file_location[:file_location.rfind('/')])
sys.path.append(file_location)
import surgery
import score
import nets
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


def writehdf5txt(FCN_Models_dir, outdir, data_splits):
    hdf5txt_locations = []
    for split in data_splits:
        hdf5filenames = glob.glob(os.path.join(
            FCN_Models_dir, 'data/cs-trip/' + split + '_hdf5/*.h5'))
        hdf5txt_locations.append(os.path.join(outdir, split + '_hdf5.txt'))
        txtfile = open(hdf5txt_locations[-1], 'w')
        for filename in hdf5filenames:
            txtfile.write(filename + '\n')
        txtfile.close()
    print 'solve: hdf5 file locations written.'
    return hdf5txt_locations


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

    snapshot_dir = os.path.join(file_location + '/fusionSnapshot/secondTrain')
    if not os.path.isdir(snapshot_dir):
        os.mkdir(snapshot_dir)
    s.snapshot_prefix = snapshot_dir
    s.test_initialization = False
    return s


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

data_splits = ['train', 'val']
# Create fusion_test prototxt files
[train_hdf5, val_hdf5] = writehdf5txt(
    file_parent_dir, file_location, data_splits)
val_net_path = file_location + '/fusion_val.prototxt'
train_net_path = file_location + '/fusion_train.prototxt'
val_batchSize = 1
train_batchSize = 1
# Create network architectuers
with open(train_net_path, 'w') as f:
    f.write(str(nets.convFusionNet(train_hdf5, train_batchSize)))
with open(val_net_path, 'w') as f:
    f.write(str(nets.convFusionNet(val_hdf5, val_batchSize)))
# Create and load solver
solver_path = os.path.join(file_location, 'fusion_solver.prototxt')
with open(solver_path, 'w') as f:
    f.write(str(fusion_solver(train_net_path, val_net_path, file_location)))
solver = caffe.SGDSolver(solver_path)

# Net surgery, filling the deconvolution layer
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
print 'performing surgery on {}'.format(interp_layers)
surgery.interp(solver.net, interp_layers)

val_imgs = np.loadtxt(
    file_parent_dir + '/data/cs-trip/val.txt', dtype=str)
for _ in range(50):
    print '------------------------------'
    print 'Running solver.step iter {}'.format(_)
    print '------------------------------'
    solver.step(1000)
    score.seg_tests(solver, False, val_imgs, layer='score')
