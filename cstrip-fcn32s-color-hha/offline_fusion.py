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
import h5py
file_location = os.path.realpath(os.path.join(
    os.getcwd(), os.path.dirname(__file__)))
sys.path.append(file_location[:file_location.rfind('/')])
import surgery
import score


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


def fusionNet(hf5_path):
    n = caffe.NetSpec()
    n.color_features, n.hha_features, n.label = layers.HDF5Data(
        batch_size=1, source=hf5_path, ntop=2)
    n.features_fused = L.Eltwise(n.color_features, n.hha_features,
                                 operation=params.Eltwise.SUM, coeff=[0.5, 0.5])
    n.upcore = layers.Deconvolution(n.score_fused,
                                    convolution_param=dict(num_output=2, kernel_size=64, stride=32,
                                                           bias_term=False),
                                    param=[dict(lr_mult=0)])
    n.score = crop(n.upscore, n.color)
    return n.to_proto()

# add '../' directory to path for importing score.py, surgery.py and
# pycaffe layer

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
layer = 'score'
gt = 'label'

# You may want to change these initialisation weights,
# they differ slightly to the models being loaded
color_weights = file_parent_dir + \
    '/cstrip-fcn32s-color/colorSnapshot/_iter_8000.caffemodel'
color_proto = file_parent_dir + '/cstrip-fcn32s-color/val.prototxt'
color_net = caffe.Net(color_proto, color_weights, caffe.TEST)
color_net.forward()
score_colour = color_net.blobs[layer].data[0]
print '--------------------------------------------------------'
print 'Shape and size of score_colour is {} & {} \nscore_colour unique values {}'.format(np.shape(score_colour), sys.getsizeof(score_colour), np.unique(score_colour))
del color_net
print 'After Delete: Shape and size of score_colour is {} & {} \nscore_colour unique values {}'.format(np.shape(score_colour), sys.getsizeof(score_colour), np.unique(score_colour))
print '--------------------------------------------------------'

hha_weights = file_parent_dir + \
    '/cstrip-fcn32s-hha/HHAsnapshot/train_iter_8000.caffemodel'
hha_proto = file_parent_dir + '/cstrip-fcn32s-hha/val.prototxt'
hha_net = caffe.Net(hha_proto, hha_weights, caffe.TEST)
hha_net.forward()
score_hha = hha_net.blobs[layer].data[0]
gt_hha = hha_net.blobs[gt].data[0, 0].astype(np.uint8)
print '--------------------------------------------------------'
print 'Shape and size of score_hha is {} & {} \nscore_hha unique values {}'.format(np.shape(score_hha), sys.getsizeof(score_hha), np.unique(score_hha))
del hha_net
print 'After Delete: Shape and size of score_hha is {} & {} \nscore_hha unique values {}'.format(np.shape(score_hha), sys.getsizeof(score_hha), np.unique(score_hha))
print '--------------------------------------------------------'

val_hdf5_location = os.path.join(file_location, 'hdfFive.h5')

with h5py.File(val_hdf5_location, 'w') as f:
    f['color_features'] = score_colour
    f['hha_features'] = score_hha
    f['label'] = gt_hha

# Create fusion_test prototxt files
test_net_path = file_location + '/fusion_test.prototxt'
with open(test_net_path, 'w') as f:
    f.write(str(fusionNet(val_hdf5_location)))

# Create and load solver
# solver_path = file_location + '/fusion_solver.prototxt'
# with open(solver_path, 'w') as f:
#     f.write(str(fusion_solver(train_net_path, test_net_path)))
# solver = caffe.SGDSolver(file_location + '/fusion_solver.prototxt')

fusion_fcn = caffe.Net(test_net_path, caffe.TEST)

# Net surgery, filling the decolvution layer
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
print 'performing surgery on {}'.format(interp_layers)
surgery.interp(fusion_fcn, interp_layers)
