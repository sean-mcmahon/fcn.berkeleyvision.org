#! /usr/bin/python
"""
cstrip COLOUR only

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


def solver(train_net_path, test_net_path, params_dict):
    s = caffe_pb2.SolverParameter()
    s.train_net = train_net_path
    s.test_net.append(test_net_path)
    s.test_interval = params_dict.get(
        'test_interval', 999999999)  # do not invoke tests here
    s.test_iter.append(654)
    s.max_iter = params_dict.get('max_iter', 300000)
    s.base_lr = params_dict['base_lr']
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

# User Input
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='gpu')
parser.add_argument('--pretrain', default='NYU')
parser.add_argument('--save_weights', default=True)
args = parser.parse_args()
if args.save_weights == 'True' or args.save_weights == 'true':
    save_weights = True
elif args.save_weights == 'False' or args.save_weights == 'false':
    save_weights = False
else:
    Exception('Invalid "save_weights" argument given ({})'.format(
        args.save_weights))
pretrain_weights = args.pretrain
print 'This is the COLOUR only solver!'

# import support functions
if 'n8307628' in home_dir:
    caffe_root = home_dir + '/Fully-Conv-Network/Resources/caffe'
    weights = home_dir + \
        '/Fully-Conv-Network/Resources/FCN_models'
elif 'sean' in home_dir:
    caffe_root = home_dir + '/src/caffe'
    weights = home_dir + '/hpc-home/Fully-Conv-Network/Resources/FCN_models'
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
import caffe

if pretrain_weights == "NYU":
    weights = os.path.join(
        weights, 'pretrained_weights/nyud-fcn32s-color-heavy.caffemodel')
    print 'Pretrain on NYU weights'
elif pretrain_weights == "CS":
    weights = os.path.join(
        weights, 'cstrip-fcn32s-color/colorSnapshot/_iter_2000.caffemodel')
    print 'Pretrain on CS weights (_iter_2000.caffemodel)'
else:
    Exception('Unrecognised pretrain weights option given ({})'.format(
        pretrain_weights))

# init
solver = caffe.SGDSolver(file_location + '/solver.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
print 'performing surgery on {}'.format(interp_layers)
surgery.interp(solver.net, interp_layers)  # calc deconv filter weights

# scoring
val = np.loadtxt(file_location[:file_location.rfind(
    '/')] + '/data/cs-trip/val.txt', dtype=str)
trainset = np.loadtxt(file_location[:file_location.rfind(
    '/')] + '/data/cs-trip/train.txt', dtype=str)
val_trip_acc_baseline = 0.45

for _ in range(80):
    print '------------------------------'
    print 'Running solver.step iter {}'.format(_)
    print '------------------------------'
    solver.step(50)

    val_trip_acc = score.seg_loss_tests(solver, val, layer='score')
    train_trip_acc = score.seg_loss_train_test(solver, trainset, layer='score')
    # print 'Checking validation acc. Acc={}, baseline={}'.format(
    #     val_trip_acc,
    #     val_trip_acc_baseline)
    if save_weights and val_trip_acc is not None:
        print 'Checking validation acc'
        if val_trip_acc > val_trip_acc_baseline:
            print 'saving snapshot'
            solver.snapshot()
            val_trip_acc_baseline = val_trip_acc
    # if getting issues on HPC try
    # export MKL_CBWR=AUTO
    # and 'export CUDA_VISIBLE_DEVICES=1'
    # print '\n>>>> Validation <<<<\n'
print '\n completed colour only train'
