#! /usr/bin/python
"""
trip trainer, designed to work with any modality

By Sean McMahpn

"""
# import caffe
import numpy as np
import os
import sys
from os.path import expanduser
import imp
import argparse
import tempfile

# add '../' directory to path for importing score.py, surgery.py and
# pycaffe layer
# for engine: 1 CAFFE 2 CUDNN
CAFFE_eng = 1
CUDNN_eng = 2
file_location = os.path.realpath(os.path.join(
    os.getcwd(), os.path.dirname(__file__)))
sys.path.append(file_location[:file_location.rfind('/')])
home_dir = expanduser("~")
# User Input
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='gpu')
parser.add_argument('--working_dir', default='rgb_1')
args = parser.parse_args()
# import support functions
if 'n8307628' in home_dir:
    caffe_root = home_dir + '/Fully-Conv-Network/Resources/caffe'
    weights_path = home_dir + \
        '/Fully-Conv-Network/Resources/FCN_models'
elif 'sean' in home_dir:
    caffe_root = home_dir + '/src/caffe'
    weights_path = home_dir + '/hpc-home/Fully-Conv-Network/Resources/FCN_models'
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
import networks
from caffe.proto import caffe_pb2


def write_dict(in_dict, work_dir):
    if not os.path.isdir(work_dir):
        os.mkdir(work_dir)
    with open(os.path.join(work_dir, 'params.txt'), 'w') as f:
        for key, value in in_dict.items():
            f.write('{}: {}\n'.format(key, value))


def createSolver(params_dict, train_net_path, test_net_path, work_dir):
    s = caffe_pb2.SolverParameter()
    s.train_net = train_net_path
    s.test_net.append(test_net_path)
    s.test_interval = params_dict.get(
        'test_interval', 999999999)  # do not invoke tests here
    s.test_iter.append(params_dict.get('test_iter', 654))
    s.max_iter = params_dict.get('max_iter', 300000)
    s.base_lr = params_dict['base_lr']
    s.lr_policy = params_dict.get('lr_policy', 'fixed')
    s.gamma = params_dict.get('gamma', 0.1)
    s.average_loss = params_dict.get('average_loss', 20)
    s.momentum = params_dict.get('momentum', 0.99)
    s.iter_size = params_dict.get('iter_size', 1)
    s.weight_decay = params_dict.get('weight_decay', 0.0005)
    s.display = params_dict.get('display', 20)
    s.snapshot = params_dict.get('snapshot', 999999999)
    s.type = params_dict['solverType']
    s.random_seed = params_dict.get('rand_seed', np.random.randint(100, 9999))

    if not os.path.isdir(work_dir):
        os.mkdir(work_dir)
    snapshot_dir = os.path.join(work_dir, 'snapshots')
    if not os.path.isdir(snapshot_dir):
        os.mkdir(snapshot_dir)
    s.snapshot_prefix = os.path.join(snapshot_dir, params_dict['type'])
    s.test_initialization = params_dict.get('test_initialization', False)

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(s))
        return f.name


def run_solver(params_dict, work_dir):
    print '\n--------------------------'
    print 'Running solver with parms:'
    for param in params_dict:
        print param, ':',  params_dict[param]
    print '--------------------------\n'
    save_weights = params_dict.get('save_weights', True)

    if params_dict['weight_init'] == "NYU_rgb":
        weights = os.path.join(
            weights_path, 'pretrained_weights/nyud-fcn32s-color-heavy.caffemodel')
        print 'Pretrain on NYU weights'
    elif params_dict['weight_init'] == "CS_rgb":
        weights = os.path.join(
            weights_path, 'cstrip-fcn32s-color/colorSnapshot/_iter_2000.caffemodel')
        print 'Pretrain on CS weights (_iter_2000.caffemodel)'
    else:
        Exception('Unrecognised pretrain weights option given ({})'.format(
            params_dict['weight_init']))

    # init network arch
    # for engine: 1 CAFFE 2 CUDNN; CUDNN non-deterministic, but is quicker than CAFFE.
    # Basically use CAFFE for exact repeatable results and CUDNN for faster
    # run time
    networks.print_net(work_dir, split='test',
                       net_type=params_dict['type'])
    val_name = 'val2'
    val_net_name = networks.createNet(val_name, net_type=params_dict['type'],
                                      f_multi=0, engine=0)
    train_net_name = networks.createNet('train', net_type=params_dict['type'],
                                        f_multi=params_dict['f_multi'],
                                        dropout_prob=params_dict['dropout'],
                                        engine=0,
                                        freeze=params_dict.get(
                                            'freeze_layers', False))

    # init solver
    solver_name = createSolver(params_dict,
                               train_net_name, val_net_name, work_dir)
    solver = caffe.get_solver(solver_name)
    solver.net.copy_from(weights)

    # surgeries
    interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
    print 'performing surgery on {}'.format(interp_layers)
    surgery.interp(solver.net, interp_layers)  # calc deconv filter weights

    # scoring
    val = np.loadtxt(os.path.join(
        file_location, 'data/cs-trip/' + val_name + '.txt'), dtype=str)
    trainset = np.loadtxt(os.path.join(file_location,
                                       'data/cs-trip/train.txt'), dtype=str)
    val_trip_acc_baseline = 0.45
    val_loss_buf = 4000000.0

    for _ in range(100):
        print '------------------------------'
        print 'Running solver.step iter {}'.format(_)
        print '------------------------------'
        solver.step(50)

        val_trip_acc, val_loss = score.seg_loss_tests(
            solver, val, layer='score')
        train_trip_acc, train_loss = score.seg_loss_train_test(
            solver, trainset, layer='score')
        # print 'Checking validation acc. Acc={}, baseline={}'.format(
        #     val_trip_acc,
        #     val_trip_acc_baseline)

        if save_weights and val_trip_acc is not None:
            print 'Checking validation acc'

            if val_trip_acc > val_trip_acc_baseline:
                print 'saving snapshot'
                solver.snapshot()
                val_trip_acc_baseline = val_trip_acc
        if val_loss < val_loss_buf:
            val_loss_buf = val_loss
            if solver.iter > 60:
                print 'minimum val loss @ iter {}, saving'.format(solver.iter)
                solver.snapshot()
    # if getting issues on HPC try
    # export MKL_CBWR=AUTO
    # and 'export CUDA_VISIBLE_DEVICES=1'
    # print '\n>>>> Validation <<<<\n'
    print '\n completed colour only train'


if __name__ == '__main__':
    work_path = args.working_dir
    if '/home' not in work_path:
        work_dir = os.path.join(file_location, work_path)
    else:
        work_dir = work_path

    dropout_regularisation = round(np.random.uniform(0.2, 0.9), 3)
    learning_rate = round(10 ** np.random.uniform(-13, -10), 16)
    final_learning_multiplier = np.random.randint(1, 10)
    freeze_lower_layers = bool(np.random.randint(0, 1))
    params_dict = {'base_lr': learning_rate, 'solverType': 'SGD',
                   'f_multi': final_learning_multiplier,
                   'dropout': dropout_regularisation,
                   'freeze_layers': freeze_lower_layers,
                   'type': 'rgb', 'weight_init': 'NYU_rgb',
                   'rand_seed': 3711}
    write_dict(params_dict, work_dir)

    run_solver(params_dict, work_dir)
