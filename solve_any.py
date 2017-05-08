#! /usr/bin/python
"""
trip trainer, designed to work with any modality

By Sean McMahon

"""
# import caffe
import numpy as np
import os
import sys
from os.path import expanduser
import imp
import argparse
import tempfile
import glob

# add '../' directory to path for importing score.py, surgery.py and
# pycaffe layer
# for engine: 1 CAFFE 2 CUDNN
CAFFE_eng = 1
CUDNN_eng = 2
file_location = os.path.realpath(os.path.join(
    os.getcwd(), os.path.dirname(__file__)))
sys.path.append(file_location)
home_dir = expanduser("~")
# User Input
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='gpu')
parser.add_argument('--working_dir', default='rgb_1')
parser.add_argument('--traintest_fold', default='1_7')
parser.add_argument('--network_modality', default='rgb')
parser.add_argument('--base_lr', default=None)
parser.add_argument('--net_type', default=None)
parser.add_argument('--net_init', default=None)
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
            weights_path,
            'pretrained_weights/nyud-fcn32s-color-heavy.caffemodel')
        print 'Pretrain on NYU weights'
    elif params_dict['weight_init'] == "CS_rgb":
        weights = os.path.join(
            weights_path,
            'cstrip-fcn32s-color/colorSnapshot/_iter_2000.caffemodel')
        print 'Pretrain on CS weights (_iter_2000.caffemodel)'
    elif params_dict['weight_init'] == "NYU_hha":
        weights = '/home/n8307628/Fully-Conv-Network/' + \
            'Resources/FCN_models/pretrained_weights/' + \
            'nyud-fcn32s-hha-heavy.caffemodel'
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
    val_net_name = networks.createNet(params_dict.get('test_set', val_name),
                                      net_type=params_dict['type'],
                                      f_multi=0, engine=0)
    train_net_name = networks.createNet(params_dict.get('train_set', 'train'),
                                        net_type=params_dict['type'],
                                        f_multi=params_dict['f_multi'],
                                        dropout_prob=params_dict['dropout'],
                                        engine=0,
                                        freeze=params_dict.get(
                                            'freeze_layers', False),
                                        conv11_multi=params_dict.get(
        'conv11_multi', 2))
    # createNet only uses conv11_multiwith early fusion!

    # init solver
    solver_name = createSolver(params_dict,
                               train_net_name, val_net_name, work_dir)
    solver = caffe.get_solver(solver_name)

    # Weight Initialisation
    if params_dict['type'] == 'depth' or params_dict['type'] == 'Depth':
        if 'hha' in params_dict['weight_init']:
            init_net_name = networks.createNet(val_name, net_type='hha2',
                                               f_multi=0, engine=0)
        elif 'rgb' in params_dict['weight_init']:
            init_net_name = networks.createNet(val_name, net_type='rgb',
                                               f_multi=0, engine=0)
        else:
            raise(Exception('Do not know how to initialise depth weights.' +
                            '\nUnkown weight_init ({})\n'.format(
                                params_dict['weight_init'])))
        init_net = caffe.Net(init_net_name, weights, caffe.TEST)
        surgery.transplant(solver.net, init_net)
        del init_net
    elif '_early' in params_dict['type']:
        # for early, need to initialise conv1_1 and then specify pre-training
        # for rest of the network
        # TODO check if net.copy_from workers properly early fuson networks
        # have different sized conv1_1 layers

        # Copy layers half of conv1_1_ and rest from another network
        base_net_name = networks.createNet('val', net_type='rgb', engine=0)
        base_net = caffe.Net(base_net_name, weights, caffe.TEST)
        surgery.transplant(solver.net, base_net)
        del base_net

        # Iniitlias conv1_1 layer
        if 'hha2' in params_dict['type'] or 'HHA2' in params_dict['type']:
            d_top = 'hha2'
            default_weights_depth = '/home/n8307628/Fully-Conv-Network/' + \
                'Resources/FCN_models/pretrained_weights/' + \
                'nyud-fcn32s-hha-heavy.caffemodel'
        elif 'rgbd' in params_dict['type'] or 'RGBD' in params_dict['type']:
            d_top = 'depth'
            default_weights_depth = '/home/n8307628/Fully-Conv-Network/' + \
                'Resources/FCN_models/cstrip-fcn32s-depth/' + \
                'DepthSnapshot/negOneNull_mean_sub_iter_8000.caffemodel'
        else:
            raise(Exception('"type" param given contains unkown modality: ' +
                            params_dict['type']))
        weights_depth = params_dict.get('weight_init_depth',
                                        default_weights_depth)
        print 'Early Fusion: using depth weights from {}'.format(weights_depth)
        base_depth_name = networks.createNet('val', net_type=d_top, f_multi=0,
                                             engine=0)
        base_net_depth = caffe.Net(base_depth_name, weights_depth,
                                   caffe.TEST)
        print 'copying Depth params from conv1_1 -> conv1_1_bgr' + d_top
        try:
            depth_filters = base_net_depth.params['conv1_1'][0].data
            solver.net.params['conv1_1_bgr' + d_top][0].data[:, 3] = np.squeeze(
                depth_filters)
        except ValueError:
            # probs 3 channel, average the weights and combine
            solver.net.params['conv1_1_bgr' + d_top][0].data[:, 3] = np.mean(
                base_net_depth.params['conv1_1'][0].data, axis=1)
        del base_net_depth
    elif '_conv' in params_dict['type']:
        raise(Exception("Have not written code to initialise conv fusion"))
    elif '_lateMix' in params_dict['type']:
        color_proto = '/home/n8307628/Fully-Conv-Network/' + \
            'Resources/FCN_models' + '/cstrip-fcn32s-color/test.prototxt'
        hha2_proto = '/home/n8307628/Fully-Conv-Network/' + \
            'Resources/FCN_models' + '/cstrip-fcn32s-hha2/test.prototxt'
        if 'CS' in params_dict['weight_init']:
            color_weights = '/home/n8307628/Fully-Conv-Network/' + \
                'Resources/FCN_models/cstrip-fcn32s-color/colorSnapshot/_iter_2000.caffemodel'
            # color_proto = '/home/n8307628/Fully-Conv-Network/' + \
            #     'Resources/FCN_models' + '/cstrip-fcn32s-color/test.prototxt'
            hha2_weights = '/home/n8307628/Fully-Conv-Network/' + \
                'Resources/FCN_models/cstrip-fcn32s-hha2/HHA2snapshot/' + \
                'secondTrain_lowerLR_iter_2000.caffemodel'
            # hha2_proto = '/home/n8307628/Fully-Conv-Network/' + \
            #     'Resources/FCN_models' + '/cstrip-fcn32s-hha2/test.prototxt'
            depth_weights = '/home/n8307628/Fully-Conv-Network/Resources/FCN_models' + \
                '/cstrip-fcn32s-depth/DepthSnapshot/' + \
                'stepLR2_lowerLR_neg1N_Msub_iter_6000.caffemodel'
            depth_proto = '/home/n8307628/Fully-Conv-Network/Resources/FCN_models' \
                + '/cstrip-fcn32s-depth/test.prototxt'
        elif 'NYU' in params_dict['weight_init']:
            color_weights = os.path.join(
                weights_path,
                'pretrained_weights/nyud-fcn32s-color-heavy.caffemodel')
            # color_proto = os.path.join(weights_path,
            #                            'nyud-fcn32s-color/test.prototxt')
            hha2_weights = os.path.join(weights_path,
                                        'pretrained_weights/' +
                                        'nyud-fcn32s-hha-heavy.caffemodel')
            # hha2_proto = os.path.join(weights_path,
            #                           'nyud-fcn32s-hha/test.prototxt')
            # currently have no depth networks trained on NYU
            depth_weights = hha2_weights
            depth_proto = hha2_proto
        else:
            raise(Exception('"type" param given contains unkown modality: ' +
                            params_dict['type']))

        # surgeries
        color_net = caffe.Net(color_proto, color_weights, caffe.TEST)
        surgery.transplant(solver.net, color_net, suffix='color')
        del color_net

        if 'hha2' in params_dict['type'] or 'HHA2' in params_dict['type']:
            hha2_net = caffe.Net(hha2_proto, hha2_weights, caffe.TEST)
            surgery.transplant(solver.net, hha2_net, suffix='hha2')
            del hha2_net
        elif 'rgbd' in params_dict['type'] or 'RGBD' in params_dict['type']:
            depth_net = caffe.Net(depth_proto, depth_weights, caffe.TEST)
            surgery.transplant(solver.net, depth_net, suffix='depth')
            del depth_net
        else:
            raise(Exception('Uknown modality for late mix fusion'))
    else:
        solver.net.copy_from(weights)

    # surgeries -> Create weights for deconv (bilinear upsampling)
    interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
    print 'performing surgery on {}'.format(interp_layers)
    surgery.interp(solver.net, interp_layers)  # calc deconv filter weights

    # scoring
    val = np.loadtxt(os.path.join(
        file_location, 'data/cs-trip/' + params_dict.get('test_set',
                                                         val_name) + '.txt'),
                     dtype=str)
    trainset = np.loadtxt(os.path.join(file_location,
                                       'data/cs-trip/train.txt'), dtype=str)
    val_trip_acc_baseline = 0.45
    val_loss_buf = 5000000.0

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
            print 'Minimum val loss @ iter {}, saving'.format(solver.iter)
            min_loss_iter = solver.iter
            solver.snapshot()
            # save the outputs!
            # test_img_save = os.path.join(work_dir,
            #                              'output_iter_{}_'.format(solver.iter) +
            #                              params_dict.get('test_set', val_name))
            # score.seg_tests(solver, test_img_save, val, layer='score')
    # if getting issues on HPC try
    # export MKL_CBWR=AUTO
    # and 'export CUDA_VISIBLE_DEVICES=1'
    # print '\n>>>> Validation <<<<\n'
    print '\n completed colour only train'
    test_net_name = networks.createNet(params_dict.get('test_set', 'test'),
                                       net_type=params_dict['type'],
                                       f_multi=params_dict['f_multi'],
                                       dropout_prob=params_dict['dropout'],
                                       engine=0,
                                       freeze=params_dict.get(
                                       'freeze_layers', False),
                                       conv11_multi=params_dict.get(
                                       'conv11_multi', 2))
    solver_name = createSolver(params_dict,
                               train_net_name, val_net_name, work_dir)
    solver = caffe.get_solver(solver_name)
    solver.net.copy_from(test_weights)
    test_net = caffe.SGDSolver()
    test_img_save = os.path.join(work_dir,
                                 'output_iter_{}_'.format(min_loss_iter) +
                                 params_dict.get('test_set', val_name))
    score.seg_tests(solver, test_img_save, val, layer='score')


if __name__ == '__main__':
    work_path = args.working_dir
    net_modal = args.network_modality
    cv_fold = args.traintest_fold
    in_base_lr = args.base_lr
    in_net_type = args.net_type
    in_net_init = args.net_init
    # TODO incorporate net_modal into params dict
    print 'Solver given working dir: ', work_path
    if '/home' not in work_path:
        work_dir = os.path.join(file_location, work_path)
    else:
        work_dir = work_path
    if os.path.isdir(work_dir):
        logfilename = glob.glob(os.path.join(work_dir, '*.log'))
        if len(logfilename) > 1:
            raise(
                Exception(
                    'work directoy: ' +
                    '\n"{}"\nalready has logfile, quitting'.format(work_dir)))

    dropout_regularisation = round(np.random.uniform(0.2, 0.9), 3)
    learning_rate = round(10 ** np.random.uniform(-13, -10), 16)
    final_learning_multiplier = np.random.randint(1, 10)
    freeze_lower_layers = bool(np.random.randint(0, 2))  # sometimes false bra
    # again 'lr_mult_conv11' will only be used for early fusion
    lr_mult_conv11 = np.random.randint(1, 6)
    # params_dict = {'base_lr': learning_rate, 'solverType': 'SGD',
    #                'f_multi': final_learning_multiplier,
    #                'dropout': dropout_regularisation,
    #                'freeze_layers': freeze_lower_layers,
    #                'type': 'rgbhha2_early', 'weight_init': 'NYU_hha',
    #                'rand_seed': 3711,
    #                'conv11_multi': lr_mult_conv11}
    if in_base_lr is None:
        cv_learning_rate = 1e-10
    else:
        cv_learning_rate = in_base_lr
    if in_net_type is None:
        cv_net_type = 'rgbhha2_early'
    else:
        cv_net_type = in_net_type
    if in_net_init is None:
        cv_weight_init = 'NYU_rgb'
    else:
        cv_weight_init = in_net_init

    if cv_fold == 'o':
        test_set = 'val2'
        train_set = 'train'
    else:
        test_set = 'val_' + cv_fold
        train_set = 'train_' + cv_fold
    cv_lr_mult_conv11 = 4
    cv_final_multi = 5
    cv_freeze = False
    params_dict_crossval = {'base_lr': cv_learning_rate, 'solverType': 'SGD',
                            'f_multi': cv_final_multi,
                            'dropout': 0.5,
                            'freeze_layers': cv_freeze,
                            'type': cv_net_type, 'weight_init': cv_weight_init,
                            'conv11_multi': cv_lr_mult_conv11,
                            'test_set': test_set,
                            'train_set': train_set}
    print 'Solver writing to dir: ', work_dir
    write_dict(params_dict_crossval, work_dir)

    run_solver(params_dict_crossval, work_dir)
