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
import re

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
parser.add_argument('--working_dir', default='solver_test')
parser.add_argument('--traintest_fold', default='o')
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


def run_test(params_dict, test_iter, work_dir, score_layer='score',
             data_layer='data'):
    """
    run the trained network on the testset. Using the lowest loss found during
    training.
    This function is designed to be used outside of the run solver function.
    Metrics will be printed, network scores, network labels and network output
    images will be saved in a new directoy within work_dir.
    """
    test_net_name = networks.createNet(params_dict['test_set'],
                                       net_type=params_dict['type'],
                                       engine=0)
    # solver_name = createSolver(params_dict,
    #                            test_net_name, test_net_name, work_dir)
    # solver = caffe.get_solver(solver_name)
    test_weights = os.path.join(work_dir, 'snapshots',
                                params_dict['type'] +
                                '_iter_' + str(test_iter) + '.caffemodel')
    # solver.net.copy_from(test_weights)
    test_img_save = os.path.join(work_dir,
                                 'test_iter_{}_'.format(test_iter) +
                                 params_dict['test_set'])
    test_net = caffe.Net(test_net_name, test_weights, caffe.TEST)
    test_txt = np.loadtxt(os.path.join(
        file_location, 'data/cs-trip/' + params_dict['test_set'] + '.txt'),
        dtype=str)
    # score.seg_tests(solver, test_img_save, test_txt, layer='score')
    score.do_seg_tests(test_net, test_iter, test_img_save,
                       test_txt, layer=score_layer, dataL=data_layer)


def test_all_cv():
    cross_val_sets = ['1', '2', '3', '4', '5']  # ['1_4', '2_4', '3_4', '4_4']
    net_types = ['rgbd_conv', 'rgbhha2_conv']
    # ['rgb', 'hha2', 'depth',
    #              'rgbd_early', 'rgbhha2_early', 'rgbd_lateMix', 'rgbhha2_lateMix']
    for test_set in cross_val_sets:
        net_dirs = ['convsearch/rgbd_conv' + test_set,
                    'convsearch/rgbhha_conv' + test_set]
        # ['rgb_crossval2/rgb_' + test_set,
        #             'hha_crossval2/hha_' + test_set,
        #             'depth_crossval2/depth_' + test_set,
        #             'earlyrgbd_crossval2/earlyrgbd_' + test_set,
        #             'earlyrgbhha_crossval2/earlyrgbhha_' + test_set,
        #             'lateMixrgbd_crossval2/lateMixrgbd_' + test_set,
        #             'lateMixrgbhha_crossval2/lateMixrgbhha_' + test_set]
        min_loss_pattern = r"Minimum val loss @ iter (?P<iter_num>\d+), saving"
        for work_dir, net_type in zip(net_dirs, net_types):
            # Get the iteration with best performance
            logfile = glob.glob(os.path.join(work_dir, '*.log'))
            if not logfile or len(logfile) > 1:
                print 'len(logfile)= {}, search str = {}'.format(
                    len(logfile),
                    os.path.join(work_dir, '*.log'))
                raise(Exception('Either no logfile found or too many found.'))
            with open(logfile[0], 'r') as f:
                logstr = f.read()
            try:
                min_loss_iter = re.findall(min_loss_pattern, logstr)[-1]
            except IndexError:
                # find highest iteration snapshot
                print 'Could not find min loss, ' + \
                    'searching snapshots for highest iteration'
                models = glob.glob(os.path.join(work_dir, 'snapshots',
                                                '*.caffemodel'))
                min_loss_iter = 0
                for model in models:
                    pattern = r'_iter_(?P<iter_num>\d+).caffemodel'
                    iteration = int(re.findall(pattern, model)[0])
                    if iteration > min_loss_iter:
                        min_loss_iter = iteration

            # Run the test, now we have the best iteration
            # because we're testing only the testset and net type matter
            params_test_cv = {'type': net_type, 'test_set': 'test_' + test_set}
            # params_test_conv = {'type': net_type, 'test_set': 'test'}
            print '\n', '=' * 50, '\n', 'testing: {} in {} \n'.format(net_type, work_dir)
            run_test(params_test_cv, min_loss_iter,
                     work_dir, data_layer='color')
            print 'tested {} in {} \n'.format(net_type, work_dir)


def checkWeightInit(net, modality='',
                    layers=('conv1_2', 'conv2_2',
                            'conv3_1', 'conv4_2', 'conv5_3')):
    # just check each phase of convolutions, they should always be initialised
    eps = 1e-6
    for raw_layer in layers:
        layer = raw_layer + modality
        w_logic = np.absolute(net.params[layer][0].data) < eps
        if np.all(w_logic.flatten()):
            print 'layer {} is uninitialised. Params:'.format(layer)
            print net.params[layer][0].data
            print '-' * 30
            return False
    print 'Layers "{}" are initialised'.format(layers)
    return True


def run_solver(params_dict, work_dir):
    print '\n--------------------------'
    print 'Running solver with parms:'
    for param in params_dict:
        print param, ':',  params_dict[param]
    print '--------------------------\n'
    save_weights = params_dict.get('save_weights', True)

    NYU_rgb_weights_path = os.path.join(
        weights_path,
        'pretrained_weights/nyud-fcn32s-color-heavy.caffemodel')
    NYU_hha_weights_path = os.path.join(
        weights_path,
        'pretrained_weights/nyud-fcn32s-hha-heavy.caffemodel')
    CS_rgb_weights_path = os.path.join(
        weights_path,
        'cstrip-fcn32s-color/colorSnapshot/_iter_2000.caffemodel')
    CS_depth_weights_path = os.path.join(
        weights_path, 'cstrip-fcn32s-depth/' +
        'DepthSnapshot/negOneNull_mean_sub_iter_8000.caffemodel')
    CS_hha2_weights_path = os.path.join(
        weights_path, 'cstrip-fcn32s-hha2/HHA2snapshot/' +
        'secondTrain_lowerLR_iter_2000.caffemodel')

    if params_dict['weight_init'] == "NYU_rgb":
        weights = NYU_rgb_weights_path
        print 'Pretrain on NYU weights'
    elif params_dict['weight_init'] == "CS_rgb":
        weights = CS_rgb_weights_path
        print 'Pretrain on CS weights (_iter_2000.caffemodel)'
    elif params_dict['weight_init'] == "NYU_hha":
        weights = NYU_hha_weights_path
        # '/home/n8307628/Fully-Conv-Network/' + \
        #     'Resources/FCN_models/pretrained_weights/' + \
        #     'nyud-fcn32s-hha-heavy.caffemodel'
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
    val_net_name = networks.createNet(params_dict.get('val_set', val_name),
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
        # ------------------- Initialise Conv Fusion --------------------------
        if 'CS' in params_dict['weight_init']:
            rgb_weights = CS_rgb_weights_path
            if 'hha2' in params_dict['type'] or 'HHA2' in params_dict['type']:
                base_depth_name = networks.createNet('val2', net_type='hha2')
                base_modal_net = caffe.Net(base_depth_name, CS_hha2_weights_path,
                                           caffe.TEST)
            elif 'rgbd' in params_dict['type'] or 'RGBD' in params_dict['type']:
                base_depth_name = networks.createNet('val2', net_type='depth')
                base_modal_net = caffe.Net(base_depth_name, CS_depth_weights_path,
                                           caffe.TEST)
            else:
                raise(Exception('Unkown modalities given for conv fusion'))
        elif 'NYU' in params_dict['weight_init']:
            rgb_weights = NYU_rgb_weights_path
            # only have hha for NYU. surgery.transplant will fit hha2 weigths to a
            # depth network
            base_depth_name = networks.createNet('val2', net_type='hha2')
            base_modal_net = caffe.Net(base_depth_name, NYU_hha_weights_path,
                                       caffe.TEST)
        else:
            raise(Exception('Unkown weight initailsation given for conv fusion'))

        if 'hha2' in params_dict['type'] or 'HHA2' in params_dict['type']:
            surgery.transplant(solver.net, base_modal_net, suffix='hha2')
        elif 'rgbd' in params_dict['type'] or 'RGBD' in params_dict['type']:
            surgery.transplant(solver.net, base_modal_net, suffix='depth')
        del base_modal_net
        print '\nRGB Init\n', '-' * 76, '\n'
        base_rgb_name = networks.createNet('val2', net_type='rgb')
        base_rgb_net = caffe.Net(base_rgb_name, rgb_weights, caffe.TEST)
        surgery.transplant(solver.net, base_rgb_net, suffix='color')
        del base_rgb_net

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

    # Check if some of the weights are initialised from each modality
    # TODO write coode to check all. They should all at least have random init
    if 'rgbd' in params_dict['type'] or 'RGBD' in params_dict['type']:
        rgb_status = checkWeightInit(solver.net, modality='color')
        depth_status = checkWeightInit(solver.net, modality='depth')
        if rgb_status is False:
            raise(Exception('RGB weight init check failed'))
        elif depth_status is False:
            raise(Exception('Depth weight init check failed'))
    elif 'hha2' in params_dict['type'] or 'HHA2' in params_dict['type']:
        rgb_status = checkWeightInit(solver.net, modality='color')
        hha_status = checkWeightInit(solver.net, modality='hha2')
        if rgb_status is False:
            raise(Exception('RGB weight init check failed'))
        elif hha_status is False:
            raise(Exception('HHA2 weight init check failed'))
    else:
        net_status = checkWeightInit(solver.net)
        if net_status is False:
            raise(Exception('Weight init check failed'))

    # surgeries -> Create weights for deconv (bilinear upsampling)
    interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
    print 'performing surgery on {}'.format(interp_layers)
    surgery.interp(solver.net, interp_layers)  # calc deconv filter weights

    # scoring
    val = np.loadtxt(os.path.join(
        file_location, 'data/cs-trip/' + params_dict.get('val_set',
                                                         val_name) + '.txt'),
                     dtype=str)
    trainset = np.loadtxt(os.path.join(file_location,
                                       'data/cs-trip/train.txt'), dtype=str)
    val_trip_acc_baseline = 0.35
    val_loss_buf = 90000000.0

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
                print 'Minimum val accuracy @ {}, saving'.format(solver.iter)
                solver.snapshot()
                val_trip_acc_baseline = val_trip_acc
                max_val_acc_iter = solver.iter
        if val_loss < val_loss_buf:
            val_loss_buf = val_loss
            print 'Minimum val loss @ iter {}, saving'.format(solver.iter)
            min_loss_iter = solver.iter
            solver.snapshot()
            # save the outputs!
            # test_img_save = os.path.join(work_dir,
            #                              'output_iter_{}_'.format(solver.iter) +
            #                              params_dict.get('val_set', val_name))
            # score.seg_tests(solver, test_img_save, val, layer='score')
    # if getting issues on HPC try
    # export MKL_CBWR=AUTO
    # and 'export CUDA_VISIBLE_DEVICES=1'
    # print '\n>>>> Validation <<<<\n'
    print '\n completed colour only train'
    print '-' * 50
    print 'Testing lowest loss iteration on test set.'
    try:
        min_loss_iter
    except NameError:
        # if no min loss found use best val acc
        min_loss_iter = max_val_acc_iter
    print '\ncompleted testing'
    return min_loss_iter

# ==============================================================================
# =========================== Main =======================================
# ==============================================================================

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
    if cv_fold == 'o':
        val_set = 'val2'
        train_set = 'train'
        test_set = 'test'
    else:
        val_set = 'val_' + cv_fold
        train_set = 'train_' + cv_fold
        test_set = 'test_' + cv_fold

    # --------------------------------------------------------------------------
    # set network type, net initialisaion and hyperparameters.
    # --------------------------------------------------------------------------
    dropout_regularisation = round(np.random.uniform(0.2, 0.9), 3)
    learning_rate = round(10 ** np.random.uniform(-13, -10), 16)
    final_learning_multiplier = np.random.randint(1, 10)
    freeze_lower_layers = bool(np.random.randint(0, 2))  # sometimes false bra
    # again 'lr_mult_conv11' will only be used for early fusion
    lr_mult_conv11 = np.random.randint(1, 6)
    params_dict = {'base_lr': learning_rate, 'solverType': 'SGD',
                   'f_multi': final_learning_multiplier,
                   'dropout': dropout_regularisation,
                   'freeze_layers': freeze_lower_layers,
                   'type': 'rgbhha2_early', 'weight_init': 'NYU_hha',
                   'rand_seed': 3711,
                   'conv11_multi': lr_mult_conv11,
                   'val_set': val_set,
                   'train_set': train_set,
                   'test_set': test_set}

    cv_learning_rate = 1e-11
    cv_net_type = 'rgbhha2_conv'
    cv_weight_init = 'NYU_rgb'
    cv_lr_mult_conv11 = 4
    cv_final_multi = 5
    cv_freeze = False
    if '_conv' in cv_net_type:
        dropout_reg = 0.75
    else:
        dropout_reg = 0.5
    params_dict_crossval = {'base_lr': cv_learning_rate, 'solverType': 'SGD',
                            'f_multi': cv_final_multi,
                            'dropout': dropout_reg,
                            'freeze_layers': cv_freeze,
                            'type': cv_net_type, 'weight_init': cv_weight_init,
                            'conv11_multi': cv_lr_mult_conv11,
                            'val_set': val_set,
                            'train_set': train_set,
                            'test_set': test_set}
    print 'Solver writing to dir: ', work_dir

    # --------------------------------------------------------------------------
    # Save params to text file, train and validate network, then test net.
    # --------------------------------------------------------------------------
    # write_dict(params_dict_crossval, work_dir)
    # best_val_per_iter = run_solver(params_dict_crossval, work_dir)
    # run_test(params_dict_crossval, best_val_per_iter, work_dir)
    test_all_cv()
