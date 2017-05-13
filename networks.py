try:
    import caffe
except ImportError:
    import imp
    caffe_root = '/home/n8307628/Fully-Conv-Network/Resources/caffe'
    filename, path, desc = imp.find_module('caffe', [caffe_root + '/python/'])
    caffe = imp.load_module('caffe', filename, path, desc)
    caffe.set_mode_gpu()
from caffe import layers as L, params as P
from caffe.coord_map import crop
import tempfile
import os.path
import net_archs
# for engine: 1 CAFFE 2 CUDNN
defEngine = 1


def createNet(split, net_type='rgb', f_multi=5, dropout_prob=0.5,
              engine=1, freeze=False, conv11_multi=4):

    if net_type == 'rgb' or net_type == 'RGB':
        tops = ['color', 'label']
        net = net_archs.fcn(split, tops, engineNum=engine, final_multi=f_multi,
                            dropout_prob=dropout_prob, freeze=freeze)
    elif net_type == 'depth' or net_type == 'Depth':
        tops = ['depth', 'label']
        net = net_archs.fcn(split, tops, engineNum=engine, final_multi=f_multi,
                            dropout_prob=dropout_prob, freeze=freeze)
    elif net_type == 'hha2' or net_type == 'HHA2':
        tops = ['hha2', 'label']
        net = net_archs.fcn(split, tops, engineNum=engine, final_multi=f_multi,
                            dropout_prob=dropout_prob, freeze=freeze)
    elif net_type == 'rgbd_early' or net_type == 'RGBD_early':
        tops = ['color', 'depth', 'label']
        net = net_archs.fcn_early(split, tops, engineNum=engine,
                                  conv1_1_lr_multi=conv11_multi,
                                  final_multi=f_multi, dropout_prob=dropout_prob,
                                  freeze=freeze)
    elif net_type == 'rgbhha2_early' or net_type == 'rgbHHA2_early':
        tops = ['color', 'hha2', 'label']
        net = net_archs.fcn_early(split, tops, engineNum=engine,
                                  conv1_1_lr_multi=conv11_multi,
                                  final_multi=f_multi, dropout_prob=dropout_prob,
                                  freeze=freeze)
    elif net_type == 'rgbd_conv' or net_type == 'RGBD_conv':
        tops = ['color', 'depth', 'label']
        net = net_archs.fcn_conv(split, tops, engineNum=engine,
                                 final_multi=f_multi, dropout_prob=dropout_prob,
                                 freeze=freeze)
    elif net_type == 'rgbhha2_conv' or net_type == 'rgbHHA2_conv':
        tops = ['color', 'hha2', 'label']
        net = net_archs.fcn_conv(split, tops, engineNum=engine,
                                 final_multi=f_multi, dropout_prob=dropout_prob,
                                 freeze=freeze)
    elif net_type == 'rgbd_lateMix' or net_type == 'RGBD_lateMix':
        tops = ['color', 'depth', 'label']
        net = net_archs.mixfcn(split, tops, engineNum=engine,
                               final_multi=f_multi, dropout_prob=dropout_prob,
                               freeze=freeze)
    elif net_type == 'rgbhha2_lateMix' or net_type == 'rgbHHA2_lateMix':
        tops = ['color', 'hha2', 'label']
        net = net_archs.mixfcn(split, tops, engineNum=engine,
                               final_multi=f_multi, dropout_prob=dropout_prob,
                               freeze=freeze)
    else:
        raise(Exception('net_type "' + net_type +
                        '" unrecognised, create case for new network here.'))
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(net.to_proto()))
        return f.name


def print_net(path, split='test', net_type='rgb'):
    if net_type == 'rgb' or net_type == 'RGB':
        tops = ['color', 'label']
        net = net_archs.fcn(split, tops)
    elif net_type == 'depth' or net_type == 'Depth':
        tops = ['depth', 'label']
        net = net_archs.fcn(split, tops)
    elif net_type == 'hha2' or net_type == 'HHA2':
        tops = ['hha2', 'label']
        net = net_archs.fcn(split, tops)
    elif '_early' in net_type:
        if 'd' in net_type or 'D' in net_type:
            tops = ['color', 'depth', 'label']
        elif 'hha2' in net_type or 'HHA2' in net_type:
            tops = ['color', 'hha2', 'label']
        else:
            raise(Exception('Unkown modality for early fusion: ' + net_type))
        net = net_archs.fcn_early(split, tops)
    elif '_conv' in net_type:
        if 'd' in net_type or 'D' in net_type:
            tops = ['color', 'depth', 'label']
        elif 'hha2' in net_type or 'HHA2' in net_type:
            tops = ['color', 'hha2', 'label']
        else:
            raise(Exception('Unkown modality for conv fusion: ' + net_type))
        net = net_archs.fcn_conv(split, tops)
    elif '_lateMix' in net_type:
        if 'd' in net_type or 'D' in net_type:
            tops = ['color', 'depth', 'label']
        elif 'hha2' in net_type or 'HHA2' in net_type:
            tops = ['color', 'hha2', 'label']
        else:
            raise(Exception('Unkown modality for lateMix fusion: ' + net_type))
        net = net_archs.mixfcn(split, tops)
    else:
        raise(Exception('net_type "' + net_type +
                        '" unrecognised, create case for new network here.'))
    with open(os.path.join(path, split + '.prototxt'), 'w') as f:
        f.write(str(net.to_proto()))


def print_rgb_nets():
    tops = ['color', 'label']
    with open('trainval.prototxt', 'w') as f:
        f.write(str(net_archs.fcn('train', tops).to_proto()))

    with open('val.prototxt', 'w') as f:
        f.write(str(net_archs.fcn('val', tops).to_proto()))

    with open('test.prototxt', 'w') as f:
        f.write(str(net_archs.fcn('test', tops).to_proto()))


def test_print(net_type='rgb'):
    n = createNet('trainval', net_type=net_type)
    print_net('', split='trainval', net_type=net_type)
    print_net('', split='val', net_type=net_type)
    print_net('', split='test', net_type=net_type)

if __name__ == '__main__':
    n_type = 'rgbd_conv'
    test_print(n_type)
