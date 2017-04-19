import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop
import tempfile
import os.path
# for engine: 1 CAFFE 2 CUDNN
defEngine = 1


def createNet(split, net_type='rgb', f_multi=5, dropout_prob=0.5,
              engine=1, freeze=False):

    if net_type == 'rgb' or net_type == 'RGB':
        tops = ['color', 'label']
        net = fcn_rgb(split, tops, engineNum=engine, final_multi=f_multi,
                      dropout_prob=dropout_prob, freeze=freeze)
    elif net_type == 'depth' or net_type == 'Depth':
        tops = ['depth', 'label']
        net = fcn_rgb(split, tops, engineNum=engine, final_multi=f_multi,
                      dropout_prob=dropout_prob, freeze=freeze)
    elif net_type == 'hha2' or net_type == 'HHA2':
        tops = ['hha2', 'label']
        net = fcn_rgb(split, tops, engineNum=engine, final_multi=f_multi,
                      dropout_prob=dropout_prob, freeze=freeze)
    else:
        raise(Exception('net_type "' + net_type +
                        '" unrecognised, create case for new network here.'))
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(net.to_proto()))
        return f.name


def print_net(path, split='test', net_type='rgb'):
    if net_type == 'rgb' or net_type == 'RGB':
        tops = ['color', 'label']
        with open(os.path.join(path, split + '.prototxt'), 'w') as f:
            f.write(str(fcn_rgb(split, tops).to_proto()))
    else:
        raise(Exception('net_type "' + net_type +
                        '" unrecognised, create case for new network here.'))


def print_all_nets():
    tops = ['color', 'label']
    with open('trainval.prototxt', 'w') as f:
        f.write(str(fcn_rgb('train', tops).to_proto()))

    with open('val.prototxt', 'w') as f:
        f.write(str(fcn_rgb('val', tops).to_proto()))

    with open('test.prototxt', 'w') as f:
        f.write(str(fcn_rgb('test', tops).to_proto()))

if __name__ == '__main__':
    print_all_nets()
