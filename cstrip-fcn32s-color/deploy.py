
# This import might casuse trouble, my network requires a custom version
# of caffe
import numpy as np
from PIL import Image
import os
import glob
import imp
import time
try:
    import cv2
except:
    print 'failed to load opencv2'
else:
    print 'opencv2 sucessfully loaded'

home_dir = os.path.expanduser("~")
if 'n8307628' in home_dir:
    caffe_root = os.path.join(home_dir, 'Fully-Conv-Network/Resources/caffe')
    base_dir = '/home/n8307628/'
elif 'sean' in home_dir:
    caffe_root = os.path.join(home_dir, 'src/caffe')
    base_dir = '/home/sean/hpc-home/'
else:
    print 'unknown directory'
    raise
print 'loading caffe from ', caffe_root
filename, path, desc = imp.find_module('caffe', [caffe_root + '/python/'])
caffe = imp.load_module('caffe', filename, path, desc)


def prepImage(img):
    # perform image preprocessing for CNN with caffe

    # no mean subtraction for now
    mean_bgr = np.array((0, 0, 0), dtype=np.float32)
    n_img = np.array(img, dtype=np.float32)
    # make bgr, may not be necessary
    n_img = n_img[:, :, ::-1]
    # mean subtract (nothing subtracted for now)
    n_img -= mean_bgr
    # reshape from w*h*3 to 3*w*h (or 3*h*w??)
    n_img = n_img.transpose((2, 0, 1))

    return n_img


def deploy(net, data, visualise=True, image_name='overlay_image'):
    score_layer = 'softmax_score'
    data_layer = 'data'
    in_img = prepImage(data)
    # reshape input for any sized image (data blob is N x C x H x W)
    net.blobs[data_layer].reshape(1, *in_img.shape)
    net.blobs[data_layer].data[...] = in_img
    start_time = time.time()
    net.forward()
    print 'foward pass took {} seconds'.format(time.time() - start_time)

    if visualise:
        # out = net.blobs[score_layer].data[0].argmax(axis=0)
        out = net.blobs[score_layer].data[0][1]
        sums = np.sum(net.blobs[score_layer].data[0], axis=0)
        # print 'out[0] shape ', np.shape(net.blobs[score_layer].data[0][1])
        print 'out[0][0] unique', np.unique(out)
        print 'softmax sum (should be all 1 ) shape', np.shape(
            sums), 'unique=', np.unique(sums)
        # give array img values, do i need to convert to BGR?
        out_img = out.astype(np.uint8) * 255

        # load img as PIL
        img = np.array(data, dtype=np.uint8)
        print 'img[0][0] unique', np.unique(img)
        colorIm = Image.fromarray(img)
        # Network prediction as PIL image
        im = Image.fromarray(out_img, mode='P')
        overlay = Image.blend(colorIm.convert(
            "RGBA"), im.convert("RGBA"), 0.7)
        basename = os.path.splitext(os.path.basename(image_name))[0]

        if 'n8307628' in home_dir:
            overlay.save(os.path.join(home_dir, 'baysian_cnn_images',
                                      basename + '_overlay.png'))
        else:
            cv_overlay = np.array(overlay)
            cv2.imshow('Image', out_img)
            cv2.waitKey(0)
            cv2.imshow('Image2', cv_overlay)
            cv2.waitKey(0)
        # convert from rgb to bgr - needed?
        # cv_overlay = cv_overlay[:, :, ::-1].copy()
    return net

if __name__ == '__main__':
    file_location = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    # load image
    test_dir = os.path.join(
        base_dir, 'Construction_Site/Springfield/12Aug16/K2/2016-08-12-10-09-26_groundfloorCarPark/labelled_colour/')
    img_names = glob.glob(os.path.join(test_dir, '*.png'))
    # initialise network
    arch = os.path.join(file_location, 'deploy_col.prototxt')
    weights = os.path.join(
        base_dir, 'Fully-Conv-Network/Resources/FCN_models/cstrip-fcn32s-color/colorSnapshot/_iter_6000.caffemodel')
    caffe.set_mode_gpu()
    net = caffe.Net(arch, weights, caffe.TRAIN)
    num_images = len(img_names)
    for count, name in enumerate(img_names):
        image = Image.open(name)
        # print 'loaded image ', os.path.basename(name), ' has shape ', np.shape(image)
        # forward pass
        print 'foward pass', count + 1, 'of', num_images
        net = deploy(net, image, visualise=True, image_name=name)
        break
