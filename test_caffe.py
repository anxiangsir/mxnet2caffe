import argparse

import caffe
import numpy as np

parser = argparse.ArgumentParser(usage="it's usage tip.", description="help info.")
parser.add_argument('--cf-prototxt', type=str, default='model_caffe/mf.prototxt')
parser.add_argument('--cf-model', type=str, default='model_caffe/mf.caffemodel')

args = parser.parse_args()
caffe.set_mode_cpu()
caffe.set_device(0)
deploy = args.cf_prototxt
caffe_model = args.cf_model

net = caffe.Net(deploy, caffe_model, caffe.TEST)
net.blobs['data'].data[...] = np.ones((1, 3, 108, 108))
out = net.forward()
pre_fc_caffe = net.blobs['fc1'].data
output_flatten = np.array(pre_fc_caffe, dtype=np.float32).flatten()
print(output_flatten.shape)