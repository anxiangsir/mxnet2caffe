import sys, argparse
import find_mxnet, find_caffe
import mxnet as mx
import caffe

parser = argparse.ArgumentParser(description='Convert MXNet model to Caffe model')
parser.add_argument('--mx-param',    type=str)
parser.add_argument('--cf-prototxt', type=str)
parser.add_argument('--cf-model',    type=str)
args = parser.parse_args()

def load_params(params_path):
  save_dict = mx.nd.load(params_path)
  arg_params = {}
  aux_params = {}
  if not save_dict:
    raise ValueError
  for k, v in save_dict.items():
    tp, name = k.split(":", 1)
    if tp == "arg":
      arg_params[name] = v
    if tp == "aux":
      aux_params[name] = v
  return arg_params, aux_params

# ------------------------------------------
# Load
arg_params, aux_params = load_params(args.mx_param)
net = caffe.Net(args.cf_prototxt, caffe.TRAIN)   

# ------------------------------------------
# Convert
all_keys = arg_params.keys() + aux_params.keys()
all_keys.sort()

print('----------------------------------\n')
print('ALL KEYS IN MXNET:')
print(all_keys)
print('%d KEYS' %len(all_keys))

print('----------------------------------\n')
print('VALID KEYS:')
for i_key,key_i in enumerate(all_keys):

  try:    
    
    if 'data' is key_i:
      pass
    elif '_weight' in key_i:
      key_caffe = key_i.replace('_weight','')
      if 'fc' in key_i:
        print key_i
        print arg_params[key_i].shape
        print net.params[key_caffe][0].data.shape
      net.params[key_caffe][0].data.flat = arg_params[key_i].asnumpy().flat      
    elif '_bias' in key_i:
      key_caffe = key_i.replace('_bias','')
      net.params[key_caffe][1].data.flat = arg_params[key_i].asnumpy().flat   
    elif '_gamma' in key_i and 'relu' not in key_i:
      key_caffe = key_i.replace('_gamma','_scale')
      net.params[key_caffe][0].data.flat = arg_params[key_i].asnumpy().flat
    # TODO: support prelu
    elif '_gamma' in key_i and 'relu' in key_i:   # for prelu
      key_caffe = key_i.replace('_gamma','')
      assert (len(net.params[key_caffe]) == 1)
      net.params[key_caffe][0].data.flat = arg_params[key_i].asnumpy().flat
    elif '_beta' in key_i:
      key_caffe = key_i.replace('_beta','_scale')
      net.params[key_caffe][1].data.flat = arg_params[key_i].asnumpy().flat    
    elif '_moving_mean' in key_i:
      key_caffe = key_i.replace('_moving_mean','')
      net.params[key_caffe][0].data.flat = aux_params[key_i].asnumpy().flat 
      net.params[key_caffe][2].data[...] = 1 
    elif '_moving_var' in key_i:
      key_caffe = key_i.replace('_moving_var','')
      net.params[key_caffe][1].data.flat = aux_params[key_i].asnumpy().flat    
      net.params[key_caffe][2].data[...] = 1 
    else:
      sys.exit("Warning!  Unknown mxnet:{}".format(key_i))
  
    print("% 3d | %s -> %s, initialized." 
           %(i_key, key_i.ljust(40), key_caffe.ljust(30)))
    
  except KeyError:
    print("\nError!  key error mxnet:{}".format(key_i))  
      
# ------------------------------------------
# Finish
net.save(args.cf_model)
print("\n- Finished.\n")

