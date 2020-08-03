# configs
mx_model=/train/trainset/1/modelzoo/FaceFeatureArm_250ms/FaceFeatureArm_250ms
mx_json="$mx_model".json
mx_param="$mx_model".params

caffe_prototxt="$mx_model".prototxt
caffe_model="$mx_model".caffemodel

python find_caffe.py
python find_mxnet.py

# json to prototxt
python json2prototxt.py \
  --mx-json $mx_json \
  --cf-prototxt $caffe_prototxt

# mxnet2caffe
python mxnet2caffe.py \
  --mx-param $mx_param \
  --cf-prototxt $caffe_prototxt\
  --cf-model $caffe_model
