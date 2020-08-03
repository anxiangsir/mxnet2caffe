# configs
# mx_model=/anxiang/modelzoo/FaceFeatureArm_250ms/FaceFeatureArm_250ms
mx_model=/train/trainset/1/modelzoo/FaceFeatureArm_250ms/FaceFeatureArm_250ms
mx_json="{$mx_model}".json
epoch=0

caffe_prototxt="{$mx_model}".prototxt
caffe_model="{$mx_model}".caffemodel

python find_caffe.py
python find_mxnet.py

# json to prototxt
python json2prototxt.py \
  --mx-json $mx_json \
  --cf-prototxt $caffe_prototxt

# mxnet2caffe
python mxnet2caffe.py \
  --mx-model $mx_model\
  --mx-epoch $epoch \
  --cf-prototxt $caffe_prototxt\
  --cf-model $caffe_model
