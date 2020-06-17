# configs
mx_model="/train/trainset/1/modelzoo/FaceFeatureArm30M/FaceFeatureArm30M"
mx_json="/train/trainset/1/modelzoo/FaceFeatureArm30M/FaceFeatureArm30M.json"
epoch=0

caffe_prototxt="FaceFeatureArm30M.prototxt"
caffe_model="FaceFeatureArm30M.caffemodel"

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
