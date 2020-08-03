# configs
mx_model="model"
epoch=300000

caffe_prototxt="FaceFeature80M.prototxt"
caffe_model="FaceFeature80M.caffemodel"

# mxnet2caffe
python test_caffe.py \
  --mx-model $mx_model\
  --mx-epoch $epoch \
  --cf-prototxt $caffe_prototxt\
  --cf-model $caffe_model
