# configs
mx_model="model"
epoch=100000

caffe_prototxt="mobileface.prototxt"
caffe_model="mobileface.caffemodel"



# mxnet2caffe
python test_caffe.py \
  --mx-model $mx_model\
  --mx-epoch $epoch \
  --cf-prototxt $caffe_prototxt\
  --cf-model $caffe_model
