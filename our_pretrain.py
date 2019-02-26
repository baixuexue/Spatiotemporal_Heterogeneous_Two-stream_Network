# -*- coding: utf-8 -*-

import numpy as np

caffe_root = '/home/myprogram/Spatiotemporal_Heterogeneous_Two-stream_Network/lib/caffe-action/'  # 设置caffe的根目录
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe  #导入caffe
caffe.set_mode_gpu()
net0 = caffe.Net('spatial_ResNet_50_deploy.prototxt',\
    'ResNet-50-model_init_rgb.caffemodel',caffe.TEST) #TEST/TRAIN
net1 = caffe.Net('temporal_ResNet_50_deploy.prototxt',\
    'ucf101_flow_res50_fromscratch.caffemodel',caffe.TEST)
#
conv1_w = net0.params['conv1'][0].data
#conv1_b = net0.params['conv1'][1].data

bn_w = net0.params['bn_conv1'][0].data

bn_b = net0.params['bn_conv1'][1].data

conv1_1 = net1.params['conv1_fintune'][0].data

#conv1_2 = net1.params['conv1'][1].data
#可以打印相应的参数和参数的维度等信息
print conv1_1
keys00 = net0.params.keys()
keys11 = net1.params.keys()
#print conv1_1.size,conv1_2.size
#net0.params.pop('conv1')
net0.params['conv1'][0].reshape(64,10,7,7)
net0.params['conv1'][0].data[:]=net1.params['conv1_fintune'][0].data[:]

conv1_w1= net0.params['conv1'][0].data
#conv1_b1 = net0.params['conv1'][1].data
keys0 = net0.params.keys()
keys1 = net1.params.keys()

bn_w1 = net0.params['bn_conv1'][0].data
bn_b1= net0.params['bn_conv1'][1].data

 #net0的参数另存为新caffemodel
net0.save('ResNet-50-model_init_flow_our.caffemodel')