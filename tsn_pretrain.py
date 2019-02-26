# -*- coding: utf-8 -*-

import numpy as np

caffe_root = '/home/myprogram/Spatiotemporal_Heterogeneous_Two-stream_Network/lib/caffe-action/'  # 设置caffe的根目录
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe  #导入caffe
caffe.set_mode_gpu()
net0 = caffe.Net('spatial_ResNet_50_deploy.prototxt',\
    'ResNet-50-model.caffemodel',caffe.TEST) #TEST/TRAIN

conv1_w = net0.params['conv1'][0].data

bn_w = net0.params['bn_conv1'][0].data

bn_b = net0.params['bn_conv1'][1].data

#可以打印相应的参数和参数的维度等信息

w_0 = net0.params['conv1'][0].data    #(64,3,7,7)
new = [np.array((i.sum(axis=0)/3).tolist() * 10).reshape(10,7,7) for i in w_0]
new_w0 = np.array(new).reshape(64,10,7,7)          #(64,10,7,7)
net0.params['conv1'][0].reshape(64,10,7,7)
net0.params['conv1'][0].data[:] = new_w0


keys0 = net0.params.keys()
conv1_w2= net0.params['conv1'][0].data
bn_w1 = net0.params['bn_conv1'][0].data
bn_b1= net0.params['bn_conv1'][1].data

 #net0的参数另存为新caffemodel
net0.save('ResNet-50-model_init_flow.caffemodel')