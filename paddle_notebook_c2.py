import paddle
import paddle.fluid as flulid
import paddle.fluid.dygraph.nn as nn
import os
import numpy as np
import gzip
import json
import random

"""
Conv2D(num_channels, num_filters, filter_size, stride=1, padding=0, dilation=1, groups=None, param_attr=None, bias_attr=None, use_cudnn=True, act=None, dtype='float32')

param_attr 指定权重参数属性的对象，默认为None，表示使用默认的权重参数属性。创建一个参数属性对象，可以设置参数的名称、初始化方式、学习率、正则化规则、是否需要训练、梯度裁剪方式、是否做模型平均等属性，**(见参考)**

Conv2D(XXX,param_attr=fluid.ParamAttr(),XXX)
fluid.ParamAttr(name=None, initializer=None, learning_rate=1.0, regularizer=None, trainable=True, do_model_average=False)
用于指定权重参数属性；
name:默认为None，表示框架自动创建参数的名称
initalizer: 参数的初始化方式，默认为none；fluid.initializer.NumpyArrayInitializer(value=np.array(XXX)) 使用Numpy型数组来初始化参数变量
learning_rate: 实际参数的学习率等于全局学习率乘以参数的学习率，再乘以learning rate schedule的系数
regularizer: 正则化方法，支持两种正则化L1Decay,L2Decay,如果在optimizer中设置了正则化，optimizer中的正则化将被忽略，默认值为None，表示没有正则化，fluid.regularizer.L2Decay(XXX)
trainable: 参数是否需要训练，默认为True,表示需要训练
do_model_average 是否做模型平均，默认为False,表示不做模型平均


例子：
w=np.array([1,0,-1],dtype='float32')
w=w.reshape([1,1,1,3])
conv=Conv2D(num_channels=1,num_filters=1,filter_size=[1,3],param_attr=fluid.ParamAttr(initalizer=NumpyArrayInitializer(value=w)))
"""

"""
BN
BatchNorm(num_channels, act=None, is_test=False, momentum=0.9, epsilon=1e-05, param_attr=None, bias_attr=None, dtype='float32', data_layout='NCHW', in_place=False, moving_mean_name=None, moving_variance_name=None, do_model_average_for_mean_and_var=False, use_global_stats=False, trainable_statistics=False)
可对卷积和全连接层进行批归一化处理；根据当前批次数据，按通道计算均值和方差
见文档

"""

"""
dropout
class paddle.fluid.dygraph.Dropout(p=0.5, seed=None, dropout_implementation='downgrade_in_infer', is_test=False)
见文档

"""
