"""
version 4
训练调试与优化-计算准确率
"""
"""
计算分类准确率，观察模型训练效果
检查模型训练过程，识别潜在问题
加入校验或测试，更好的评价模型效果
加入正则化项，避免模型过拟合
可视化分析
"""
import paddle
import paddle.fluid as flulid
import paddle.fluid.dygraph.nn as nn
import os
import numpy as np
import gzip
import json
import random

#====================================================计算分类准确率
"""
fluid.layers.accuracy可以直接计算准确率，API的输入参数Input为预测的分类结果predict,输入参数label为数据真值的label

在前项计算过程forward函数中计算分类准确率，并在训练时打印每个批次样本的分类准确率
"""

#加载数据
PATH='../mnist.json.gz'
data_json=gzip.open(PATH)
data=json.load(data_json)

#训练集、验证集、测试集
train_set,val_set,eval_set=data #list(list(list),list),list,list

IMG_ROWS=28
IMG_COLS=28

train_imgs,train_lables=train_set #(list(list),list)

#索引号
idx_list=list(range(len(train_imgs)))
#随机打乱
random.shuffle(idx_list)
#batch_size
BATCH_SIZE=100

#定义数据生成器，返回批次数据
def data_generator():
    img_list=[]
    labels_list=[]

    for i in idx_list:
        #将数据处理成模型期望的格式

        img=np.reshape(train_imgs[i],[1,IMG_ROWS,IMG_COLS]).astype('float32') #(c,h,w)=(1,28,28)
        #对应于均方差损失函数标签格式
        # label=np.reshape(train_lables[i],[1]).astype('float32') #(1,1)
        #对应于交叉熵损失函数标签格式
        label=np.reshape(train_lables[i],[1]).astype('int64') #(1,1)

        img_list.append(img) #list(ndarray) (n,c,h,w)=(batch_size,1,28,28)
        labels_list.append(label) #list(ndarray) (n,1,1)

        if len(img_list)==BATCH_SIZE:
            #获得一个batch size的数据，并返回
            yield np.array(img_list),np.array(labels_list) #转换成ndarray(ndarray) (n,c,h,w)=(batch_size,1,28,28);(n,1,1)

            #清空数据读取列表
            img_list=[]
            labels_list=[]

    #如果有剩余数据的数据小于BATCH_SIZE
    #则剩余数据一起构成一个大小为len(img_list)的mini-batch
    if len(img_list)>0:
        yield np.array(img_list),np.array(labels_list)
    return data_generator #img:(batch_size,1,28,28);label,(n,1,1)

assert len(train_imgs)==len(train_lables),"wrong!"

#定义网络
class MNIST(flulid.dygraph.Layer):
    def __init__(self):
        super(MNIST,self).__init__()

        #对单张图像来说
        self.conv1=nn.Conv2D(num_channels=1,num_filters=20,filter_size=5,stride=1,padding=2,act='relu')
        self.pool1=nn.Pool2D(pool_size=2,pool_stride=2,pool_type='max')
        self.conv2=nn.Conv2D(num_channels=20,num_filters=20,filter_size=5,stride=1,padding=2,act='relu')
        self.pool2=nn.Pool2D(pool_size=2,pool_stride=2,pool_type='max')
        #对应于均方差损失函数
        # self.fc=nn.Linear(input_dim=980,output_dim=1,act=None)
        #对应于交叉熵损失函数
        self.fc=nn.Linear(input_dim=980,output_dim=10,act='softmax')


    def forward(self,inputs,label):
        x=self.conv1(inputs)
        x=self.pool1(x)
        x=self.conv2(x)
        x=self.pool2(x)
        x=flulid.layers.reshape(x,[x.shape[0],-1]) #x为[N,C,H,W]，转换维度，[N,*]
        x=self.fc(x)
        if label is not None:
            acc=flulid.layers.accuracy(input=x,label=label)
            return x,acc
        else:
            return x

#gpu训练
use_gpu=True
place=flulid.CUDAPlace(0) if use_gpu else flulid.CPUPlace()
# import time
# with flulid.dygraph.guard(place):
#     # strat=time.process_time()
#     model=MNIST()
#     model.train()
#
#     train_loader=data_generator #img:(batch_size,1,28,28);label,(n,1,1)
#
#     optimizer=flulid.optimizer.SGDOptimizer(learning_rate=0.001,parameter_list=model.parameters())
#
#     EPOCH_NUM=10
#     for epoch_id in range(EPOCH_NUM):
#         for batch_id,data in enumerate(train_loader()):
#             img_data,label_data=data #img:(batch_size,1,28,28);label,(n,1,1)，ndarray
#             img=flulid.dygraph.to_variable(img_data) #因为只能接受ndarray的数据
#             label=flulid.dygraph.to_variable(label_data)
#
#             predict=model(img) #batch
#             #对应于均方差损失函数
#             # loss=flulid.layers.square_error_cost(predict,label) #因为只能计算单个样本
#             #对应于交叉熵损失函数
#             loss=flulid.layers.cross_entropy(predict,label) #因为只能计算单个样本
#             avg_loss=flulid.layers.mean(loss) #对batch进行平均
#
#             if batch_id%200==0:
#                 print('epoch:{},batch:{},loss is:{}'.format(epoch_id,batch_id,avg_loss.numpy()))
#
#             avg_loss.backward()
#             optimizer.minimize(avg_loss)
#             model.clear_gradients()
#
#     # print(time.process_time()-strat)

#====================================================加入正则项，避免模型过拟合
"""
问题：
训练数据存在噪声 --> 使用数据清洗和修正
模型强大复杂，训练数据较少 --> 更多的数据或限制模型能力
推荐：限制模型的能力 --> 正则化
"""
"""
正则化：
降低模型的复杂度，在损失(模型优化)中认为加入对参数规模的惩罚项，当参数越多或取值越大时，该惩罚项越大，通过调整惩罚项的权重系数，可以使模型，在尽量减少训练损失和保持模型的泛化能力之间取得平衡。
正则化项的存在，增加了模型在训练集上的损失

paddle支持为所有参数统一加上正则化项，也支持为特定的参数添加正则化项
统一加上正则项：在优化器中设置regularization参数即可，使用参数regularization_coeff调节正则化项的权重，权重越大，对模型复杂度的惩罚越高
各种优化算法均可以加入正则化项，避免过拟合，
"""
with flulid.dygraph.guard(place):
    model=MNIST()
    model.train()

    train_loader = data_generator

    optimizer=flulid.optimizer.AdamOptimizer(learning_rate=0.01,regularization=flulid.regularizer.L2Decay(regularization_coeff=0.1),parameter_list=model.parameters())

    EPOCH_NUM=10
    for epoch_id in range(EPOCH_NUM):
        for batch_id,data in enumerate(train_loader()):
            img_data,label_data=data #img:(batch_size,1,28,28);label,(n,1,1)，ndarray
            img=flulid.dygraph.to_variable(img_data) #因为只能接受ndarray的数据
            label=flulid.dygraph.to_variable(label_data)

            predict,acc=model(img,label) #batch
            #对应于均方差损失函数
            # loss=flulid.layers.square_error_cost(predict,label) #因为只能计算单个样本
            #对应于交叉熵损失函数
            loss=flulid.layers.cross_entropy(predict,label) #因为只能计算单个样本
            avg_loss=flulid.layers.mean(loss) #对batch进行平均

            if batch_id%200==0:
                print('epoch:{},batch:{},loss is:{},acc is {}'.format(epoch_id,batch_id,avg_loss.numpy(),acc.numpy()))

            avg_loss.backward()
            optimizer.minimize(avg_loss)
            model.clear_gradients()
