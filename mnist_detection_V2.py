"""
version 2
"""

#==数据处理
"""
步骤：
读入数据
划分数据集
生成批次数据
训练样本乱序
校验数据有效性

训练集：确定模型参数
验证集：调节参数：多个网络结构、正则化权重选择
测试集：用于估计应用效果
"""

import paddle
import paddle.fluid as flulid
import paddle.fluid.dygraph.nn as nn
import os
import numpy as np
import gzip
import json
import random

#===数据同步读取
#==加载json文件
PATH='../mnist.json.gz'
data_json=gzip.open(PATH)
data=json.load(data_json)

#训练集、验证集、测试集
train_set,val_set,eval_set=data #list(list(list),list),list,list

IMG_ROWS=28
IMG_COLS=28

train_imgs,train_lables=train_set #(list(list),list)
val_imgs,val_labels=val_set
eval_imgs,eval_labels=eval_set
print('imgs,train:{},val:{},eval:{}'.format(len(train_imgs),len(val_imgs),len(eval_imgs)))
print('labels,train:{},val:{},eval:{}'.format(len(train_lables),len(val_labels),len(eval_labels)))

#==数据乱序
"""
实验表明，模型对最后出现的数据印象更加深刻，训练数据导入后，越接近模型训练结束，最后几个批次数据对模型参数的影响越大

先设置合理的batch_size,再讲数据转变符合模型输入要求的np.array格式返回，同时在返回数据时将python生成器设置为yield模式，减少内存占用
"""
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

        img=np.reshape(train_imgs[i],[1,IMG_ROWS,IMG_COLS]).astype('float32') #(c,h,w)
        label=np.reshape(train_lables[i],[1]).astype('float32')

        img_list.append(img) #list(ndarray)
        labels_list.append(label) #list(ndarray)

        if len(img_list)==BATCH_SIZE:
            #获得一个batch size的数据，并返回
            yield np.array(img_list),np.array(labels_list) #转换成ndarray(ndarray)

            #清空数据读取列表
            img_list=[]
            labels_list=[]

    #如果有剩余数据的数据小于BATCH_SIZE
    #则剩余数据一起构成一个大小为len(img_list)的mini-batch
    if len(img_list)>0:
        yield np.array(img_list),np.array(labels_list)
    return data_generator()

#实例化数据读取器
# train_loader=data_generator
#
# #(batch_size,1,28,28),(batch,1)
# for batch_id,data in enumerate(train_loader()):
#     imgs,labels=data
#     pass

#==校验数据有效性
"""
实际应用中，原始数据可能存在标注不明确，数据杂乱、格式不统一等问题，因此在完成数据处理流程中，还需要进行数据校验
机器校验：加入一些校验和清理数据的操作
人工校验：先打印数据输出结果，观察是否是设置的格式，再总训练的结果验证数据处理和读取的有效性
"""
#机器校验
#当数据集中图片数量和标签数量不等时
assert len(train_imgs)==len(train_lables),"wrong!"

#人工校验
#打印输出输出结果，观察是否是预期的格式，实现数据处理和加载函数后，可以调用它读取一次数据，观察数据的shape和类型是否与函数中设置的一致，即在data_generator后打印一次


#==将上述封装后
#伪代码
"""
def load_data():
    pass
    def data_generator():
        pass
    return data_generator()

#定义网络
pass
with flulid.dygraph.guard():
    pass
    #实例化网络
    #加载数据
    train_loader=load_data() #本质上就是train_load=data_generator
    #优化器
    #训练
    #保存模型
"""


#===数据异步读取
"""
上述为同步数据读取，对于样本量较大，数据读取较慢的场景，应该采用异步处理。
异步读取数据时，数据读取和模型训练并行执行，能够加快数据读取速度，牺牲一小部分内存换区数据读取效率的提升
同步读取：数据读取与模型训练串行：当模型需要数据时，才运行数据读取函数获得当前批次的数据，在读取数据期间，模型一直等待数据读取结束才进行训练
异步读取：数据读取和模型训练并行，读取到的数据不断的放入缓存区，无需等待模型训练就可以启动下一轮数据读取，当模型训练完一个批次后，不用等待数据读取过程，直接从缓存区获得下一批次数据进行训练，从而加快了数据读取速度
异步队列：数据读取和模型训练交互的仓库，二值均可从仓库中读取数据，
"""
"""
fluid.io.Dateloader.from_generator(feed_list=None, capacity=None, use_double_buffer=True, iterable=True, return_list=False, use_multiprocess=False, drop_last=True)
capacity: DataLoader对象内部维护队列的容量大小，单位为batch数量
use_double_buffer: 异步地预读取下一个batch的数据，加速但会占用少量GPU和CPU内存
iterable: True表示创建的DataLoader对象可迭代
return_list: 动态图为False
use_multiprocess: 多进程加载，仅动态图有效

"""
#定义数据读取后存放的位置，cpu或gpu
place=flulid.CPUPlace()
# place=flulid.CUDAPlace(0) #数据读取到GPU上
with flulid.dygraph.guard(place):
    #加载训练数据
    train_loader=data_generator

    #定义dataloader对象用于加载python生成器产生的数据
    data_loader=flulid.io.DataLoader.from_generator(capacity=5,return_list=True)
    #设置数据生成器
    data_loader.set_batch_generator(train_loader,places=place)
    #迭代的读取数据并打印数据的形状
    for i,data in enumerate(data_loader):
        img_data,label_data=data
        print(i,img_data.shape,label_data.shape)
        if i>5:
            break












pass


