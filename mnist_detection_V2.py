"""
version 2
"""

#=========================================================================================数据处理
#=========================================================================================
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

#=========================================================================================加载json文件
PATH='../mnist.json.gz'
data_json=gzip.open(PATH)
data=json.load(data_json)

#训练集、验证集、测试集
train_set,val_set,eval_set=data #list(list(list),list),list,list

IMG_ROWS=28
IMG_COLS=28

train_imgs,train_lables=train_set #(list(list),list)
# val_imgs,val_labels=val_set
# eval_imgs,eval_labels=eval_set
# print('imgs,train:{},val:{},eval:{}'.format(len(train_imgs),len(val_imgs),len(eval_imgs)))
# print('labels,train:{},val:{},eval:{}'.format(len(train_lables),len(val_labels),len(eval_labels)))

#=========================================================================================数据乱序
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

#=========================================================================================同步数据读取
#实例化数据读取器
# train_loader=data_generator
#
# #(batch_size,1,28,28),(batch,1)
# for batch_id,data in enumerate(train_loader()):
#     imgs,labels=data
#     pass

#=========================================================================================校验数据有效性
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


#=========================================================================================将上述封装后
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


#=========================================================================================数据异步读取
"""
上述为同步数据读取，对于样本量较大，数据读取较慢的场景，应该采用异步处理。
异步读取数据时，数据读取和模型训练并行执行，能够加快数据读取速度，牺牲一小部分内存换区数据读取效率的提升
同步读取：数据读取与模型训练串行：当模型需要数据时，才运行数据读取函数获得当前批次的数据，在读取数据期间，模型一直等待数据读取结束才进行训练
异步读取：数据读取和模型训练并行，读取到的数据不断的放入缓存区，无需等待模型训练就可以启动下一轮数据读取，当模型训练完一个批次后，不用等待数据读取过程，直接从缓存区获得下一批次数据进行训练，从而加快了数据读取速度
异步队列：数据读取和模型训练交互的仓库，二值均可从仓库中读取数据，

注：异步读取数据只在数据规模巨大时会带来显著的性能提升，对于大多数场景，同步数据读取的方式已经足够
"""
"""
fluid.io.Dateloader.from_generator(feed_list=None, capacity=None, use_double_buffer=True, iterable=True, return_list=False, use_multiprocess=False, drop_last=True)
feed_list:仅在静态图中使用，
capacity: DataLoader对象内部维护队列的容量大小，单位为batch数量，读取数据速度很快，设置为更大的值
use_double_buffer: 异步地预先读取下一个batch的数据并放到缓存区，加速但会占用少量GPU和CPU内存
iterable: True表示创建的DataLoader对象可迭代
return_list: 动态图为True
use_multiprocess: 多进程加载，仅动态图有效
"""
"""
同步 vs 异步的区别
train_loader=data_generator
#同步
for i,data in enumerate(train_loader):
    pass

#异步
train_loader=data_generator
place=fluid.CPUPlace()
data_loader=flulid.io.DataLoader.from_generator(capacity=5,return_list=True)
data_loader.set_batch_generator(train_loader,places=place)
for i,data in enumerate(data_loader):
    pass
"""

# #定义数据读取后存放的位置，cpu或gpu
# #CPUPlace表示一个设备描述符，表示一个分配或将要分配Tensor或LoDTensor的CPU设备
# place=flulid.CPUPlace() #读取的数据是放在GPU还是CPU上
#
# #GPUPlace表示一个设备描述符，表示一个分配或将要分配Tensor或LoDTensor的GPU设备，
# # 每个cudaplace有一个dev_id(设备id)来表明当前得到CUDAPlace所代表的显卡编号，编号从0开始，dev_id不同的CUDAPlace所对应的内存不可互相访问，
# #这里的编号指的是可见显卡的逻辑编号，而不是显卡实际的编号，通过CUDA_VISIBLE_DEVICE环境变量限制程序能够使用的GPU设备，程序启动时会遍历当前的可见设备，并从0开始编号，如果没有设置CUDA_VISIBLE_DEVICE，则默认所有设备都是可见的，
# #如果为None,则默认会使用id为0的编号，默认值为None
# # place=flulid.CUDAPlace(0) #数据读取到GPU上
# with flulid.dygraph.guard(place):
#     #加载训练数据
#     train_loader=data_generator
#
#     #定义创建一个dataloader对象用于加载python生成器产生的数据，只是先定义
#     data_loader=flulid.io.DataLoader.from_generator(capacity=5,return_list=True)
#     #用创建的DataLoader对象设置一个数据生成器set_batch_generator,输入的参数是一个python数据生成器train_loader和服务器资源类型palce(cpu或gpu)
#     data_loader.set_batch_generator(train_loader,places=place)
#     #迭代的读取数据并打印数据的形状
#     for i,data in enumerate(data_loader):
#         img_data,label_data=data
#         print(i,img_data.shape,label_data.shape)
#         if i>5:
#             break

#=========================================================================================网络结构
#=========================================================================================
#卷积神经网络
"""
Conv2D(num_channels, num_filters, filter_size, stride=1, padding=0, dilation=1, groups=None, param_attr=None, bias_attr=None, use_cudnn=True, act=None, dtype='float32')
输入输出NCHW
卷积核维度[M,C,H,W] M是输出特征图的个数，C是输入特征图的个数，H/W是滤波器的高度/宽度
如果组数大于1，C等于输入特征图个数除以组数的结果

卷积sigma(W*X+b) 
X:4D tensor [N,C,H,W]
W:4D tensor [M,C,H,W]
b:2D tensor [M,1]
输出：(x+2*padding-(dilation*(W-1)+1))/stride+1

num_channels 输入图像通道数
num_filters 滤波器个数，=输出特征图个数
filter_size 滤波器大小，一个数：高和宽方向相同；元组tuple:(高，宽)
stride 滤波器滑动步长，默认为1，一个数：垂直和水平相同，元组tuple(垂直，水平)
padding 边界填充大小，默认为0，一个数：垂直和水平相同，元组tuple(垂直，水平)
dilation 膨胀系数大小，默认为1(不膨胀)，一个数：垂直和水平相同，元组tuple(垂直，水平)
groups 分组卷积中二维卷积层的组数，默认为1，**(见参考)**
param_attr 指定权重参数属性的对象，默认为None，表示使用默认的权重参数属性。创建一个参数属性对象，可以设置参数的名称、初始化方式、学习率、正则化规则、是否需要训练、梯度裁剪方式、是否做模型平均等属性，**(见参考)**
bias_attr 如上
use_cudnn 使用cudnn库，默认为True
act 应用于输出上的激活函数，可以单独设置(显示调用)
dtype 数据类型，默认float32,也可以是float64


Pool2D(pool_size=-1, pool_type='max', pool_stride=1, pool_padding=0, global_pooling=False, use_cudnn=True, ceil_mode=False, exclusive=True)
**(见参考)**


Linear(input_dim, output_dim, param_attr=None, bias_attr=None, act=None, dtype='float32')
**(见参考)**
Linear层只能接受一个tensor的输入，输出[N,*,outputdim],*为附加尺寸 **(没懂)**
"""
#=========================================================================================均方差
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


    def forward(self,inputs):
        x=self.conv1(inputs)
        x=self.pool1(x)
        x=self.conv2(x)
        x=self.pool2(x)
        x=flulid.layers.reshape(x,[x.shape[0],-1]) #x为[N,C,H,W]，转换维度，[N,*]
        x=self.fc(x)
        return x

#训练
with flulid.dygraph.guard():
    model=MNIST()
    model.train()

    train_loader=data_generator #img:(batch_size,1,28,28);label,(n,1,1)

    optimizer=flulid.optimizer.SGDOptimizer(learning_rate=0.001,parameter_list=model.parameters())

    EPOCH_NUM=10
    for epoch_id in range(EPOCH_NUM):
        for batch_id,data in enumerate(train_loader()):
            img_data,label_data=data #img:(batch_size,1,28,28);label,(n,1,1)，ndarray
            img=flulid.dygraph.to_variable(img_data) #因为只能接受ndarray的数据
            label=flulid.dygraph.to_variable(label_data)

            predict=model(img) #batch
            #对应于均方差损失函数
            # loss=flulid.layers.square_error_cost(predict,label) #因为只能计算单个样本
            #对应于交叉熵损失函数
            loss=flulid.layers.cross_entropy(predict,label) #因为只能计算单个样本
            avg_loss=flulid.layers.mean(loss) #对batch进行平均

            if batch_id%200==0:
                print('epoch:{},batch:{},loss is:{}'.format(epoch_id,batch_id,avg_loss.numpy()))

            avg_loss.backward()
            optimizer.minimize(avg_loss)
            model.clear_gradients()

#=========================================================================================交叉熵,(软硬标签的概念还需要理解)(硬标签可以理解为one-hot)
#修改标签
# label=np.reshape(train_lables[i],[1]).astype('int64') #(1,1)
#修改网络结构
# self.fc=nn.Linear(input_dim=980,output_dim=10,act='softmax')
#修改损失函数
# loss=flulid.layers.cross_entropy(predict,label) #因为只能计算单个样本

#=========================================================================================模型预测
#=========================================================================================
#最终输出为最大的那个值
results=model(img)
lab=np.argsort(results.numpy()) #返回的升序的索引

#=========================================================================================优化器
#=========================================================================================
#需要研究理论


pass


