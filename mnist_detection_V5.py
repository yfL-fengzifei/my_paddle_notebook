"""
version 5
训练调试与优化-查看参数
可视化
"""
import paddle
import paddle.fluid as flulid
import paddle.fluid.dygraph.nn as nn
import os
import numpy as np
import gzip
import json
import random

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

# net=MNIST()
"""
总结
net.sublayers() -> net.conv1
                   net.pool1
                   net.fc
                   ...
net.parameters()/net.named_parameters() -> net.conv1.weight
                                           net.conv1.bias
                                           net.fc.weight
                                           net.fc.bias
                                           ...
                                           没有pool
                                           
net.conv1.parameters()/net.conv1.named_parameters() -> net.conv1.weight
                                                    -> net.conv1.bias
                                                    net.conv1._parameters.keys()
                                                    net.conv1._parameters.values()
                                                    
net.pool.paramters()

注：
查看数据用.numpy()

"""
"""
#注：等价于 不是 等于

#net.conv1 
是一个对象
net.conv1._full_name/_num_channels/_act/_dilation/_dtype/_filter_size/_num_filters/_padding/_stride/_groups/_use_cudnn

net.conv1.weight
net.conv1.weight.name
net.conv1.weight.dtype
net.conv1.weight.shape
net.conv1.weight.persistable
net.conv1.weight.trainable
net.conb1.bias
net.conv1.bias.name
net.conb1.bias.dtype
net.conv1.bias.shape
net.conb1.bias.presistable
net.conv1.bias.trainable

#net.conv1.parameters()
net.conv1.parameters() #type(net.conv1.parameters)=method,type(net.conv1.parameters())=list
net.conv1.parameters()[0].name/shape/persistable #一般对应于权重
net.conv1.parameters()[1].name/shape/persitable #一般对应于偏置

#net.conv1.named_parameters()
net.conv1.named_parameters() #type(net.conv1.named_parameters)=generator
for name,param in net.conv1.named_parameters(): pass

#net.conv1._sub_layers
net.conv1._sub_layers #此时为空 ,type(net.conv1._sub_layers)=OrderDict()

#net.conv1._parameters
net.conv1._parameters #type(net.conv1._parameters)=OrderDict
net.conv1._parameters.keys() #type(order_keys)=odict_keys
net.conv1._parameters.values() #type(odict_keys())=odict_values
net.conv1._parameters['weight']
net.conv1._parameters['bias']


#注
net.conv1.weight 等价于 net.conv1.parameters()[0] 等价于 param in net.conv1.named_parameters() 等价于 net.conv1._parameters['weight'] 
net.conv1.bias 等价于 net.conv1.parameters()[1]  等价于 param in net.conv1.named_parameters() 等价于 net.conv1._parameters['bias']

#net.pool1
#基本如上，就是没有参数

#net.parameters() 
是一个list. type(net.parameters())=list 
包含卷积和全连接层的权重和偏执，离散构建，len(net.parameters())
#注
net.conv1.parameters()是net.parameters()的子元素，结合上面的等价可以再推
net.parameters()[0] 仅等价于 net.conv1.parameters()[0] 

net.parameters()[0]
net.parameters()[0].name 名字
net.parameters()[0].shape 维度
net.parameters()[0].type 参数类型
net.parameters()[0].shape 参数数据类型
...与上面类似

#net.named_parameters()
for name,param in net.named_parameters(): pass

#注！
net.conv1; net.pool1 均可单独访问
net.parameters();net.named_parameters() 保存的只可学习层

#net.sublayers()
type(net.sublayers())=list
ner.sublayers()[0] 等价于 net.conv1 访问内部参数的方法也一样

"""

#卷积核就是权重，维度是MCHW，M是输出特征图个数，C是输入通道个数
# net=MNIST()
# print(net.conv1.weight.shape)
# print(net.conv1.bias.shape)
# print(net.conv2.weight.shape)
# print(net.conv2.bias.shape)
# print(net.fc.weight.shape)
# print(net.fc.bias.shape)

# with flulid.dygraph.guard(place):
#     model=MNIST()
#     model.train()
#
#     train_loader = data_generator
#
#     optimizer=flulid.optimizer.AdamOptimizer(learning_rate=0.01,regularization=flulid.regularizer.L2Decay(regularization_coeff=0.1),parameter_list=model.parameters())
#
#     EPOCH_NUM=10
#     for epoch_id in range(EPOCH_NUM):
#         for batch_id,data in enumerate(train_loader()):
#             img_data,label_data=data #img:(batch_size,1,28,28);label,(n,1,1)，ndarray
#             img=flulid.dygraph.to_variable(img_data) #因为只能接受ndarray的数据
#             label=flulid.dygraph.to_variable(label_data)
#
#             predict,acc=model(img,label) #batch
#             #对应于均方差损失函数
#             # loss=flulid.layers.square_error_cost(predict,label) #因为只能计算单个样本
#             #对应于交叉熵损失函数
#             loss=flulid.layers.cross_entropy(predict,label) #因为只能计算单个样本
#             avg_loss=flulid.layers.mean(loss) #对batch进行平均
#
#             if batch_id%200==0:
#                 print('epoch:{},batch:{},loss is:{},acc is {}'.format(epoch_id,batch_id,avg_loss.numpy(),acc.numpy()))
#
#             avg_loss.backward()
#             optimizer.minimize(avg_loss)
#             model.clear_gradients()

#使用visualDL
"""
visualdl 将训练过程中的数据、参数等信息存储至日志文件后，启动面板即可查看可视化结果
可以通过LogWriter定制一个日志记录器
class LogWriter(logdir=None,commnet='',max_queue=10,flush_secs=120,filename_suffix='',write_to_disk=True,**kwargs)
logdir: 日志文件所在路径，visualdl在此路径下建立日志文件并进行记录，最好填
comment:为日志文件夹名添加后缀，如果定制了logdir则无效
max_queue：日志记录消息队列的最大容量，达到此容量则立即写入日志文件
flush_secs: 日志记录消息队列的最大缓存时间，达到此时间则立即写入日志文件
filename_suffix: 为默认的日志文件添加后缀
wirte_to_disk: 是否写入磁盘
"""
"""
建立一个日志文件
在训练过程中插入作图语句，当每100个batch训练完成后，将当前损失所谓一个新增的数据点(iter和acc的映射对)存储到第一步设置的文件中
使用变量iter记录下已经训练的批次数，作为作图的x轴坐标
log_writer.add_scalar(tag='acc',setp=iter,value=avg_acc.numpy())
log_writer.add_scalar(tag='loss',setp=iter,value=avg_loss.numpy())
iter+=100
"""
from visualdl import LogWriter
log_writer=LogWriter('./log')

with flulid.dygraph.guard(place):
    model=MNIST()
    model.train()

    train_loader = data_generator

    optimizer=flulid.optimizer.AdamOptimizer(learning_rate=0.01,regularization=flulid.regularizer.L2Decay(regularization_coeff=0.1),parameter_list=model.parameters())

    EPOCH_NUM=10
    iter=0
    for epoch_id in range(EPOCH_NUM):
        for batch_id,data in enumerate(train_loader()):
            img_data,label_data=data #img:(batch_size,1,28,28);label,(n,1,1)，ndarray
            img=flulid.dygraph.to_variable(img_data) #因为只能接受ndarray的数据
            label=flulid.dygraph.to_variable(label_data)

            predict,avg_acc=model(img,label) #batch
            #对应于均方差损失函数
            # loss=flulid.layers.square_error_cost(predict,label) #因为只能计算单个样本
            #对应于交叉熵损失函数
            loss=flulid.layers.cross_entropy(predict,label) #因为只能计算单个样本
            avg_loss=flulid.layers.mean(loss) #对batch进行平均

            if batch_id%100==0:
                print('epoch:{},batch:{},loss is:{},acc is {}'.format(epoch_id,batch_id,avg_loss.numpy(),avg_acc.numpy()))
                log_writer.add_scalar(tag='acc',step=iter,value=avg_acc.numpy())
                log_writer.add_scalar(tag='loss',step=iter,value=avg_loss.numpy())
                iter+=100

            avg_loss.backward()
            optimizer.minimize(avg_loss)
            model.clear_gradients()



