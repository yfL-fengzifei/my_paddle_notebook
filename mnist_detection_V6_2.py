"""
version 5
训练调试与优化-恢复训练
"""
"""
参数加载时，如果模型参数文件和优化器参数文件是相同的，可以同时加载,如果模型参数文件和优化参数文件的名字不同，需要调用两次load_dygraph
params_dict,opt_dict=fluid.load_dygraph(params_path)
理想的恢复训练是模型状态回到训练中断的时刻，恢复训练之后的梯度更新走向是和恢复训练前的梯度走向完全相同，基于此，可以通过恢复训练后的损失变化，判断上述方法是否准确的恢复训练。即从epoch0结束时保存的模型参数和优化器状态恢复训练，校验气候训练损失变化epoch1是否和不中断时的训练完全一致
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

# from visualdl import LogWriter
# log_writer=LogWriter('./log2_re')

"""
例子：使用adam优化器，学习率以多项式曲线从0.01衰减到0.01
lr=fluid.dygrapgh.PolynomialDecay(0.01,total_steps,0.001)
learing_rate:初始学习率
decay_steps:衰减步数
end_learing_rate:最终学习率
power:多项式的幂
cycle:下降后是否重新上升，polynomial decay的变化曲线下图所示
"""

print('recovery from epoch0(1)')
params_path='./checkpoint/mnist_epoch4'
with flulid.dygraph.guard(place):
    #加载模型和优化器
    params_dict,opt_dict=flulid.load_dygraph(params_path)

    model=MNIST()
    model.set_dict(params_dict)

    EPOCH_NUM=10
    #定义学习率，并加载优化器参数到模型中
    total_steps=(int(60000/BATCH_SIZE)+1)*EPOCH_NUM
    lr=flulid.dygraph.PolynomialDecay(0.01,total_steps,0.001)
    optimizer=flulid.optimizer.AdamOptimizer(learning_rate=lr,regularization=flulid.regularizer.L2Decay(regularization_coeff=0.1),parameter_list=model.parameters())
    optimizer.set_dict(opt_dict)

    model.train()

    train_loader = data_generator

    for epoch_id in range(5,EPOCH_NUM):
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
                # log_writer.add_scalar(tag='acc',step=iter,value=avg_acc.numpy())
                # log_writer.add_scalar(tag='loss',step=iter,value=avg_loss.numpy())
                # iter+=100

            avg_loss.backward()
            optimizer.minimize(avg_loss)
            model.clear_gradients()

        #保存模型参数和优化器参数
        # flulid.save_dygraph(model.state_dict(),'./checkpoint/mnist_epoch{}'.format(epoch_id))
        # flulid.save_dygraph(optimizer.state_dict(),'./checkpoint/mnist_epoch{}'.format(epoch_id))

# params_path1='./checkpoint/mnist_epoch0'
# params_path2='./checkpoint/mnist_epoch4'
# with flulid.dygraph.guard(place):
#     #加载模型和优化器
#     params_dict1,opt_dict1=flulid.load_dygraph(params_path1)
#     params_dict2,opt_dict2=flulid.load_dygraph(params_path2)
#
#     model1=MNIST()
#     model1.set_dict(params_dict1)
#     model2=MNIST()
#     model2.state_dict(params_dict2)
#
#     EPOCH_NUM=10
#     #定义学习率，并加载优化器参数到模型中
#     total_steps=(int(60000/BATCH_SIZE)+1)*EPOCH_NUM
#     lr1=flulid.dygraph.PolynomialDecay(0.01,total_steps,0.001)
#     lr2=flulid.dygraph.PolynomialDecay(0.01,total_steps,0.001)
#     optimizer1=flulid.optimizer.AdamOptimizer(learning_rate=lr1,regularization=flulid.regularizer.L2Decay(regularization_coeff=0.1),parameter_list=model1.parameters())
#     optimizer1.set_dict(opt_dict1)
#     optimizer2=flulid.optimizer.AdamOptimizer(learning_rate=lr2,regularization=flulid.regularizer.L2Decay(regularization_coeff=0.1),parameter_list=model2.parameters())
#     optimizer2.set_dict(opt_dict2)
#
#     model1.train()
#     model2.train()
#
#     train_loader = data_generator
#
#     for epoch_id in range(1,EPOCH_NUM):
#         for batch_id,data in enumerate(train_loader()):
#             img_data,label_data=data #img:(batch_size,1,28,28);label,(n,1,1)，ndarray
#             img=flulid.dygraph.to_variable(img_data) #因为只能接受ndarray的数据
#             label=flulid.dygraph.to_variable(label_data)
#
#             predict1,avg_acc1=model1(img,label) #batch
#             #对应于均方差损失函数
#             # loss=flulid.layers.square_error_cost(predict,label) #因为只能计算单个样本
#             #对应于交叉熵损失函数
#             loss1=flulid.layers.cross_entropy(predict1,label) #因为只能计算单个样本
#             avg_loss1=flulid.layers.mean(loss1) #对batch进行平均
#
#             predict2,avg_acc2=model2(img,label) #batch
#             #对应于均方差损失函数
#             # loss=flulid.layers.square_error_cost(predict,label) #因为只能计算单个样本
#             #对应于交叉熵损失函数
#             loss2=flulid.layers.cross_entropy(predict2,label) #因为只能计算单个样本
#             avg_loss2=flulid.layers.mean(loss2) #对batch进行平均
#
#             if batch_id%100==0:
#                 print('model:1,epoch:{},batch:{},loss is:{},acc is {}'.format(epoch_id,batch_id,avg_loss1.numpy(),avg_acc1.numpy()))
#                 # log_writer.add_scalar(tag='acc',step=iter,value=avg_acc.numpy())
#                 # log_writer.add_scalar(tag='loss',step=iter,value=avg_loss.numpy())
#                 # iter+=100
#                 print('model:2,epoch:{},batch:{},loss is:{},acc is {}'.format(epoch_id,batch_id,avg_loss2.numpy(),avg_acc2.numpy()))
#
#             avg_loss1.backward()
#             optimizer1.minimize(avg_loss1)
#             model1.clear_gradients()
#             avg_loss2.backward()
#             optimizer2.minimize(avg_loss2)
#             model2.clear_gradients()
#
#         #保存模型参数和优化器参数
#         # flulid.save_dygraph(model.state_dict(),'./checkpoint/mnist_epoch{}'.format(epoch_id))
#         # flulid.save_dygraph(optimizer.state_dict(),'./checkpoint/mnist_epoch{}'.format(epoch_id))
