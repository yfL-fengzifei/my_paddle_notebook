"""
拉胯式
"""
import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph.nn as nn
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

#下载数据
#默认地址C:\Users\LYF_fengzifei\.cache\paddle\dataset
trainset=paddle.dataset.mnist.train()
#trainset是tuple的形式，(img,label),其中img的类型是numpy,dtype=float32
#测试数量
# for i,_ in enumerate(trainset()):
#     sum=i
# print(sum+1)
#显示测试
# img_iter=iter(trainset())
# img_data=next(img_iter) #img是一个tuple(data,label),type(data)=np.ndarray
# img,label=img_data
# plt.imshow(img.reshape((28,28)))
# plt.show()

#数据打包
#一共60000个数据
train_reader=paddle.batch(trainset,batch_size=16) #paddle.batch默认丢弃最后不满足batch的操作
#train_reader为batch_size的list,每个元素是一个tuple,每个tuple是一个(data,label),data是ndarray

#利用全连接网络构建
#网络继承与fluid.dygraph.Layer
class MNIST(fluid.dygraph.Layer):
    def __init__(self):
        super(MNIST,self).__init__()

        self.fc=nn.Linear(input_dim=784,output_dim=1,act=None)

    def forward(self,inputs):
        outputs=self.fc(inputs)
        return outputs

#训练网络
#定义动态图工作环境
#通过with语句创建一个dygraph运行的context
#动态图下需要在guard下进行
with fluid.dygraph.guard():
    #实例化网络
    model=MNIST()

    #设置为训练模式
    model.train()

    #实例化优化器
    optimizer=fluid.optimizer.SGDOptimizer(learning_rate=0.001,parameter_list=model.parameters())

    #开始训练
    #epoch
    EPOCH_NUM=10

    for epoch_id in range(EPOCH_NUM):
        for batch_id,data in enumerate(train_reader()):
            img_data=np.array([x[0] for x in data]) #img.dtype='float32',(batch_size,data.shape)
            label_data=np.array([x[1] for x in data]).astype('float32').reshape(-1,1) #(batch_size,data.shape)

            #转换为variable(tensor),因为paddle仅能利用ndarray->tensor,其他不行
            img=fluid.dygraph.to_variable(img_data)
            label=fluid.dygraph.to_variable(label_data)

            #前项计算
            predict=model(img) #shape=(batch_size,1)

            #计算损失
            loss=fluid.layers.square_error_cost(predict,label) #维度要匹配上
            avg_loss=fluid.layers.mean(loss)

            #打印
            if batch_id!=0 and batch_id%1000==0:
                print('epoch:{},batch:{},loss is {}'.format(epoch_id,batch_id,avg_loss.numpy()))

            #更新数据
            avg_loss.backward()
            optimizer.minimize(avg_loss)
            model.clear_gradients()
