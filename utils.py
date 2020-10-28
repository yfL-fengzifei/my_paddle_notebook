#function test
import paddle
import paddle.fluid as flulid
import paddle.fluid.dygraph.nn as nn
import os
import numpy as np
import gzip
import json
import random

#=========================================cross_entropy
# PATH='../mnist.json.gz'
#标准形式
JSON_PATH='./new_mnist.json'
with open(JSON_PATH,'r') as path:
    # data_json=gzip.open(PATH)
    data=json.load(path)
train_set,val_set,eval_set=data #list(list(list),list),list,list
IMG_ROWS=28
IMG_COLS=28
train_imgs,train_lables=train_set #(list(list),list)
img=np.reshape(train_imgs[0],[1,28,28]) #(1,28,28),ndarray
label=np.reshape(train_lables[0],[1]).astype('int64') #(1),ndarray
# import matplotlib.pyplot as plt
# plt.imshow(img.transpose((1,2,0))) #(h,w,c)
# plt.show()
print(img.shape,label.shape)

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

with flulid.dygraph.guard():
    model=MNIST()
    model.train()
    img_data=np.expand_dims(img,axis=0).astype('float32')
    img_tensor=flulid.dygraph.to_variable(img_data)
    label_data=np.expand_dims(label,axis=0).astype('int64')
    label_tensor=flulid.dygraph.to_variable(label_data)
    predict=model(img_tensor)

    print(predict)
    loss=flulid.layers.cross_entropy(predict,label_tensor)
    print(loss)




