#Variable and tensor and LoDTensor
#operator and program and executor
#动态图dygraph and 编程细节

#variable
"""
variable 可包含任何类型的值变量
variable 基本等于tensor
"""
#paddle中的三种varibale
#1.模型中的可学习参数
"""
以variable中的子类parameter表示
可学习参数：权重、偏执等
可学习参数的生存期和整个训练任务一样长，会接受优化算法的更新，

#==可学习参数的创建(内部解释后续说明)
#单独创建
w=fluid.layers.create_parameter(name='w',shape=[1],dtype='float32')
print(w)
#隐式创建
#创建全连接层时，隐式对权值和偏执创建可学习参数
y=fluid.layers.fc(input=x,size=128,bias_attr=True) #不能直接运行，没有定义x
"""
#2.占位variable
"""
一般是用在静态图下
"""
#3.常量variable
"""
用fluid.layers.fill_constant来创建常量variable，执行包含的tensor形状、类型、常量值
data=fluid.layers.fill_constant(shape=[1],value=0,dtype='int64')
print(data)
"""


#tensor
"""
和其他框架的tensor一样

fluid.data用来接收输入数据，需要提供tensor的形状信息
fulid.data(name='x',shape=[3,None],dtype='int64')
"""
#注
"""
对于一些任务中 batch样本大小不一致的问题，有两种解决方法
1.padding 将大小不一致的样本padding到同样的大小，常用且推荐
2.Lod-Tensor 记录每个样本的大小，减少无用的计算量，LoD牺牲灵活性来提升性能；(如果一个batch内的样本无法通过分桶、排序等方式使得大小接近，建议使用Lod-Tensor)
"""

#Lod-Tensor
"""
对大部分用户无需关注具体用法
"""

import paddle.fluid as fluid
w=fluid.layers.create_parameter(name='w',shape=[1],dtype='float32')
print(w)
data=fluid.layers.fill_constant(shape=[1],value=0,dtype='int64')
print(data)


#operator and program and executor
"""
见文档

所有对数据的操作都由operator表示
operator被封装入paddle.fluid.layers,paddle.fluid.nets等
"""


#动态图dygraph and 编程细节
import paddle.fluid

#数据读取
"""
reader函数：从文件、网络、生成器等读取数据并生成数据项
reader creater 返回reader函数的函数
reader decorator 函数，接收一个或多个reader，并返回一个reader
batch reder 函数，从reader、网络、文件、生成器等从读取数据，并生成一批数据项
"""

#损失函数
"""
paddle提供的算子一般是针对一条样本的，当输入一个batch的数据时，损失算子的输出有多个值，每个值对应一个样本的损失，所以通常在损失算子后面使用mean算子，对损失进行归约
"""

#模型参数
"""
variable中的presistable=True表示长期变量
长期变量：在整个训练过程中持续存在，不会因为一个迭代的结束而销毁结束。所有的模型参数都是长期变量，并未所有的长期变量都是模型参数
"""

#模型保存
"""
执行预测：仅保存模型参数就行
恢复训练：保存一个checkpoint，需要将各种长期变量保存下来，需要记录当前的epoch和step的id

推荐使用
save_inderence_model：会根据用户配置的feeded_var_names和target_vars进行网络裁剪，保存下裁剪后的网络结构的__model__以及裁剪后网络中的长期变量；即会保存网络参数以裁剪后的模型，如果后续要做预测相关的工作，选择此进行变量和网络的保存
save_parmas不会保存网络结构，会保存网络中的全部长期变量到指定位置；即保存的网络参数是最全面的，如果是增量驯良或恢复训练，选择此
"""