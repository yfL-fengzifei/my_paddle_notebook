#Variable and tensor and LoDTensor

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

