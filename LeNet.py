import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph.nn as nn
import numpy as np

class LeNet(fluid.dygraph.Layer):
    def __init__(self,num_classes=1):
        super(LeNet,self).__init__()
        self.conv1=nn.Conv2D(num_channels=1,num_filters=6,filter_size=5,act='sigmoid') #输出[N,6,H,W],权重[6,C,5,5]
        self.pool1=nn.Pool2D(pool_size=2,pool_stride=2,pool_type='max')
        self.conv2=nn.Conv2D(num_channels=6,num_filters=16,filter_size=5,act='sigmoid') #
        self.pool2=nn.Pool2D(pool_size=2,pool_stride=2,pool_type='max')
        self.conv3=nn.Conv2D(num_channels=16,num_filters=120,filter_size=4,act='sigmoid')
        self.fc1=nn.Linear(input_dim=120,output_dim=64,act='sigmoid')
        self.fc2=nn.Linear(input_dim=64,output_dim=num_classes)

    def forward(self, x):
        x=self.conv1(x)
        x=self.pool1(x)
        x=self.conv2(x)
        x=self.pool2(x)
        x=self.conv3(x)
        x=fluid.layers.reshape(x,[x.shape[0],-1])
        x=self.fc1(x)
        x=self.fc2(x)
        return x

# x=np.random.randn(3,1,28,28).astype('float32')
# with fluid.dygraph.guard():
#     m=LeNet(num_classes=10)
#     print(m.sublayers())
#     x=fluid.dygraph.to_variable(x)
#     for item in m.sublayers():
#         try:
#             x=item(x)
#         except:
#             x=fluid.layers.reshape(x,[x.shape[0],-1])
#             x=item(x)
#         if len(item.parameters())==2:
#             print(item.full_name(),x.shape,item.parameters()[0].shape,item.parameters()[1].shape)
#         else:
#             print(item.full_name(),x.shape)


#手写字符识别
trainset=paddle.dataset.mnist.train()
validset=paddle.dataset.mnist.test()
train_loader=paddle.batch(trainset,batch_size=10)
valid_loder=paddle.batch(trainset,batch_size=10)
with fluid.dygraph.guard():
    model=LeNet(num_classes=10)
    model.train()

    opt=fluid.optimizer.Momentum(learning_rate=0.001,momentum=0.9,parameter_list=model.parameters())

    epoch_num=5
    for epoch in range(epoch_num):
        for batch_id,data in enumerate(train_loader()):
            x_data=np.array([item[0] for item in data],dtype='float32').reshape((-1,1,28,28))
            y_data=np.array([item[1] for item in data],dtype='int64').reshape((-1,1))

            img=fluid.dygraph.to_variable(x_data)
            label=fluid.dygraph.to_variable(y_data) # [10,1]


            logits=model(img)
            loss=fluid.layers.softmax_with_cross_entropy(logits,label)

            avg_loss=fluid.layers.mean(loss)

            # if batch_id%100==0:
                # print('epoch:{},batch_id:{},loss:{}'.format(epoch,batch_id,avg_loss.numpy()))


            avg_loss.backward()
            opt.minimize(avg_loss)
            model.clear_gradients()

        model.eval()
        accuracies=[]
        losses=[]
        for batch_id,data in enumerate(valid_loder()):

            x_data=np.array([item[0] for item in data],dtype='float32').reshape((-1,1,28,28))
            y_data=np.array([item[1] for item in data],dtype='int64').reshape((-1,1))

            img=fluid.dygraph.to_variable(x_data)
            label=fluid.dygraph.to_variable(y_data) # [10,1]

            logits=model(img)
            loss=fluid.layers.softmax_with_cross_entropy(logits,label)

            pred=fluid.layers.softmax(logits)
            acc=fluid.layers.accuracy(pred,label)

            losses.append(loss.numpy())
            accuracies.append(acc.numpy())
            # print('[validation] batch loss/acc:{}/{}'.format(fluid.layers.mean(loss).numpy(),acc.numpy()))
        print('[validation] accuracy/loss:{}/{''}'.format(np.mean(accuracies),np.mean(losses)))

        model.train()

