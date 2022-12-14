#准备数据集
import torch.optim
import torchvision
from torch.utils.tensorboard import SummaryWriter

from model import *
from torch import nn
from torch.utils.data import DataLoader

train_data = torchvision.datasets.CIFAR10("dataset",train=True,transform=torchvision.transforms.ToTensor(),download=True)

test_data = torchvision.datasets.CIFAR10("dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)

train_data_size=len(train_data)
test_data_size = len(train_data)

print("训练数据集的长度为:{}".format(train_data_size))
print("测试数据集的长度为:{}".format(test_data_size))
# 利用dataLoader 来加载数据
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

#搭建神经网络
class xck(nn.Module):
    def __init__(self):
        super(xck,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10),
        )

    def forward(self,x):
        x=self.model(x)
        return x

xck =xck()
# 利用GPU进行训练 - cuda ---这里可以用Device写法来指定利用cpu还是gpu训练
xck =xck.cuda()

#损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn =loss_fn.cuda()

#优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(xck.parameters(),lr=learning_rate)
# 其他优化方式还包括 mini-batch adam ---

#设置训练网络的一些参数
#记录训练的次数 全局变量
total_train_step = 0
#记录测试的次数 全局变量
total_test_step = 0
#训练的轮数
epoch = 10


#添加tensorboard
writer  = SummaryWriter("logs_train")

for i in range(epoch):
    print("-----第{}轮,训练开始-----".format(i+1))

    #训练步骤开始 train --> ?
    xck.train()
    for data in train_dataloader:
        imgs,targets =data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = xck(imgs)
        loss = loss_fn(outputs,targets)
        optimizer.zero_grad()
        # 进行梯度清零,优化器优化模型
        loss.backward()
        #反向传播
        optimizer.step()
        # optimizer.step 执行一次优化步骤，通过梯度下降法来更新参数的值
        total_train_step = total_train_step+1
        #更新训练的次数
        if total_train_step % 100 ==0:
            print("训练次数，{},Loss:{}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)
    #测试步骤开始 eval --> ?
    xck.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = xck(imgs)
            loss = loss_fn(outputs,targets)
            total_test_loss=total_test_step+loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy


        print("整体测试集上的Loss：{}".format(total_test_loss))
        print("整体测试集上面的正确率：{}".format(total_accuracy/test_data_size))

        #写入tensorboard
        # writer.add_scalar("test_loss",total_test_loss,total_test_step)
        # writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_step)
        total_test_step = total_test_step+1


        # torch.save(xck,"xck_{}.pth".format(i))
        print("model saved successfully")
# writer.close()