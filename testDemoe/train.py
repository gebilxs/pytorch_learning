#准备数据集
import torch.optim
import torchvision
# from torch.utils.tensorboard import SummaryWriter

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

xck =xck()
#损失函数
loss_fn = nn.CrossEntropyLoss()

#优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(xck.parameters(),lr=learning_rate)

#设置训练网络的一些参数
#记录训练的次数
total_train_step = 0
#记录测试的次数
total_test_step = 0
#训练的轮数
epoch = 10


#添加tensorboard
# writer  = SummaryWriter("logs_train")
for i in range(epoch):
    print("-----第{}轮,训练开始-----".format(i+1))

    #训练步骤开始
    xck.train()
    for data in train_dataloader:
        imgs,targets =data
        outputs = xck(imgs)
        loss = loss_fn(outputs,targets)
        #进行梯度清零,优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step+1
        if total_train_step % 100 ==0:
            print("训练次数，{},Loss:{}".format(total_train_step,loss.item()))
            # writer.add_scalar("train_loss",loss.item(),total_train_step)
    #测试步骤开始
    xck.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets = data
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


        torch.save(xck,"xck_{}.pth".format(i))
        print("model saved successfully")
# writer.close()