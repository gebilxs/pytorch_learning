import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 让他训练的更加的快

dataset = torchvision.datasets.CIFAR10("/dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=64)

# input = torch.tensor([[1,2,0,3,1],
#                       [0,1,2,3,1],
#                       [1,2,1,0,0],
#                       [5,2,3,1,1],
#                       [2,1,0,1,1]],dtype=torch.float32)
#
# input = torch.reshape(input,(-1,1,5,5))
# print(input.shape)

class xck(nn.Module):
    def __init__(self):
        super(xck,self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3,ceil_mode=False)

    def forward(self,input):
        output = self.maxpool1(input)
        return output
xck = xck()
# xck(input)
# output = xck(input)
# print(output)

writer = SummaryWriter("log_maxpool")
step =0
for data in dataloader:
    imgs,targets=data
    writer.add_images("input",imgs,step)
    output  = xck(imgs)
    writer.add_images("output",output,step)
    step = step +1

writer.close()