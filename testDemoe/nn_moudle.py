import torch
from torch import nn

class xck(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self,input,):
        output = input + 1
        return output

xck =xck()
x=torch.tensor(1.0)
output=xck(x)
print(output)