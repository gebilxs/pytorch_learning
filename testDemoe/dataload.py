import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#准备的测试数据集
test_data = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())

#batch_size 选取4个数据进行打包
test_loader = DataLoader(dataset=test_data,batch_size=64,shuffle=False,num_workers=0,drop_last=True)

#测试数据集中第一张样本图片集
img,target=test_data[0]
print(img.shape)
print(target)


writer = SummaryWriter("dataloader")
step =0
# for data in test_loader:
#     imgs,targets = data
#     # print(imgs.shape)
#     # print(targets)
#     writer.add_images("test_data_drop_list",imgs,step)
#     step=step+1
for epoch in range(2):
    step=0
    for data in test_loader:
        imgs,targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("Epoch:{}".format(epoch),imgs,step)
        step=step+1

writer.close()