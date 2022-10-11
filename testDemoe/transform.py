from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = "images/Screenshot 2022-10-11 192536.png"
img = Image.open(img_path)
# print(img)
# 1.transforms 该如何使用(python)
writer = SummaryWriter("logs")

tensor_trans=transforms.ToTensor()
tensor_img = tensor_trans(img)
# print(tensor_img)

writer.add_image("Tensor_img",tensor_img)

writer.close()
