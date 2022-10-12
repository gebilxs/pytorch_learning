from PIL import Image, ImageFile
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# 不知道返回值的时候 print print(type()) debug
writer =SummaryWriter("logs")
img = Image.open("images/SVM_08.png")
# img = Image.open("images/Screenshot 2022-10-11 223458.png").convert('RGB')
# 四通道图片在RGB的基础之上增加了透明度的维度
print(img)

#ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor",img_tensor)

#Normalize
print(img_tensor[0][0][0])
# trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
# trans_norm = transforms.Normalize([1,3,5],[3,2,1])
trans_norm = transforms.Normalize([6,3,2],[9,3,5])

img_norm = trans_norm(img_tensor)

print(img_norm[0][0][0])
#writer.add_image("Normalize",img_norm)
# writer.add_image("Normalize",img_norm,1)
writer.add_image("Normalize",img_norm,2)


#resize
print(img.size)
trans_resize=transforms.Resize((512,512))
img_resize=trans_resize(img)
img_resize=trans_totensor(img_resize)
writer.add_image("Resize",img_resize,0)
print(img_resize)

#compose - resize -2
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2,trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize",img_resize_2,1)


# RandomCrop
trans_random = transforms.RandomCrop(256)
trans_compose_2=transforms.Compose([trans_random,trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop",img_crop,i)

writer.close()