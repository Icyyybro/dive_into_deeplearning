import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l



"""下载图片"""
# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0～1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root=".//data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root=".//data", train=False, transform=trans, download=True)
#print(f'num_train: {len(mnist_train)}\nnum_test: {len(mnist_test)}')



"""取图片"""
def get_fashion_mnist_labels(labels):  #@save
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]



"""可视化图片"""
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
d2l.plt.show()



batch_size = 256
"""使用4个进程来读取数据"""
def get_dataloader_workers():  #@save
    return 1



"""读取所有数据"""
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,num_workers=get_dataloader_workers())
timer = d2l.Timer()
for X, y in train_iter:
    continue
f'{timer.stop():.2f} sec'