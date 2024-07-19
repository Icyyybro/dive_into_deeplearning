import torch

#arange 创建一个行向量 x。这个行向量包含以0开始的前12个整数
x = torch.arange(12)
print(x)

#shape属性来访问张量（沿每个轴的长度）的形状
print(x.shape)

#张量中元素的总数
print(x.numel())

#改变一个张量的形状而不改变元素数量和元素值
x = x.reshape(3,4)
print(x)

#创建一个形状为（3,4）的张量。 其中的每个元素都从均值为0、标准差为1的标准高斯分布（正态分布）中随机采样
x = torch.randn(3,4)
print(x)

#指定元素的值来创建
x = torch.tensor([[2,1,3],[5,6,7],[3,1,2]])
print(x)