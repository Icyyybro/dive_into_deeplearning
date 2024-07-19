import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l



"""生成y=Xw+b+噪声"""
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape(-1, 1)

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

print("features: ", features[0], "\nlabel: ", labels[0])
d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
plt.show()



"""读取数据集，并切片分成多个小组"""
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)                 #shuffle函数用于随机打乱一个列表
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

batch_size = 10
for x, y in data_iter(batch_size, features, labels):
    print(x, "\n", y)



#以下为优化
"""初始化参数"""
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)



"""定义模型"""
def linreg(x, w, b):
    return torch.matmul(x, w) + b



"""定义损失函数"""
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2/2



"""梯度下降"""
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()          #将梯度清零，防止影响下一个梯度计算



"""开始训练"""
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for x,y in data_iter(batch_size, features, labels):
        l = loss(net(x, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch: {epoch + 1}, loss: {train_l.mean()}')




