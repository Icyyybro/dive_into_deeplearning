import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from LeNet.module import LeNet

# 导入训练集和测试集
train_data = torchvision.datasets.CIFAR10(root=".//data", train=True, transform=torchvision.transforms.ToTensor,
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root=".//data", train=False, transform=torchvision.transforms.ToTensor,
                                         download=True)
# 训练集和测试集大小
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f'The size of the train is : {train_data_size}')
print(f'The size of the test is : {test_data_size}')

# 加载数据
train_dataloader = DataLoader(dataset=train_data, batch_size=64)
test_dataloader = DataLoader(dataset=test_data, batch_size=64)

# 构建网络
net = LeNet()

# 损失函数
loss_func = nn.CrossEntropyLoss()

# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# 训练次数
epoch = 10
total_train_step = 0
total_test_step = 0

for i in range(epoch):
    print(f'-------第{i+1}次训练开始------')

    # 每次训练开始
    for data in train_dataloader:
        images, targets = data
        output = net(images)
        loss = loss_func(output, targets)

        # 调用优化器优化参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新参数
        total_train_step += 1
        print(f'训练次数 : {total_train_step}\tloss : {loss.item()}')

    # 测试
    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            images, targets = data
            output = net(images)
            loss = loss_func(output, loss)
            total_test_loss += loss
    print(f'******整体测试集上的loss : {total_test_loss}')