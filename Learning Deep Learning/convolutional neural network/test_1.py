import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(".//data", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


class nn_test(nn.Module):
    def __init__(self):
        super(nn_test, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
        self.conv2 = Conv2d(in_channels=6, out_channels=3, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        y = self.conv2(x)
        return y


my_network = nn_test()

for data in dataloader:
    imgs, targets = data
    output = my_network.forward(imgs)
    print(output)
