import torch
from matplotlib import pyplot as plt
from numpy import random


def set_figsize(figsize=(3.5, 2.5)):
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize


# 每次返回batch_size（批量大小）个随机样本的特征和标签。
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个batch
        yield features.index_select(0, j), labels.index_select(0, j)

