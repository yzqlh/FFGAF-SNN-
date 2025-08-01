
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt
from spikingjelly import visualizing
import numpy as np
import seaborn as sns
import torch.nn.functional as F
from torch.optim import Adam, SGD
import fxpmath as fxp
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.datasets import MNIST,CIFAR10


from config import *






def split_by_proportion(y, proportions=[2/40,3/40,4/40,5/40,4/40,4/40,4/40,4/40,5/40,5/40], dim=1):
    """
    按比例分割张量 y 的指定维度

    参数:
        y: 要分割的张量
        proportions: 分割比例，例如 [0.2, 0.3, 0.5] 表示分成三部分，比例为 2:3:5
        dim: 要分割的维度，默认为 1
    返回:
        一个包含分割后张量的列表
    """
    # 检查并归一化比例
    proportions = torch.tensor(proportions, dtype=torch.float32)
    proportions_sum = proportions.sum()
    # 如果比例总和不等于1，进行归一化处理
    if not torch.isclose(proportions_sum, torch.tensor(1.0)):
        proportions = proportions / proportions_sum

    # 获取指定维度的总大小
    total_size = y.size(dim)

    # 计算每个子集的大小
    subset_sizes = [int(p * total_size) for p in proportions]

    # 调整最后一个子集的大小以确保总和正确
    subset_sizes[-1] = total_size - sum(subset_sizes[:-1])

    # 分割张量
    y_sets = torch.split(y, subset_sizes, dim=dim)

    return y_sets

def visualize_mem(mem_rec):
    for id, rec in enumerate(mem_rec):
        arr = []
        rec = [i.cpu().detach() for i in rec]
        for i in rec:
            neuron_id = np.random.randint(0, 1500)
            x = i[np.random.randint(1, 500), neuron_id:neuron_id + 250]
            arr.append(x)

        visualizing.plot_2d_heatmap(array=np.asarray(arr), title=f'layer {id}', xlabel='Simulating Step',
                                    ylabel='Neuron Index', int_x_ticks=True, x_max=10, dpi=200)

    plt.show()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    sns.kdeplot(np.asarray(torch.stack(mem_rec[0], dim=0).cpu().detach()).flatten(), color='skyblue', ax=axs[0],
                clip=(0.25, 5))
    sns.kdeplot(np.asarray(torch.stack(mem_rec[1], dim=0).cpu().detach()).flatten(), color='skyblue', ax=axs[1],
                clip=(0.25, 5))
    sns.kdeplot(np.asarray(torch.stack(mem_rec[2], dim=0).cpu().detach()).flatten(), color='skyblue', ax=axs[2],
                clip=(0.25, 5))
    axs[0].set_title('layer 0')
    axs[1].set_title('layer 1')
    axs[2].set_title('layer 2')
    plt.show()


def visualize_activ(h):
    fig, axs = plt.subplots(1, 1, figsize=(15, 5))
    sns.kdeplot(h.flatten().data.cpu(), color='skyblue', ax=axs, clip=[-30, 30])
    axs.set_title("activ")
    plt.show()





def plot_layer_gradients(layer_gradients):
    """
    分层绘制梯度变化图。

    参数:
        layer_gradients: 一个二维列表，其中每个子列表表示一层的梯度值。
    """
    # 获取每层的梯度值
    gradients = []
    for layer in layer_gradients:
        # 提取梯度值（假设梯度是张量，使用 .item() 获取标量值）
        layer_grads = [grad.item() if hasattr(grad, 'item') else grad for grad in layer]
        gradients.append(layer_grads)

    # 分层绘制梯度变化曲线
    for i, layer_grad in enumerate(gradients):
        plt.figure(figsize=(10, 6))
        plt.plot(layer_grad)
        plt.title(f'Layer {i + 1} Gradient Over Time Steps')
        plt.xlabel('Time Step')
        plt.ylabel('Gradient Value')
        plt.grid(True)
        plt.savefig(f'Fmnist_layer_{i + 1}_gradients.png')
        plt.close()

def plot_layer_accuracies(layer_accuracies_epochs):
    """
    绘制每层每个 epoch 的精度变化图。

    参数:
        layer_accuracies_epochs: 一个二维列表，其中每个子列表表示一个 epoch 中五层的精度。
    """
    # 转置列表以获取每层的精度列表
    layer_accuracies = list(zip(*layer_accuracies_epochs))

    # 获取 epoch 数量
    epochs = list(range(1, len(layer_accuracies_epochs) + 1))

    # 绘制每层的精度变化曲线
    plt.figure(figsize=(10, 6))
    for i, accuracies in enumerate(layer_accuracies):
        plt.plot(epochs, accuracies, label=f'Layer {i + 1}')

    # 添加图表标题和轴标签
    plt.title('Layer-wise Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    # 添加图例
    plt.legend()

    # 保存图表
    plt.savefig('Fmnist_layer_accuracies.png')

    # 显示图表
    plt.show()

class ann_BatchNorm(nn.BatchNorm2d):

    def __init__(self, num_features=batch_size,channels = None, eps=1e-05, momentum=0.1, alpha=1, affine=False, track_running_stats=True,layer_thresh = 3):
        super(ann_BatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha
        self.running_mean = torch.ones([num_features,channels],device="cuda",requires_grad=False)
        self.running_var = torch.ones([num_features,channels],device="cuda",requires_grad=False)
        self.layer_thresh = layer_thresh

    def forward(self, input):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0,2,3])
            var = input.var([0,2,3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var
        input = self.layer_thresh * (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))

        return input,torch.sqrt(self.running_var)



class tdBatchNorm(nn.BatchNorm2d):

    def __init__(self, num_features=batch_size,channels = None, eps=1e-05, momentum=0.1, alpha=1, affine=False, track_running_stats=True,layer_thresh = 3):
        super(tdBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha
        self.weight = nn.Parameter(torch.ones([num_features,channels],device="cuda",requires_grad=True))
        self.bias = nn.Parameter(torch.zeros([num_features,channels],device="cuda",requires_grad=True))
        self.running_mean = torch.ones([num_features,channels],device="cuda",requires_grad=False)
        self.running_var = torch.ones([num_features,channels],device="cuda",requires_grad=False)
        self.layer_thresh = layer_thresh

    def forward(self, input):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0,3,4])
            var = input.var([0,3,4], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var
        input = self.layer_thresh * (input - mean[None, :, :, None, None]) / (torch.sqrt(var[None, :, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, :, None, None] ##  + self.bias[None, :,:, None, None]

        return input,torch.sqrt(self.running_var)



class convBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features=batch_size,channels = None, eps=1e-05, momentum=0.1, alpha=1, affine=False, track_running_stats=True,layer_thresh = 1):
        super(convBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha
        self.weight = nn.Parameter(torch.ones([time_steps,channels],device="cuda",requires_grad=True))
        self.bias = nn.Parameter(torch.zeros([time_steps,channels],device="cuda",requires_grad=True))
        self.running_mean = torch.zeros([channels],device="cuda",requires_grad=False)
        self.running_var = torch.ones([channels],device="cuda",requires_grad=False)
        self.layer_thresh = layer_thresh

    def forward(self, input):


        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0,1,3,4])
            var = input.var([0,1,3,4], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var
        input = self.layer_thresh * (input - self.running_mean[None,None ,:, None, None]) / (torch.sqrt(self.running_var[None,None , :, None, None] + self.eps))

        return input,torch.sqrt(self.running_var)

class tdBatchNorm(nn.BatchNorm2d):

    def __init__(self, num_features=batch_size,channels = None, eps=1e-05, momentum=0.1, alpha=1, affine=False, track_running_stats=True,layer_thresh = 1):
        super(tdBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha
        self.weight = nn.Parameter(torch.ones([num_features,channels],device="cuda",requires_grad=True))
        self.bias = nn.Parameter(torch.zeros([num_features,channels],device="cuda",requires_grad=True))
        self.running_mean = torch.ones([num_features,channels],device="cuda",requires_grad=False)
        self.running_var = torch.ones([num_features,channels],device="cuda",requires_grad=False)
        self.layer_thresh = layer_thresh

    def forward(self, input):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0,3,4])
            var = input.var([0,3,4], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var
        input = self.layer_thresh * (input - mean[None, :, :, None, None]) / (torch.sqrt(var[None, :, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, :, None, None] ##  + self.bias[None, :,:, None, None]

        return input,torch.sqrt(self.running_var)