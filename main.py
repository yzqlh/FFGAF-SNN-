import argparse
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.datasets import MNIST,CIFAR10
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam, SGD
import numpy as np
from config import *
import logging
import os
from  function import *



class Net(torch.nn.Module):
    def __init__(self,dims):
        super().__init__()
        self.conv_layers = []

        for i, out_channels in enumerate(archi):

            in_channels = dims[-1][0]
            layer = Conv_Layer(dims[-1], in_channels=in_channels, out_channels=out_channels,
                              kernel_size=3, maxpool=maxpool[i],stride=stride[i],lr = lr[i] ,droprate=0, loss_criterion=None,layer_i=i).cuda()
            self.conv_layers.append(layer)
            dims.append(layer.next_dims)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        if load_model:
            for i in range(len(archi)):
                self.conv_layers[i].load_state_dict(torch.load(f'FM-ff+csnn+cwc{i}.pth'))

        self.n_classes = 10


    def predict(self, x):
        goodness_per_label = []
        out_list = []
        h_s = x
        h_t = []
        spike_sum = []
        for i, layer in enumerate(self.conv_layers):

            spike_sum.append(torch.sum(h_s,)/batch_size)
            if i==0:
                h_s,h= layer.forward_encode(h_s)
            else:
                h_s,h= layer(h_s)
            h_t.append(h)

        for i in range(len(h_t)):
            h = h_t[i]
            h_reshaped = split_by_proportion(h,proportions[i])
            goodness_factors = [y_set.pow(2).mean((1, 2, 3)).unsqueeze(-1) for y_set in h_reshaped]
            mean_squared_values = torch.cat(goodness_factors, 1)
            _, predicted_classes = torch.max(mean_squared_values, dim=1)

            goodness_per_label.append(predicted_classes)
            out_list.append(mean_squared_values)
        return goodness_per_label,out_list[-1]


    def train(self, x_pos,y,epoch):
        h_pos = x_pos
        total_loss = 0
        for i, layer in enumerate(self.conv_layers):
            layer.training =True
            if fast:
                s, e = start_end[i]
                if epoch in list(range(s, e)):
                    h_pos, loss = layer.train(h_pos.detach(), y,i)
                    total_loss += loss
                else:
                    h_pos,_ = layer(h_pos)
            else:
                h_pos, loss,grad = layer.train(h_pos.detach(), y,i)
                total_loss += loss
        return total_loss




def actfun( input,i,th):
    a =input.ge(th/time_steps).float()
    return a

def mem_update(x,mem,spike,i,th):
    mem = mem -(spike) + x
    spike = actfun(mem,i,th)
    return mem,spike




class Conv_Layer(nn.Conv2d):
    def __init__(self, in_dims, in_channels=1, out_channels=8, num_classes=10, kernel_size=7, stride=1,lr = 0.1, padding=1,
                 maxpool=True, droprate=0, loss_criterion='CwC_CE', ClassGroups=None,layer_i=None):
        super(Conv_Layer, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                         padding=padding)

        self.dims = in_dims
        self.outc = out_channels
        self.kernel_size = kernel_size
        self.outsize = int(((self.dims[1] - self.kernel_size + (2 * padding)) / stride) + 1)
        self.ismaxpool = maxpool
        self.loss = 10000
        self.steps = time_steps
        self.ClassGroups = ClassGroups
        self.relu = nn.ReLU()
        self.rfy_relu = IF(L=L[layer_i],thresh=thresh_act[layer_i])
        if maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.outsize = int(self.outsize / 2)  # if maxpool

        self.wsz = self.outsize  # for visualisation of features
        self.N_neurons_out = self.outc * self.outsize ** 2  # total number of output neurons
        self.next_dims = [self.outc, self.outsize, self.outsize]
        self.td_bn = tdBatchNorm(channels=self.outc,layer_thresh=layer_thresh[layer_i])
        self.conv_bn = convBatchNorm(channels=self.outc,layer_thresh=layer_thresh[layer_i])
        self.anbn = ann_BatchNorm(channels=self.outc,layer_thresh=layer_thresh[layer_i])

        self.dropout = nn.Dropout(0.5)
        if layer_i==0:
            self.weight_w = nn.Parameter(
                0.5 * torch.ones([out_channels], device=device, requires_grad=True))
            self.bias_b = nn.Parameter(torch.zeros([out_channels], device=device, requires_grad=True))
        else:
            self.weight_w = nn.Parameter(0.5*torch.ones([out_channels],device=device,requires_grad=True))
            self.bias_b = nn.Parameter(torch.zeros([out_channels], device=device, requires_grad=True))
        self.out_weight = nn.Parameter(torch.ones(num_classes,device=device,requires_grad=True))
        self.loss_criterion = loss_criterion
        self.criterion = nn.CrossEntropyLoss()
        self.ep_losses = []
        self.num_classes = num_classes
        self.gf = None
        self.lr = lr # 0.01
        self.opt = Adam(self.parameters(), lr=self.lr)
        print(self.dims, self.N_neurons_out)
        for name, param in self.named_parameters():
            print(f"参数名称: {name}, 参数值的形状: {param.shape}")

    def forward(self, x):

        hidden_out = []
        x = self.dropout(x)
        # Forward Pass
        x_a = torch.zeros((self.steps,) + self._conv_forward(x[0], self.weight, self.bias).shape, device=x.device).detach()
        for step in range(self.steps):
            x_a[step, ...] = self._conv_forward(x[step, ...], self.weight, self.bias)
        a = self.weight_w
        y = x_a * a[None, None, :, None, None] + self.bias_b[None, None, :, None, None]
        y, _ = self.td_bn(y)
        y = y.sum(0)
        x_act,th= self.rfy_relu(y)

        if self.ismaxpool:
            x_act = self.maxpool(x_act)

        x_h = [x_act.detach()]
        for i in range(time_steps-1):
            x_h.append(torch.zeros_like(x_act))
        mem = torch.zeros(x_act.shape[0:])
        spike = torch.zeros(x_act.shape[0:])
        mem = mem.to(device)
        spike = spike.to(device)
        for i in range(time_steps):
            mem, spike = mem_update(x=x_h[i], mem=mem, spike=spike,i=i,th=th)
            hidden_out.append(spike)
        h_s = torch.stack(hidden_out,dim=0)

        return h_s, x_act

    def forward_encode(self, x):

        hidden_out = []
        x_a  = self._conv_forward(x, self.weight, self.bias)
        a = self.weight_w
        y = x_a * a[None, :, None, None] + self.bias_b[None, :, None, None]
        y, _ = self.anbn(y)
        x_act,th= self.rfy_relu(y)


        if self.ismaxpool:
            x_act = self.maxpool(x_act)


        x_h = [x_act.detach()]
        for i in range(time_steps - 1):
            x_h.append(torch.zeros_like(x_act))
        mem = torch.zeros(x_act.shape[0:])
        spike = torch.zeros(x_act.shape[0:])
        mem = mem.to(device)
        spike = spike.to(device)
        for i in range(time_steps):
            mem, spike = mem_update(x=x_h[i], mem=mem, spike=spike,i=i,th=th)
            hidden_out.append(spike)
        h_s = torch.stack(hidden_out, dim=0)

        return h_s, x_act

    def goodness_factorCW(self, y, gt,i):

        pos_mask = torch.zeros((gt.shape[0], self.num_classes), dtype=torch.uint8, device=y.device)
        arange_tensor = torch.arange(gt.shape[0], device=y.device)
        pos_mask[arange_tensor, gt] = 1
        pos_mask = pos_mask.bool()
        neg_mask = ~pos_mask

        y_sets = split_by_proportion(y,proportions[i])

        goodness_factors = [y_set.pow(2).mean((1, 2, 3)).unsqueeze(-1) for y_set in y_sets]
        gf = torch.cat(goodness_factors, 1)
        g_pos = gf[pos_mask].view(-1, 1)
        g_neg = gf[neg_mask].view(gf.shape[0], -1).mean(1).unsqueeze(-1)

        return g_pos, g_neg, gf


    def train(self, x_pos, gt,i):
        self.opt.zero_grad()
        # forward pass
        if i==0:
            y_s,y_h = self.forward_encode(x_pos)
        else:
            y_s,y_h = self.forward(x_pos)
        #y_h = torch.sum(y_h,dim=0)
        gt = gt.cuda()
        g_pos, g_neg, self.gf = self.goodness_factorCW(y_h, gt,i)
        loss = loss_fn(self.gf, gt)
        loss.backward()
        self.weight_w.grad = self.weight_w.grad * torch.sigmoid(8*weight_c_grad[i] * (self.weight_w - 0.5)) * (
                  1 - torch.sigmoid(8*weight_c_grad[i] * (self.weight_w) - 0.5))
        self.bias_b.grad = self.bias_b.grad * torch.sigmoid(8 *weight_c_grad[i]* (self.bias_b)) * (
                    1 - torch.sigmoid(8*weight_c_grad[i] * (self.bias_b)))

        self.opt.step()

        return y_s,loss,self.weight.grad.mean()








class CwCLoss(nn.Module):
    def __init__(self):
        super(CwCLoss, self).__init__()
        self.eps = 1e-9

    def forward(self, g_pos, logits):
        # Ensure that values are not too large/small for exp function
        logits = torch.clamp(logits, min=-50, max=50)
        g_pos = torch.clamp(g_pos, min=-50, max=50)

        # Calculate the sum of the exponential of all goodness scores for each sample
        exp_sum = torch.sum(torch.exp(logits), dim=1)

        # Compute the CwC loss using the formula
        loss = -torch.mean(torch.log((torch.exp(g_pos) + self.eps) / (exp_sum + self.eps)))

        return loss

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",

        type=int,
        default=120,
        metavar="N",
        help="number of epochs to train (default: 1000)",
    )
    args = parser.parse_args()

    device = "cuda"
    loss_fn = nn.CrossEntropyLoss()
    ############################
    ###########################
    save_model = True
    load_model = False
    train = True
    visualize_h = False
    acc = 0
    best_acc = 0.68
    plot_acc_vr = True
    ############################
    ###########################
    transform = Compose(
        [
            ToTensor(),
        ]
    )
    train_loader = DataLoader(
        CIFAR10("./data/", train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True,drop_last=True)

    test_loader = DataLoader(
        CIFAR10("./data/", train=False, download=True, transform=transform), batch_size=batch_size, shuffle=True,drop_last=True  )

    net = Net(dims = [[3, 32, 32]])
    net.to(device)
    torch.autograd.set_detect_anomaly(True)
    val_accuracies = []
    val_accuracies_w = []
    layer_accuracies = []
    # visualize_weight(net)
    for epoch in range(epochs):
        loss = 0
        total = 0
        correct = 0
        correct_w = 0
        total_te = 0
        correct_te = 0
        correct_te_w = 0
        epoch_losses =  [[] for _ in range(len(archi))]
        epoch_errors = [[] for _ in range(len(archi))]


        if train:
            for x, y in tqdm(train_loader):
                b, C, H, W = x.shape


                code_spike = x.to(device)
                y = y.to(device)
                loss += net.train(code_spike, y,epoch)
                total += y.size(0)
                pred,_ = net.predict(code_spike)

                for i in range(len(pred)):
                    batch_loss =  pred[i].eq(y.cuda()).float().mean().item()
                    epoch_losses[i].append(batch_loss)

            print('Epoch: {}'.format(epoch))
            print(f"loss:{loss}")
            for i in range(len(epoch_losses)):
                print(f'Avg Pred - Train  layer{i}  = {np.asarray(epoch_losses[i]).mean()}')

        wrong_pred = []
        wrong_goodness = []


        for idx, (x, y) in enumerate(test_loader):
            original_spike_train = []
            b, C, H, W = x.shape

            code_spike = x.to(device)

            y = y.to(device)
            total += y.size(0)
            pred,out = net.predict(code_spike)
            for i in range(len(pred)):
                batch_loss = pred[i].eq(y.cuda()).float().mean().item()
                epoch_errors[i].append(batch_loss)
        for i in range(len(epoch_losses)):
            print(f'Avg Pred - Test  layer{i}  = {np.asarray(epoch_errors[i]).mean()}')
        layer_accuracies.append([np.asarray(epoch_errors[i].copy()).mean() for i in range(len(epoch_losses))])

        if save_model:
            acc = np.asarray(epoch_errors).mean(1).max()
            if i >= 0:
                if acc > best_acc:
                    best_acc = acc
                    for i in range(len(archi)):
                        torch.save(net.conv_layers[i].state_dict(), f"FM-ff+csnn+cwc{i}.pth")
                    print('Saving..')
            print(' best acc:', best_acc)
    if plot_acc_vr:
        plot_layer_accuracies(layer_accuracies_epochs=layer_accuracies)


