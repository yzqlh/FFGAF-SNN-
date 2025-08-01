import logging
import os
import torch
import torch.nn as nn
# 参数定义
fast = False
start_end = [[0, 0], [0, 60], [0, 60], [0, 60]]
batch_size = 128
proportions = [[4/40,4/40,5/40,5/40,5/40,4/40,3/40,4/40,3/40,3/40],[4/40,4/40,5/40,5/40,5/40,4/40,3/40,4/40,3/40,3/40],[4/40,4/40,5/40,5/40,5/40,4/40,3/40,4/40,3/40,3/40],
                [4/40,4/40,5/40,5/40,5/40,4/40,3/40,4/40,3/40,3/40],[4/40,4/40,5/40,5/40,5/40,4/40,3/40,4/40,3/40,3/40]]

stride = [1,2,1,2,1,1,1]
lr = [0.1,0.1,0.1,0.1,0.1,0.1]
CW_lr = [1,10,10,100,100,100]
weight_c_grad = [1.5,1,0.8,0.5,0.25,0.1]

maxpool = [False,False,False,False,False,False,False]
archi = [40, 120, 120, 240]


time_steps = 10
layer_thresh = [2,1,1,1,1]
epochs = 100





class GradFloor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

myfloor = GradFloor.apply



thresh_act = [8,15,15,15,15]
L = [8,8,8,8,8]


class IF(nn.Module):
    def __init__(self,L=20, thresh=15.0, tau=1., gama=1.0):
        super(IF, self).__init__()
        self.thresh = nn.Parameter(torch.tensor([thresh],dtype=float), requires_grad=True)
        self.tau = tau
        self.gama = gama
        self.L = L
        self.loss = 0

    def forward(self, x):
        x = x / self.thresh
        x = torch.clamp(x,0,1)
        x = myfloor(x*self.L+0.5)/self.L
        x = x * self.thresh
        return x,self.thresh

