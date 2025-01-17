#!/usr/bin/env python
import ctypes
import pdb
import math
from modules import utils
import torch.nn.functional as F
import numpy as np
from pretrain import pretrain
import torch
from torch import nn
import math
import matplotlib.pyplot as plt
import cuda_matmul

# _lib = ctypes.CDLL('/home/yiping/Downloads/wire/modules/matrixMul.so')

class GaussLayer(nn.Module):
    def __init__(self,opt, in_features, out_features,rank_k ,fre,is_first=False):

        super().__init__()
        self.in_features = in_features
        self.k =rank_k
        print(self.k)
        self.w=fre
        self.opt = opt
        self.is_first = is_first
        if self.is_first or self.k ==0:

                self.linear = nn.Linear(in_features, out_features)
        else:
            
            self.A = nn.Parameter(torch.empty(out_features,self.k))
            self.B = nn.Parameter(torch.empty(self.k,in_features))

            # print(self.A.max(),self.B.max())
            nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.B,a=math.sqrt(5))
            self.bias = nn.Parameter(torch.empty(out_features))
            nn.init.uniform_(self.bias,-1/16,1/16)

    def gaussian(self, x, sigma = 0.1):
        return (-0.5*(x)**2/sigma**2).exp()
    def forward(self, input):
        if self.is_first:
            return self.gaussian(self.linear(input))
        
        if self.k == 0:
            return self.gaussian(self.linear(input))

        elif self.opt.choice == 'naive':
            return self.gaussian(input@self.A@self.B+self.bias)
            # return self.gaussian(F.linear(input,self.A@self.B,self.bias))
        
        elif self.opt.choice == 'sin':
            # x = torch.sin(self.w*self.A@self.B)/16

            return torch.exp(-F.linear(input,torch.sin(self.w*self.A@self.B)/16,self.bias)**2/(0.1**2)/2)
class INR(nn.Module):

    def __init__(self,opt, in_features,
                 hidden_features, hidden_layers, 
                 out_features,rank_k,
                 freq):
        super().__init__()
        self.nonlin = GaussLayer
        self.net = []
        self.net.append(self.nonlin(opt,in_features, hidden_features, rank_k,freq,is_first=True))



        for i in range(hidden_layers):

            self.net.append(self.nonlin(opt,hidden_features, hidden_features, rank_k,freq))


        final_linear = nn.Linear(hidden_features,
                                    out_features,
                                    dtype=torch.float)

        # final_linear = self.nonlin(opt,hidden_features, out_features, 0,freq,is_last=True)
        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
                    
        return output