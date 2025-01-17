import argparse
import os
import numpy as np
import torch
def add_config(parser):
    parser.add_argument('--nonlin',type=str,default = 'gauss')
    parser.add_argument('--niters', type = int,default = 10000)
    parser.add_argument('--lr',type = float,default=1e-3)
    parser.add_argument('--expname',type = str, default = 'dragon_512')
    parser.add_argument('--hidden_layers',type = int,default=2)
    parser.add_argument('--hidden_features',type = int,default = 256)
    parser.add_argument('--in_features',type = int,default = 3)
    parser.add_argument('--out_features',type = int,default = 1)
    parser.add_argument('--rank_k',type = int,default = 0)
    parser.add_argument('--seed',type = int,default = 1234)
    parser.add_argument('--omega', type = float,default = 1)
    parser.add_argument('--choice', type = str,default = 'naive')


                                                