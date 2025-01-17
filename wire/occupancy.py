#!/usr/bin/env python

import os
import sys
import glob
import tqdm
import importlib
import time
import pdb
import copy
import argparse
import config_occupancy as config
import numpy as np
from scipy import io
from scipy import ndimage
import imageio.v2 as imageio
import cv2
import cuda_matmul
import torch
from torch.optim.lr_scheduler import LambdaLR

import matplotlib.pyplot as plt
plt.gray()
import modules.sin
from modules import utils
from modules import volutils

def get_coords(H, W, T=None):
    '''
        Get 2D/3D coordinates
    '''
    if T is None:
        X, Y = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
        coords = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))
    else:
        X, Y, Z = np.meshgrid(np.linspace(-1, 1, W),
                              np.linspace(-1, 1, H),
                              np.linspace(-1, 1, T))
        coords = np.hstack((X.reshape(-1, 1),
                            Y.reshape(-1, 1),
                            Z.reshape(-1, 1)))
    
    return torch.tensor(coords.astype(np.float32))
def main(opt):

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    im = io.loadmat('data/%s.mat'%opt.expname)['hypercube'].astype(np.float32)


    im = ndimage.zoom(im/im.max(), [1,1, 1], order=0)
    mcubes_thres = 0.5 
    
    hidx, widx, tidx = np.where(im > 0.99)
    im = im[hidx.min():hidx.max(),
            widx.min():widx.max(),
            tidx.min():tidx.max()]
    H, W, T = im.shape
    maxpoints = min(H*W*T, int(2e5))
    
    #input image shape [H*W,3] 
    imten = torch.tensor(im).cuda().reshape(H*W*T, 1)

    # Create model
    
    model = modules.sin.INR(
                   
                    opt=opt,
                    in_features=opt.in_features,
                    out_features=opt.out_features, 
                    hidden_features=opt.hidden_features,
                    hidden_layers=opt.hidden_layers,
                    rank_k = opt.rank_k,
                    freq=opt.frequency
                    ).cuda()
    # Optimizer
    optim = torch.optim.Adam(lr=opt.lr, params=model.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
    # Schedule to 0.1 times the initial rate
    scheduler = LambdaLR(optim, lambda x: 0.1**min(x/opt.niters, 1))

    criterion = torch.nn.MSELoss()
    # scheduler = LambdaLR(optim, lambda x: 0.1**min(x/opt.niter, 1))

    # Create input coords shape [H*W,2]
    mse_array = np.zeros(opt.niters)
    time_array = np.zeros(opt.niters)
    best_mse = float('inf')
    best_img = None
    
    
    tbar = tqdm.tqdm(range(opt.niters))
    # predicted im shape[H*W,3]
    coords = get_coords(H,W,T)
    
    im_estim = torch.zeros((H*W*T, 1), device='cuda')
    tic = time.time()

    

    trainloss =[]
    psnr_value = []
    for idx in tbar:
        indices = torch.randperm(H*W*T)
        
        train_loss = 0
        nchunks = 0
        for b_idx in range(0, H*W*T, maxpoints):
            b_indices = indices[b_idx:min(H*W*T, b_idx+maxpoints)]
            b_coords = coords[b_indices, ...].cuda()
            b_indices = b_indices.cuda()
            pixelvalues = model(b_coords[None, ...]).squeeze()[:, None]
            
            with torch.no_grad():
                im_estim[b_indices, :] = pixelvalues
        
            loss = criterion(pixelvalues, imten[b_indices, :])
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()
         
            lossval = loss.item()
            train_loss += lossval
            nchunks += 1
        mse_array[idx] = volutils.get_IoU(im_estim, imten, mcubes_thres)        
        time_array[idx] = time.time()
        
        
        if lossval < best_mse:
            best_mse = lossval
            best_img = copy.deepcopy(im_estim)




        trainloss.append(lossval)
        mse_array[idx] = train_loss/nchunks
        time_array[idx] = time.time()
        
        tbar.set_description('%.4e'%mse_array[idx])
        tbar.refresh()
        
        psnr_value.append(utils.psnr(im, im_estim.reshape(H, W, T).detach().cpu().numpy()))
        print(volutils.get_IoU(im_estim.reshape(H, W, T).detach().cpu().numpy(), im, mcubes_thres))
   

    total_time = time.time() - tic
    nparams = utils.count_parameters(model)
    print(nparams)
    best_img = best_img.reshape(H, W, T).detach().cpu().numpy()

    # savename = 'results/rank1_sin900.dae'
    # volutils.march_and_save(best_img, mcubes_thres, savename, True)
    print('IoU: ', volutils.get_IoU(best_img, im, mcubes_thres))
    print(opt.frequency,opt.rank_k)
    print(opt.expname)


if __name__ =='__main__':
    parser = argparse.ArgumentParser(description= 'low_rank MLP')
    config.add_config(parser)
    opt = parser.parse_args()

    opt.niters = 200
    # opt.pth = f"results/{opt.expname}/sin/"
    opt.rank_k = 1
    opt.frequency = 100

    opt.sigma = 0.1

    # os.makedirs(opt.pth,exist_ok=True)
    main(opt)
 