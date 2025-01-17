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
import config
import numpy as np
from scipy import io
from scipy import ndimage
import imageio.v2 as imageio
import torch
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
from modules import utils
import modules.gauss
os.environ['CUDA_LAUNCH_BLOCKING']='1'
def zero_non_diagonal_gradients(model):
    grad_count = 0  # 初始化计数器
    for name, param in model.named_parameters():
        if 'Aa' in name or 'Ab' in name:  # 检查参数名称
            if param.grad is not None:
                diagonal_mask = torch.eye(param.size(0), dtype=torch.bool).to(param.device)
                non_diagonal_grad = ~diagonal_mask
                grad_count += non_diagonal_grad.sum().item()  # 累计非对角线梯度数量
                param.grad.data[non_diagonal_grad] = 0
    return grad_count
def main(opt):

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    im = imageio.imread('images/%s.png'%opt.expname).astype(np.float32)
    im = ndimage.zoom(im/im.max(), [1,1, 1], order=0)
    
    H, W, T = im.shape
    
    
    #input image shape [H*W,3] 
    imten = torch.tensor(im).cuda().reshape(H*W, T)

    # Create model
    model = modules.gauss.INR(
                   
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
    scheduler = LambdaLR(optim, lambda x: 0.1**min(x/opt.niter, 1))
    criterion = torch.nn.MSELoss()
    
    # Create input coords shape [H*W,2]
    mse_array = np.zeros(opt.niter)
    time_array = np.zeros(opt.niter)
    tbar = tqdm.tqdm(range(opt.niter))
    # predicted im shape[H*W,3]
    im_estim = torch.zeros((H*W, T), device='cuda')
    tic = time.time()
    coords = torch.linspace(-1,1,im.shape[0])
    x_test = torch.stack(torch.meshgrid(coords,coords))
    x_test = x_test.permute(1,2,0)

    test_data = [x_test.view(-1,2),imten]

    train_data = [(x_test[::2,::2]).contiguous().view(-1,2), torch.tensor(im[::2,::2]).contiguous().view(-1,T).cuda()]
    trainloss =[]
    psnr_value = []
    for idx in tbar:
        train_loss = 0
        nchunks = 0
        inputs,labels = train_data
        perm = torch.randperm(inputs.size(0))
        outputs = model(inputs[perm].cuda())

        loss = criterion(outputs,labels[perm].cuda())
        optim.zero_grad()
        loss.backward(retain_graph=True)

        # num=zero_non_diagonal_gradients(model)
        
        optim.step()
        scheduler.step()

        with torch.no_grad():
            inputs,labels=test_data
            im_estim = model(inputs.cuda())
            
        lossval = loss.item()
        train_loss += lossval
        nchunks += 1

        trainloss.append(lossval)
        mse_array[idx] = train_loss/nchunks
        time_array[idx] = time.time()
        
        tbar.set_description('%.4e'%mse_array[idx])
        tbar.refresh()
        
        psnr_value.append(utils.psnr(im, im_estim.reshape(H, W, T).detach().cpu().numpy()))

   
    print(total_params)
    total_time = time.time() - tic
    nparams = utils.count_parameters(model)
    best_img = im_estim.reshape(H, W, T).detach().cpu().numpy()

    indices, = np.where(time_array > 0)
    time_array = time_array[indices]
    mse_array = mse_array[indices]
    

    print('Total time %.2f minutes'%(total_time/60))
    print('PSNR: ',np.mean(psnr_value[-100:]))
    print('Total pararmeters: %.2f million'%(nparams/1e6))

    img = np.clip(best_img, 0, 1)
    plt.imsave('full_rank.png',img.reshape(H,W,T))
    print(opt.rank_k)
    np.max(psnr_value[-100:])
    torch.save(model.state_dict(), 'model_params.pth')

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description= 'low_rank MLP')
    config.add_config(parser)

    opt = parser.parse_args()

    main(opt)

######################################
    # opt.choice = 'full'
    # opt.omega = np.pi
   
    # if opt.choice == 'full':
    #     opt.pth = f"results/{opt.expname}/full/"
    #     opt.rank_k = 0
    #     os.makedirs(opt.pth,exist_ok=True)
    # elif opt.choice == 'naive':
    #     opt.pth = f"results/{opt.expname}/naive/"
    #     os.makedirs(opt.pth,exist_ok=True)
    # elif opt.choice == 'sine':
    #     opt.pth = f"results/{opt.expname}/sin{int(opt.omega/np.pi)}pi/"
    #     os.makedirs(opt.pth,exist_ok=True)

    # psnrs = []
    # sranks = []
    # nums = []
    # psnr, srank, num = main(opt)

# ##############################################################
#     opt.choice = 'full'
#     opt.omega = np.pi
   
#     if opt.choice == 'full':
#         opt.pth = f"results/{opt.expname}/full/"
#         opt.rank_k = 0
#         os.makedirs(opt.pth,exist_ok=True)
#     elif opt.choice == 'naive':
#         opt.pth = f"results/{opt.expname}/naive/"
#         os.makedirs(opt.pth,exist_ok=True)
#     elif opt.choice == 'sine':
#         opt.pth = f"results/{opt.expname}/sin{int(opt.omega/np.pi)}pi/"
#         os.makedirs(opt.pth,exist_ok=True)

#     psnrs = []
#     sranks = []
#     nums = []
#     for opt.rank_k in range(1,20):
#         psnr, srank, num = main(opt)
#         print(srank)
#         psnrs.append(psnr)
#         sranks.append(srank)
#         nums.append(num)
#     for opt.rank_k in range(20,120,5):
#         psnr, srank, num = main(opt)
#         print(srank)
#         psnrs.append(psnr)
#         sranks.append(srank)
#         nums.append(num)
#     torch.save(psnrs,opt.pth+'psnr.pth')
#     torch.save(sranks,opt.pth+'srank.pth')
#     torch.save(nums,opt.pth+'nums.pth')

# ##############################################################
    # opt.choice = 'naive'
    # opt.omega = 2*np.pi
    # opt.lr = 5e-4
    # opt.niters = 10000
    # if opt.choice == 'full':
    #     opt.pth = f"results/{opt.expname}/full/"
    #     opt.rank_k = 0
    #     os.makedirs(opt.pth,exist_ok=True)
    # elif opt.choice == 'naive':
    #     opt.pth = f"results/{opt.expname}/naive/"
    #     os.makedirs(opt.pth,exist_ok=True)
    # elif opt.choice == 'sine':
    #     opt.pth = f"results/{opt.expname}/sin{int(opt.omega/np.pi)}pi/"
    #     os.makedirs(opt.pth,exist_ok=True)

    # psnrs = []
    # sranks = []
    # nums = []
    # for opt.std_B in [0.1]:
    #     opt.std_C = 0.3
    #     print(opt.std_B)
    #     for opt.rank_k in [200]:
    #         psnr, srank, num = main(opt)
    #         print(srank)
    #         psnrs.append(psnr)
    #         sranks.append(srank)
    #         nums.append(num)
    # for opt.rank_k in range(20,120,5):
    #     psnr, srank, num = main(opt)
    #     print(srank)
    #     psnrs.append(psnr)
    #     sranks.append(srank)
    #     nums.append(num)
    # torch.save(psnrs,opt.pth+'psnr.pth')
    # torch.save(sranks,opt.pth+'srank.pth')
    # torch.save(nums,opt.pth+'nums.pth')

# ##############################################################
    # opt.choice = 'sin'
    # opt.rank_k = 1
    # opt.device = 'cuda:0'
    # opt.omega = 525
    # opt.w_sinin = 35
    # opt.sin_sin_w = 1
    # psnrs = []
    # sranks = []
    # nums = []
    # opt.o = 16
    # opt.BN = False
    # opt.sigma = 0.1
    # opt.hidden_layers = 4
    # if opt.choice == 'full':
    #     opt.pth = f"results/{opt.expname}/full/num_layers_{int(opt.hidden_layers)}/"
    #     opt.rank_k = 0
    #     os.makedirs(opt.pth,exist_ok=True)
    # elif opt.choice == 'naive':
    #     # opt.pth = f"results/{opt.expname}/rank_{int(opt.rank_k)}/naive/"
    #     opt.pth = f"results/{opt.expname}/naive/rank_{int(opt.rank_k)}/num_layers_{int(opt.hidden_layers)}/"
    #     os.makedirs(opt.pth,exist_ok=True)
    # elif opt.choice == 'sin':
    #     opt.pth = f"results___/{opt.expname}/rank_{int(opt.rank_k)}/num_layers_{int(opt.hidden_layers)}/"
    #     os.makedirs(opt.pth,exist_ok=True)
    # elif opt.choice == 'sin_sin':
    #     opt.pth = f"results/{opt.expname}/rank_{int(opt.rank_k)}/sin_sin{int(opt.sin_sin_w)}/"
    #     os.makedirs(opt.pth,exist_ok=True)
    # elif opt.choice == 'cos':
    #     opt.pth = f"results/{opt.expname}/cos{int(opt.omega)}/"
    #     os.makedirs(opt.pth,exist_ok=True)
    # elif opt.choice == 'matmul':
    #     opt.pth = f"results/{opt.expname}/rank_{int(opt.rank_k)}/matmul{int(opt.w_sinin)}/"
    #     os.makedirs(opt.pth,exist_ok=True)

    # psnr, srank, num = main(opt)
    # print('srank: ',srank)
    # # print('#layers: ',opt.hidden_layers)
    # psnrs.append(psnr)
    # torch.save(psnrs,opt.pth+'psnr.pth')
    # torch.save(sranks,opt.pth+'srank.pth')
    # torch.save(nums,opt.pth+'nums.pth')

    # opt.rank_k = 50
    # for opt.omega in range(10,70,5):
    #     if opt.choice == 'full':
    #         opt.pth = f"results_/{opt.expname}/full/"
    #         opt.rank_k = 0
    #         os.makedirs(opt.pth,exist_ok=True)
    #     elif opt.choice == 'naive':
    #         # opt.pth = f"results/{opt.expname}/rank_{int(opt.rank_k)}/naive/"
    #         opt.pth = f"results/{opt.expname}/naive/"
    #         os.makedirs(opt.pth,exist_ok=True)
    #     elif opt.choice == 'sin':
    #         opt.pth = f"results/{opt.expname}/rank_{int(opt.rank_k)}/sin{int(opt.omega)}/"
    #         os.makedirs(opt.pth,exist_ok=True)
    #     elif opt.choice == 'cos':
    #         opt.pth = f"results/{opt.expname}/cos{int(opt.omega)}/"
    #         os.makedirs(opt.pth,exist_ok=True)
    #     elif opt.choice == 'matmul':
    #         opt.pth = f"results/{opt.expname}/rank_{int(opt.rank_k)}/matmul{int(opt.w_sinin)}/"
    #         os.makedirs(opt.pth,exist_ok=True)
    #     psnrs = []
    #     sranks = []
    #     nums = []
    #     psnr, srank, num = main(opt)
    #     print('srank: ',srank)
    #     print('omega: ',opt.omega)
    # torch.save(psnrs,opt.pth+'psnr.pth')
    # torch.save(sranks,opt.pth+'srank.pth')
    # torch.save(nums,opt.pth+'nums.pth')












#     '''
#     psnr_value =[]
#     srank_ =[]
#     idx = []
#     nums =[]
#     rank_k=200
#     #im, im_estim=main(rank_k)
#     #best_psnr.append(utils.psnr(im, im_estim))
#     #idx.append(rank_k)
#     stdb = 0.1
#     stdc = 0.35
#     w_ = 10*np.pi
#     for rank_k in [4]:
#         psnr,srank,num=main(rank_k,offline=False,stdb=stdb,stdc=stdc,w_=w_,std_bias=std_bias)
#         #print(std_bias)
#         nums.append(num)
#         print(srank)
#         srank_.append(srank)
#         psnr_value.append(psnr)
#         idx.append(rank_k)
    
#     for rank_k in range(20,200,3):
#         psnr,srank,num=main(rank_k,offline=False,stdb=stdb,stdc=stdc,w_=w_,std_bias=std_bias)
#         #print(std_bias)
#         nums.append(num)
#         print(srank)
#         srank_.append(srank)
#         psnr_value.append(psnr)
#         idx.append(rank_k)
#     #smooth_psnr = utils.moving_average(psnr,window_size=5)

#     plt.figure()
#     plt.plot(idx,psnr_value,linestyle='-')
#     plt.xlabel('k')
#     plt.ylabel('PSNR')
#     plt.title('PSNR vs k')
#     plt.savefig('results/psnr_vs_k')
#     torch.save(psnr_value,'results/psnr.pth')
#     torch.save(srank_,'results/srank.pth')
#     torch.save(nums,'results/nums.pth')

#     psnr =[]
#     idx = []
#     for rank_k in range(1,20):
#         im, im_estim=main(rank_k,offline=True)
#         psnr.append(utils.psnr(im, im_estim))
#         idx.append(rank_k)
#     for rank_k in range(20,200,3):
#         im, im_estim=main(rank_k,offline=True)
#         psnr.append(utils.psnr(im, im_estim))
#         idx.append(rank_k)
#     smooth_psnr = utils.moving_average(psnr,window_size=5)

#     plt.figure()
#     plt.plot(idx,psnr,linestyle='-')
#     plt.xlabel('value of low-rank k')
#     plt.ylabel('PSNR')
#     plt.title('PSNR vs k')
#     plt.savefig('results/psnr_vs_k_pretrain')
#     torch.save(psnr,'results/psnr_pretrain.pth')
# '''