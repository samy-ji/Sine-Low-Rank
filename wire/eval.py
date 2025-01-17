import torch
import imageio
from modules import utils
import os
import re
from scipy import ndimage
import numpy as np

def main(pth):
    expname = 'fox'     # Volume to load

    im = imageio.imread('images/%s.png'%expname).astype(np.float32)
    im = im/im.max()
    H, W, T = im.shape

    model = torch.load(pth)

    imten = torch.tensor(im).cuda().reshape(H*W, T)

    coords = torch.linspace(-1,1,im.shape[0])
    x_test = torch.stack(torch.meshgrid(coords,coords))
    x_test = x_test.permute(1,2,0)

    test_data = [x_test.view(-1,2),imten]
    inputs,labels=test_data
    im_estim = model(inputs.cuda())
    A1 = torch.sin(3*np.pi*model.net[1].A@model.net[1].B)
    A2 = torch.sin(3*np.pi*model.net[2].A@model.net[2].B)
    A3 = torch.sin(3*np.pi*model.net[3].A@model.net[3].B)
    A4 = torch.sin(3*np.pi*model.net[4].A@model.net[4].B)
    # A = model.net[1].linear.weight.data
    # A2 = model.net[2].linear.weight.data
    # A3 = model.net[3].linear.weight.data
    # A4 = model.net[4].linear.weight.data
    psnr=utils.psnr(im, im_estim.reshape(H, W, T).detach().cpu().numpy())
    srank1=utils.srank(A1).detach().cpu().numpy()
    srank2=utils.srank(A2).detach().cpu().numpy()
    srank3=utils.srank(A3).detach().cpu().numpy()
    srank4=utils.srank(A4).detach().cpu().numpy()

    _,s1,_ = torch.svd(A1)
    _,s2,_ = torch.svd(A2)
    _,s3,_ = torch.svd(A3)
    _,s4,_ = torch.svd(A4)

    return s1.detach().cpu().numpy(),s2.detach().cpu().numpy(),s3.detach().cpu().numpy(),s4.detach().cpu().numpy()

def rename(pth):
    file_list = os.listdir(pth)

    for fn in file_list:
        old_path = os.path.join(pth,fn)

        number = re.search(r'_([0-9]+)_',fn)

        if number:
            number = number.group(1)

            new_name = f'{int(number):03d}'
            new_path = os.path.join(pth,new_name)

            os.rename(old_path,new_path)

def add_pth(pth,extension='.pth'):
    file_list = os.listdir(pth)

    for fn in file_list:
        old_pth = os.path.join(pth,fn)

        new_file_name = f'{fn}{extension}'
        new_pth = os.path.join(pth,new_file_name)
        os.rename(old_pth,new_pth)



# add_pth('/home/yiping/Downloads/wire/results/fox/sin3pi/models')
#rename('/home/yiping/Downloads/wire/results/fox/sin2pi')
# main('/home/yiping/Downloads/wire/results/fox/sin3pi/models/001')

if __name__ =='__main__':
    for j in ['sin3pi']:
        for i in ['198.pth']:
            pth = '/home/yiping/Downloads/wire/results/fox/'+str(j)+'/'+i
                        # psnr = []
            sranks = []
            s1,s2,s3,s4=main(pth)


            torch.save(s1,pth+'_s1.pth')
            torch.save(s2,pth+'_s2.pth')
            torch.save(s3,pth+'_s3.pth')
            torch.save(s4,pth+'_s4.pth')
            # psnr.append(psnr_)
    # torch.save(psnr,'/home/yiping/Downloads/wire/results/fox/sin3pi/psnr.pth')