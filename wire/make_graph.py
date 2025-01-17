import matplotlib.pyplot as plt
import torch
import numpy as np

idx =[]
for i in range(1,30,3):
    idx.append(i)
for i in range(30,70,5):
    idx.append(i)



# 264707

####################################################
# psnr vs k
data = torch.load('results/fox/rank_1/psnr.pth')
data=[20.15,20.09,20.44,20.77,20.60,20.92,20.94,21.10,21.09,21.18,21.33,21.31,21.41,21.37,21.32,21.48,21.43,21.39,21.34,21.47,21.51,21.34,21.43,21.40,21.33,21.37,21.36]
data=[21.31,21.76,22.08,22.11,22.25,22.33,22.22,22.01,22.09,21.99,21.98,21.92,21.90,21.87,21.79,21.80,21.76,21.73,21.86,21.82,21.76,21.71,21.76,21.67,21.65,21.60,21.52]

data=[23.14,23.18,23.35,23.35,23.23,23.32,23.32,23.29,23.22,23.28,23.25,23.18,23.14,23.19,23.11,22.94]
data=[
24.23,24.35,24.45,24.52,24.7,24.74,24.7,24.73,24.65,24.57,24.59,24.56,24.5,24.39,24.39,24.36,24.42,24.3]
data=[24.67,24.73,24.98,24.78,25.16,25.19,25.25,25.19,24.87,25.10,24.75,24.75,24.90,24.88,24.90,24.85,24.71,24.62]
# # data1 = torch.load('results/fox/sin1/psnr.pth')
# data2 = torch.load('results/fox/sin2/psnr.pth')[:60]
# # data3 = torch.load('results/fox/sin3/psnr.pth')
# data4 = torch.load('results/fox/sin4/psnr.pth')[:60]


# # data4pi = torch.load('results/fox/sin3pi/psnr.pth')
# num = torch.load('results/fox/naive/nums.pth')
# print(num[40])
print(data)

# #print(data_srank[9])
# plt.figure()
# idx =idx[:60]
plt.axhline(y=25.04, color='red', label='AB')
plt.plot(idx,data, color='blue', label='a*sin(w*AB)')
# # plt.plot(idx,data1, color='red', label='low_rank_sin1')
# plt.plot(idx,data2, color='yellow', label='low_rank_sin2')
# # plt.plot(idx,data3, color='black', label='low_rank_sin3')
# plt.plot(idx,data4, color='blue', label='low_rank_sin4')
# plt.axvline(x=20,color='red',linestyle = '-',linewidth=2,label='~16.4% numbers of parameters')
# plt.axvline(x=30,color='red',linestyle = '--',linewidth=2,label='~24.2% numbers of parameters')
# plt.axvline(x=40,color='red',linestyle = '-.',linewidth=2,label='~49.7% numbers of parameters')
# plt.axvline(x=50,color='red',linestyle = ':',linewidth=2,label='~72.9% numbers of parameters')



# # plt.plot(idx,data4pi, color='purple', label='low_rank_sin3pi')




plt.title('PSNR vs w, when k = 50')
plt.xlabel('w')
plt.ylabel('PSNR')
plt.xlim(1,65)
plt.ylim(22,27)
plt.grid(True)
plt.legend()

plt.savefig('results/fox/psnr_vs_omega_k50')
# plt.show()


####################################################
# # srank vs k
# data_srank = torch.load('results/fox/naive/srank.pth')[:60]
# data_srank2 = torch.load('results/fox/sin2/srank.pth')[:60]
# data_srank3 = torch.load('results/fox/sin4/srank.pth')[:60]
# # data_srank4 = torch.load('results/fox/sin3pi/srank.pth')
# # # # # data_srank5 = torch.load('fox/sin4pi/srank.pth')
# # # # # for i in data_srank_:
# # # # #     data_srank.append(i.detach().cpu().numpy())
# # plt.figure()
# idx=idx[:60]
# plt.axhline(y=66,color = 'green', label = 'full_rank')
# plt.plot(idx,data_srank,color = 'blue',label='low_rank AB')
# plt.plot(idx,data_srank2,color = 'red',label='sin(2*AB)')
# plt.plot(idx,data_srank3,color = 'yellow',label='sin(4*AB)')
# # # print(data_srank1)
# # plt.plot(idx,data_srank4,color = 'gray',label='low_rank_sin2pi')
# # # # # plt.plot(idx,data_srank5,color = 'blue',label='low_rank_sin5pi')

# plt.title('SRANK vs k')
# plt.xlabel('k')
# plt.ylabel('SRANK')
# plt.grid(True)
# plt.legend()
# plt.savefig('results/fox/srank60')



# # ####################################################
# s1 = torch.load('/home/yiping/Downloads/wire/results/fox/naive/005.pth_s1.pth')
# s2 = torch.load('/home/yiping/Downloads/wire/results/fox/naive/005.pth_s2.pth')
# s3 = torch.load('/home/yiping/Downloads/wire/results/fox/naive/005.pth_s3.pth')
# s4 = torch.load('/home/yiping/Downloads/wire/results/fox/naive/005.pth_s4.pth')

# plt.title('SRANK vs k in the 1st layer')
# plt.plot(s1/s1.max())
# plt.xlabel('index')
# plt.ylabel('SRANK')
# plt.grid(True)
# plt.legend()
# plt.savefig('results/fox/full/srank1')

# plt.title('SRANK vs k in the 2nd layer')
# plt.plot(s2/s2.max())
# plt.xlabel('index')
# plt.ylabel('SRANK')
# plt.grid(True)
# plt.legend()
# plt.savefig('results/fox/full/srank2')

# plt.title('SRANK vs k in the 3rd layer')
# plt.plot(s3/s3.max())
# plt.xlabel('index')
# plt.ylabel('SRANK')
# plt.grid(True)
# plt.legend()
# plt.savefig('results/fox/full/srank3')

# plt.title('SRANK vs k in the 4th layer')
# plt.plot(s4/s4.max())
# plt.xlabel('index')
# plt.ylabel('SRANK')
# plt.grid(True)
# plt.legend()
# plt.savefig('results/fox/full/srank4')

# plt.figure()
# plt.title('AB k=5')
# plt.plot(s1/s1.max(),color='blue',label = '1st layer')
# plt.plot(s2/s2.max(),color='red',label = '2nd layer')
# plt.plot(s3/s3.max(),color='black',label = '3rd layer')
# plt.plot(s4/s4.max(),color='yellow',label = '4th layer')
# plt.xlabel('index')
# plt.ylabel('normalized sigular value')
# plt.ylim(0,1)

# plt.grid(True)
# plt.legend()
# plt.savefig('results/fox/naive/svd_005')
# srank = torch.load('/home/yiping/Downloads/wire/results/fox/naive/srank.pth')[60]
# print(srank)
# sig = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
# k10 = [20.61,21.07,20.97,21.61,21.90,22.18,22.40,22.57,22.23,21.04]
# k20 =[19.34,21.30,21.73,22.26,22.77,23.38,23.23,22.05,20.33,20.10]
# k50 = [20.66,21.68,22.55,23.41,24.14,23.31,22.23,21.56,20.58,19.33]
# k110 = [20.99,22.25,23.40,24.56,23.94,22.99,22.30,20.73,17.74,15.05]
# k200 =[21.51,22.94,24.07,24.18,23.65,22.27,19.80,14.58,14.48,12.34]

# plt.figure()
# plt.title('PSNR vs sigma')
# plt.plot(sig,k10,color='blue',label = 'k=10')
# plt.plot(sig,k20,color='red',label = 'k=20')
# plt.plot(sig,k50,color='black',label = 'k=50')
# plt.plot(sig,k110,color='yellow',label = 'k=110')
# plt.plot(sig,k200,color='grey',label = 'k=200')
# plt.xlabel('sigma')
# plt.ylabel('PSNR')
# plt.grid(True)
# plt.legend()
# plt.savefig('results/fox/srank_vs_sigma')
