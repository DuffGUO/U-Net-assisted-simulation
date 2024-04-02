import torch
from Unet import UNet
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from Dataset_try import SEGData
from excel_to_matrix import *
from PIL import Image
import numpy as np
import pandas as pd
from torchvision import utils
import time

# device = torch.device("cuda")
device = torch.device("cpu")
net = UNet()
net = net.to(device)

optimizer = torch.optim.Adam(net.parameters())
loss_func = nn.MSELoss()
loss_func = loss_func.to(device)

# 加载训练集
input_dir_train = 'inputfigure96pixel/train_400'
output_dir_train = 'outputmatrix96pixel/train_400'
data_train = SEGData(input_dir_train, output_dir_train)
# 加载验证集
input_dir_valid = 'inputfigure96pixel/valid_400'
output_dir_valid = 'outputmatrix96pixel/valid_400'
data_valid = SEGData(input_dir_valid, output_dir_valid)


dataloader_train = DataLoader(data_train, batch_size=5, shuffle=True, num_workers=0, drop_last=True)
dataloader_valid = DataLoader(data_valid, batch_size=5, shuffle=True, num_workers=0, drop_last=True)
# summary = SummaryWriter(r'Log')
EPOCH = 120
loss_recording = np.zeros((EPOCH,2))
# start_time = time.time()
print('load success')
for epoch in range(EPOCH):
    print('开始第{}轮'.format(epoch))
    net.train()
    loss_train_sum = 0
    for data in dataloader_train:
        imgs, labels = data
        imgs = imgs.to(device)
        labels = labels.to(device)

        img_out = net(imgs)
        loss_train = loss_func(img_out, labels)
        loss_train_sum = loss_train_sum + loss_train
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
    print("训练集的loss: {}", format(loss_train))
    print("训练集的平均loss: {}", format(loss_train_sum/64))
    loss_recording[epoch, 0] = loss_train_sum/64
    torch.save(net.state_dict(), './trained parameters_cpu/{}.pth'.format(epoch))

    net.eval()
    with torch.no_grad():
        loss_valid_sum = 0
        for data in dataloader_valid:
            imgs, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)

            img_out = net(imgs)
            loss_valid = loss_func(img_out, labels)
            loss_valid_sum = loss_valid + loss_valid_sum
        print("验证集的loss: {}", format(loss_valid))
        print("验证集的平均loss: {}", format(loss_valid_sum/8))
        loss_recording[epoch, 1] = loss_valid_sum/8

# 记录loss的迭代于excel中
# arr = pd.DataFrame(loss_recording)
# arr.to_excel('A_with 400_2.xlsx', float_format='%.24f')

