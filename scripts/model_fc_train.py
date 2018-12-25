# coding: utf-8

# ## LSTM的window版本
# 每一个输出都是前面SeqLen个输入进入lstm循环的结果

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle as pkl
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
import sys

import pandas as pd
import numpy as np
import time
import datetime
import copy
import os

sys.path.append('../')
from utils.util import *
from utils.model_collections import *

# In[3]:


import datetime
dt = datetime.datetime.now()

dataset_name = 'KT'
ct_train = False
BatchSize = 1
SeqLength = 5
LR = 5e-4
tflog_interval = 5
LSTM_hiddenDim = 256
# Linear_hiddenDim = 128
num_epochs  = 200
feature_dim = 10 * SeqLength
model_load_str = '../models/ctTrainLstm_yg_params_lr0.005_epoch200_batch1_SeqLen5_DP0.25_LSTMHSize96_Time2018-12-13_10:34:58/lstm_yg_epoch199_BestValLoss0.030.pkl'
model_save_dirname = 'fc_{}_params_lr{}_epoch{}_batch{}_SeqLen{}_Time{}'.format(dataset_name, LR, num_epochs, BatchSize, SeqLength, dt.strftime('%Y-%m-%d_%H:%M:%S'))
model_save_path = '../models/{}'.format(model_save_dirname)
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

output_dim = 2

with open('../pickle/train_test_df_{}.pkl'.format(dataset_name), 'rb')  as f:
    train_test = pkl.load(f)
df_train = train_test['train_dfs']
df_test = train_test['test_dfs']

train_dataloader = []
test_dataloader = []
train_len = test_len = 0
for df in df_train:
    train_dataloader.append(DataLoader(oxts_dataset(df, seqLen=SeqLength), batch_size=BatchSize))
    train_len += len(df)
for df in df_test:
    test_dataloader.append(DataLoader(oxts_dataset(df, seqLen=SeqLength), batch_size=BatchSize))
    test_len += len(df)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = FC_REG(feature_dim, LSTM_hiddenDim, 2)
model = model.double()
model.to(device)
print(model)
if ct_train:
    model.load_state_dict(torch.load(model_load_str))

loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=10, factor=0.5)
dataloaders = {'train': train_dataloader, 'val': test_dataloader}
dataset_sizes = {'train': train_len, 'val': test_len}

since = time.time()
writer = SummaryWriter(log_dir='../tflog/' + model_save_dirname)
best_model_wts = copy.deepcopy(model.state_dict)
# model.load_state_dict(best_model_wts)
best_loss = 1e8

for epoch in range(num_epochs):
    print("Epoch {}/{}".format(epoch+1, num_epochs))
    print('-' * 10)

    for phase in ['train', 'val']:
        num_iters = 0
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        for dataloader in dataloaders[phase]:
            for i, (X, y) in enumerate(dataloader):
                num_iters += 1
                X_tensor = X.view(BatchSize, -1).to(device)
                y_tensor = y.view(SeqLength, 2).to(device)[-1].unsqueeze(0)
                model.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    y_pred = model(X_tensor)
                    loss = loss_function(y_pred, y_tensor)
                    if (epoch+1) % tflog_interval == 0:
                        writer.add_scalars('{}/Epoch{}_Lat'.format(phase, epoch+1), {
                            'pred':y_pred.view(-1)[0],
                            'truth':y_tensor[0, 0]
                        }, num_iters)
                        writer.add_scalars('{}/Epoch{}_Lon'.format(phase, epoch+1), {
                            'pred':y_pred.view(-1)[1],
                            'truth':y_tensor[0, 1]
                        }, num_iters)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * X_tensor.size(0)

        epoch_loss = running_loss / dataset_sizes[phase]
        writer.add_scalar('loss/Phase_{}_EpochLoss'.format(phase), epoch_loss, epoch)

        print('{} Loss: {:.4f}'.format(phase, epoch_loss))
        # deep copy the model
        if phase == 'val' and epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            model_savePath = os.path.join(model_save_path, 'fc_{}_epoch{}_BestValLoss{:.3f}.pkl' \
                                .format(dataset_name, epoch, best_loss))
            torch.save(best_model_wts, model_savePath)


print()
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
print('Best val loss: {:4f}'.format(best_loss))
model_savePath = os.path.join(model_save_path, 'fc_{}_epoch{}_BestValLoss{:.3f}.pkl' \
                                .format(dataset_name, epoch, best_loss))
torch.save(model.state_dict(), model_savePath)
