
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pickle as pkl

import sys
import pandas as pd
import numpy as np
import time
import datetime
import copy
import os

sys.path.append('../')
from utils.util import *
from utils.prePlot import *

with open('../pickle/train_test_df.pkl', 'rb')  as f:
    train_test = pkl.load(f)
df_train = train_test['train_dfs']
df_test = train_test['test_dfs']


BatchSize = 1
SeqLength = 5
DropOut = 0.25
LSTM_hiddenDim = 96
# Linear_hiddenDim = 128
num_epochs  = 50
feature_dim = 10
output_dim = 2
model_savePath = '../models/window_lstm_params_AllSeq_lr1e-1-1e-3_epoch20_batch1_SeqLen5_DP0.25_LSTMHSize96_BestValLoss0.3652880839153577_Time2018-11-20 17:25:58.265553.pkl'


# In[5]:


train_dataloader = []
test_dataloader = []
train_len = test_len = 0
for df in df_train:
    train_dataloader.append(DataLoader(oxts_dataset(df, seqLen=SeqLength), batch_size=BatchSize))
    train_len += len(df)
for df in df_test:
    test_dataloader.append(DataLoader(oxts_dataset(df, seqLen=SeqLength), batch_size=BatchSize))
    test_len += len(df)

class LSTM_REG(nn.Module):
    def __init__(self, feature_dim, hidden_dim, reg_dim, device ,dropout=0,
                    num_layers=2, bidirectional=True, batch_size=1):
        super(LSTM_REG, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_directions = 2 if bidirectional == True else 1
        self.lstm = nn.LSTM(feature_dim, hidden_dim, dropout=dropout,
                            num_layers=num_layers, bidirectional=bidirectional).double()

        self.hidden2reg = nn.Linear(hidden_dim  * self.num_directions, reg_dim).double()
        self.hidden = self.init_hidden()
        self.dropout = nn.Dropout(p=dropout)


    def init_hidden(self):
        return (torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_dim, device=device).double(),
                torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_dim, device=device).double())

    def forward(self, inputs):
        #inputs = torch.from_numpy(inputs).float()
        lstm_out, self.hidden = self.lstm(inputs, self.hidden)
        lstm_out = self.dropout(lstm_out)
        reg = self.hidden2reg(lstm_out)
        return reg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = LSTM_REG(feature_dim, LSTM_hiddenDim, 2, device=device,
                 bidirectional=True, dropout=DropOut)
model.to(device)
model.load_state_dict(torch.load(model_savePath))

dataloaders = {'train': train_dataloader, 'val': test_dataloader}
dataset_sizes = {'train': train_len, 'val': test_len}

since = time.time()
phase = 'val'
y_pred_lat_his = []
y_pred_lon_his = []
y_lat_his = []
y_lon_his = []

with torch.no_grad():
    model.eval()
    running_loss = 0.0
    for dataloader in dataloaders[phase]:

        y_pred_lat_his.append(np.zeros(len(dataloader)))
        y_pred_lon_his.append(np.zeros(len(dataloader)))
        y_lat_his.append(np.zeros(len(dataloader)))
        y_lon_his.append(np.zeros(len(dataloader)))
        pos = 0
        for i, (X, y) in enumerate(dataloader):
            model.hidden = model.init_hidden()
            X_tensor = X.view(-1, BatchSize, feature_dim).to(device)
            y_tensor = y.view(-1, BatchSize, output_dim).to(device)[-1]
            y_pred = model(X_tensor)[-1]
            loss = loss_function(y_pred, y_tensor)
            running_loss += loss.item() * X_tensor.size(0)

            y_pred_lat_his[-1][pos] = y_pred[0, 0].cpu().numpy()
            y_pred_lon_his[-1][pos] = y_pred[0, 1].cpu().numpy()
            y_lat_his[-1][pos] = y_tensor[0, 0].cpu().numpy()
            y_lon_his[-1][pos] = y_tensor[0, 1].cpu().numpy()
            pos += 1

    epoch_loss = running_loss / dataset_sizes[phase]
    print('{} Loss: {:.4f}'.format(phase, epoch_loss))
