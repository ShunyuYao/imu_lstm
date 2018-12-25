
# coding: utf-8

# ## LSTM的window版本
# 每一个输出都是前面SeqLen个输入进入lstm循环的结果

# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
import sys

import pandas as pd
import numpy as np
import time
import datetime
import copy
import os

sys.path.append('../')
from utils import *

# In[3]:


import datetime
dt = datetime.datetime.now()
# In[11]:


BatchSize = 1
SeqLength = 5
LR = 1e-3
DropOut = 0.25
LSTM_hiddenDim = 96
# Linear_hiddenDim = 128
num_epochs  = 50
feature_dim = 11
model_save_str = 'window_lstm_params_lr{}_epoch{}_batch{}_SeqLen{}_DP_{}_LSTMHSize{}_Time_{}'                         .format(LR, num_epochs, BatchSize, SeqLength,                                DropOut, LSTM_hiddenDim, str(datetime.datetime.now().ctime()))
output_dim = 2
block_size = 20 + 2 * SeqLength

print(model_save_str)


# In[4]:


df_select = []
df_len = []
df_name_list = []
date_list = os.listdir('../processed_data/')
for date in date_list:
    df_path = os.path.join('../processed_data/', date)
    df_list = os.listdir(df_path)
    for df_name in df_list:
        df = pd.read_csv(os.path.join(df_path, df_name))
        # print(len(df))
        if len(df) > block_size:
            df_select.append(df)
            df_len.append(len(df))
            df_name_list.append(df_name)
        else:
            print(df_name, len(df))
    # 20 + 2 * SeqLength
print(len(df_select))


# In[5]:


total_length = 0
for length in df_len:
    total_length += length
print(total_length)


# In[6]:


def sample_train_test(df_select, total_length, test_rate):
    select_len = 0
    choice_list = []
    while select_len < int(total_length * test_rate):
        choice = np.random.choice(len(df_select), 1, replace=False)
        if choice not in choice_list:
            choice_list.append(choice[0])
        select_len += len(df_select[choice[0]])
        print('the length of df is:', len(df_select[choice[0]]))
    print('total length:', select_len)
    return choice_list, select_len

test_idxs, test_len = sample_train_test(df_select, total_length, test_rate=0.15)
# df_select[test_idxs]


# In[7]:


len(df_select)


# In[8]:


df_train = []
df_test = []
for i in test_idxs:
    df_test.append(df_select[i])

for i in range(len(df_select)):
    if i not in test_idxs:
        df_train.append(df_select[i])
del df_select


# In[9]:


import pickle as pkl
pkl_save = {'train_dfs': df_train, 'test_dfs': df_test, 'df_name_list': df_name_list,
           'test_idxs': test_idxs}
with open('./pickle/train_test_df.pkl', 'wb') as f:
    pkl.dump(pkl_save, f)


# In[10]:


class oxts_dataset(Dataset):

    def __init__(self, df, seqLen):
        self.X, self.y = processDf(df, split_test=False)
        self.seqLen = seqLen

    def __len__(self):
        return len(self.X)  - self.seqLen
    def __getitem__(self, idx):

        idx += self.seqLen
        X_sample = self.X[idx-self.seqLen : idx].copy()
        y_sample = self.y[idx-self.seqLen : idx].copy()
        # print(y_sample)
        start_lat = y_sample[0, 0]
        start_lon = y_sample[0, 1]
        for i in range(1, y_sample.shape[0]):
            lat_lon_dist = latLonDist(start_lat, y_sample[i, 0], start_lon, y_sample[i, 1])
            y_sample[i, 0] = lat_lon_dist[0, 0]
            y_sample[i, 1] = lat_lon_dist[0, 1]
        y_sample[0, :] = np.zeros((1, 2))

        sample = (X_sample, y_sample)
        return sample


# In[11]:


train_dataloader = []
test_dataloader = []
for df in df_train:
    train_dataloader.append(DataLoader(oxts_dataset(df, seqLen=SeqLength), batch_size=BatchSize))
for df in df_test:
    test_dataloader.append(DataLoader(oxts_dataset(df, seqLen=SeqLength), batch_size=BatchSize))


# In[12]:


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


# In[13]:


torch.backends.cudnn.enabled=False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[14]:


# device = torch.device("cpu")
model = LSTM_REG(feature_dim, LSTM_hiddenDim, 2, device=device,
                 bidirectional=True, dropout=DropOut)
model.to(device)
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=LR)

train_len = total_length - test_len
dataloaders = {'train': train_dataloader, 'val': test_dataloader}
dataset_sizes = {'train': train_len, 'val': test_len}


# In[15]:


model


# In[16]:


# optimizer = optim.SGD(model.parameters(), lr=LR)
# num_epochs = 60


# In[17]:


since = time.time()
writer = SummaryWriter(log_dir='runs/' + model_save_str)
best_model_wrts = copy.deepcopy(model.state_dict)
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
                X_tensor = X.view(-1, BatchSize, feature_dim).to(device)
                y_tensor = y.view(-1, BatchSize, output_dim).to(device)[-1]

                model.zero_grad()
                model.hidden = model.init_hidden()

                with torch.set_grad_enabled(phase == 'train'):

                    y_pred = model(X_tensor)[-1]
                    loss = loss_function(y_pred, y_tensor)
                    writer.add_scalars('pred/Epoch{}_Phase_{}_Lat'.format(epoch+1, phase), {
                        'pred':y_pred.view(-1)[0],
                        'truth':y_tensor[0, 0]
                    }, i)
                    writer.add_scalars('pred/Epoch{}_Phase_{}_Lon'.format(epoch+1, phase), {
                        'pred':y_pred.view(-1)[1],
                        'truth':y_tensor[0, 1]
                    }, i)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * X_tensor.size(0)
                writer.add_scalar('loss/Phase_{}_RunningLoss'.format(phase)
                                  , running_loss, i)

        epoch_loss = running_loss / dataset_sizes[phase]
        writer.add_scalar('loss/Phase_{}_EpochLoss'.format(phase)
                                  ,epoch_loss, epoch)

        print('{} Loss: {:.4f}'.format(phase, epoch_loss))
        # deep copy the model
        if phase == 'val' and epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

print()
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
print('Best val loss: {:4f}'.format(best_loss))


# Epoch = 40的时候，虽然val_loss = 0.004785但是发现出现了非常严重的过拟合　　
#

# In[18]:


torch.save(model.state_dict(),
           'window_lstm_params_AllSeq_lr{}_epoch{}_batch{}_SeqLen{}_DP{}_LSTMHSize{}_BestValLoss{}_Time{}.pkl'
           .format('1e-1-1e-3', '20', BatchSize, SeqLength,  \
                   DropOut, LSTM_hiddenDim, best_loss, str(datetime.datetime.now())))


# In[22]:


import pickle as pkl

with open('./pickle/'+'dfListTrain_'+model_save_str+'.pkl', 'wb') as f:
    pkl.dump(df_train, f)
with open('./pickle/'+'dfListTest_'+model_save_str+'.pkl', 'wb') as f:
    pkl.dump(df_test, f)
