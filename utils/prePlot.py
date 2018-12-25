import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from .util import *

# def absDist(test_df, i, seqLen=2, bs=1):
#     test = DataLoader(oxts_dataset(test_df[i], seqLen), batch_size=bs)
#     y_lat = [0]
#     y_lon = [0]
#     for i, (X, y) in enumerate(test):
#         y_lat_i = y[:,-1,0] # + y_lat[-1]
#         y_lon_i = y[:,-1,1] # + y_lon[-1]
#         y_lat.append(y_lat_i.numpy())
#         y_lon.append(y_lon_i.numpy())
#
#     return (y_lat, y_lon)

def absDist(test_df, i, seqLen=2, bs=1):
    test = DataLoader(oxts_dataset(test_df[i], seqLen), batch_size=bs)
    y_lat = np.zeros(len(test.dataset) + seqLen)
    y_lon = np.zeros(len(test.dataset) + seqLen)
    y_lat[:seqLen] = test.dataset.y_init[:, 0]
    y_lon[:seqLen] = test.dataset.y_init[:, 1]
    pos = seqLen
    for i, (X, y) in enumerate(test):
        y_lat_i = y[:,-1,0] # + y_lat[-1]
        y_lon_i = y[:,-1,1] # + y_lon[-1]
        y_lat[pos] = y_lat_i.numpy()
        y_lon[pos] = y_lon_i.numpy()
        pos += 1

    return (y_lat, y_lon)
