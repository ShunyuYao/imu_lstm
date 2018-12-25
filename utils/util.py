import pandas as pd
import numpy as np
from math import cos, asin, sqrt
from torch.utils.data import Dataset, DataLoader
from scipy.integrate import quad, simps

def createTimeDelta(ts_se):
    ts_arr = np.zeros(len(ts_se))
    start_time = ts_se[0]
    ts_se = ts_se.drop(0, axis=0)
    for i in range(len(ts_se)):
        td = ts_se.iloc[i] - start_time
        ts_arr[i+1] = td.total_seconds()
        # print(td.total_seconds())
    return ts_arr

def distFromV(time, ve, vn, end=-1, beg=0):
    if end == -1:
        ve = ve[beg:]
        vn = vn[beg:]
        time = time - time[beg]
        time = time[beg:]
    else:
        end += 1
        ve = ve[beg:end]
        vn = vn[beg:end]
        time = time - time[beg]
        time = time[beg:end]

    # print(ve, vn, time)
    dist_ve = simps(ve, time)
    dist_vn = simps(vn ,time)
    dist = np.sqrt(dist_ve ** 2 + dist_vn ** 2)
    return dist

def distanceFromCood(df, beg=0, end=-1):
    lat1 = df.lat.iloc[beg]
    lat2 = df.lat.iloc[end]
    lon1 = df.lon.iloc[beg]
    lon2 = df.lon.iloc[end]
    p = 0.017453292519943295     #Pi/180
    a = 0.5 - cos((lat2 - lat1) * p)/2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    return 12742 * asin(sqrt(a)) * 1e3 #2*R*asin...

def processDf(df, split_test=True, train_rate=0.8):
    # df['timestamp'] = pd.to_datetime(df['timestamp'])
    # if df_abs == True:
    #     df = df.drop(['alt', 'vf', 'vl', 'vu', 'ax', 'ay', 'az',
    #                       'wx', 'wy', 'wz', 'pos_accuracy', 'vel_accuracy',
    #                       'navstat', 'numsats', 'posmode', 'velmode', 'orimode'], axis=1)
    # else:
    #     df = df.drop(['alt', 'vn', 've', 'af', 'al', 'au',
    #                       'wf', 'wl', 'wu', 'pos_accuracy', 'vel_accuracy',
    #                       'navstat', 'numsats', 'posmode', 'velmode', 'orimode'], axis=1)
    # time_arr = createTimeDelta(df['timestamp'])
    # df['timeDelta'] = time_arr
    # df = df.set_index('timestamp')
    y = df.loc[:, ['lat', 'lon']].values
    X = df.drop(['lat', 'lon'], axis=1).values

    if split_test == True:
        train_size = int(train_rate * len(X))
        X_train, X_test, y_train, y_test = X[:train_size], X[(train_size-SeqLength):] \
                                            ,y[:train_size], y[(train_size-SeqLength):]
        return X_train, X_test, y_train, y_test
    else:
        return X, y

def latLonDist(lat1, lat2, lon1, lon2):
    p = 0.017453292519943295     #Pi/180
    a = 0.5 - cos((lat2 - lat1) * p)/2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    dist = 12742 * asin(sqrt(a)) #2*R*asin...
    lat_delt = lat2 - lat1
    lon_delt = lon2 - lon1
    lat_lon = np.sqrt(lat_delt ** 2 + lon_delt**2)
    lat_dist = dist * lat_delt / (lat_lon + 1e-8)
    lon_dist = dist * lon_delt / (lat_lon + 1e-8)
    lat_lon_arr = np.zeros((1, 2))
    lat_lon_arr[0, 0] = lat_dist * 1e3
    lat_lon_arr[0, 1] = lon_dist * 1e3

    return lat_lon_arr

class oxts_dataset(Dataset):

    def __init__(self, df, seqLen):
        self.X, self.y = processDf(df, split_test=False)
        self.seqLen = seqLen
        self.X_init = self.X[: self.seqLen].copy()
        self.y_init = self.y[: self.seqLen].copy()
        start_lat = self.y_init[0, 0]
        start_lon = self.y_init[0, 1]
        for i in range(0, self.y_init.shape[0]):
            lat_lon_dist = latLonDist(start_lat, self.y_init[i, 0], start_lon, self.y_init[i, 1])
            self.y_init[i, 0] = lat_lon_dist[0, 0]
            self.y_init[i, 1] = lat_lon_dist[0, 1]

    def __len__(self):
        return len(self.X) - self.seqLen
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
        if self.seqLen > 1:
            y_sample[0, :] = np.zeros((1, 2))

        sample = (X_sample, y_sample)
        return sample
