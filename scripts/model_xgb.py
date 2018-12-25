import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import pickle as pkl

sys.path.append('../')
from utils.util import *

xgb_param = {
    'learning_rate': 0.001,
    'n_estimators': 50,  #5000
    'max_depth': 3,
    'min_child_weight': 1,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.9,
    'reg_alpha': 0,
    'reg_lambda': 0,
    'seed': 666,
    'silent': 1,
    'eta': 1,
    'eval_metric': 'rmse'
}

with open('../pickle/train_test_df_KT.pkl', 'rb')  as f:
    train_test = pkl.load(f)
dfs_train = train_test['train_dfs']
dfs_test = train_test['test_dfs']

df_train = pd.DataFrame([])
df_test = pd.DataFrame([])

for df in dfs_train:
    df_lat_shift = df['lat'].shift(-1).dropna()
    df_lat = df['lat'][:-1].copy()
    df_lon_shift = df['lon'].shift(-1).dropna()
    df_lon = df['lon'][:-1].copy()
    df = df[:-1].copy()

    lat_lon_dist = list(map(latLonDist,
                       df_lat,
                       df_lat_shift,
                       df_lon,
                       df_lon_shift))
    lat = np.concatenate(lat_lon_dist)[:, 0]
    lon = np.concatenate(lat_lon_dist)[:, 1]
    df['lat'] = lat
    df['lon'] = lon
    df_train = pd.concat([df_train, df])

for df in dfs_test:
    df_lat_shift = df['lat'].shift(-1).dropna()
    df_lat = df['lat'][:-1].copy()
    df_lon_shift = df['lon'].shift(-1).dropna()
    df_lon = df['lon'][:-1].copy()
    df = df[:-1].copy()

    lat_lon_dist = list(map(latLonDist,
                       df_lat,
                       df_lat_shift,
                       df_lon,
                       df_lon_shift))

    lat = np.concatenate(lat_lon_dist)[:, 0]
    lon = np.concatenate(lat_lon_dist)[:, 1]
    df['lat'] = lat
    df['lon'] = lon
    df_test = pd.concat([df_test, df])

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
y_train = df_train.loc[:, ['lat', 'lon']].values
X_train = df_train.drop(['lat', 'lon'], axis=1).values
y_test = df_test.loc[:, ['lat', 'lon']].values
X_test = df_test.drop(['lat', 'lon'], axis=1).values

train_dataloader = [X_train, y_train]
test_dataloader = [X_test, y_test]

class XGBHelper(object):
    def __init__(self, num_boost_round=200, params=None):
        self.params = params
        self.num_boost_round = num_boost_round

    def train(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, y_train)
        self.model = xgb.train(self.params, dtrain, verbose_eval=10, num_boost_round=self.num_boost_round)

    def predict(self, x):
        dtest = xgb.DMatrix(x)
        return self.model.predict(dtest)

def calc_loss(true_df,pred_df):
    loss = np.sum((true_df - pred_df) ** 2) / len(true_df)
    return loss

num_boost_round = 5000
dataloader = {'train': train_dataloader, 'val': test_dataloader}
datasize = {'train': len(y_train), 'val': len(y_test)}
for target in ['lat', 'lon']:
    # print(target)
    model = XGBHelper(num_boost_round=num_boost_round, params=xgb_param)
    for phase in ['train', 'val']:
        running_loss = 0.0
        dl = dataloader[phase]
        X = dl[0]
        y = dl[1]
        if target == 'lat':
            y = y[:, 0]
        elif target == 'lon':
            y = y[:, 1]
            # print(y)
        if phase == 'train':
            model.train(X, y)

        y_pred = model.predict(X)
        total_loss = calc_loss(y, y_pred)

        print("{} {} loss is: {:.3f}".format(phase, target, total_loss))

from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
gbr = GradientBoostingRegressor()
for target in ['lat', 'lon']:
    # print(target)
    model = GradientBoostingRegressor()
    for phase in ['train', 'val']:
        running_loss = 0.0
        dl = dataloader[phase]
        X = dl[0]
        y = dl[1]
        if target == 'lat':
            y = y[:, 0]
        elif target == 'lon':
            y = y[:, 1]
            # print(y)
        if phase == 'train':
            model.fit(X, y)

        y_pred = model.predict(X)
        total_loss = calc_loss(y, y_pred)

        print("{} {} loss is: {:.5f}".format(phase, target, total_loss))

for target in ['lat', 'lon']:
    # print(target)
    model = BaggingRegressor(n_estimators=1, max_features=0.2)
    for phase in ['train', 'val']:
        running_loss = 0.0
        dl = dataloader[phase]
        X = dl[0]
        y = dl[1]
        if target == 'lat':
            y = y[:, 0]
        elif target == 'lon':
            y = y[:, 1]
            # print(y)
        if phase == 'train':
            model.fit(X, y)

        y_pred = model.predict(X)
        total_loss = calc_loss(y, y_pred)

        print("{} {} loss is: {:.5f}".format(phase, target, total_loss))
