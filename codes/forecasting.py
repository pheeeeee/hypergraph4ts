
import matplotlib.pyplot as plt
from multiprocessing import Pool
from counting import *
from application import *
from utils import *
from metric import *

import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error


import time




from metric import *
from dataloader import *


"""Data Loading"""
#Import Data
import pandas as pd
import numpy as np
from copy import copy, deepcopy
import json

#caiso = pd.read_csv('/Users/piprober/Desktop/projects/g4tsempirical/20160322-20240323 CAISO Actual Load.csv')
traffic = pd.read_excel('/Users/piprober/Desktop/projects/g4tsempirical/trafficpems_output.xlsx')
electricity = pd.read_csv('/Users/piprober/Desktop/projects/g4tsempirical/electricityLD2011_2014.txt', delimiter=';')
weather1 = pd.read_csv('/Users/piprober/Desktop/projects/g4tsempirical/mpi_roof_2020a.csv', encoding='ISO-8859-1')
weather2 = pd.read_csv('/Users/piprober/Desktop/projects/g4tsempirical/mpi_roof_2020b.csv', encoding='ISO-8859-1')
etth1 = pd.read_csv('/Users/piprober/Desktop/projects/g4tsempirical/ETTh1.csv')
ettm1 = pd.read_csv('/Users/piprober/Desktop/projects/g4tsempirical/ETTm1.csv')
wind = pd.read_csv('/Users/piprober/Desktop/projects/g4tsempirical/wind.csv')
solar = pd.read_csv('/Users/piprober/Desktop/projects/g4tsempirical/solar_AL.txt.gz')
weather = pd.concat([weather1, weather2], ignore_index=True)


ts = {}
#ts['caiso'] = caiso[caiso['zone']=='CA ISO']['load'].dropna()
ts['traffic'] = traffic['% Observed'].dropna()
ts['electricity'] = electricity['MT_370'][70176:]
ts['electricity'] = pd.Series(np.array([float(str(x).split(',')[0]) for x in ts['electricity']])).dropna()
ts['weather'] = weather.iloc[:, -1].dropna()
ts['etth1'] = etth1.iloc[:,-1].dropna()
ts['ettm1'] = ettm1.iloc[:,-1].dropna()
ts['wind'] = wind.iloc[:,-1].dropna()
ts['solar'] = solar.iloc[:,-1].dropna()

#Pre-decided hyperparameters
degrees={}
nodes={}
degrees['caiso'] = [80.0, 120.0, 80.0, 80.0, 80.0, 120.0, 80.0, 80.0, 120.0, 80.0]
nodes['caiso'] = [800.0, 800.0, 500.0, 500.0, 800.0, 800.0, 500.0, 500.0, 800.0, 500.0]
degrees['traffic'] = [5.0, 5.0, 5.0, 25.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
nodes['traffic'] = [2000.0, 5000.0, 3000.0, 3000.0, 800.0, 10000.0, 3000.0, 3000.0, 1500.0, 800.0]
degrees['electricity'] = [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0]
nodes['electricity'] = [1000.0, 500.0, 1000.0, 1000.0, 500.0, 1000.0, 500.0, 500.0, 500.0, 1000.0]
degrees['weather'] = [24.0, 110.0, 24.0, 24.0, 24.0, 110.0, 24.0, 24.0, 24.0, 55.0]
nodes['weather'] = [5000.0, 2000.0, 3000.0, 5000.0, 3000.0, 2000.0, 500.0, 10000.0, 500.0, 800.0]
degrees['etth1'] = [3.0, 5.0, 2.0, 5.0, 2.0, 3.0, 2.0, 2.0, 5.0, 2.0]
nodes['etth1'] = [500.0, 3000.0, 800.0, 1000.0, 1000.0, 500.0, 800.0, 800.0, 3000.0, 10000.0]
degrees['ettm1'] = [4.0, 4.0, 5.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 4.0]
nodes['ettm1'] = [5000.0, 10000.0, 800.0, 5000.0, 1000.0, 500.0, 10000.0, 10000.0, 800.0, 5000.0]
degrees['wind'] = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
nodes['wind'] = [10000.0, 10000.0, 500.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 500.0]
degrees['solar'] = [80.0, 110.0, 80.0, 110.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0]
nodes['solar'] = [5000.0, 10000.0, 5000.0, 10000.0, 1000.0, 500.0, 500.0, 10000.0, 1500.0, 500.0]


metricfunction = mean_absolute_error
lookback_windows = [1440, 720, 336, 192, 168, 96 ]
prediction_window = [168,672,1008,168,192,144,192] #For each time series data of the name list.
names = ['traffic','electricity','weather','etth1','ettm1','solar','wind']


prediction = {}
for i , name in enumerate(names):
    prediction[name] = prediction_window[i]

ts1 = dataloader(ts, lookback_window=lookback_windows, prediction_window=prediction)
ts1.segment()

name = random.choice(names)
l = 1440
train_data = ts1.data[name][l]['train_data']
target_data = ts1.data[name][l]['target']
node = int(random.choice(nodes[name]))
degree = int(random.choice(degrees[name]))


### Using Only One data (single lookback window)
def zero_shot(name, lookback, length ,degree, node, how='uniform', index='random'):
    train_data = ts1.data[name][lookback]['train_data']
    if index == 'random':
        index = random.randint(0, len(train_data))
    train = train_data[index]
    obj = application(train, bin=node, n_tail=degree, last=degree, parallel=False, timer=True)
    generated_data = obj.generate(n=length, initialnodes=obj.lastnodes, how=how)
    return generated_data

index = random.randint(0, len(train_data))
target = target_data[index]
generated_data = zero_shot(name, lookback=l, length=len(target), degree=degree, node=node, how='uniform', index=index)
loss = mean_absolute_error(generated_data, target)
print(f"MAE of G4TS using only look-back window (Zero-Shot) on data {name} with lookback_window {l} is {loss} ")









name = random.choice(names)
l = 1440
train_data = ts1.data[name][l]['train_data']
target_data = ts1.data[name][l]['target']
node = int(random.choice(nodes[name]))
degree = int(random.choice(degrees[name]))

### Using random p percentage of available training data
p = random.random() # p is between 0 and 1
n_train = round(p*(len(train_data)-1)) #-1 is to make sure target data is not used in training.

target_index = random.randint(0, len(target_data))
target = target_data[target_index]
lookback_data = train_data[target_index]

indices = list(range(len(train_data)))
indices.remove(target_index)
train_index = np.random.choice(indices, n_train, replace=False)
train = []
for i in train_index:
    train.append(np.concatenate((train_data[i], target_data[i])))
train.append(train_data[target_index])

def forecasting(train_data, lookback_window_data, length, degree, node, how='uniform'):
    obj = application(train_data, bin=node, n_tail=degree, last=degree, parallel=False, timer=False)
    lookback_window = application(lookback_window_data, bin=node, n_tail=degree, last=degree, parallel=False, timer=False)
    generated_data = obj.generate(n=length, initialnodes= lookback_window.lastnodes, how=how)
    return generated_data

generated_data = forecasting(train_data=train, lookback_window_data=lookback_data, length=len(target), degree=degree, node=node, how='uniform')
loss = mean_absolute_error(generated_data, target)
print(f"MAE of G4TS using {p*100} percent of data {name} with lookback window size {l} is {loss} ")
