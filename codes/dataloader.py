#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 11:58:25 2024

@author: piprober
"""


import pandas as pd
import numpy as np
import math

class dataloader:
    def __init__(self, series_data, lookback_window, prediction_window):
        """_summary_

        Args:        
        :param series_data: A dictionary of pandas Series.
        :param lookback_window: A list of integers for the look-back windows.
        :param prediction_window: A dictionary of integers for the prediction_window for each time series data.
        :param ratio: A tuple of two floats for the training/validation separation.
        """
        self.series_data = series_data
        self.lookback_window = lookback_window
        self.prediction_window = prediction_window
        self.data = {key: {} for key in series_data}
        self.names = list(series_data.keys())
        
        
    def length(self, name, lookback):
        return len(self.data[name][lookback]['normalized_data'])
        
    def getid(self, name, lookback, idx, type = 'all'):
        """
        Args:
            name (string or list of strs): name of time series
            lookback (int or list of int): lookback window length
            idx (int or list of int): index
        """
        if isinstance(name, str):
            name = [name]
        if isinstance(lookback, int):
            lookback = [lookback]
        
        data = {}
        for key in name:
            data[key] = {}
            for l in lookback:
                if type == 'all':
                    data[key][l] = {
                        'data':self.data.get(key).get(l)['normalized_data'][idx],
                        'mean':self.data.get(key).get(l)['mean'][idx],
                        'std':self.data.get(key).get(l)['std'][idx]
                    }
                elif type == 'data':
                    data[key][l] = self.data.get(key).get(l)['normalized_data'][idx]
        return data
        
    
    def segment(self, split=True, overlap=0, drop_last=True, padding=False):
        """Segment all the time series into time windows."""
        
        for name, series in self.series_data.items():
            for l in self.lookback_window:
                h = self.prediction_window[name]
                length = l + h
                windows = self.segment_data(d=series, window_length=length, overlap=overlap, drop_last=drop_last, padding=padding)
                m = np.empty(len(windows))
                s = np.empty(len(windows))
                for id,w in enumerate(windows):
                    mean = np.mean(w[:l])
                    std = np.std(w[:l])
                    if math.isinf(std) or math.isnan(std) or (std==0):
                        continue
                    windows[id] = (w-mean)/std
                    m[id] = mean
                    s[id] = std
                if split:
                    lookback = [w[:l] for w in windows]
                    target = [w[l:] for w in windows]
                    self.data[name][l] ={           
                                        'normalized_data':windows,         
                                        'train_data': lookback,
                                        'target': target,
                                        'mean': m,
                                        'std': s
                                        }
                if not split:
                    self.data[name][l] = {
                    'normalized_data': windows,
                    'mean': m,
                    'std': s
                    }
        
        
    def segment_data(self,d, window_length, overlap=0, drop_last=True, padding=False):
        """
        Segments the data into windows of a specific length.

        Args:
            d (np.array): The input time series data as a NumPy array.
            window_length (int): The length of each window.
            overlap (int): The number of points to overlap between windows.
            drop_last (bool): If True, disregard the last window.
            padding (bool): If True, pad the last window with zeros if it's not long enough.

        Returns:
            np.array: An array of segmented windows.
        """
        data = np.array(d)
        # Initialize variables
        stride = window_length - overlap
        num_windows = (len(data) - overlap) // stride
        extra_points = (len(data) - overlap) % stride

        # Create windows
        windows = [data[i * stride:i * stride + window_length] for i in range(num_windows)]

        # Handle the last window
        if not drop_last:
            if extra_points > 0:
                last_window = data[-extra_points:] if overlap else data[-window_length:]
                if padding:
                    # Pad the last window if it's not the full length
                    last_window = np.pad(last_window, (mean(last_window), window_length - len(last_window)), 'constant')
                windows.append(last_window)

        return np.array(windows)
        
        
