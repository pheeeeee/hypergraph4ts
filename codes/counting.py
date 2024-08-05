#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 11:54:53 2024

@author: piprober
"""


import pickle
import numpy as np
import torch
import random
import string

import numba
from numba import njit, prange, cuda

import torch

from collections import Counter
from multiprocessing import Pool
import os
from collections.abc import Iterable


from copy import deepcopy

#import hypernetx as hnx

from utils import *

import time 


def count_sequences(sequence, k):
    # 길이 k의 부분 수열을 저장할 카운터 생성
    subseq_counter = Counter()
    
    # 전체 수열을 순회하면서 길이 k의 부분 수열을 카운트
    for i in range(len(sequence) - k + 1):
        subseq = tuple(sequence[i:i+k])  # 부분 수열을 추출 (튜플로 변환하여 해시 가능하게 함)
        subseq_counter[subseq] += 1

    return dict(subseq_counter)

def count_sequences_4parallel(args):
    sequence, k = args
    
    # 길이 k의 부분 수열을 저장할 카운터 생성
    subseq_counter = Counter()
    
    # 전체 수열을 순회하면서 길이 k의 부분 수열을 카운트
    for i in range(len(sequence) - k + 1):
        subseq = tuple(sequence[i:i+k])  # 부분 수열을 추출 (튜플로 변환하여 해시 가능하게 함)
        subseq_counter[subseq] += 1

    return subseq_counter

def parallel_count_sequences(sequence, k, num_processes=None):
    # Determine the chunk size for each process
    n = len(sequence)
    if num_processes is None:
        num_processes = os.cpu_count()  # Default to the number of CPUs available

    chunk_size = (n - k + 1) // num_processes
    pool = Pool(processes=num_processes)
    results = pool.map(count_sequences_4parallel, 
                       [(sequence[i * chunk_size : (i + 1) * chunk_size], k) for i in range(num_processes)])

    # Combine the results from all processes
    combined_counter = Counter()
    for counter in results:
        combined_counter.update(counter)

    return dict(combined_counter)


        
        
class g4ts:
    def __init__(self, data, bin,  initial=0, last=0, device = 'cpu', parallel = True, fixbin=False, timer=False):
        """
        Args:
            data (numpy ndarray or list): timeseries data 
            n_tail (int): number of tails of each edge.
            bin (int or np): 
            initial (int) : The number of initial history of nodes saved
            last (int) : The number of last history of nodes saved
        """
        self.data = data
        self.device = device
        self.parallel = parallel
        
        self.n = len(data)
        self.M = np.max(data)
        self.m = np.min(data)
        
        self.timer = timer
        
        
            
        if not fixbin:
            if isinstance(bin, (int,float)):
                bin = int(bin)
                bin_len = (self.M-self.m)/bin
                bin_num = bin
                nodes = np.arange(bin_num)
                bin = []
                for i in range(bin_num-1):
                    bin.append(self.m + (i+1)*bin_len)
                
            elif isinstance(bin, np.ndarray):
                bin = bin[bin > self.m]
                bin = bin[bin < self.M]
                bin = np.sort(bin)
                bin_num = len(bin) + 1
                nodes = np.arange(bin_num)
        else:
            bin_num = len(bin) + 1
            nodes = np.arange(bin_num)
        
        self.bin = bin
        self.bin_num = bin_num
        self.nodes = nodes
        
        
        if parallel:
            self.nodehistory = self.node_history_parallel()
            
        elif device == 'cpu':
            self.nodehistory = self.node_history_cpu()
            
        else:
            assert torch.cuda.is_available(), "Cuda is not available."
            
            if isinstance(device, int):
                cuda.select_device(int(device))  # Assuming the device numbering starts at 0
            elif  device[:-1]=='cuda':
                cuda.select_device(int(device[-1]))

            # Transfer data to GPU
            d_signal = cuda.to_device(self.data)
            d_bin = cuda.to_device(self.bin)
            d_nodes = cuda.to_device(self.nodes)
            node_history = np.empty(self.n, dtype=np.int32)
            d_node_history = cuda.to_device(node_history)  # node_history needs to be on GPU before kernel execution
            
                        
            # Define the number of threads and blocks
            threads_per_block = 256
            blocks_per_grid = (d_signal.size + threads_per_block - 1) // threads_per_block

            # Launch the kernel
            self.node_history_gpu[blocks_per_grid, threads_per_block](data = d_signal, bin = d_bin, nodes = d_nodes, node_history = d_node_history)
            # Copy result back to host
            node_history = d_node_history.copy_to_host()
            self.nodehistory = node_history
    
        initialnodes = self.nodehistory[:initial]
        self.initialnodes = initialnodes
        
        lastnodes = self.nodehistory[-last:]
        self.lastnodes = lastnodes
        

    #@njit(parallel=True)
    def node_history_parallel(self):
        node_history = np.empty(self.n, dtype=np.int32)
        
        for i in prange(self.n):
            if self.data[i] == self.M:
                node_history[i] = self.nodes[-1]
            else:
                count = 0
                for threshold in self.bin:
                    if self.data[i] < threshold:
                        break
                    count += 1
                node_history[i] = count
                
        return node_history
    
    def node_history_cpu(self):
        node_history = np.empty(self.n, dtype=np.int32)

        # Mask where signal equals M
        mask_M = self.data == self.M
        node_history[mask_M] = int(self.nodes[-1])  # Assuming nodes[-1] can be cast to int and is defined

        # Handle where signal is not M
        mask_not_M = ~mask_M

        # For each element in signal that is not M, find the first bin that is greater
        # Use searchsorted to find the first index in 'bin' where elements should be inserted to maintain order
        indices = np.searchsorted(self.bin, self.data[mask_not_M], side='right')
        node_history[mask_not_M] = indices
        
        return node_history
    
    @cuda.jit
    def node_history_gpu(self, data, bin , nodes, node_history):
        idx = cuda.grid(1)
        if idx < data.size:
            if data[idx] == self.M:
                node_history[idx] = nodes[-1]
            else:
                count = 0
                for threshold in bin:
                    if data[idx] < threshold:
                        break
                    count += 1
                node_history[idx] = count  
                

    def compute(self, n_tail, n_workers = 1, print_computing_time=False):
        """_summary_

        Args:
            n_tail (int): number of tail for each hyperedge
            n_workers (_type_, optional): _description_. Defaults to 1. We can make counting easy.

        Returns:
            dictionary : ex) {(2, 3, 5, 2): 1, (3, 5, 2, 2): 1, (5, 2, 2, 2): 1, (2, 2, 2, 2): 10}
        """
        self.n_tail = n_tail
        if n_workers == 1:
            start_time = time.perf_counter()
            edges = count_sequences(self.nodehistory, k = n_tail+1)
            end_time = time.perf_counter()
        else:
            start_time = time.perf_counter()
            edges = parallel_count_sequences(self.nodehistory, k = n_tail+1)
            end_time = time.perf_counter()
            
        duration = end_time - start_time
        if self.timer:
            if print_computing_time:
                print(f"Training completed in {duration} seconds. Recored as .timer attr")
            self.duration = duration
            
        if not isinstance(edges, dict):
            edges = dict(edges)
        
        self.edges = edges
        self.edgetype = 'original'
        return edges
    
    def to_lower_tail(self,n_tail,overwrite=False):
        new_dict = {}
        
        # 원본 딕셔너리의 모든 키와 값을 순회합니다.
        for key, value in self.weights.items():
            n = len(key)
            # 가능한 모든 subtuple에 대해 반복합니다.
            for i in range(n - n_tail + 1):
                subtuple = key[i:i + n_tail]
                if subtuple in new_dict:
                    # subtuple이 이미 존재하면 값을 더합니다.
                    new_dict[subtuple] += value
                else:
                    # 새로운 subtuple을 딕셔너리에 추가합니다.
                    new_dict[subtuple] = value
        
        if overwrite:
            self.weights = new_dict 
                   
        return new_dict
        
    
    
    def tograph(self, type = 'hypergraph'):
        if not hasattr(self, 'edges'):
            edges = self.compute(n_tail=self.n_tail)
            self.edges = edges        
            
        edgenames = {}
        edgeweights = {}
        for index, name in enumerate(edges.keys()):
            edgenames['e'+str(index)] = name  #ex. {'e1': (3,4,5)}
            edgeweights['e'+str(index)] = edges[name] #ex. {'e1' : 3, ... }
                
        if type == 'hypergraph':
            G = hnx.Hypergraph(edgenames)
            for edge_name, weight in edgeweights.items():
                G.edges[edge_name].weight = weight
                
        self.edgenames = edgenames
        self.edgeweights = edgeweights
        print("The attribute .edges returns a dictionary of real edge name and weights.")
        
        return G
    
    def make_each_vertex_distinct(self, to_graph = False, how='decimal'):
        """ Convert each vertex at different location of edge into different vertex.
        Args:
            how (str, optional): if 'alphabet', each vertex at edge is turned distinguished using alphabets
                                 if 'decimal' or smt else, each vertex at edge is distinguished using the decimal point. 
                                 so 1.1 means the first vertex of the edge is 1. 1.4 means the fourth vertex of the edge is 1. Of course it should start from 1.
                                 So it should be 9-jin bup. for ex, 1.10 should not exist. Instead, it should be 1.8 , 1.9, 1.11, 1.12, ... 
                                 Defaults to 'alphabet'.
            to_graph (Boolean, optional): if true, hnx hypergraph is also changed.
        """
        
        self.edgetype = how
        
        if not hasattr(self, 'edges'):
            edges = self.compute(n_tail=self.n_tail)
            self.edges = edges
        
        if not hasattr(self, 'edgenames'):
            edgenames = {}
            edgeweights = {}
            for index, name in enumerate(edges.keys()):
                edgenames['e'+str(index)] = name  #ex. {'e1': (3,4,5)}
                edgeweights['e'+str(index)] = edges[name] #ex. {'e1' : 3, ... }
            self.edgenames = edgenames
            self.edgeweights = edgeweights
        
        if how == 'alphabet':
            new_edge = {}
            new_edgenames = {}
            for edgenumber, edgee in self.edgenames.items():
                edgee = list(edgee)
                alphabet = infinite_alphabetic()
                for i in range(len(edgee)):
                    edgee[i] = next(alphabet) + str(edgee[i])
                new_edgenames[edgenumber] = tuple(edgee)
            
            for edge, weights in self.edges.items():
                edge = list(edge)
                alphabet = infinite_alphabetic()
                for i in range(len(edge)):
                    edge[i] = next(alphabet) + str(edge[i])
                new_edge[tuple(edge)] = weights
                
                
        else:
            new_edge = {}
            new_edgenames = {}
            for edgenumber, edge in self.edgenames.items():
                edge = list(edge)
                decimal = infinite_decimal()
                for i in range(len(edge)):
                    edge[i] = edge[i] + next(decimal)
                new_edgenames[edgenumber] = tuple(edge)
                
            for edge, weights in self.edges.items():
                edge = list(edge)
                decimal = infinite_decimal()
                for i in range(len(edge)):
                    edge[i] = edge[i] + next(decimal)
                new_edge[tuple(edge)] = weights
        
        self.edges = new_edge
        self.edgenames = new_edgenames
        print('Vertex Location distinguished')
        
        if to_graph:
            G = hnx.Hypergraph(new_edgenames)
            for edge_name, weight in self.edgeweights.items():
                G.edges[edge_name].weight = weight
            print("Graph is also changed.")
            return G
    
    def make_each_vertex_original(self, to_graph = False):
    
        """ Convert each vertex at different location of edge into different vertex.
    Args:
        how (str, optional): if 'alphabet', each vertex at edge is turned distinguished using alphabets
                                if 'decimal' or smt else, each vertex at edge is distinguished using the decimal point. 
                                so 1.1 means the first vertex of the edge is 1. 1.4 means the fourth vertex of the edge is 1. Of course it should start from 1.
                                So it should be 9-jin bup. for ex, 1.10 should not exist. Instead, it should be 1.8 , 1.9, 1.11, 1.12, ... 
                                Defaults to 'alphabet'.
        to_graph (Boolean, optional): if true, hnx hypergraph is also changed.
    """
    
        if self.edgetype == 'decimal':
            self.edges = {tuple(int(x) for x in key): value for key, value in self.edges.items()}
            self.edgenames = {key : tuple(int(x) for x in value) for key, value in self.edgenames.items()}

        elif self.edgetype == 'alphabet':
            import re
            self.edges = {tuple(int(re.search(r'\d+', s).group()) for s in key): value for key, value in self.edges.items()}
            self.edgenames = {key : tuple(int(re.search(r'\d+', s).group()) for s in value) for key, value in self.edgenames.items()}
    
        self.edgetype = 'original'
        
        if to_graph:
            G = hnx.Hypergraph(self.edgenames)
            for edge_name, weight in self.edgeweights.items():
                G.edges[edge_name].weight = weight
            print("Graph is also changed.")
            return G
        
        
        
        
        
        
        
        
        
        
        

class multig4ts:
    def __init__(self, data, bin, n_tail, initial=0, last=0, device = 'cpu', parallel = True, timer=False):
        
        assert self.is_list_of_sequences(data), "There is Only One data sequence"
        # Verify data structure
        assert isinstance(data, list), "Data should be a list"
        assert all(isinstance(arr, np.ndarray) for arr in data), "Each element of data should be a NumPy array"

        self.dataset = data
        n_data = len(data)
        flattened_data = [arr.flatten() for arr in data]
        concatenated_data = np.concatenate(flattened_data)
        self.M = np.max(concatenated_data)
        self.m = np.min(concatenated_data)
        if isinstance(bin, (int,float)):
            bin = int(bin)
            bin_len = (self.M-self.m)/bin
            bin_num = bin
            nodes = np.arange(bin_num)
            bin = []
            for i in range(bin_num-1):
                bin.append(self.m + (i+1)*bin_len)
        elif isinstance(bin, np.ndarray):
            bin = bin[(bin > self.m) & (bin < self.M)]
            print("Filtered bin values:", bin)
            bin = np.sort(bin)
            bin_num = len(bin) + 1
            nodes = np.arange(bin_num)
        self.bin = bin
        self.bin_num = bin_num
        self.nodes = nodes
        
        obj = g4ts(data[0],bin=self.bin, initial=initial, last=last, device=device, parallel=parallel, fixbin=True, timer=timer)
        if timer == True:
            duration = obj.timer
        
        ini = [obj.initialnodes]
        la = [obj.lastnodes]
        edges = obj.compute(n_tail=n_tail)
        for idx in range(n_data-1):
            obj = g4ts(data[idx+1],bin=self.bin, initial=initial, last=last, device=device, parallel=parallel, fixbin=True, timer=timer)
            otheredges = obj.compute(n_tail=n_tail)
            ini.append(obj.initialnodes)
            la.append(obj.lastnodes)
            edges = {key: edges.get(key, 0) + otheredges.get(key, 0) for key in edges.keys() | otheredges.keys()}
            if timer == True:
                duration += obj.timer
        
        self.edges = edges
        self.initialnodes = ini
        self.lastnodes = la
        if timer == True:
            self.timer = duration
    
    def is_list_of_sequences(self, dataset):
        # Check if the dataset itself is a list
        if not isinstance(dataset, Iterable):
            return False
        
        # Check each element in the list to confirm it's a sequence
        for sequence in dataset:
            # Ensure the element is an iterable (list, tuple, etc.) but not string-like
            if not isinstance(sequence, (list, tuple, np.ndarray)) or isinstance(sequence, str):
                return False
        
        # If all elements are sequences, return True
        return True
    
    def tograph(self):
        edges = self.edges
        edgenames = {}
        edgeweights = {}
        for index, name in enumerate(edges.keys()):
            edgenames['e'+str(index)] = name  #ex. {'e1': (3,4,5)}
            edgeweights['e'+str(index)] = edges[name] #ex. {'e1' : 3, ... }
                

        G = hnx.Hypergraph(edgenames)
        for edge_name, weight in edgeweights.items():
            G.edges[edge_name].weight = weight
                
        self.edgenames = edgenames
        self.edgeweights = edgeweights
        print("The attribute .edges returns a dictionary of real edge name and weights.")
        
        return G
    
    def make_each_vertex_distinct(self, to_graph = False, how='decimal'):
        """ Convert each vertex at different location of edge into different vertex.
        Args:
            how (str, optional): if 'alphabet', each vertex at edge is turned distinguished using alphabets
                                 if 'decimal' or smt else, each vertex at edge is distinguished using the decimal point. 
                                 so 1.1 means the first vertex of the edge is 1. 1.4 means the fourth vertex of the edge is 1. Of course it should start from 1.
                                 So it should be 9-jin bup. for ex, 1.10 should not exist. Instead, it should be 1.8 , 1.9, 1.11, 1.12, ... 
                                 Defaults to 'alphabet'.
            to_graph (Boolean, optional): if true, hnx hypergraph is also changed.
        """
        if how == 'alphabet':
            new_edge = {}
            new_edgenames = {}
            for edgenumber, edgee in self.edgenames.items():
                edgee = list(edgee)
                alphabet = infinite_alphabetic()
                for i in range(len(edgee)):
                    edgee[i] = next(alphabet) + str(edgee[i])
                new_edgenames[edgenumber] = tuple(edgee)
            
            for edge, weights in self.edges.items():
                edge = list(edge)
                alphabet = infinite_alphabetic()
                for i in range(len(edge)):
                    edge[i] = next(alphabet) + str(edge[i])
                new_edge[tuple(edge)] = weights
                
                
        else:
            new_edge = {}
            new_edgenames = {}
            for edgenumber, edge in self.edgenames.items():
                edge = list(edge)
                decimal = infinite_decimal()
                for i in range(len(edge)):
                    edge[i] = edge[i] + next(decimal)
                new_edgenames[edgenumber] = tuple(edge)
                
            for edge, weights in self.edges.items():
                edge = list(edge)
                decimal = infinite_decimal()
                for i in range(len(edge)):
                    edge[i] = edge[i] + next(decimal)
                new_edge[tuple(edge)] = weights
        
        self.edges = new_edge
        self.edgenames = new_edgenames
        print('Vertex Location distinguished')
        
        if to_graph:
            G = hnx.Hypergraph(new_edgenames)
            for edge_name, weight in self.edgeweights.items():
                G.edges[edge_name].weight = weight
            print("Graph is also changed.")
            return G



