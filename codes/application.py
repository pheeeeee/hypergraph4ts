#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import random
import string

from collections import Counter
from multiprocessing import Pool
import os
from collections.abc import Iterable

from copy import deepcopy

from codes.counting import *
from codes.utils import *

class application:
    def __init__(self,  data, bin, n_tail,  initial=0, last=0, device = 'cpu', parallel = False, timer=False, print_computing_time=False):
        self.data = data
        self.n_tail = n_tail
        self.device = device
        self.parallel = parallel
        self.initial = initial
        self.last = last
        assert timer - print_computing_time > -1, "Training time cannot be printed if timer is False."

        
        if self.is_list_of_sequences(data):
            G = multig4ts(data, bin=bin, n_tail = n_tail, initial=initial, last=last, device=device, parallel=parallel,timer=timer, print_computing_time=print_computing_time)
            self.edges = G.edges
        else:
            G = g4ts(data, bin=bin,  initial=initial, last=last, device=device, parallel=parallel, timer=timer)
            self.edges = G.compute(n_tail=self.n_tail, print_computing_time = print_computing_time)

        self.G = G
        self.bin = G.bin
        self.initialnodes = G.initialnodes
        self.lastnodes = G.lastnodes
        self.newnodes = np.array([])
        if timer:
            self.timer = G.duration
        
    def is_list_of_sequences(self,dataset):
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
    
        
    def generate_newnodes(self, n , initialnodes=None):
        if initialnodes is None:
            initialnodes = random.choices(list(self.edges.keys()), weights=self.edges.values(), k=1)[0]
        
        newnodes = np.empty(n+len(initialnodes))
        newnodes[:len(initialnodes)] = initialnodes
        newnodes = self.update_newnodes(newnodes = newnodes, initialnodes=initialnodes)
        self.newnodes = newnodes
        return newnodes
        
    def process_element(self, previous):
        """
        Args:
            edges (dictionary): key is edge, value is counting
            previous (tuple): _description_

        Returns:
            int: A following node after the 'previous' input given the edges.
        """
        previous = tuple(previous)            
        candidate, prob, score, reserve = [], [], [], []
        for edge in list(self.edges.keys()):
            if previous == edge[:-1]:
                candidate.append(edge[-1])
                prob.append(self.edges[edge])
            elif not candidate:
                reserve.append(edge)
                score.append(np.linalg.norm(np.array(edge[:-1]) - np.array(previous)))

        if not prob:
            min_value = min(score)
            min_indices = [index for index, value in enumerate(score) if value == min_value]
            candidate = [reserve[ind][-1] for ind in min_indices]
            prob = np.ones(len(candidate))

        prob = np.array(prob) / sum(prob)
        return int(random.choices(candidate, weights=prob, k=1)[0])

    def update_newnodes(self, newnodes, initialnodes=None):
        n_tail = self.n_tail
        if initialnodes is None:
            initialnodes = newnodes[:n_tail]
        elif len(initialnodes) > n_tail:
            initialnodes = initialnodes[-n_tail:]
        elif len(initialnodes) < n_tail:
            AssertionError("There should be more Initial Values.")
        newnodes[:n_tail] = initialnodes
        n = len(newnodes)
        for i in range(n-n_tail):
            newnodes[n_tail+i] = self.process_element(newnodes[i:n_tail+i])
        newnodes = newnodes[n_tail:]
        self.newnodes = newnodes
        return newnodes.astype(int)
    
    def generate(self, n, initialnodes, how='average', inference_time=False):
        """
        It generates new data of length n.
        It converts a seq of nodes into real data list.
        newnodes is numpy array of integers.
        initial (list) is a list of initial nodes.
        """
        #Now generate into continuous ones. 
        newnodes = self.newnodes
        if len(newnodes) == 0:
            inferece_node_start = time.perf_counter()
            newnodes = self.generate_newnodes(n=n, initialnodes=initialnodes)
            inferece_node_end = time.perf_counter()
            inference_node_time = inferece_node_end - inferece_node_start
        gendata = np.empty(len(newnodes))
        bins = self.bin
        bins.append(self.G.M)
        bins.append(self.G.m)
        bins = sorted(bins)
        
        if how=='average':
            for i in range(len(gendata)):
                gendata[i] = sum((bins[int(newnodes[i])],bins[int(newnodes[i]+1)]))/2
        elif how=='uniform':
            inference_start = time.perf_counter()
            for i in range(len(gendata)):
                gendata[i] = random.uniform(bins[int(newnodes[i])],bins[int(newnodes[i]+1)])
            inference_end = time.perf_counter()
        inference_timer = inference_end - inference_start
        inference_timer = inference_timer + inference_node_time
        if inference_time is True:
            print(f"Inference Time for time length {len(gendata)} is {inference_timer}")
        elif how=='median':
            print('You need to write code')
            #for i in range(len(gendata)):
            #    gendata[i] = random.uniform(bins[int(newnodes[i])],bins[int(newnodes[i]+1)])
        elif how=='mode':
            print('You need to write code')
            #for i in range(len(gendata)):
            #    gendata[i] = random.uniform(bins[int(newnodes[i])],bins[int(newnodes[i]+1)])
           
              
        return gendata
    
    
    def compute_betti(self, how='decimal'):
        self.graph = self.G.tograph()
        self.G.make_each_vertex_distinct()
        edges = self.G.edges
        
    



        
def simulate_markov_bridge(P, initial_state, final_state, n_steps):
    n_states = P.shape[0]
    current_state = initial_state
    path = [current_state]

    for step in range(1, n_steps):
        if step == n_steps - 1:
            next_state = final_state
        else:
            # Adjusted probabilities to gradually move towards the final state
            remaining_steps = n_steps - step
            # Temporary probabilities, modifying them to reach the final state
            probabilities = np.zeros(n_states)
            for i in range(n_states):
                probabilities[i] = P[current_state, i] * (P ** (remaining_steps - 1))[i, final_state]
            probabilities /= probabilities.sum()  # Normalize to make it a valid probability vector

        next_state = np.random.choice(np.arange(n_states), p=probabilities)
        path.append(next_state)
        current_state = next_state

    return path
        
        










