#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 11:55:42 2024

@author: piprober
"""


import numpy as np 
from collections.abc import Iterable



def infinite_alphabetic():
    from itertools import product
    import string

    # Start with single letters, then increase length
    length = 1
    while True:
        for s in product(string.ascii_lowercase, repeat=length):
            yield ''.join(s)
        length += 1

    
def digit_count(num):
    return len(str(abs(num)))


def truncate(number, digits):
    import math
    stepper = 10.0 ** digits
    return math.floor(number * stepper) / stepper

def infinite_decimal():
    length = int(1)
    while True:
        if length%10 != 0:
            deci = digit_count(length)
            output = int(length)*(10**(-int(deci)))
            output = truncate(output,int(deci))
            yield output
        length = int(length + 1)
        
def khat(seq, k):
    assert isinstance(seq, (tuple, list, np.ndarray)), "The input sequence should be tuple, list or np.ndarray"
    assert len(seq) > k, "Boundary Operator of degree k must have its input length longer than k. (ex. k=0, len(seq)=1)"
    
    sign = (-1)**k
    if isinstance(seq, tuple):
        seq = list(seq)
        del seq[k]
        return (sign,tuple(seq))
    
    elif isinstance(seq, list):
        del seq[k]
        return (sign, seq)
    
    else:
        seq = np.concatenate((seq[:k], seq[k+1:]))
        return (sign, seq)

def boundary_operator(seq):
    n = len(seq)
    coefficient = []
    basis = []
    for k in range(n):
        cop = copy(seq)
        kth_sign, output = khat(cop, k)
        coefficient.append(kth_sign)
        basis.append(output)
        
    return coefficient, basis

def is_list_of_sequences( dataset):
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
