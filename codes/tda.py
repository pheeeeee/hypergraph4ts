#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 11:56:59 2024

@author: piprober
"""


import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import math

import torch


from github.counting import *
from utils import  khat, is_list_of_sequences

"""
data = np.random.normal(size=100000)
data = np.sin(data)
g = g4ts(data, bin=5)
edges = g.compute(n_tail=10)

"""


class CustomVectorSpace:
    basis = []
    def __init__(self, basis):
        self.add_basis(basis)
    
    def find_index(self, vectors):
        not_in_basis = []
        ind_vec = []
        for i,vec in enumerate(vectors):
            if vec not in self.basis:
                not_in_basis.append(vec)
                del vectors[i]
            else:
                ind_vec.append((self.basis.index(vec),vec))
        return ind_vec
    
    
    @classmethod
    def add_basis(cls, new_vectors):
        """add basis

        Args:
            new_vectors (list): list of new basis as a tuple.
        """
        new_vectors = list(set(new_vectors))
        cls.basis = sorted(cls.basis + new_vectors)
        
        
    def coefficient(clc,coefficient=None, basis=None):
        """
        With given coefficient and basis input, it aligns the basis in order and returns a tuple (coefficient, basis)
        It generates an element of the vector space with corresponding coordinates.
        
        Input
        coefficient (list)
        basis (list of tuples / tuple)
        """
        if basis is None:
            basis = clc.basis
        elif isinstance(basis,tuple):
            basis = [basis]
            if coefficient is None:
                coefficient = [1]
        if coefficient is None:
            coefficient = [0]*len(basis)
            
        assert len(coefficient) == len(basis), "Number of coefficients must match the number of basis"
        true_basis = clc.basis
        true_coefficient = [0]*len(true_basis)
        for i, base in enumerate(basis):
            if base in true_basis:
                idx = true_basis.index(base)
                true_coefficient[idx] += coefficient[i]
        return (true_coefficient, true_basis)
    
    def remove_element_from_tuple(t,i):
        """Remove the element at index i from the tuple t."""
        return t[:i] + t[i+1:]
    
    def boundary_operator(self, edge):
        """computes a boundary of edge.

        Args:
            edge (tuple): edge in self.edges
        """
        vectors = []
        sign = []
        for i in range(len(edge)):
            vectors.append( self.remove_element_from_tuple(edge, i))
            if i%2 ==0:
                sign.append(1)
            else:
                sign.append(-1)
                
        return (sign, vectors)
    
    
    def inner_product(self, element1, element2):
        """It computes the inner product of element1 and element2

        Args:
            element1 (tuple): it's a tuple. (coefficients, basis) Get it from .coefficient()
            element2 (tuple): it's a tuple. (coefficients, basis)
        """
        element1 = self.coefficient(element1)
        element2 = self.coefficient(element2)
        assert element1[1] == element2[1], "Basis of two elements are different from each other."
        coef1, coef2 = np.array(element1[0]), np.array(element2[0])
        output = np.dot(coef1, coef2)
        return output

        
        
    

def remove_element_from_tuple(t, i):
    """Remove the element at index i from the tuple t."""
    return t[:i] + t[i+1:]

def boundary_operator(edge):
    """computes a boundary of edge.

    Args:
        edge (tuple): edge in self.edges
    """
    vectors = []
    sign = []
    for i in range(len(edge)):
        vectors.append( remove_element_from_tuple(edge, i))
        if i%2 ==0:
            sign.append(1)
        else:
            sign.append(-1)
    return (sign, vectors)


def make_vertex_original(edges):
    edges = {tuple(int(x) for x in key): value for key, value in edges.items()}
    return edges

def to_lower_tail(edges,n_tail):
    edges = make_vertex_original(edges) #Should I make vertices distinct and then get the lower degree
    new_edge = {}
    
    # 원본 딕셔너리의 모든 키와 값을 순회합니다.
    for key, value in edges.items():
        n = len(key)
        # 가능한 모든 subtuple에 대해 반복합니다.
        for i in range(n - n_tail ):
            subtuple = key[i:i + n_tail+1]
            if subtuple in new_edge:
                # subtuple이 이미 존재하면 값을 더합니다.
                new_edge[subtuple] += value
            else:
                # 새로운 subtuple을 딕셔너리에 추가합니다.
                new_edge[subtuple] = value
                
    return new_edge

def find_indices(input_list):
    # Dictionary to hold element as key and their indices as value
    index_map = {}
    # Iterate through the list and populate the dictionary
    for index, element in enumerate(input_list):
        if element in index_map:
            index_map[element].append(index)
        else:
            index_map[element] = [index]
    
    # Filter and return only those entries that have more than one index (duplicates)
    return index_map




class F_complex(CustomVectorSpace):
    def __init__(self, edges, p=None):
        assert isinstance(edges, dict), "edges should be dictionary of ((v1,v2):w_12)"
        if p is not None:
            assert p < len(list(edges.keys())[0]), "p should be smaller or equal than the n_tail"
        else:
            p = int(len(list(edges.keys())[0])-1)
        self.order = p
        edges = to_lower_tail(edges,p)
        self.basis = list(edges.keys())
        #super().__init__(basis = list(set(edges.keys())))

        
class Omega_complex(CustomVectorSpace):
    def __init__(self,  edges, p=None):
        assert isinstance(edges, dict), "edges should be dictionary of ((v1,v2):w_12)"
        if p is None:
            p = int(len(list(edges.keys())[0])-1)
        else:
            assert p < len(list(edges.keys())[0]), "p should be smaller or equal than the n_tail"
        self.order = p
        edges = to_lower_tail(edges, p)
        
        Fp = F_complex(edges, p=p)
        Fp_1 = F_complex(edges, p=p-1)
        
        new_basis=[]
        
        if p != 1:
            for base in Fp.basis:
                signs, vectors = boundary_operator(base)
                vectors = sorted(vectors)
                index_map = find_indices(vectors)
                count = 1
                for ver, ind in index_map.items():
                    c = sum([signs[i] for i in ind])
                    if c != 0:
                        if ver not in Fp_1.basis:
                            count = count*0
                            break
                        else:
                            count = count*1
                if count==1:
                    new_basis.append(base)
        else:
            new_basis=Fp.basis
        
        self.basis = list(set(new_basis))
        #super().__init__(basis = new_basis)



class hyperdigraph_homology():
    def __init__(self, edges): #  , device_id=3):
        self.edges = edges
        #self.device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
            
        
    def adjoint_boundary_operator_as_matrix(self,omega1, omega2):
        """Returns the adjoint of the Matrix representation of boundary operator from basis1 to basis2.
            (i.e. d_{p}(b1) = A(b2) : we are computing A. A is adjoint of d_p )
        Args:
            omega1 (OmegaComplex type): Omega Complex object 
            omega2 (OmegaComplex type): Omega Complex object
        """
        basis1 = omega1.basis
        basis2 = omega2.basis
        
        assert omega1.order == omega2.order + 1, "Basis1 should be ONE dim higher than Basis2."
        n1, n2 = len(basis1), len(basis2)
        linear_operator = torch.rand(n1, n2)
        for i, base1 in enumerate(basis1):
            sign, vectors = boundary_operator(base1)
            (true_coefficient1, true_basis1) = omega2.coefficient(sign, vectors)
            linear_operator[i,:] = torch.tensor(true_coefficient1)
        return linear_operator

    def adjoint_boundary_operator_p_order_as_matrix_from_edges(self, p):
        edges = self.edges
        omega1 = Omega_complex(edges=edges, p=p)
        omega2 = Omega_complex(edges=edges, p=p-1)
        linear_operator = self.adjoint_boundary_operator_as_matrix(omega1, omega2)
        return linear_operator


    def laplacian_matrix(self, p):
        if p > 0:
            B_p_star = self.adjoint_boundary_operator_p_order_as_matrix_from_edges( p=p)
            B_p_star = np.array(B_p_star)
            B_p = np.conjugate(B_p_star.T)
            B_p1_star = self.adjoint_boundary_operator_p_order_as_matrix_from_edges( p=p+1)
            B_p1_star = np.array(B_p1_star)
            B_p1 = np.conjugate(B_p1_star.T)
            
            B = B_p1 @ B_p1_star + B_p_star @ B_p

        else:
            B_p1_star = self.adjoint_boundary_operator_p_order_as_matrix_from_edges( p=p+1)
            B_p1_star = np.array(B_p1_star)
            B_p1 = np.conjugate(B_p1_star.T)
            
            B = B_p1 @ B_p1_star

        return B


    def Betti_Number(self, p, epsilon=0.001):
        A = self.laplacian_matrix( p)    
        eigenvalues, eigenvectors = np.linalg.eig(A)
        count = 0
        for eig in eigenvalues:
            if eig<=epsilon:
                count += 1
        return count
    
    def spectrum(self, p, epsilon=0.001):
        A = self.laplacian_matrix( p)    
        eigenvalues, eigenvectors = np.linalg.eig(A)
        eigenvalues = [element for element in eigenvalues if element < epsilon]
        return sorted(eigenvalues)
        

    def hard_weight_persistent_homology(self, max_dim, a=0, b=1, dt=0.1, spectrum=False):
        """Computes Persistent Homology

        Args:
            max_dim (int): the max dimension of the homology you want to compute.
            a (int, optional): starting survival threshold . Defaults to 0.
            b (float between 0 and 1, optional): Ending survival threshold
            dt (float, optional): The change from a to b. Defaults to 0.01.
        """
        edges = self.edges
        assert max_dim < len(list(edges.keys())[0]), "max_dim should be smaller or equal than the n_tail"
        filtrations = np.arange(a, b, dt)
        Betti_Numbers = {}
        spectrums = {}
        for p in range(max_dim):
            Betti_Numbers['beta'+str(p)] = np.empty(len(filtrations))
            if spectrum:
                spectrums[p] = {}
            for i,t in enumerate(filtrations):
                if t == 0:
                    edges_filt = {key:0 for key in edges.keys()}
                else:
                    edges_filt = {key: (val if val > 1/t else 0) for key, val in edges.items()}
                homology = hyperdigraph_homology(edges_filt)
                Betti_Numbers['beta'+str(p)][i] = homology.Betti_Number(p=p)
                if spectrum:
                    spectrums[p][t] = homology.spectrum(p=p)
        
        if spectrum:
            return Betti_Numbers, spectrums
        else:
            return Betti_Numbers
        
    
    def soft_weight_persistent_homology(self, max_dim, filtrations, spectrum=False):
        """Computes Persistent Homology

        Args:
            max_dim (int): the max dimension of the homology you want to compute.
            filtration (list) : list of filtration t number from 0 to 1. ex) [0,0.1, 0.5,0.9]
        """
        edges = self.edges
        assert max_dim < len(list(edges.keys())[0]), "max_dim should be smaller or equal than the n_tail"
        Betti_Numbers = {}
        spectrums = {}
        for p in range(max_dim):
            Betti_Numbers['beta'+str(p)] = np.empty(len(filtrations))
            if spectrum:
                spectrums[p] = {}
            for i,t in enumerate(filtrations):
                if t == 0:
                    edges_filt = {key:0 for key in edges.keys()}
                else:
                    edges_filt = {key: (val if val > 1/t else 0) for key, val in edges.items()}
                homology = hyperdigraph_homology(edges_filt)
                Betti_Numbers['beta'+str(p)][i] = homology.Betti_Number(p=p)
                if spectrum:
                    spectrums[p][t] = homology.spectrum(p=p)
        
        if spectrum:
            return Betti_Numbers, spectrums
        else:
            return Betti_Numbers
    
"""
#Plot the Betti Numbers
h = hyperdigraph_homology(edges)
betti = h.hard_weight_persistent_homology(8)

for i, beta in enumerate(list(betti.keys())):
    x = np.arange(0,1,0.01)
    fig, ax = plt.subplots()
    ax.step(x, betti[beta])
    ax.set_title('Persistence Diagram of '+str(beta))
    ax.xlabel('Filtration')
    x.ylabel('Betti Number')
    fig.savefig(f'//home/pheeeeee/g4ts/tda/{str(beta)}.png')"""

    



    

"""


from itertools import combinations
        
def compute_persistence(edges):
    #Compute persistence pairs for a list of simplices sorted by filtration value.
    birth = {}
    death = {}
    for simplex, time in edges:
        if len(simplex) == 1:
            birth[simplex] = time
        else:
            for subsimplex in combinations(simplex, len(simplex) - 1):
                if subsimplex in death:
                    continue
                death[subsimplex] = time
    # Find unmatched births (infinite persistence)
    persistence = [(b, death.get(b, float('inf'))) for b in birth]
    return [p for p in persistence if p[1] != float('inf')]
"""
    
        
        
        

