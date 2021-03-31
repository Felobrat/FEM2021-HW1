# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 20:16:55 2021

@author: fbrav
"""
import numpy as np
from numpy import array, pi, zeros
from beam import beam_element

xy = array([
    [0,0],
    [4,3],
    [9,3],
    [9,0]])

conec = array([
    [0,1],
    [1,2],
    [2,3]])

t = 20e-3
r = 400e-3

properties = {}
properties["E"] = 200e9
properties["A"] = np.pi*(r**2 - (r-t)**2)
properties["I"] = np.pi*(r**4/4-(r-t)**4/4)
properties["qx"] = 1.0
properties["qy"] = -5.0
 
Nnodes = xy.shape[0]
Nelems = conec.shape[0]

NDOFs_node = 3
NDOFs = NDOFs_node*Nnodes 

K = zeros((NDOFs, NDOFs))
f = zeros((NDOFs, 1))

for e in range(Nelems):
    ni = conec[e,0]
    nj = conec[e,1]
    
    print(f"e = {e} ni = {ni} nj = {nj}")
    
    xy_e = xy[[ni, nj], :]
    
    ke, fe = beam_element(xy_e, properties)
    
    d = [3*ni, 3*ni+1, 3*ni+2, 3*nj, 3*nj+1, 3*nj+2] #global DOFs from local DOFs
    
    #direct stiffness method
    for i in range(2*NDOFs_node):
        p = d[i]
        for j in range(2*NDOFs_node):
            q = d[j]
            K[p,q] += ke[i,j]
        f[p]+= fe[i]
        

print(f"K = {K}")
print(f"f = {f}")
        
    